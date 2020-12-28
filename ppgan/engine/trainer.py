#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import copy

import logging
import datetime

import paddle
from paddle.distributed import ParallelEnv

from ..datasets.builder import build_dataloader
from ..models.builder import build_model
from ..utils.visual import tensor2img, save_image
from ..utils.filesystem import makedirs, save, load
from ..utils.timer import TimeAverager


class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 1

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class Trainer:
    """
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               model.setup_lr_schedulers                  ||    lr_scheduler
    #                     |                                    ||
    #               model.setup_optimizers                     ||    optimizers
    #                     |                                    ||
    #     train loop (model.setup_input + model.train_iter)    ||    train loop
    #                     |                                    ||
    #         print log (model.get_current_losses)             ||
    #                     |                                    ||
    #         save checkpoint (model.nets)                     \/
    """
    def __init__(self, cfg):

        # build model
        self.model = build_model(cfg.model)
        # multiple gpus prepare
        if ParallelEnv().nranks > 1:
            self.distributed_data_parallel()

        # build train dataloader
        self.train_dataloader = build_dataloader(cfg.dataset.train)
        self.iters_per_epoch = len(self.train_dataloader)

        # build lr scheduler
        # TODO: has a better way?
        if 'lr_scheduler' in cfg and 'iters_per_epoch' in cfg.lr_scheduler:
            cfg.lr_scheduler.iters_per_epoch = self.iters_per_epoch
        self.lr_schedulers = self.model.setup_lr_schedulers(cfg.lr_scheduler)

        # build optimizers
        self.optimizers = self.model.setup_optimizers(self.lr_schedulers,
                                                      cfg.optimizer)

        # build metrics
        self.metrics = None
        validate_cfg = cfg.get('validate', None)
        if validate_cfg and 'metrics' in validate_cfg:
            self.metrics = self.model.setup_metrics(validate_cfg['metrics'])

        self.logger = logging.getLogger(__name__)

        # base config
        self.output_dir = cfg.output_dir
        self.epochs = cfg.get('epochs', None)
        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

        self.start_epoch = 1
        self.current_epoch = 1
        self.current_iter = 1
        self.inner_iter = 1
        self.weight_interval = cfg.snapshot_config.interval
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.validate_interval = -1
        if cfg.get('validate', None) is not None:
            self.validate_interval = cfg.validate.get('interval', -1)
        self.cfg = cfg

        self.local_rank = ParallelEnv().local_rank

        self.time_count = {}
        self.best_metric = {}

    def distributed_data_parallel(self):
        strategy = paddle.distributed.prepare_context()
        for net_name, net in self.model.nets.items():
            self.model.nets[net_name] = paddle.DataParallel(net, strategy)

    def train(self):
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()

        iter_loader = IterLoader(self.train_dataloader)

        while self.current_iter < (self.total_iters + 1):
            self.current_epoch = iter_loader.epoch
            self.inner_iter = self.current_iter % self.iters_per_epoch

            start_time = step_start_time = time.time()
            data = next(iter_loader)
            reader_cost_averager.record(time.time() - step_start_time)
            # unpack data from dataset and apply preprocessing
            # data input should be dict
            self.model.setup_input(data)
            self.model.train_iter(self.optimizers)

            batch_cost_averager.record(time.time() - step_start_time,
                                       num_samples=self.cfg.get(
                                           'batch_size', 1))
            if self.current_iter % self.log_interval == 0:
                self.data_time = reader_cost_averager.get_average()
                self.step_time = batch_cost_averager.get_average()
                self.ips = batch_cost_averager.get_ips_average()
                self.print_log()

                reader_cost_averager.reset()
                batch_cost_averager.reset()

            if self.current_iter % self.visual_interval == 0:
                self.visual('visual_train')

            step_start_time = time.time()

            self.model.lr_scheduler.step()

            if self.by_epoch:
                temp = self.current_epoch
            else:
                temp = self.current_iter
            if self.validate_interval > -1 and temp % self.validate_interval == 0:
                self.test()

            if temp % self.weight_interval == 0:
                self.save(temp, 'weight', keep=-1)
                self.save(temp)

            self.current_iter += 1

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(self.cfg.dataset.test,
                                                    is_train=False,
                                                    distributed=False)

        if self.metrics:
            for metric in self.metrics.values():
                metric.reset()

        # data[0]: img, data[1]: img path index
        # test batch size must be 1
        for i, data in enumerate(self.test_dataloader):

            self.model.setup_input(data)
            self.model.test_iter(metrics=self.metrics)

            visual_results = {}
            current_paths = self.model.get_image_paths()
            current_visuals = self.model.get_current_visuals()

            for j in range(len(current_paths)):
                short_path = os.path.basename(current_paths[j])
                basename = os.path.splitext(short_path)[0]
                for k, img_tensor in current_visuals.items():
                    name = '%s_%s' % (basename, k)
                    visual_results.update({name: img_tensor[j]})

            self.visual('visual_test', visual_results=visual_results)
            if i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' %
                                 (i, len(self.test_dataloader)))

        if self.metrics:
            for metric_name, metric in self.metrics.items():
                self.logger.info("Metric {}: {:.4f}".format(
                    metric_name, metric.accumulate()))

    def print_log(self):
        losses = self.model.get_current_losses()

        message = ''
        if self.by_epoch:
            message += 'Epoch: %d/%d, iter: %d/%d ' % (
                self.current_epoch, self.epochs, self.inner_iter,
                self.iters_per_epoch)
        else:
            message += 'Iter: %d/%d ' % (self.current_iter, self.total_iters)

        message += f'lr: {self.current_learning_rate:.3e} '

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        if hasattr(self, 'step_time'):
            message += 'batch_cost: %.5f sec ' % self.step_time

        if hasattr(self, 'data_time'):
            message += 'reader_cost: %.5f sec ' % self.data_time

        if hasattr(self, 'ips'):
            message += 'ips: %.5f images/s ' % self.ips

        if hasattr(self, 'step_time'):
            eta = self.step_time * (self.total_iters - self.current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            message += f'eta: {eta_str}'

        # print the message
        self.logger.info(message)

    @property
    def current_learning_rate(self):
        for optimizer in self.model.optimizers.values():
            return optimizer.get_lr()

    def visual(self, results_dir, visual_results=None):
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        if self.cfg.is_train:
            msg = 'epoch%.3d_' % self.current_epoch
        else:
            msg = ''

        makedirs(os.path.join(self.output_dir, results_dir))
        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)

        for label, image in visual_results.items():

            image_numpy = tensor2img(image, min_max)
            img_path = os.path.join(self.output_dir, results_dir,
                                    msg + '%s.png' % (label))
            save_image(image_numpy, img_path)

    def save(self, epoch, name='checkpoint', keep=1):
        if self.local_rank != 0:
            return

        assert name in ['checkpoint', 'weight']

        state_dicts = {}
        save_filename = 'epoch_%s_%s.pdparams' % (epoch, name)
        save_path = os.path.join(self.output_dir, save_filename)
        for net_name, net in self.model.nets.items():
            state_dicts[net_name] = net.state_dict()

        if name == 'weight':
            save(state_dicts, save_path)
            return

        state_dicts['epoch'] = epoch

        for opt_name, opt in self.model.optimizers.items():
            state_dicts[opt_name] = opt.state_dict()

        save(state_dicts, save_path)

        if keep > 0:
            try:
                checkpoint_name_to_be_removed = os.path.join(
                    self.output_dir,
                    'epoch_%s_%s.pdparams' % (epoch - keep, name))
                if os.path.exists(checkpoint_name_to_be_removed):
                    os.remove(checkpoint_name_to_be_removed)

            except Exception as e:
                self.logger.info('remove old checkpoints error: {}'.format(e))

    def resume(self, checkpoint_path):
        state_dicts = load(checkpoint_path)
        if state_dicts.get('epoch', None) is not None:
            self.start_epoch = state_dicts['epoch'] + 1

        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])

        for opt_name, opt in self.model.optimizers.items():
            opt.set_state_dict(state_dicts[opt_name])

    def load(self, weight_path):
        state_dicts = load(weight_path)

        for net_name, net in self.model.nets.items():
            if net_name in state_dicts:
                net.set_state_dict(state_dicts[net_name])
                self.logger.info(
                    'Loaded pretrained weight for net {}'.format(net_name))
            else:
                self.logger.warning(
                    'Can not find state dict of net {}. Skip load pretrained weight for net {}'
                    .format(net_name, net_name))
