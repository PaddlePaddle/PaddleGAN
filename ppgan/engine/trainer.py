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
from ..metric.psnr_ssim import calculate_psnr, calculate_ssim


class Trainer:
    def __init__(self, cfg):

        # build train dataloader
        self.train_dataloader = build_dataloader(cfg.dataset.train)

        if 'lr_scheduler' in cfg.optimizer:
            cfg.optimizer.lr_scheduler.step_per_epoch = len(
                self.train_dataloader)

        # build model
        self.model = build_model(cfg)
        # multiple gpus prepare
        if ParallelEnv().nranks > 1:
            self.distributed_data_parallel()

        self.logger = logging.getLogger(__name__)
        self.enable_visualdl = cfg.get('enable_visualdl', False)
        if self.enable_visualdl:
            import visualdl
            self.vdl_logger = visualdl.LogWriter(logdir=cfg.output_dir)

        # base config
        self.output_dir = cfg.output_dir
        self.epochs = cfg.epochs
        self.start_epoch = 1
        self.current_epoch = 1
        self.batch_id = 0
        self.global_steps = 0
        self.weight_interval = cfg.snapshot_config.interval
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.validate_interval = -1
        if cfg.get('validate', None) is not None:
            self.validate_interval = cfg.validate.get('interval', -1)
        self.cfg = cfg

        self.local_rank = ParallelEnv().local_rank

        # time count
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.epochs * self.steps_per_epoch

        self.time_count = {}
        self.best_metric = {}

    def distributed_data_parallel(self):
        strategy = paddle.distributed.prepare_context()
        for net_name, net in self.model.nets.items():
            self.model.nets[net_name] = paddle.DataParallel(net, strategy)

    def train(self):
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            start_time = step_start_time = time.time()
            for i, data in enumerate(self.train_dataloader):
                reader_cost_averager.record(time.time() - step_start_time)

                self.batch_id = i
                # unpack data from dataset and apply preprocessing
                # data input should be dict
                self.model.set_input(data)
                self.model.optimize_parameters()

                batch_cost_averager.record(time.time() - step_start_time,
                                           num_samples=self.cfg.get(
                                               'batch_size', 1))

                step_start_time = time.time()
                
                if i % self.log_interval == 0:
                    self.data_time = reader_cost_averager.get_average()
                    self.step_time = batch_cost_averager.get_average()
                    self.ips = batch_cost_averager.get_ips_average()
                    self.print_log()

                    reader_cost_averager.reset()
                    batch_cost_averager.reset()

                if i % self.visual_interval == 0:
                    self.visual('visual_train')

                self.global_steps += 1

            self.logger.info(
                'train one epoch use time: {:.3f} seconds.'.format(time.time() -
                                                                   start_time))
            if self.validate_interval > -1 and epoch % self.validate_interval:
                self.validate()
            self.model.lr_scheduler.step()
            if epoch % self.weight_interval == 0:
                self.save(epoch, 'weight', keep=-1)
            self.save(epoch)

    def validate(self):
        if not hasattr(self, 'val_dataloader'):
            self.val_dataloader = build_dataloader(self.cfg.dataset.val,
                                                   is_train=False)

        metric_result = {}

        for i, data in enumerate(self.val_dataloader):
            self.batch_id = i

            self.model.set_input(data)
            self.model.test()

            visual_results = {}
            current_paths = self.model.get_image_paths()
            current_visuals = self.model.get_current_visuals()

            for j in range(len(current_paths)):
                short_path = os.path.basename(current_paths[j])
                basename = os.path.splitext(short_path)[0]
                for k, img_tensor in current_visuals.items():
                    name = '%s_%s' % (basename, k)
                    visual_results.update({name: img_tensor[j]})
                if 'psnr' in self.cfg.validate.metrics:
                    if 'psnr' not in metric_result:
                        metric_result['psnr'] = calculate_psnr(
                            tensor2img(current_visuals['output'][j], (0., 1.)),
                            tensor2img(current_visuals['gt'][j], (0., 1.)),
                            **self.cfg.validate.metrics.psnr)
                    else:
                        metric_result['psnr'] += calculate_psnr(
                            tensor2img(current_visuals['output'][j], (0., 1.)),
                            tensor2img(current_visuals['gt'][j], (0., 1.)),
                            **self.cfg.validate.metrics.psnr)
                if 'ssim' in self.cfg.validate.metrics:
                    if 'ssim' not in metric_result:
                        metric_result['ssim'] = calculate_ssim(
                            tensor2img(current_visuals['output'][j], (0., 1.)),
                            tensor2img(current_visuals['gt'][j], (0., 1.)),
                            **self.cfg.validate.metrics.ssim)
                    else:
                        metric_result['ssim'] += calculate_ssim(
                            tensor2img(current_visuals['output'][j], (0., 1.)),
                            tensor2img(current_visuals['gt'][j], (0., 1.)),
                            **self.cfg.validate.metrics.ssim)

            self.visual('visual_val',
                        visual_results=visual_results,
                        step=self.batch_id)

            if i % self.log_interval == 0:
                self.logger.info('val iter: [%d/%d]' %
                                 (i, len(self.val_dataloader)))

        for metric_name in metric_result.keys():
            metric_result[metric_name] /= len(self.val_dataloader.dataset)

        self.logger.info('Epoch {} validate end: {}'.format(
            self.current_epoch, metric_result))

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(self.cfg.dataset.test,
                                                    is_train=False)

        # data[0]: img, data[1]: img path index
        # test batch size must be 1
        for i, data in enumerate(self.test_dataloader):
            self.batch_id = i

            self.model.set_input(data)
            self.model.test()

            visual_results = {}
            current_paths = self.model.get_image_paths()
            current_visuals = self.model.get_current_visuals()

            for j in range(len(current_paths)):
                short_path = os.path.basename(current_paths[j])
                basename = os.path.splitext(short_path)[0]
                for k, img_tensor in current_visuals.items():
                    name = '%s_%s' % (basename, k)
                    visual_results.update({name: img_tensor[j]})

            self.visual('visual_test',
                        visual_results=visual_results,
                        step=self.batch_id,
                        is_save_image=True)

            if i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' %
                                 (i, len(self.test_dataloader)))

    def print_log(self):
        losses = self.model.get_current_losses()
        message = 'Epoch: %d, iters: %d ' % (self.current_epoch, self.batch_id)

        message += '%s: %.6f ' % ('lr', self.current_learning_rate)

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            if self.enable_visualdl:
                self.vdl_logger.add_scalar(k, v, step=self.global_steps)

        if hasattr(self, 'step_time'):
            message += 'batch_cost: %.5f sec ' % self.step_time

        if hasattr(self, 'data_time'):
            message += 'reader_cost: %.5f sec ' % self.data_time

        if hasattr(self, 'ips'):
            message += 'ips: %.5f images/s ' % self.ips

        if hasattr(self, 'step_time'):
            cur_step = self.steps_per_epoch * (self.current_epoch -
                                               1) + self.batch_id
            eta = self.step_time * (self.total_steps - cur_step - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            message += f'eta: {eta_str}'

        # print the message
        self.logger.info(message)

    @property
    def current_learning_rate(self):
        for optimizer in self.model.optimizers.values():
            return optimizer.get_lr()

    def visual(self,
               results_dir,
               visual_results=None,
               step=None,
               is_save_image=False):
        """
        visual the images, use visualdl or directly write to the directory

        Parameters:
            results_dir (str)     --  directory name which contains saved images
            visual_results (dict) --  the results images dict
            step (int)            --  global steps, used in visualdl
            is_save_image (bool)  --  weather write to the directory or visualdl
        """
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        min_max = self.cfg.get('min_max', None)
        if min_max is None:
            min_max = (-1., 1.)
        image_num = self.cfg.get('image_num', None)
        if (image_num is None) or (not self.enable_visualdl):
            image_num = 1
        for label, image in visual_results.items():
            image_numpy = tensor2img(image, min_max, image_num)
            if (not is_save_image) and self.enable_visualdl:
                self.vdl_logger.add_image(
                    results_dir + '/' + label,
                    image_numpy,
                    step=step if step else self.global_steps,
                    dataformats="HWC" if image_num == 1 else "NCHW")
            else:
                if self.cfg.is_train:
                    msg = 'epoch%.3d_' % self.current_epoch
                else:
                    msg = ''
                makedirs(os.path.join(self.output_dir, results_dir))
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
            self.global_steps = self.steps_per_epoch * state_dicts['epoch']

        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])

        for opt_name, opt in self.model.optimizers.items():
            opt.set_state_dict(state_dicts[opt_name])

    def load(self, weight_path):
        state_dicts = load(weight_path)

        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])

    def close(self):
        """
        when finish the training need close file handler or other.

        """
        if self.enable_visualdl:
            self.vdl_logger.close()
