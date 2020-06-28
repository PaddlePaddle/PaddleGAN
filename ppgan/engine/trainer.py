import os
import time
import logging

from paddle.imperative import ParallelEnv

from ..datasets.builder import build_dataloader
from ..models.builder import build_model
from ..utils.visual import tensor2img, save_image
from ..utils.filesystems import save, load, makedirs


class Trainer:
    def __init__(self, cfg):

        # build train dataloader
        self.train_dataloader = build_dataloader(cfg.dataset.train)
        
        if 'lr_scheduler' in cfg.optimizer:
            cfg.optimizer.lr_scheduler.step_per_epoch = len(self.train_dataloader)
        
        # build model
        self.model = build_model(cfg)

        self.logger = logging.getLogger(__name__)
        # base config
        # self.timestamp = time.strftime('-%Y-%m-%d-%H-%M', time.localtime())
        self.output_dir = cfg.output_dir
        self.epochs = cfg.epochs
        self.start_epoch = 0
        self.current_epoch = 0
        self.batch_id = 0
        self.weight_interval = cfg.snapshot_config.interval
        self.log_interval = cfg.log_config.interval
        self.visual_interval = cfg.log_config.visiual_interval
        self.cfg = cfg

        self.local_rank = ParallelEnv().local_rank
    
    def train(self):
        
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            self.current_epoch = epoch
            for i, data in enumerate(self.train_dataloader):
                self.batch_id = i
                # unpack data from dataset and apply preprocessing
                self.model.set_input(data)
                self.model.optimize_parameters()

                if i % self.log_interval == 0:
                    self.print_log()
                    
                if i % self.visual_interval == 0:
                    self.visual('visual_train')

            self.logger.info('train one epoch time: {}'.format(time.time() - start_time))
            if epoch % self.weight_interval == 0:
                self.save(epoch, 'weight', keep=-1)
            self.save(epoch)

    def test(self):
        if not hasattr(self, 'test_dataloader'):
            self.test_dataloader = build_dataloader(self.cfg.dataset.test, is_train=False)

        # data[0]: img, data[1]: img path index
        # test batch size must be 1
        for i, data in enumerate(self.test_dataloader):
            self.batch_id = i
            # FIXME: dataloader not support map input, hard code now!!!
            if self.cfg.dataset.test.name == 'AlignedDataset':
                if self.cfg.dataset.test.direction == 'BtoA':
                    fake = self.model.test(data[1])
                else:
                    fake = self.model.test(data[0])
            elif self.cfg.dataset.test.name == 'SingleDataset':
                fake = self.model.test(data[0])
                
            current_paths = self.test_dataloader.dataset.get_path_by_indexs(data[-1])

            visual_results = {}
            for j in range(len(current_paths)):
                name = os.path.basename(current_paths[j])
                name = os.path.splitext(name)[0]

                visual_results.update({name + '_fakeB': fake[j]})
            visual_results.update({name + '_realA': data[1]})
            visual_results.update({name + '_realB': data[0]})
            # visual_results.update({'realB': data[1]})
            self.visual('visual_test', visual_results=visual_results)
            
            if i % self.log_interval == 0:
                self.logger.info('Test iter: [%d/%d]' % (i, len(self.test_dataloader)))

    def print_log(self):
        losses = self.model.get_current_losses()
        message = 'Epoch: %d, iters: %d ' % (self.current_epoch, self.batch_id)
        
        message += '%s: %.6f ' % ('lr', self.current_learning_rate)

        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        # print the message
        self.logger.info(message)

    @property
    def current_learning_rate(self):
        return self.model.optimizers[0].current_step_lr()

    def visual(self, results_dir, visual_results=None):
        self.model.compute_visuals()

        if visual_results is None:
            visual_results = self.model.get_current_visuals()

        if self.cfg.isTrain:
            msg = 'epoch%.3d_' % self.current_epoch
        else:
            msg = ''

        makedirs(os.path.join(self.output_dir, results_dir))
        for label, image in visual_results.items():
            image_numpy = tensor2img(image)
            img_path = os.path.join(self.output_dir, results_dir, msg + '%s.png' % (label))
            save_image(image_numpy, img_path)

    def save(self, epoch, name='checkpoint', keep=1):
        if self.local_rank != 0:
            return
            
        assert name in ['checkpoint', 'weight']

        state_dicts = {}
        save_filename = 'epoch_%s_%s.pkl' % (epoch, name)
        save_path = os.path.join(self.output_dir, save_filename)
        for net_name in self.model.model_names:
            if isinstance(net_name, str):
                net = getattr(self.model, 'net' + net_name)
                state_dicts['net' + net_name] = net.state_dict()

        if name == 'weight':
            save(state_dicts, save_path)
            return

        state_dicts['epoch'] = epoch

        for opt_name in self.model.optimizer_names:
            if isinstance(opt_name, str):
                opt = getattr(self.model, opt_name)
                state_dicts[opt_name] = opt.state_dict()

        save(state_dicts, save_path)

        if keep > 0:
            try:
                checkpoint_name_to_be_removed = os.path.join(self.output_dir, 
                                            'epoch_%s_%s.pkl' % (epoch - keep, name))
                if os.path.exists(checkpoint_name_to_be_removed):
                    os.remove(checkpoint_name_to_be_removed)

            except Exception as e:
                self.logger.info('remove old checkpoints error: {}'.format(e))

    def resume(self, checkpoint_path):
        state_dicts = load(checkpoint_path)
        if state_dicts.get('epoch', None) is not None:
            self.start_epoch = state_dicts['epoch'] + 1
        
        for name in self.model.model_names:
            if isinstance(name, str):
                net = getattr(self.model, 'net' + name)
                net.set_dict(state_dicts['net' + name])

        for name in self.model.optimizer_names:
            if isinstance(name, str):
                opt = getattr(self.model, name)
                opt.set_dict(state_dicts[name])

    def load(self, weight_path):
        state_dicts = load(weight_path)
        
        for name in self.model.model_names:
            if isinstance(name, str):
                net = getattr(self.model, 'net' + name)
                net.set_dict(state_dicts['net' + name])
 