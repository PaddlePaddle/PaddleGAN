import paddle

from .discriminators.builder import build_discriminator
from .generators.builder import build_generator
from .base_model import BaseModel
from .builder import MODELS
from ..modules.init import init_weights
from ..solver import build_optimizer


import logging
TEST_MODE = False
if TEST_MODE:
    import numpy as np
    logging.warning('TEST MODE: run.py')
    fake_batch_size = 2
    fake_input = np.transpose(np.tile(np.load('/home/aistudio/work/src/img.npy')[:1, ...], (fake_batch_size, 1, 1, 1)).astype(np.float32)/255, (0, 3, 1, 2))  #Shape:[fake_batch_size, 3, 256, 256]
    
    
@MODELS.register()
class FirstOrderModel(BaseModel):
    def __init__(self, cfg):
        super(FirstOrderModel, self).__init__(cfg)
        
        # def local var
        self.input_data = None
        self.generated = None
        self.losses_generator = None
        
        # define networks
        generator_cfg = cfg.model.generator
        generator_cfg.update({'common_params': cfg.model.common_params})
        generator_cfg.update({'train_params': cfg.train_params})
        generator_cfg.update({'dis_scales': cfg.model.discriminator.discriminator_cfg.scales})
        self.nets['Gen_Full'] = build_generator(generator_cfg)
        discriminator_cfg = cfg.model.discriminator
        discriminator_cfg.update({'common_params': cfg.model.common_params})
        discriminator_cfg.update({'train_params': cfg.train_params})
        self.nets['Dis'] = build_discriminator(discriminator_cfg)
        
        # init params
        # it will reinit AADownSample param
        # init_weights(self.nets['Gen_Full'])
        # init_weights(self.nets['Dis'])
        
        if self.is_train:
            # TODO: Add loss
            self.losses = {}
            # define loss functions
            # self.criterionGAN = GANLoss(cfg.model.gan_mode)
            # self.criterionL1 = paddle.nn.L1Loss()

            # build optimizers
            # self.build_lr_scheduler()
            from paddle.optimizer.lr import MultiStepDecay
            lr_cfg = cfg.lr_scheduler
            self.kp_lr = MultiStepDecay(learning_rate=lr_cfg['lr_kp_detector'],
                                        milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            self.gen_lr = MultiStepDecay(learning_rate=lr_cfg['lr_generator'],
                                    milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            self.dis_lr = MultiStepDecay(learning_rate=lr_cfg['lr_discriminator'],
                                    milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            self.optimizers['optimizer_KP'] = build_optimizer(
                cfg.optimizer,
                self.kp_lr,
                parameter_list=self.nets['Gen_Full'].kp_extractor.parameters())
            self.optimizers['optimizer_Gen'] = build_optimizer(
                cfg.optimizer,
                self.gen_lr,
                parameter_list=self.nets['Gen_Full'].generator.parameters())
            self.optimizers['optimizer_Dis'] = build_optimizer(
                cfg.optimizer,
                self.dis_lr,
                parameter_list=self.nets['Dis'].parameters())

    def set_input(self, input):
        if TEST_MODE:
            logging.warning('TEST MODE: Input is Fixed')
            x = dict()
            x['driving'] = paddle.to_tensor(fake_input)
            x['source'] = paddle.to_tensor(fake_input)
            self.input_data = x
        else:
            self.input_data = input
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.losses_generator, self.generated = self.nets['Gen_Full'](self.input_data.copy(), self.nets['Dis'].discriminator)
        self.visual_items['generated'] = self.generated['prediction'].detach()
    
    def backward_G(self):
        loss_values = [val.sum() for val in self.losses_generator.values()]
        loss = paddle.add_n(loss_values)
        loss.backward()
        self.losses = dict(zip(self.losses_generator.keys(), loss_values))

    def backward_D(self):
        losses_discriminator = self.nets['Dis'](self.input_data.copy(), self.generated)
        loss_values = [val.mean() for val in losses_discriminator.values()]
        loss = paddle.add_n(loss_values)
        loss.backward()
        self.losses.update(dict(zip(losses_discriminator.keys(), loss_values)))

    def optimize_parameters(self):
        self.forward()
        
        # update G
        self.set_requires_grad(self.nets['Dis'], False)
        self.optimizers['optimizer_KP'].clear_grad()
        self.optimizers['optimizer_Gen'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_KP'].step()
        self.optimizers['optimizer_Gen'].step()

        # update D
        self.set_requires_grad(self.nets['Dis'], True)
        self.optimizers['optimizer_Dis'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_Dis'].step()
        if TEST_MODE:
            logging.warning('\n'+'\n'.join(['%s:%1.4f'%(k,v) for k,v,in self.losses.items()]))
            import pdb;pdb.set_trace()
