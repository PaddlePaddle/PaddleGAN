import paddle
from paddle.optimizer.lr import MultiStepDecay

from .base_model import BaseModel
from .builder import MODELS
from .discriminators.builder import build_discriminator
from .generators.builder import build_generator
from ..modules.init import init_weights
from ..solver import build_optimizer


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
        init_weights(self.nets['Gen_Full'].generator)
        init_weights(self.nets['Gen_Full'].kp_extractor)
        init_weights(self.nets['Dis'].discriminator)
        # A pre-trained vgg19 model should be loaded to self.nets['Gen_Full'].generator.vgg before training
        
        if self.is_train:
            # define loss functions
            self.losses = {}
            
            # build optimizers
            lr_cfg = cfg.lr_scheduler
            self.kp_lr = MultiStepDecay(learning_rate=lr_cfg['lr_kp_detector'],
                                        milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            self.gen_lr = MultiStepDecay(learning_rate=lr_cfg['lr_generator'],
                                    milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            self.dis_lr = MultiStepDecay(learning_rate=lr_cfg['lr_discriminator'],
                                    milestones=lr_cfg['epoch_milestones'], gamma=0.1)
            
            class lr_scheduler():
                def __init__(self, kp_lr, gen_lr, dis_lr):
                    self.kp_lr = kp_lr
                    self.gen_lr = gen_lr
                    self.dis_lr = dis_lr
                
                def step(self):
                    self.kp_lr.step()
                    self.gen_lr.step()
                    self.dis_lr.step()
            self.lr_scheduler = lr_scheduler(self.kp_lr, self.gen_lr, self.dis_lr)
            
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
        self.input_data = input
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.losses_generator, self.generated = self.nets['Gen_Full'](self.input_data.copy(), self.nets['Dis'].discriminator)
        
        self.visual_items['driving_source_gen'] = paddle.concat((
            self.input_data['driving'].detach(),
            self.input_data['source'].detach(),
            self.generated['prediction'].detach()),
            axis=-1)
    
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
        self.set_requires_grad(self.nets['Dis'].discriminator, False)
        self.optimizers['optimizer_KP'].clear_grad()
        self.optimizers['optimizer_Gen'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_KP'].step()
        self.optimizers['optimizer_Gen'].step()

        # update D
        self.set_requires_grad(self.nets['Dis'].discriminator, True)
        self.optimizers['optimizer_Dis'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_Dis'].step()
