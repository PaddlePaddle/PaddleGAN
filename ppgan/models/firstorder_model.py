import logging

import paddle

from .base_model import BaseModel
from .builder import MODELS
from .discriminators.builder import build_discriminator
from .generators.builder import build_generator
from ..modules.init import init_weights
from ..solver import build_optimizer

TEST_MODE = False
if TEST_MODE:
    import numpy as np
    logging.warning('TEST MODE')
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
        # self.nets['Gen_Full'].pyramid
        # self.nets['Gen_Full'].vgg
        init_weights(self.nets['Gen_Full'].generator)
        init_weights(self.nets['Gen_Full'].kp_extractor)
        init_weights(self.nets['Dis'].discriminator)
        if 'pytorch_ckpt' in cfg.keys():
            ckpt_config = cfg.pytorch_ckpt
        else:
            ckpt_config = {
                'vgg19_model': '/home/aistudio/work/pre-trained/vgg19_np.npz',
                'generator': '/home/aistudio/work/pre-trained/mgif/G_param.npz',
                'discriminator': '/home/aistudio/work/pre-trained/mgif/D_param.npz',
                'kp': '/home/aistudio/work/pre-trained/mgif/KP_param.npz'
            }
        load_ckpt(ckpt_config, generator=self.nets['Gen_Full'].generator, optimizer_generator=None,
                  kp_detector=self.nets['Gen_Full'].kp_extractor, optimizer_kp_detector=None,
                  discriminator=self.nets['Dis'].discriminator, optimizer_discriminator=None,
                  vgg=self.nets['Gen_Full'].vgg)
        if self.is_train:
            # TODO: Add loss
            self.losses = {}
            # define loss functions

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
        if TEST_MODE:
            logging.warning('\n'+'\n'.join(['%s:%1.4f'%(k,v) for k,v,in self.losses.items()]))
            import pdb;pdb.set_trace()


def load_ckpt(ckpt_config, generator=None, optimizer_generator=None, kp_detector=None, optimizer_kp_detector=None,
              discriminator=None, optimizer_discriminator=None, vgg=None):
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    new_dict = lambda name, valu: dict([(k2, v1) for (k1,v1), (k2,v2) in zip(valu.items(), name.items())])
    if has_key('generator') and generator is not None:
        if ckpt_config['generator'][-3:] == 'npz':
            G_param = np.load(ckpt_config['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = dict([(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i])
            diff_num = np.array([list(i.shape) != list(j.shape) for i, j in
                                 zip(generator.state_dict().values(), G_param_clean.values())]).sum()
            if diff_num == 0:
                generator.set_state_dict(new_dict(generator.state_dict(), G_param_clean))
                logging.info('G is loaded from *.npz')
            else:
                logging.warning('G cannot load from *.npz')
        else:
            param, optim = paddle.fluid.load_dygraph(ckpt_config['generator'])
            generator.set_dict(param)
            if optim is not None and optimizer_generator is not None:
                optimizer_generator.set_state_dict(optim)
            else:
                logging.info('Optimizer of G is not loaded')
            logging.info('Generator is loaded from *.pdparams')
    if has_key('kp') and kp_detector is not None:
        if ckpt_config['kp'][-3:] == 'npz':
            KD_param = np.load(ckpt_config['kp'], allow_pickle=True)['arr_0'].item()
            KD_param_clean = dict([(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i])
            diff_num = np.array([list(i.shape) != list(j.shape) for i, j in
                                 zip(kp_detector.state_dict().values(), KD_param_clean.values())]).sum()
            if diff_num == 0:
                kp_detector.set_state_dict(new_dict(kp_detector.state_dict(), KD_param_clean))
                logging.info('KP is loaded from *.npz')
            else:
                logging.warning('KP cannot load from *.npz')
        else:
            param, optim = paddle.fluid.load_dygraph(ckpt_config['kp'])
            kp_detector.set_dict(param)
            if optim is not None and optimizer_kp_detector is not None:
                optimizer_kp_detector.set_state_dict(optim)
            else:
                logging.info('Optimizer of KP is not loaded')
            logging.info('KP is loaded from *.pdparams')
    if has_key('discriminator') and discriminator is not None:
        if ckpt_config['discriminator'][-3:] == 'npz':
            D_param = np.load(ckpt_config['discriminator'], allow_pickle=True)['arr_0'].item()
            if 'NULL Place' in ckpt_config['discriminator']:
                # 针对未开启spectral_norm的Fashion数据集模型
                ## fashion数据集的默认设置中未启用spectral_norm，但其官方ckpt文件中存在spectral_norm特有的参数 需要重排顺序
                ## 已提相关issue，作者回应加了sn也没什么影响 https://github.com/AliaksandrSiarohin/first-order-model/issues/264
                ## 若在配置文件中开启sn则可通过else语句中的常规方法读取，故现已在配置中开启sn。
                D_param_clean = [(i, D_param[i]) for i in D_param if
                                 'num_batches_tracked' not in i and 'weight_v' not in i and 'weight_u' not in i]
                for idx in range(len(D_param_clean) // 2):
                    if 'conv.bias' in D_param_clean[idx * 2][0]:
                        D_param_clean[idx * 2], D_param_clean[idx * 2 + 1] = D_param_clean[idx * 2 + 1], \
                                                                             D_param_clean[
                                                                                 idx * 2]
                parameter_clean = discriminator.parameters()
                for v, b in zip(parameter_clean, D_param_clean):
                    v.set_value(b[1])
            else:
                D_param_clean = list(D_param.items())
                parameter_clean = discriminator.parameters()
                assert len(D_param_clean) == len(parameter_clean)
                # resort
                ## PP:        [conv.weight,   conv.bias,          weight_u, weight_v]
                ## pytorch:   [conv.bias,     conv.weight_orig,   weight_u, weight_v]
                for idx in range(len(parameter_clean)):
                    if list(parameter_clean[idx].shape) == list(D_param_clean[idx][1].shape):
                        parameter_clean[idx].set_value(D_param_clean[idx][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'w_0' and D_param_clean[idx + 1][0].split('.')[
                        -1] == 'weight_orig':
                        parameter_clean[idx].set_value(D_param_clean[idx + 1][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'b_0' and D_param_clean[idx - 1][0].split('.')[
                        -1] == 'bias':
                        parameter_clean[idx].set_value(D_param_clean[idx - 1][1])
                    else:
                        logging.error('Error', idx)
            logging.info('Discriminator is loaded from *.npz')
        else:
            param, optim = paddle.fluid.load_dygraph(ckpt_config['discriminator'])
            discriminator.set_dict(param)
            if optim is not None and optimizer_discriminator is not None:
                optimizer_discriminator.set_state_dict(optim)
            else:
                logging.info('Optimizer of Discriminator is not loaded')
            logging.info('Discriminator is loaded from *.pdparams')
    if has_key('vgg19_model'):
        vggVarList = [i for i in vgg.parameters()]
        paramset = np.load(ckpt_config['vgg19_model'], allow_pickle=True)['arr_0']
        for var, v in zip(vggVarList, paramset):
            if list(var.shape) == list(v.shape):
                var.set_value(v)
            else:
                logging.warning('VGG19 cannot be loaded')
        logging.info('Pre-trained VGG19 is loaded from *.npz')
