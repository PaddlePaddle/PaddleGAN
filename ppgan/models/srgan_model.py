# import logging
from collections import OrderedDict
import paddle
import paddle.nn as nn
# import torch.nn.parallel as P
# from torch.nn.parallel import DataParallel, DistributedDataParallel
# import models.networks as networks
# import models.lr_scheduler as lr_scheduler
from .generators.builder import build_generator
from .base_model import BaseModel
from .losses import GANLoss
from .builder import MODELS
# logger = logging.getLogger('base')

@MODELS.register()
class SRGANModel(BaseModel):
    def __init__(self, cfg):
        super(SRGANModel, self).__init__(cfg)
        # if opt['dist']:
        #     self.rank = torch.distributed.get_rank()
        # else:
        #     self.rank = -1  # non dist training
        # train_opt = opt['train']

        # define networks and load pretrained models
        self.model_names = ['G']
        
        self.netG = build_generator(cfg.model.generator)
        self.visual_names = ['LQ', 'GT', 'fake_H']
        # self.netG = networks.define_G(opt).to(self.device)
        # if opt['dist']:
        #     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # else:
        #     self.netG = DataParallel(self.netG)

        if False:#self.is_train:
            self.netD = build_discriminator(cfg.model.discriminator)
            # if opt['dist']:
            #     self.netD = DistributedDataParallel(self.netD,
            #                                         device_ids=[torch.cuda.current_device()])
            # else:
            #     self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        # if self.is_train:
        #     pass
            # G pixel loss
            # if train_opt['pixel_weight'] > 0:
            #     l_pix_type = train_opt['pixel_criterion']
            #     if l_pix_type == 'l1':
            #         self.cri_pix = nn.L1Loss().to(self.device)
            #     elif l_pix_type == 'l2':
            #         self.cri_pix = nn.MSELoss().to(self.device)
            #     else:
            #         raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
            #     self.l_pix_w = train_opt['pixel_weight']
            # else:
            #     # logger.info('Remove pixel loss.')
            #     self.cri_pix = None

            # # G feature loss
            # if train_opt['feature_weight'] > 0:
            #     l_fea_type = train_opt['feature_criterion']
            #     if l_fea_type == 'l1':
            #         self.cri_fea = nn.L1Loss().to(self.device)
            #     elif l_fea_type == 'l2':
            #         self.cri_fea = nn.MSELoss().to(self.device)
            #     else:
            #         raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
            #     self.l_fea_w = train_opt['feature_weight']
            # else:
            #     logger.info('Remove feature loss.')
            #     self.cri_fea = None
            # if self.cri_fea:  # load VGG perceptual loss
            #     self.netF = networks.define_F(opt, use_bn=False).to(self.device)
            #     if opt['dist']:
            #         self.netF = DistributedDataParallel(self.netF,
            #                                             device_ids=[torch.cuda.current_device()])
            #     else:
            #         self.netF = DataParallel(self.netF)

            # # GD gan loss
            # self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            # self.l_gan_w = train_opt['gan_weight']
            # # D_update_ratio and D_init_iters
            # self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            # self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # # optimizers
            # # G
            # wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            # optim_params = []
            # for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            #     if v.requires_grad:
            #         optim_params.append(v)
            #     else:
            #         if self.rank <= 0:
            #             logger.warning('Params [{:s}] will not optimize.'.format(k))
            # self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
            #                                     weight_decay=wd_G,
            #                                     betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            # self.optimizers.append(self.optimizer_G)
            # # D
            # wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
            #                                     weight_decay=wd_D,
            #                                     betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            # self.optimizers.append(self.optimizer_D)

            # # schedulers
            # if train_opt['lr_scheme'] == 'MultiStepLR':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
            #                                              restarts=train_opt['restarts'],
            #                                              weights=train_opt['restart_weights'],
            #                                              gamma=train_opt['lr_gamma'],
            #                                              clear_state=train_opt['clear_state']))
            # elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             lr_scheduler.CosineAnnealingLR_Restart(
            #                 optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
            #                 restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            # else:
            #     raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            # self.log_dict = OrderedDict()

        # self.print_network()  # print network
        # self.load()  # load G and D if needed

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        # AtoB = self.opt.dataset.train.direction == 'AtoB'
        if 'A' in input:
            self.LQ = paddle.to_tensor(input['A'])
        if 'B' in input:
            self.GT = paddle.to_tensor(input['B'])
        if 'A_paths' in input:
            self.image_paths = input['A_paths']

    # def feed_data(self, data, need_GT=True):
    #     self.var_L = data['LQ'].to(self.device)  # LQ
    #     if need_GT:
    #         self.var_H = data['GT'].to(self.device)  # GT
    #         input_ref = data['ref'] if 'ref' in data else data['GT']
    #         self.var_ref = input_ref.to(self.device)

    def forward(self):
        self.fake_H = self.netG(self.LQ)

    def optimize_parameters(self, step):
        pass
    #     # G
    #     for p in self.netD.parameters():
    #         p.requires_grad = False

    #     self.optimizer_G.zero_grad()
    #     self.fake_H = self.netG(self.var_L.detach())

    #     l_g_total = 0
    #     if step % self.D_update_ratio == 0 and step > self.D_init_iters:
    #         if self.cri_pix:  # pixel loss
    #             l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
    #             l_g_total += l_g_pix
    #         if self.cri_fea:  # feature loss
    #             real_fea = self.netF(self.var_H).detach()
    #             fake_fea = self.netF(self.fake_H)
    #             l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
    #             l_g_total += l_g_fea

    #         pred_g_fake = self.netD(self.fake_H)
    #         if self.opt['train']['gan_type'] == 'gan':
    #             l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
    #         elif self.opt['train']['gan_type'] == 'ragan':
    #             pred_d_real = self.netD(self.var_ref).detach()
    #             l_g_gan = self.l_gan_w * (
    #                 self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
    #                 self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
    #         l_g_total += l_g_gan

    #         l_g_total.backward()
    #         self.optimizer_G.step()

    #     # D
    #     for p in self.netD.parameters():
    #         p.requires_grad = True

    #     self.optimizer_D.zero_grad()
    #     l_d_total = 0
    #     pred_d_real = self.netD(self.var_ref)
    #     pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
    #     if self.opt['train']['gan_type'] == 'gan':
    #         l_d_real = self.cri_gan(pred_d_real, True)
    #         l_d_fake = self.cri_gan(pred_d_fake, False)
    #         l_d_total = l_d_real + l_d_fake
    #     elif self.opt['train']['gan_type'] == 'ragan':
    #         l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
    #         l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
    #         l_d_total = (l_d_real + l_d_fake) / 2

    #     l_d_total.backward()
    #     self.optimizer_D.step()

    #     # set log
    #     if step % self.D_update_ratio == 0 and step > self.D_init_iters:
    #         if self.cri_pix:
    #             self.log_dict['l_g_pix'] = l_g_pix.item()
    #             # self.log_dict['l_g_mean_color'] = l_g_mean_color.item()
    #         if self.cri_fea:
    #             self.log_dict['l_g_fea'] = l_g_fea.item()
    #         self.log_dict['l_g_gan'] = l_g_gan.item()

    #     self.log_dict['l_d_real'] = l_d_real.item()
    #     self.log_dict['l_d_fake'] = l_d_fake.item()
    #     self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
    #     self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    # def test(self):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         self.fake_H = self.netG(self.var_L)
    #     self.netG.train()

    # def back_projection(self):
    #     lr_error = self.var_L - torch.nn.functional.interpolate(self.fake_H,
    #                                                             scale_factor=1/self.opt['scale'],
    #                                                             mode='bicubic',
    #                                                             align_corners=False)
    #     us_error = torch.nn.functional.interpolate(lr_error,
    #                                                scale_factor=self.opt['scale'],
    #                                                mode='bicubic',
    #                                                align_corners=False)
    #     self.fake_H += self.opt['back_projection_lamda'] * us_error
    #     torch.clamp(self.fake_H, 0, 1)

    # def test_chop(self):
    #     self.netG.eval()
    #     with torch.no_grad():
    #         self.fake_H = self.forward_chop(self.var_L)
    #     self.netG.train()

    # def forward_chop(self, *args, shave=10, min_size=160000):
    #     # scale = 1 if self.input_large else self.scale[self.idx_scale]
    #     scale = self.opt['scale']
    #     n_GPUs = min(torch.cuda.device_count(), 4)
    #     args = [a.squeeze().unsqueeze(0) for a in args]

    #     # height, width
    #     h, w = args[0].size()[-2:]
    #     # print('len(args)', len(args))
    #     # print('args[0].size()', args[0].size())

    #     top = slice(0, h//2 + shave)
    #     bottom = slice(h - h//2 - shave, h)
    #     left = slice(0, w//2 + shave)
    #     right = slice(w - w//2 - shave, w)
    #     x_chops = [torch.cat([
    #         a[..., top, left],
    #         a[..., top, right],
    #         a[..., bottom, left],
    #         a[..., bottom, right]
    #     ]) for a in args]
    #     # print('len(x_chops)', len(x_chops))
    #     # print('x_chops[0].size()', x_chops[0].size())

    #     y_chops = []
    #     if h * w < 4 * min_size:
    #         for i in range(0, 4, n_GPUs):
    #             x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
    #             # print(len(x))
    #             # print(x[0].size())
    #             y = P.data_parallel(self.netG, *x, range(n_GPUs))
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y):
    #                     y_chop.extend(_y.chunk(n_GPUs, dim=0))
    #     else:

    #         # print(x_chops[0].size())
    #         for p in zip(*x_chops):
    #             # print('len(p)', len(p))
    #             # print('p[0].size()', p[0].size())
    #             y = self.forward_chop(*p, shave=shave, min_size=min_size)
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[_y] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

    #     h *= scale
    #     w *= scale
    #     top = slice(0, h//2)
    #     bottom = slice(h - h//2, h)
    #     bottom_r = slice(h//2 - h, None)
    #     left = slice(0, w//2)
    #     right = slice(w - w//2, w)
    #     right_r = slice(w//2 - w, None)

    #     # batch size, number of color channels
    #     b, c = y_chops[0][0].size()[:-2]
    #     y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    #     for y_chop, _y in zip(y_chops, y):
    #         _y[..., top, left] = y_chop[0][..., top, left]
    #         _y[..., top, right] = y_chop[1][..., top, right_r]
    #         _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
    #         _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

    #     if len(y) == 1:
    #         y = y[0]

    #     return y

    # def get_current_log(self):
    #     return self.log_dict

    # def get_current_visuals(self, need_GT=True):
    #     out_dict = OrderedDict()
    #     out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
    #     out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
    #     if need_GT:
    #         out_dict['GT'] = self.var_H.detach()[0].float().cpu()
    #     return out_dict

    # def print_network(self):
    #     # Generator
    #     s, n = self.get_network_description(self.netG)
    #     if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
    #         net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
    #                                          self.netG.module.__class__.__name__)
    #     else:
    #         net_struc_str = '{}'.format(self.netG.__class__.__name__)
    #     if self.rank <= 0:
    #         logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
    #         logger.info(s)
    #     if self.is_train:
    #         # Discriminator
    #         s, n = self.get_network_description(self.netD)
    #         if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
    #                                                                 DistributedDataParallel):
    #             net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
    #                                              self.netD.module.__class__.__name__)
    #         else:
    #             net_struc_str = '{}'.format(self.netD.__class__.__name__)
    #         if self.rank <= 0:
    #             logger.info('Network D structure: {}, with parameters: {:,d}'.format(
    #                 net_struc_str, n))
    #             logger.info(s)

    #         if self.cri_fea:  # F, Perceptual Network
    #             s, n = self.get_network_description(self.netF)
    #             if isinstance(self.netF, nn.DataParallel) or isinstance(
    #                     self.netF, DistributedDataParallel):
    #                 net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
    #                                                  self.netF.module.__class__.__name__)
    #             else:
    #                 net_struc_str = '{}'.format(self.netF.__class__.__name__)
    #             if self.rank <= 0:
    #                 logger.info('Network F structure: {}, with parameters: {:,d}'.format(
    #                     net_struc_str, n))
    #                 logger.info(s)

    # def load(self):
    #     load_path_G = self.opt['path']['pretrain_model_G']
    #     if load_path_G is not None:
    #         logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
    #         self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    #     load_path_D = self.opt['path']['pretrain_model_D']
    #     if self.opt['is_train'] and load_path_D is not None:
    #         logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
    #         self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    # def save(self, iter_step):
    #     self.save_network(self.netG, 'G', iter_step)
    #     self.save_network(self.netD, 'D', iter_step)
