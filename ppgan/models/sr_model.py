from collections import OrderedDict
import paddle
import paddle.nn as nn

from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..solver import build_optimizer
from .base_model import BaseModel
from .losses import GANLoss
from .builder import MODELS

import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from .builder import MODELS


@MODELS.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""
    def __init__(self, cfg):
        super(SRModel, self).__init__(cfg)

        self.model_names = ['G']

        self.netG = build_generator(cfg.model.generator)
        self.visual_names = ['lq', 'output', 'gt']

        self.loss_names = ['l_total']
        # define network
        # self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        # self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        # load_path = self.opt['path'].get('pretrain_model_g', None)
        # if load_path is not None:
        #     self.load_network(self.net_g, load_path,
        #                       self.opt['path']['strict_load'])
        self.optimizers = []
        if self.isTrain:
            self.criterionL1 = paddle.nn.L1Loss()

            self.build_lr_scheduler()
            self.optimizer_G = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.netG.parameters())
            self.optimizers.append(self.optimizer_G)
            # self.optimizer_D = build_optimizer(
            #     opt.optimizer,
            #     self.lr_scheduler,
            #     parameter_list=self.netD.parameters())

            # self.init_training_settings()

    # def init_training_settings(self):
    #     self.net_g.train()
    #     train_opt = self.opt['train']

    #     # define losses
    #     if train_opt.get('pixel_opt'):
    #         pixel_type = train_opt['pixel_opt'].pop('type')
    #         cri_pix_cls = getattr(loss_module, pixel_type)
    #         self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
    #             self.device)
    #     else:
    #         self.cri_pix = None

    #     if train_opt.get('perceptual_opt'):
    #         percep_type = train_opt['perceptual_opt'].pop('type')
    #         cri_perceptual_cls = getattr(loss_module, percep_type)
    #         self.cri_perceptual = cri_perceptual_cls(
    #             **train_opt['perceptual_opt']).to(self.device)
    #     else:
    #         self.cri_perceptual = None

    #     if self.cri_pix is None and self.cri_perceptual is None:
    #         raise ValueError('Both pixel and perceptual losses are None.')

    #     # set up optimizers and schedulers
    #     self.setup_optimizers()
    #     self.setup_schedulers()

    # def setup_optimizers(self):
    #     train_opt = self.opt['train']
    #     optim_params = []
    #     for k, v in self.net_g.named_parameters():
    #         if v.requires_grad:
    #             optim_params.append(v)
    #         else:
    #             logger = get_root_logger()
    #             logger.warning(f'Params {k} will not be optimized.')

    #     optim_type = train_opt['optim_g'].pop('type')
    #     if optim_type == 'Adam':
    #         self.optimizer_g = torch.optim.Adam(optim_params,
    #                                             **train_opt['optim_g'])
    #     else:
    #         raise NotImplementedError(
    #             f'optimizer {optim_type} is not supperted yet.')
    #     self.optimizers.append(self.optimizer_g)

    def set_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])
        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])
        self.image_paths = input['lq_path']
        # self.lq = data['lq'].to(self.device)
        # if 'gt' in data:
        #     self.gt = data['gt'].to(self.device)

    def forward(self):
        pass

    def test(self):
        """Forward function used in test time.
        """
        with paddle.no_grad():
            self.output = self.netG(self.lq)

    def optimize_parameters(self):
        self.optimizer_G.clear_grad()
        self.output = self.netG(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.criterionL1:
            l_pix = self.criterionL1(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.loss_l_total = l_total
        self.optimizer_G.step()
