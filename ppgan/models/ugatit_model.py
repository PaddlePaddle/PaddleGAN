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

import paddle
import paddle.nn as nn
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..modules.nn import RhoClipper
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class UGATITModel(BaseModel):
    """
    This class implements the UGATIT model, for learning image-to-image translation without paired data.

    UGATIT paper: https://arxiv.org/pdf/1907.10830.pdf
    """
    def __init__(self, cfg):
        """Initialize the CycleGAN class.

        Parameters:
            opt (config)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(UGATITModel, self).__init__(cfg)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        self.nets['genA2B'] = build_generator(cfg.model.generator)
        self.nets['genB2A'] = build_generator(cfg.model.generator)
        init_weights(self.nets['genA2B'])
        init_weights(self.nets['genB2A'])

        if self.is_train:
            # define discriminators
            self.nets['disGA'] = build_discriminator(cfg.model.discriminator_g)
            self.nets['disGB'] = build_discriminator(cfg.model.discriminator_g)
            self.nets['disLA'] = build_discriminator(cfg.model.discriminator_l)
            self.nets['disLB'] = build_discriminator(cfg.model.discriminator_l)
            init_weights(self.nets['disGA'])
            init_weights(self.nets['disGB'])
            init_weights(self.nets['disLA'])
            init_weights(self.nets['disLB'])

        if self.is_train:
            # define loss functions
            self.BCE_loss = nn.BCEWithLogitsLoss()
            self.L1_loss = nn.L1Loss()
            self.MSE_loss = nn.MSELoss()

            self.build_lr_scheduler()
            self.optimizers['optimizer_G'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['genA2B'].parameters() +
                self.nets['genB2A'].parameters())
            self.optimizers['optimizer_D'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['disGA'].parameters() +
                self.nets['disGB'].parameters() +
                self.nets['disLA'].parameters() +
                self.nets['disLB'].parameters())
            self.Rho_clipper = RhoClipper(0, 1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        mode = 'train' if self.is_train else 'test'
        AtoB = self.cfg.dataset[mode].direction == 'AtoB'

        if AtoB:
            if 'A' in input:
                self.real_A = paddle.to_tensor(input['A'])
            if 'B' in input:
                self.real_B = paddle.to_tensor(input['B'])
        else:
            if 'B' in input:
                self.real_A = paddle.to_tensor(input['B'])
            if 'A' in input:
                self.real_B = paddle.to_tensor(input['A'])

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'real_A'):
            self.fake_A2B, _, _ = self.nets['genA2B'](self.real_A)

            # visual
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_A2B'] = self.fake_A2B

        if hasattr(self, 'real_B'):
            self.fake_B2A, _, _ = self.nets['genB2A'](self.real_B)

            # visual
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_B2A'] = self.fake_B2A

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        self.nets['genA2B'].eval()
        self.nets['genB2A'].eval()
        with paddle.no_grad():
            self.forward()
            self.compute_visuals()

        self.nets['genA2B'].train()
        self.nets['genB2A'].train()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        def _criterion(loss_func, logit, is_real):
            if is_real:
                target = paddle.ones_like(logit)
            else:
                target = paddle.zeros_like(logit)
            return loss_func(logit, target)

        # forward
        # compute fake images and reconstruction images.
        self.forward()

        # update D
        self.optimizers['optimizer_D'].clear_grad()
        real_GA_logit, real_GA_cam_logit, _ = self.nets['disGA'](self.real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.nets['disLA'](self.real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.nets['disGB'](self.real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.nets['disLB'](self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.nets['disGA'](self.fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.nets['disLA'](self.fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.nets['disGB'](self.fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.nets['disLB'](self.fake_A2B)

        D_ad_loss_GA = _criterion(self.MSE_loss,
                                  real_GA_logit, True) + _criterion(
                                      self.MSE_loss, fake_GA_logit, False)

        D_ad_cam_loss_GA = _criterion(
            self.MSE_loss, real_GA_cam_logit, True) + _criterion(
                self.MSE_loss, fake_GA_cam_logit, False)

        D_ad_loss_LA = _criterion(self.MSE_loss,
                                  real_LA_logit, True) + _criterion(
                                      self.MSE_loss, fake_LA_logit, False)

        D_ad_cam_loss_LA = _criterion(
            self.MSE_loss, real_LA_cam_logit, True) + _criterion(
                self.MSE_loss, fake_LA_cam_logit, False)

        D_ad_loss_GB = _criterion(self.MSE_loss,
                                  real_GB_logit, True) + _criterion(
                                      self.MSE_loss, fake_GB_logit, False)

        D_ad_cam_loss_GB = _criterion(
            self.MSE_loss, real_GB_cam_logit, True) + _criterion(
                self.MSE_loss, fake_GB_cam_logit, False)

        D_ad_loss_LB = _criterion(self.MSE_loss,
                                  real_LB_logit, True) + _criterion(
                                      self.MSE_loss, fake_LB_logit, False)

        D_ad_cam_loss_LB = _criterion(
            self.MSE_loss, real_LB_cam_logit, True) + _criterion(
                self.MSE_loss, fake_LB_cam_logit, False)

        D_loss_A = self.cfg.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA +
                                          D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.cfg.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB +
                                          D_ad_loss_LB + D_ad_cam_loss_LB)

        Discriminator_loss = D_loss_A + D_loss_B
        Discriminator_loss.backward()
        self.optimizers['optimizer_D'].step()

        # update G
        self.optimizers['optimizer_G'].clear_grad()

        fake_A2B, fake_A2B_cam_logit, _ = self.nets['genA2B'](self.real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.nets['genB2A'](self.real_B)

        fake_A2B2A, _, _ = self.nets['genB2A'](fake_A2B)
        fake_B2A2B, _, _ = self.nets['genA2B'](fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.nets['genB2A'](self.real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.nets['genA2B'](self.real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.nets['disGA'](fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.nets['disLA'](fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.nets['disGB'](fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.nets['disLB'](fake_A2B)

        G_ad_loss_GA = _criterion(self.MSE_loss, fake_GA_logit, True)
        G_ad_cam_loss_GA = _criterion(self.MSE_loss, fake_GA_cam_logit, True)
        G_ad_loss_LA = _criterion(self.MSE_loss, fake_LA_logit, True)
        G_ad_cam_loss_LA = _criterion(self.MSE_loss, fake_LA_cam_logit, True)
        G_ad_loss_GB = _criterion(self.MSE_loss, fake_GB_logit, True)
        G_ad_cam_loss_GB = _criterion(self.MSE_loss, fake_GB_cam_logit, True)
        G_ad_loss_LB = _criterion(self.MSE_loss, fake_LB_logit, True)
        G_ad_cam_loss_LB = _criterion(self.MSE_loss, fake_LB_cam_logit, True)

        G_recon_loss_A = self.L1_loss(fake_A2B2A, self.real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, self.real_B)

        G_identity_loss_A = self.L1_loss(fake_A2A, self.real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, self.real_B)

        G_cam_loss_A = _criterion(self.BCE_loss,
                                  fake_B2A_cam_logit, True) + _criterion(
                                      self.BCE_loss, fake_A2A_cam_logit, False)

        G_cam_loss_B = _criterion(self.BCE_loss,
                                  fake_A2B_cam_logit, True) + _criterion(
                                      self.BCE_loss, fake_B2B_cam_logit, False)

        G_loss_A = self.cfg.adv_weight * (
            G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA
        ) + self.cfg.cycle_weight * G_recon_loss_A + self.cfg.identity_weight * G_identity_loss_A + self.cfg.cam_weight * G_cam_loss_A
        G_loss_B = self.cfg.adv_weight * (
            G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB
        ) + self.cfg.cycle_weight * G_recon_loss_B + self.cfg.identity_weight * G_identity_loss_B + self.cfg.cam_weight * G_cam_loss_B

        Generator_loss = G_loss_A + G_loss_B
        Generator_loss.backward()
        self.optimizers['optimizer_G'].step()

        # clip parameter of AdaILN and ILN, applied after optimizer step
        self.nets['genA2B'].apply(self.Rho_clipper)
        self.nets['genB2A'].apply(self.Rho_clipper)

        self.losses['discriminator_loss'] = Discriminator_loss
        self.losses['generator_loss'] = Generator_loss
