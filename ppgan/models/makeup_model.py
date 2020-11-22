# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import vgg16
from paddle.utils.download import get_path_from_url
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss
from ..modules.init import init_weights
from ..solver import build_optimizer
from ..utils.image_pool import ImagePool
from ..utils.preprocess import *
from ..datasets.makeup_dataset import MakeupDataset

VGGFACE_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/vggface.pdparams'


@MODELS.register()
class MakeupModel(BaseModel):
    """
    PSGAN paper: https://arxiv.org/pdf/1909.06956.pdf
    """
    def __init__(self, cfg):
        """Initialize the PSGAN class.

        Parameters:
            cfg (dict)-- config of model.
        """
        super(MakeupModel, self).__init__(cfg)

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.nets['netG'] = build_generator(cfg.model.generator)
        init_weights(self.nets['netG'], init_type='xavier', init_gain=1.0)

        if self.is_train:  # define discriminators
            vgg = vgg16(pretrained=False)
            self.vgg = vgg.features
            cur_path = os.path.abspath(os.path.dirname(__file__))
            vgg_weight_path = get_path_from_url(VGGFACE_WEIGHT_URL, cur_path)
            param = paddle.load(vgg_weight_path)
            vgg.load_dict(param)

            self.nets['netD_A'] = build_discriminator(cfg.model.discriminator)
            self.nets['netD_B'] = build_discriminator(cfg.model.discriminator)
            init_weights(self.nets['netD_A'], init_type='xavier', init_gain=1.0)
            init_weights(self.nets['netD_B'], init_type='xavier', init_gain=1.0)

            self.fake_A_pool = ImagePool(
                cfg.dataset.train.pool_size
            )  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(
                cfg.dataset.train.pool_size
            )  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss(
                cfg.model.gan_mode)  #.to(self.device)  # define GAN loss.
            self.criterionCycle = paddle.nn.L1Loss()
            self.criterionIdt = paddle.nn.L1Loss()
            self.criterionL1 = paddle.nn.L1Loss()
            self.criterionL2 = paddle.nn.MSELoss()

            self.build_lr_scheduler()
            self.optimizers['optimizer_G'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netG'].parameters())
            self.optimizers['optimizer_DA'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netD_A'].parameters())
            self.optimizers['optimizer_DB'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netD_B'].parameters())

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = paddle.to_tensor(input['image_A'])
        self.real_B = paddle.to_tensor(input['image_B'])
        self.c_m = paddle.to_tensor(input['consis_mask'])
        self.P_A = paddle.to_tensor(input['P_A'])
        self.P_B = paddle.to_tensor(input['P_B'])
        self.mask_A_aug = paddle.to_tensor(input['mask_A_aug'])
        self.mask_B_aug = paddle.to_tensor(input['mask_B_aug'])
        self.c_m_t = paddle.transpose(self.c_m, perm=[0, 2, 1])
        if self.is_train:
            self.mask_A = paddle.to_tensor(input['mask_A'])
            self.mask_B = paddle.to_tensor(input['mask_B'])
            self.c_m_idt_a = paddle.to_tensor(input['consis_mask_idt_A'])
            self.c_m_idt_b = paddle.to_tensor(input['consis_mask_idt_B'])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A, amm = self.nets['netG'](self.real_A, self.real_B, self.P_A,
                                             self.P_B, self.c_m,
                                             self.mask_A_aug,
                                             self.mask_B_aug)  # G_A(A)
        self.fake_B, _ = self.nets['netG'](self.real_B, self.real_A, self.P_B,
                                           self.P_A, self.c_m_t,
                                           self.mask_A_aug,
                                           self.mask_B_aug)  # G_A(A)
        self.rec_A, _ = self.nets['netG'](self.fake_A, self.real_A, self.P_A,
                                          self.P_A, self.c_m_idt_a,
                                          self.mask_A_aug,
                                          self.mask_B_aug)  # G_A(A)
        self.rec_B, _ = self.nets['netG'](self.fake_B, self.real_B, self.P_B,
                                          self.P_B, self.c_m_idt_b,
                                          self.mask_A_aug,
                                          self.mask_B_aug)  # G_A(A)

        # visual
        self.visual_items['real_A'] = self.real_A
        self.visual_items['fake_B'] = self.fake_B
        self.visual_items['rec_A'] = self.rec_A
        self.visual_items['real_B'] = self.real_B
        self.visual_items['fake_A'] = self.fake_A
        self.visual_items['rec_B'] = self.rec_B

    def forward_test(self, input):
        '''
        not implement now
        '''
        return self.nets['netG'](input['image_A'], input['image_B'],
                                 input['P_A'], input['P_B'],
                                 input['consis_mask'], input['mask_A_aug'],
                                 input['mask_B_aug'])

    def test(self, input):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with paddle.no_grad():
            return self.forward_test(input)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.nets['netD_A'], self.real_B,
                                              fake_B)
        self.losses['D_A_loss'] = self.loss_D_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.nets['netD_B'], self.real_A,
                                              fake_A)
        self.losses['D_B_loss'] = self.loss_D_B

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.cfg.lambda_identity
        lambda_A = self.cfg.lambda_A
        lambda_B = self.cfg.lambda_B
        lambda_vgg = 5e-3
        # Identity loss
        if lambda_idt > 0:
            self.idt_A, _ = self.nets['netG'](self.real_A, self.real_A,
                                              self.P_A, self.P_A,
                                              self.c_m_idt_a, self.mask_A_aug,
                                              self.mask_B_aug)  # G_A(A)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_A) * lambda_A * lambda_idt
            self.idt_B, _ = self.nets['netG'](self.real_B, self.real_B,
                                              self.P_B, self.P_B,
                                              self.c_m_idt_b, self.mask_A_aug,
                                              self.mask_B_aug)  # G_A(A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_B) * lambda_B * lambda_idt

            # visual
            self.visual_items['idt_A'] = self.idt_A
            self.visual_items['idt_B'] = self.idt_B
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.nets['netD_A'](self.fake_A),
                                          True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.nets['netD_B'](self.fake_B),
                                          True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A,
                                                self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B,
                                                self.real_B) * lambda_B

        self.losses['G_A_adv_loss'] = self.loss_G_A
        self.losses['G_B_adv_loss'] = self.loss_G_B

        mask_A_lip = self.mask_A_aug[:, 0].unsqueeze(1)
        mask_B_lip = self.mask_B_aug[:, 0].unsqueeze(1)

        mask_A_lip_np = mask_A_lip.numpy().squeeze()
        mask_B_lip_np = mask_B_lip.numpy().squeeze()
        mask_A_lip_np, mask_B_lip_np, index_A_lip, index_B_lip = mask_preprocess(
            mask_A_lip_np, mask_B_lip_np)
        real_A = paddle.nn.clip((self.real_A + 1.0) / 2.0, 0.0, 1.0) * 255.0
        real_A_np = real_A.numpy().squeeze()
        real_B = paddle.nn.clip((self.real_B + 1.0) / 2.0, 0.0, 1.0) * 255.0
        real_B_np = real_B.numpy().squeeze()
        fake_A = paddle.nn.clip((self.fake_A + 1.0) / 2.0, 0.0, 1.0) * 255.0
        fake_A_np = fake_A.numpy().squeeze()
        fake_B = paddle.nn.clip((self.fake_B + 1.0) / 2.0, 0.0, 1.0) * 255.0
        fake_B_np = fake_B.numpy().squeeze()

        fake_match_lip_A = hisMatch(fake_A_np, real_B_np, mask_A_lip_np,
                                    mask_B_lip_np, index_A_lip)
        fake_match_lip_B = hisMatch(fake_B_np, real_A_np, mask_B_lip_np,
                                    mask_A_lip_np, index_B_lip)
        fake_match_lip_A = paddle.to_tensor(fake_match_lip_A)
        fake_match_lip_A.stop_gradient = True
        fake_match_lip_A = fake_match_lip_A.unsqueeze(0)
        fake_match_lip_B = paddle.to_tensor(fake_match_lip_B)
        fake_match_lip_B.stop_gradient = True
        fake_match_lip_B = fake_match_lip_B.unsqueeze(0)
        fake_A_lip_masked = fake_A * mask_A_lip
        fake_B_lip_masked = fake_B * mask_B_lip
        g_A_lip_loss_his = self.criterionL1(fake_A_lip_masked, fake_match_lip_A)
        g_B_lip_loss_his = self.criterionL1(fake_B_lip_masked, fake_match_lip_B)

        #skin
        mask_A_skin = self.mask_A_aug[:, 1].unsqueeze(1)
        mask_B_skin = self.mask_B_aug[:, 1].unsqueeze(1)

        mask_A_skin_np = mask_A_skin.numpy().squeeze()
        mask_B_skin_np = mask_B_skin.numpy().squeeze()
        mask_A_skin_np, mask_B_skin_np, index_A_skin, index_B_skin = mask_preprocess(
            mask_A_skin_np, mask_B_skin_np)

        fake_match_skin_A = hisMatch(fake_A_np, real_B_np, mask_A_skin_np,
                                     mask_B_skin_np, index_A_skin)
        fake_match_skin_B = hisMatch(fake_B_np, real_A_np, mask_B_skin_np,
                                     mask_A_skin_np, index_B_skin)
        fake_match_skin_A = paddle.to_tensor(fake_match_skin_A)
        fake_match_skin_A.stop_gradient = True
        fake_match_skin_A = fake_match_skin_A.unsqueeze(0)
        fake_match_skin_B = paddle.to_tensor(fake_match_skin_B)
        fake_match_skin_B.stop_gradient = True
        fake_match_skin_B = fake_match_skin_B.unsqueeze(0)
        fake_A_skin_masked = fake_A * mask_A_skin
        fake_B_skin_masked = fake_B * mask_B_skin
        g_A_skin_loss_his = self.criterionL1(fake_A_skin_masked,
                                             fake_match_skin_A)
        g_B_skin_loss_his = self.criterionL1(fake_B_skin_masked,
                                             fake_match_skin_B)

        #eye
        mask_A_eye = self.mask_A_aug[:, 2].unsqueeze(1)
        mask_B_eye = self.mask_B_aug[:, 2].unsqueeze(1)

        mask_A_eye_np = mask_A_eye.numpy().squeeze()
        mask_B_eye_np = mask_B_eye.numpy().squeeze()
        mask_A_eye_np, mask_B_eye_np, index_A_eye, index_B_eye = mask_preprocess(
            mask_A_eye_np, mask_B_eye_np)

        fake_match_eye_A = hisMatch(fake_A_np, real_B_np, mask_A_eye_np,
                                    mask_B_eye_np, index_A_eye)
        fake_match_eye_B = hisMatch(fake_B_np, real_A_np, mask_B_eye_np,
                                    mask_A_eye_np, index_B_eye)
        fake_match_eye_A = paddle.to_tensor(fake_match_eye_A)
        fake_match_eye_A.stop_gradient = True
        fake_match_eye_A = fake_match_eye_A.unsqueeze(0)
        fake_match_eye_B = paddle.to_tensor(fake_match_eye_B)
        fake_match_eye_B.stop_gradient = True
        fake_match_eye_B = fake_match_eye_B.unsqueeze(0)
        fake_A_eye_masked = fake_A * mask_A_eye
        fake_B_eye_masked = fake_B * mask_B_eye
        g_A_eye_loss_his = self.criterionL1(fake_A_eye_masked, fake_match_eye_A)
        g_B_eye_loss_his = self.criterionL1(fake_B_eye_masked, fake_match_eye_B)

        self.loss_G_A_his = (g_A_eye_loss_his + g_A_lip_loss_his +
                             g_A_skin_loss_his * 0.1) * 0.1
        self.loss_G_B_his = (g_B_eye_loss_his + g_B_lip_loss_his +
                             g_B_skin_loss_his * 0.1) * 0.1

        self.losses['G_A_his_loss'] = self.loss_G_A_his
        self.losses['G_B_his_loss'] = self.loss_G_B_his

        #vgg loss
        vgg_s = self.vgg(self.real_A)
        vgg_s.stop_gradient = True
        vgg_fake_A = self.vgg(self.fake_A)
        self.loss_A_vgg = self.criterionL2(vgg_fake_A,
                                           vgg_s) * lambda_A * lambda_vgg

        vgg_r = self.vgg(self.real_B)
        vgg_r.stop_gradient = True
        vgg_fake_B = self.vgg(self.fake_B)
        self.loss_B_vgg = self.criterionL2(vgg_fake_B,
                                           vgg_r) * lambda_B * lambda_vgg

        self.loss_rec = (self.loss_cycle_A * 0.2 + self.loss_cycle_B * 0.2 +
                         self.loss_A_vgg + self.loss_B_vgg) * 0.5
        self.loss_idt = (self.loss_idt_A + self.loss_idt_B) * 0.1

        self.losses['G_A_vgg_loss'] = self.loss_A_vgg
        self.losses['G_B_vgg_loss'] = self.loss_B_vgg
        self.losses['G_rec_loss'] = self.loss_rec
        self.losses['G_idt_loss'] = self.loss_idt

        # bg consistency loss
        mask_A_consis = paddle.cast(
            (self.mask_A == 0), dtype='float32') + paddle.cast(
                (self.mask_A == 10), dtype='float32') + paddle.cast(
                    (self.mask_A == 8), dtype='float32')
        mask_A_consis = paddle.unsqueeze(paddle.clip(mask_A_consis, 0, 1), 1)
        self.loss_G_bg_consis = self.criterionL1(
            self.real_A * mask_A_consis, self.fake_A * mask_A_consis) * 0.1

        # combined loss and calculate gradients

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_rec + self.loss_idt + self.loss_G_A_his + self.loss_G_B_his + self.loss_G_bg_consis
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(
            [self.nets['netD_A'], self.nets['netD_B']],
            False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.clear_gradients() #zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizers['optimizer_G'].minimize(
            self.loss_G)  #step()       # update G_A and G_B's weights
        self.optimizers['optimizer_G'].clear_gradients()
        # D_A and D_B
        self.set_requires_grad(self.nets['netD_A'], True)
        # self.optimizer_D.clear_gradients() #zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.optimizers['optimizer_DA'].minimize(
            self.loss_D_A)  #step()  # update D_A and D_B's weights
        self.optimizers['optimizer_DA'].clear_gradients()  #zero_g
        self.set_requires_grad(self.nets['netD_B'], True)

        self.backward_D_B()  # calculate graidents for D_B
        self.optimizers['optimizer_DB'].minimize(
            self.loss_D_B)  #step()  # update D_A and D_B's weights
        self.optimizers['optimizer_DB'].clear_gradients(
        )  #zero_grad()   # set D_A and D_B's gradients to zero
