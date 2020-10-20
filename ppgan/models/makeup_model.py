# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F
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
import numpy as np
from .vgg import vgg16


@MODELS.register()
class MakeupModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_B', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        self.vgg = vgg16(pretrained=True)
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = build_generator(opt.model.generator)
        init_weights(self.netG, init_type='xavier', init_gain=1.0)

        if self.isTrain:  # define discriminators
            self.netD_A = build_discriminator(opt.model.discriminator)
            self.netD_B = build_discriminator(opt.model.discriminator)
            init_weights(self.netD_A, init_type='xavier', init_gain=1.0)
            init_weights(self.netD_B, init_type='xavier', init_gain=1.0)

        if self.isTrain:
            self.fake_A_pool = ImagePool(
                opt.dataset.train.pool_size
            )  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(
                opt.dataset.train.pool_size
            )  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss(
                opt.model.gan_mode)  #.to(self.device)  # define GAN loss.
            self.criterionCycle = paddle.nn.L1Loss()
            self.criterionIdt = paddle.nn.L1Loss()
            self.criterionL1 = paddle.nn.L1Loss()
            self.criterionL2 = paddle.nn.MSELoss()

            self.build_lr_scheduler()
            self.optimizer_G = build_optimizer(
                opt.optimizer,
                self.lr_scheduler,
                parameter_list=self.netG.parameters())
            # self.optimizer_D = paddle.optimizer.Adam(learning_rate=lr_scheduler_d, parameter_list=self.netD_A.parameters() + self.netD_B.parameters(), beta1=opt.beta1)
            self.optimizer_DA = build_optimizer(
                opt.optimizer,
                self.lr_scheduler,
                parameter_list=self.netD_A.parameters())
            self.optimizer_DB = build_optimizer(
                opt.optimizer,
                self.lr_scheduler,
                parameter_list=self.netD_B.parameters())
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_DA)
            self.optimizers.append(self.optimizer_DB)
            self.optimizer_names.extend(
                ['optimizer_G', 'optimizer_DA', 'optimizer_DB'])

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
        if self.isTrain:
            self.mask_A = paddle.to_tensor(input['mask_A'])
            self.mask_B = paddle.to_tensor(input['mask_B'])
            self.c_m_idt_a = paddle.to_tensor(input['consis_mask_idt_A'])
            self.c_m_idt_b = paddle.to_tensor(input['consis_mask_idt_B'])

        #self.hm_gt_A = self.hm_gt_A_lip + self.hm_gt_A_skin + self.hm_gt_A_eye
        #self.hm_gt_B = self.hm_gt_B_lip + self.hm_gt_B_skin + self.hm_gt_B_eye

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_A, amm = self.netG(self.real_A, self.real_B, self.P_A,
                                     self.P_B, self.c_m, self.mask_A_aug,
                                     self.mask_B_aug)  # G_A(A)
        self.fake_B, _ = self.netG(self.real_B, self.real_A, self.P_B, self.P_A,
                                   self.c_m_t, self.mask_A_aug,
                                   self.mask_B_aug)  # G_A(A)
        self.rec_A, _ = self.netG(self.fake_A, self.real_A, self.P_A, self.P_A,
                                  self.c_m_idt_a, self.mask_A_aug,
                                  self.mask_B_aug)  # G_A(A)
        self.rec_B, _ = self.netG(self.fake_B, self.real_B, self.P_B, self.P_B,
                                  self.c_m_idt_b, self.mask_A_aug,
                                  self.mask_B_aug)  # G_A(A)

    def forward_test(self, input):
        '''
        not implement now
        '''
        return self.netG(input['image_A'], input['image_B'], input['P_A'],
                         input['P_B'], input['consis_mask'],
                         input['mask_A_aug'], input['mask_B_aug'])

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
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.losses['D_A_loss'] = self.loss_D_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.losses['D_B_loss'] = self.loss_D_B

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        '''
        self.loss_names = [
                'G_A_vgg',
                'G_B_vgg',
                'G_bg_consis'
                ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'amm_a']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'amm_b']
        '''
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_vgg = 5e-3
        # Identity loss
        if lambda_idt > 0:
            self.idt_A, _ = self.netG(self.real_A, self.real_A, self.P_A,
                                      self.P_A, self.c_m_idt_a, self.mask_A_aug,
                                      self.mask_B_aug)  # G_A(A)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_A) * lambda_A * lambda_idt
            self.idt_B, _ = self.netG(self.real_B, self.real_B, self.P_B,
                                      self.P_B, self.c_m_idt_b, self.mask_A_aug,
                                      self.mask_B_aug)  # G_A(A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)
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
                             g_A_skin_loss_his * 0.1) * 0.01
        self.loss_G_B_his = (g_B_eye_loss_his + g_B_lip_loss_his +
                             g_B_skin_loss_his * 0.1) * 0.01

        self.losses['G_A_his_loss'] = self.loss_G_A_his
        self.losses['G_B_his_loss'] = self.loss_G_A_his

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

        self.loss_rec = (self.loss_cycle_A + self.loss_cycle_B +
                         self.loss_A_vgg + self.loss_B_vgg) * 0.2
        self.loss_idt = (self.loss_idt_A + self.loss_idt_B) * 0.2

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
            [self.netD_A, self.netD_B],
            False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.clear_gradients() #zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.minimize(
            self.loss_G)  #step()       # update G_A and G_B's weights
        self.optimizer_G.clear_gradients()
        # self.optimizer_G.clear_gradients()
        # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.set_requires_grad(self.netD_A, True)
        # self.optimizer_D.clear_gradients() #zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.optimizer_DA.minimize(
            self.loss_D_A)  #step()  # update D_A and D_B's weights
        self.optimizer_DA.clear_gradients()  #zero_g
        self.set_requires_grad(self.netD_B, True)
        # self.optimizer_DB.clear_gradients() #zero_grad()   # set D_A and D_B's gradients to zero

        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_DB.minimize(
            self.loss_D_B)  #step()  # update D_A and D_B's weights
        self.optimizer_DB.clear_gradients(
        )  #zero_grad()   # set D_A and D_B's gradients to zero
