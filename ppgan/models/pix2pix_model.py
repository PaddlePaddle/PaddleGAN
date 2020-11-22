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
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires 'paired' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (from PatchGAN),
    and a vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, cfg):
        """Initialize the pix2pix class.

        Parameters:
            opt (config dict)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(Pix2PixModel, self).__init__(cfg)
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(cfg.model.generator)
        init_weights(self.nets['netG'])

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.is_train:
            self.nets['netD'] = build_discriminator(cfg.model.discriminator)
            init_weights(self.nets['netD'])

        if self.is_train:
            self.losses = {}
            # define loss functions
            self.criterionGAN = GANLoss(cfg.model.gan_mode)
            self.criterionL1 = paddle.nn.L1Loss()

            # build optimizers
            self.build_lr_scheduler()
            self.optimizers['optimizer_G'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netG'].parameters())
            self.optimizers['optimizer_D'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netD'].parameters())

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.cfg.dataset.train.direction == 'AtoB'

        self.real_A = paddle.fluid.dygraph.to_variable(
            input['A' if AtoB else 'B'])
        self.real_B = paddle.fluid.dygraph.to_variable(
            input['B' if AtoB else 'A'])

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.nets['netG'](self.real_A)  # G(A)

        # put items to visual dict
        self.visual_items['fake_B'] = self.fake_B
        self.visual_items['real_A'] = self.real_A
        self.visual_items['real_B'] = self.real_B

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = paddle.concat((self.real_A, self.fake_B), 1)
        pred_fake = self.nets['netD'](fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = paddle.concat((self.real_A, self.real_B), 1)
        pred_real = self.nets['netD'](real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = paddle.concat((self.real_A, self.fake_B), 1)
        pred_fake = self.nets['netD'](fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B,
                                          self.real_B) * self.cfg.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

        self.losses['G_adv_loss'] = self.loss_G_GAN
        self.losses['G_L1_loss'] = self.loss_G_L1

    def optimize_parameters(self):
        # compute fake images: G(A)
        self.forward()

        # update D
        self.set_requires_grad(self.nets['netD'], True)
        self.optimizers['optimizer_D'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_D'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        self.optimizers['optimizer_G'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()
