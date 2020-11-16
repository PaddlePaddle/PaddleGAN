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
from paddle import fluid
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class DCGANModel(BaseModel):
    """ This class implements the DCGAN model, for learning a mapping from input images to output images given paired data.

    The model training requires dataset.
    By default, it uses a '--netG DCGenerator' generator,
    a '--netD DCDiscriminator' discriminator,
    and a vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    DCGAN paper: https://arxiv.org/pdf/1511.06434
    """
    def __init__(self, cfg):
        """Initialize the pix2pix class.

        Parameters:
            opt (config dict)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        super(DCGANModel, self).__init__(cfg)
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(cfg.model.generator)
        init_weights(self.nets['netG'])
        self.cfg = cfg
        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.is_train:
            self.nets['netD'] = build_discriminator(cfg.model.discriminator)
            init_weights(self.nets['netD'])

        if self.is_train:
            self.losses = {}
            # define loss functions
            self.criterionGAN = GANLoss(cfg.model.gan_mode)

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
        """
        self.real = paddle.to_tensor(input['A'])
        self.z = paddle.randn(shape=(self.real.shape[0],self.cfg.model.generator.input_nz,1,1))
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake = self.nets['netG'](self.z) 

        # put items to visual dict
        self.visual_items['real'] = self.real
        self.visual_items['fake'] = self.fake

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.nets['netD'](self.fake.detach())
        #print("pred_fake:",pred_fake.numpy())
        pred_fake = fluid.layers.squeeze(pred_fake,[-1,-2]) 
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        #print("loss_D_fake:",self.loss_D_fake.numpy())
        # Real
        pred_real = self.nets['netD'](self.real.detach())
        #print("pred_real:",pred_real.numpy())
        pred_real = fluid.layers.squeeze(pred_real,[-1,-2]) 
        self.loss_D_real = self.criterionGAN(pred_real, True)
        #print("loss_D_real:",self.loss_D_real.numpy())
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.nets['netD'](self.fake.detach())
        pred_fake = fluid.layers.squeeze(pred_fake,[-1,-2]) 
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN

        self.loss_G.backward()

        self.losses['G_adv_loss'] = self.loss_G_GAN

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
