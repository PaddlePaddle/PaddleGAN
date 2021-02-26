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
from .criterions import build_criterion
from ..modules.init import init_weights


@MODELS.register()
class DCGANModel(BaseModel):
    """
    This class implements the DCGAN model, for learning a distribution from input images.
    DCGAN paper: https://arxiv.org/pdf/1511.06434
    """
    def __init__(self, generator, discriminator=None, gan_criterion=None):
        """Initialize the DCGAN class.
        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            pixel_criterion (dict): config of pixel criterion.
            gan_criterion (dict): config of gan criterion.
        """
        super(DCGANModel, self).__init__()
        self.gen_cfg = generator
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'])

        if self.is_train:
            self.nets['netD'] = build_discriminator(discriminator)
            init_weights(self.nets['netD'])

        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.
        """
        # get 1-channel gray image, or 3-channel color image
        self.real = paddle.to_tensor(input['img'])
        if 'img_path' in input:
            self.image_paths = input['A_path']

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""

        # generate random noise and fake image
        self.z = paddle.rand(shape=(self.real.shape[0], self.gen_cfg.input_nz,
                                    1, 1))
        self.fake = self.nets['netG'](self.z)

        # put items to visual dict
        self.visual_items['real'] = self.real
        self.visual_items['fake'] = self.fake

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake
        pred_fake = self.nets['netD'](self.fake.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)

        pred_real = self.nets['netD'](self.real)
        self.loss_D_real = self.gan_criterion(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        # G(A) should fake the discriminator
        pred_fake = self.nets['netD'](self.fake)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN

        self.loss_G.backward()

        self.losses['G_adv_loss'] = self.loss_G_GAN

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()

        #update D
        self.set_requires_grad(self.nets['netD'], True)
        self.set_requires_grad(self.nets['netG'], False)
        self.optimizers['optimizer_D'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_D'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        self.set_requires_grad(self.nets['netG'], True)
        self.optimizers['optimizer_G'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()
