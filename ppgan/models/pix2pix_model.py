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

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 pixel_criterion=None,
                 gan_criterion=None,
                 direction='a2b'):
        """Initialize the pix2pix class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            pixel_criterion (dict): config of pixel criterion.
            gan_criterion (dict): config of gan criterion.
        """
        super(Pix2PixModel, self).__init__()

        self.direction = direction
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'])

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if discriminator:
            self.nets['netD'] = build_discriminator(discriminator)
            init_weights(self.nets['netD'])

        if pixel_criterion:
            self.pixel_criterion = build_criterion(pixel_criterion)

        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.direction == 'AtoB'

        self.real_A = paddle.to_tensor(input['A' if AtoB else 'B'])
        self.real_B = paddle.to_tensor(input['B' if AtoB else 'A'])

        self.image_paths = input['A_path' if AtoB else 'B_path']

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
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        # Real
        real_AB = paddle.concat((self.real_A, self.real_B), 1)
        pred_real = self.nets['netD'](real_AB)
        self.loss_D_real = self.gan_criterion(pred_real, True)
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
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.pixel_criterion(self.fake_B, self.real_B)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

        self.losses['G_adv_loss'] = self.loss_G_GAN
        self.losses['G_L1_loss'] = self.loss_G_L1

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()

        # update D
        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimD'].clear_grad()
        self.backward_D()
        optimizers['optimD'].step()

        # update G
        self.set_requires_grad(self.nets['netD'], False)
        optimizers['optimG'].clear_grad()
        self.backward_G()
        optimizers['optimG'].step()

    def test_iter(self, metrics=None):
        self.nets['netG'].eval()
        self.forward()
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(self.fake_B, self.real_B)
        self.nets['netG'].train()
