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
from ..utils.image_pool import ImagePool


@MODELS.register()
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 cycle_criterion=None,
                 idt_criterion=None,
                 gan_criterion=None,
                 pool_size=50,
                 direction='a2b',
                 lambda_a=10.,
                 lambda_b=10.):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            cycle_criterion (dict): config of cycle criterion.
        """
        super(CycleGANModel, self).__init__()

        self.direction = direction

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        # define generators
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.nets['netG_A'] = build_generator(generator)
        self.nets['netG_B'] = build_generator(generator)
        init_weights(self.nets['netG_A'])
        init_weights(self.nets['netG_B'])

        # define discriminators
        if discriminator:
            self.nets['netD_A'] = build_discriminator(discriminator)
            self.nets['netD_B'] = build_discriminator(discriminator)
            init_weights(self.nets['netD_A'])
            init_weights(self.nets['netD_B'])

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(pool_size)

        # define loss functions
        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

        if cycle_criterion:
            self.cycle_criterion = build_criterion(cycle_criterion)

        if idt_criterion:
            self.idt_criterion = build_criterion(idt_criterion)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """

        AtoB = self.direction == 'a2b'

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
            self.fake_B = self.nets['netG_A'](self.real_A)  # G_A(A)
            self.rec_A = self.nets['netG_B'](self.fake_B)  # G_B(G_A(A))

            # visual
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_B'] = self.fake_B
            self.visual_items['rec_A'] = self.rec_A

        if hasattr(self, 'real_B'):
            self.fake_A = self.nets['netG_B'](self.real_B)  # G_B(B)
            self.rec_B = self.nets['netG_A'](self.fake_A)  # G_A(G_B(B))

            # visual
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_A'] = self.fake_A
            self.visual_items['rec_B'] = self.rec_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Args:
            netD (Layer): the discriminator D
            real (paddle.Tensor): real images
            fake (paddle.Tensor): images generated by a generator

        Return:
            the discriminator loss.

        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.gan_criterion(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.gan_criterion(pred_fake, False)
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
        # Identity loss
        if self.idt_criterion:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.nets['netG_A'](self.real_B)

            self.loss_idt_A = self.idt_criterion(self.idt_A,
                                                 self.real_B) * self.lambda_b
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.nets['netG_B'](self.real_A)

            # visual
            self.visual_items['idt_A'] = self.idt_A
            self.visual_items['idt_B'] = self.idt_B

            self.loss_idt_B = self.idt_criterion(self.idt_B,
                                                 self.real_A) * self.lambda_a
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.gan_criterion(self.nets['netD_A'](self.fake_B),
                                           True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.gan_criterion(self.nets['netD_B'](self.fake_A),
                                           True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.cycle_criterion(self.rec_A,
                                                 self.real_A) * self.lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.cycle_criterion(self.rec_B,
                                                 self.real_B) * self.lambda_b

        self.losses['G_idt_A_loss'] = self.loss_idt_A
        self.losses['G_idt_B_loss'] = self.loss_idt_B
        self.losses['G_A_adv_loss'] = self.loss_G_A
        self.losses['G_B_adv_loss'] = self.loss_G_B
        self.losses['G_A_cycle_loss'] = self.loss_cycle_A
        self.losses['G_B_cycle_loss'] = self.loss_cycle_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # compute fake images and reconstruction images.
        self.forward()
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.nets['netD_A'], self.nets['netD_B']],
                               False)
        # set G_A and G_B's gradients to zero
        optimizers['optimG'].clear_grad()
        # calculate gradients for G_A and G_B
        self.backward_G()
        # update G_A and G_B's weights
        self.optimizers['optimG'].step()
        # D_A and D_B
        self.set_requires_grad([self.nets['netD_A'], self.nets['netD_B']], True)

        # set D_A and D_B's gradients to zero
        optimizers['optimD'].clear_grad()
        # calculate gradients for D_A
        self.backward_D_A()
        # calculate graidents for D_B
        self.backward_D_B()
        # update D_A and D_B's weights
        optimizers['optimD'].step()


    def test_iter(self, metrics=None):
        self.nets['netG_A'].eval()
        self.forward()
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(self.fake_B, self.real_B)
        self.nets['netG_A'].train()
