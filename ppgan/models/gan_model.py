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
import numpy as np
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .criterions.builder import build_criterion

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.visual import make_grid


@MODELS.register()
class GANModel(BaseModel):
    """ This class implements the vanilla GAN model with some tricks.

    vanilla GAN paper: https://arxiv.org/abs/1406.2661
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_criterion=None,
                 params=None):
        """Initialize the GAN Model class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            gan_criterion (dict): config of gan criterion.
            params (dict): hyper params for train or test. Default: None.
        """
        super(GANModel, self).__init__(params)
        self.iter = 0

        self.disc_iters = 1 if self.params is None else self.params.get(
            'disc_iters', 1)
        self.disc_start_iters = (0 if self.params is None else self.params.get(
            'disc_start_iters', 0))
        self.samples_every_row = (8 if self.params is None else self.params.get(
            'samples_every_row', 8))
        self.visual_interval = (500 if self.params is None else self.params.get(
            'visual_interval', 500))

        # define generator
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'])

        # define a discriminator
        if self.is_train:
            if discriminator is not None:
                self.nets['netD'] = build_discriminator(discriminator)
                init_weights(self.nets['netD'])

            # define loss functions
            if gan_criterion:
                self.criterionGAN = build_criterion(gan_criterion)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (list): include the data itself and its metadata information.
        """
        if isinstance(input, (list, tuple)):
            input = input[0]
        if not isinstance(input, dict):
            input = {'img': input}
        self.D_real_inputs = [paddle.to_tensor(input['img'])]
        if 'class_id' in input:  # n class input
            if isinstance(self.nets['netG'], paddle.DataParallel):
                self.n_class = self.nets['netG']._layers.n_class
            else:
                self.n_class = self.nets['netG'].n_class

            self.D_real_inputs += [
                paddle.to_tensor(input['class_id'], dtype='int64')
            ]
        else:
            self.n_class = 0

        batch_size = self.D_real_inputs[0].shape[0]

        if isinstance(self.nets['netG'], paddle.DataParallel):
            self.G_inputs = self.nets['netG']._layers.random_inputs(batch_size)
        else:
            self.G_inputs = self.nets['netG'].random_inputs(batch_size)

        if not isinstance(self.G_inputs, (list, tuple)):
            self.G_inputs = [self.G_inputs]

        if not hasattr(self, 'G_fixed_inputs'):
            self.G_fixed_inputs = [t for t in self.G_inputs]
            if self.n_class > 0:
                rows_num = (batch_size - 1) // self.samples_every_row + 1
                class_ids = paddle.randint(0, self.n_class, [rows_num, 1])
                class_ids = class_ids.tile([1, self.samples_every_row])
                class_ids = class_ids.reshape([
                    -1,
                ])[:batch_size].detach()
                self.G_fixed_inputs[1] = class_ids.detach()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_imgs = self.nets['netG'](*self.G_inputs)  # G(img, class_id)

        # put items to visual dict
        self.visual_items['fake_imgs'] = make_grid(
            self.fake_imgs, self.samples_every_row).detach()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_imgs
        # use conditional GANs; we need to feed both input and output to the discriminator
        self.loss_D = 0
        self.D_fake_inputs = [self.fake_imgs.detach()]
        if len(self.G_inputs) > 1 and self.G_inputs[1] is not None:
            self.D_fake_inputs += [self.G_inputs[1]]
        pred_fake = self.nets['netD'](*self.D_fake_inputs)
        # Real
        real_imgs = self.D_real_inputs[0]
        self.visual_items['real_imgs'] = make_grid(
            real_imgs, self.samples_every_row).detach()
        pred_real = self.nets['netD'](*self.D_real_inputs)

        self.loss_D_fake = self.criterionGAN(pred_fake, False, True)
        self.loss_D_real = self.criterionGAN(pred_real, True, True)

        # combine loss and calculate gradients
        if self.criterionGAN.gan_mode in ['vanilla', 'lsgan']:
            self.loss_D = self.loss_D + (self.loss_D_fake +
                                         self.loss_D_real) * 0.5
        else:
            self.loss_D = self.loss_D + self.loss_D_fake + self.loss_D_real

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        # First, G(imgs) should fake the discriminator
        self.D_fake_inputs = [self.fake_imgs]
        if len(self.G_inputs) > 1 and self.G_inputs[1] is not None:
            self.D_fake_inputs += [self.G_inputs[1]]
        pred_fake = self.nets['netD'](*self.D_fake_inputs)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True, False)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN

        self.loss_G.backward()

        self.losses['G_adv_loss'] = self.loss_G_GAN

    def train_iter(self, optimizers=None):

        # compute fake images: G(imgs)
        self.forward()

        # update D
        self.set_requires_grad(self.nets['netD'], True)
        optimizers['optimizer_D'].clear_grad()
        self.backward_D()
        optimizers['optimizer_D'].step()
        self.set_requires_grad(self.nets['netD'], False)

        # weight clip
        if self.criterionGAN.gan_mode == 'wgan':
            with paddle.no_grad():
                for p in self.nets['netD'].parameters():
                    p[:] = p.clip(-0.01, 0.01)

        if self.iter > self.disc_start_iters and self.iter % self.disc_iters == 0:
            # update G
            optimizers['optimizer_G'].clear_grad()
            self.backward_G()
            optimizers['optimizer_G'].step()

        if self.iter % self.visual_interval == 0:
            with paddle.no_grad():
                self.visual_items['fixed_generated_imgs'] = make_grid(
                    self.nets['netG'](*self.G_fixed_inputs),
                    self.samples_every_row)

        self.iter += 1
