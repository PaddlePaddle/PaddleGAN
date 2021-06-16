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

from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .sr_model import BaseSRModel
from .builder import MODELS

from .criterions import build_criterion


@MODELS.register()
class ESRGAN(BaseSRModel):
    """
    This class implements the ESRGAN model.

    ESRGAN paper: https://arxiv.org/pdf/1809.00219.pdf
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 pixel_criterion=None,
                 perceptual_criterion=None,
                 gan_criterion=None):
        """Initialize the ESRGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            pixel_criterion (dict): config of pixel criterion.
            perceptual_criterion (dict): config of perceptual criterion.
            gan_criterion (dict): config of gan criterion.
        """
        super(ESRGAN, self).__init__(generator)

        self.nets['generator'] = build_generator(generator)

        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)

        if pixel_criterion:
            self.pixel_criterion = build_criterion(pixel_criterion)

        if perceptual_criterion:
            self.perceptual_criterion = build_criterion(perceptual_criterion)

        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

    def train_iter(self, optimizers=None):
        optimizers['optimG'].clear_grad()
        l_total = 0
        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output
        # pixel loss
        if self.pixel_criterion:
            l_pix = self.pixel_criterion(self.output, self.gt)
            l_total += l_pix
            self.losses['loss_pix'] = l_pix
        if self.perceptual_criterion:
            l_g_percep, l_g_style = self.perceptual_criterion(
                self.output, self.gt)
            # l_total += l_pix
            if l_g_percep is not None:
                l_total += l_g_percep
                self.losses['loss_percep'] = l_g_percep
            if l_g_style is not None:
                l_total += l_g_style
                self.losses['loss_style'] = l_g_style

        # gan loss (relativistic gan)
        if hasattr(self, 'gan_criterion'):
            self.set_requires_grad(self.nets['discriminator'], False)
            real_d_pred = self.nets['discriminator'](self.gt).detach()
            fake_g_pred = self.nets['discriminator'](self.output)
            l_g_real = self.gan_criterion(real_d_pred -
                                          paddle.mean(fake_g_pred),
                                          False,
                                          is_disc=False)
            l_g_fake = self.gan_criterion(fake_g_pred -
                                          paddle.mean(real_d_pred),
                                          True,
                                          is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_total += l_g_gan
            self.losses['l_g_gan'] = l_g_gan
            l_total.backward()
            optimizers['optimG'].step()

            self.set_requires_grad(self.nets['discriminator'], True)
            optimizers['optimD'].clear_grad()
            # real
            fake_d_pred = self.nets['discriminator'](self.output).detach()
            real_d_pred = self.nets['discriminator'](self.gt)
            l_d_real = self.gan_criterion(
                real_d_pred - paddle.mean(fake_d_pred), True,
                is_disc=True) * 0.5

            # fake
            fake_d_pred = self.nets['discriminator'](self.output.detach())
            l_d_fake = self.gan_criterion(
                fake_d_pred - paddle.mean(real_d_pred.detach()),
                False,
                is_disc=True) * 0.5

            (l_d_real + l_d_fake).backward()
            optimizers['optimD'].step()

            self.losses['l_d_real'] = l_d_real
            self.losses['l_d_fake'] = l_d_fake
            self.losses['out_d_real'] = paddle.mean(real_d_pred.detach())
            self.losses['out_d_fake'] = paddle.mean(fake_d_pred.detach())
        else:
            l_total.backward()
            optimizers['optimG'].step()
