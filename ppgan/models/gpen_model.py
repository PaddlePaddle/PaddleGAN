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
from ..modules.init import init_weights

from .criterions.id_loss import IDLoss
from paddle.nn import functional as F
from paddle import autograd
import math


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(outputs=real_pred.sum(),
                               inputs=real_img,
                               create_graph=True)
    grad_penalty = grad_real.pow(2).reshape([grad_real.shape[0],
                                             -1]).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred,
                         loss_funcs=None,
                         fake_img=None,
                         real_img=None,
                         input_img=None):
    smooth_l1_loss, id_loss = loss_funcs

    loss = F.softplus(-fake_pred).mean()
    loss_l1 = smooth_l1_loss(fake_img, real_img)
    loss_id = id_loss(fake_img, real_img, input_img)
    loss += 1.0 * loss_l1 + 1.0 * loss_id

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = paddle.randn(fake_img.shape) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(),
                          inputs=latents,
                          create_graph=True)
    path_lengths = paddle.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() -
                                            mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


@MODELS.register()
class GPENModel(BaseModel):
    """ This class implements the gpen model.

    """

    def __init__(self, generator, discriminator=None, direction='a2b'):

        super(GPENModel, self).__init__()

        self.direction = direction
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        self.nets['g_ema'] = build_generator(generator)
        self.nets['g_ema'].eval()

        if discriminator:
            self.nets['netD'] = build_discriminator(discriminator)

        self.accum = 0.5**(32 / (10 * 1000))
        self.mean_path_length = 0

        self.gan_criterions = []
        self.gan_criterions.append(paddle.nn.SmoothL1Loss())
        self.gan_criterions.append(IDLoss())
        self.current_iter = 0

    def setup_input(self, input):

        self.degraded_img = paddle.to_tensor(input[0])
        self.real_img = paddle.to_tensor(input[1])

    def forward(self, test_mode=False, regularize=False):
        if test_mode:
            self.fake_img, _ = self.nets['g_ema'](self.degraded_img)  # G(A)
        else:
            if regularize:
                self.fake_img, self.latents = self.nets['netG'](
                    self.degraded_img, return_latents=True)
            else:
                self.fake_img, _ = self.nets['netG'](self.degraded_img)

    def backward_D(self, regularize=False):
        """Calculate GAN loss for the discriminator"""
        if regularize:
            self.real_img.stop_gradient = False
            real_pred = self.nets['netD'](self.real_img)
            r1_loss = d_r1_loss(real_pred, self.real_img)
            (10 / 2 * r1_loss * 16).backward()
        else:
            fake_pred = self.nets['netD'](self.fake_img)
            real_pred = self.nets['netD'](self.real_img)
            self.loss_D = d_logistic_loss(real_pred, fake_pred)
            self.loss_D.backward()
            self.losses['D_loss'] = self.loss_D

    def backward_G(self, regularize):
        """Calculate GAN and L1 loss for the generator"""

        if regularize:
            path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                self.fake_img, self.latents, self.mean_path_length)
            weighted_path_loss = 2 * 4 * path_loss
            weighted_path_loss.backward()
        else:
            fake_pred = self.nets['netD'](self.fake_img)
            self.loss_G = g_nonsaturating_loss(fake_pred, self.gan_criterions,
                                               self.fake_img, self.real_img,
                                               self.degraded_img)
            self.loss_G.backward()
            self.losses['G_loss'] = self.loss_G

    def train_iter(self, optimizers=None):

        self.current_iter += 1
        # update D
        self.set_requires_grad(self.nets['netD'], True)
        self.set_requires_grad(self.nets['netG'], False)
        self.forward(test_mode=False)
        optimizers['optimD'].clear_grad()
        self.backward_D(regularize=False)
        optimizers['optimD'].step()

        d_regularize = self.current_iter % 24 == 0
        if d_regularize:
            optimizers['optimD'].clear_grad()
            self.backward_D(regularize=True)
            optimizers['optimD'].step()
        # update G
        self.set_requires_grad(self.nets['netD'], False)
        self.set_requires_grad(self.nets['netG'], True)
        self.forward(test_mode=False)
        optimizers['optimG'].clear_grad()
        self.backward_G(regularize=False)
        optimizers['optimG'].step()

        g_regularize = self.current_iter % 4 == 0
        if g_regularize:
            self.forward(test_mode=False, regularize=True)
            optimizers['optimG'].clear_grad()
            self.backward_G(regularize=True)
            optimizers['optimG'].step()

        self.accumulate(self.nets['g_ema'], self.nets['netG'], self.accum)

    def test_iter(self, metrics=None):
        self.nets['g_ema'].eval()
        self.forward(test_mode=True)

        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(self.fake_img, self.real_img)

    def accumulate(self, model1, model2, decay=0.999):
        par1 = dict(model1.state_dict())
        par2 = dict(model2.state_dict())

        for k in par1.keys():
            par1[k] = par1[k] * decay + par2[k] * (1 - decay)

        model1.load_dict(par1)
