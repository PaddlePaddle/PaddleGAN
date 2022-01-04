#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn.functional as F
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights


@MODELS.register()
class LapStyleDraModel(BaseModel):
    def __init__(self,
                 generator_encode,
                 generator_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleDraModel, self).__init__()

        # define generators
        self.nets['net_enc'] = build_generator(generator_encode)
        self.nets['net_dec'] = build_generator(generator_decode)
        init_weights(self.nets['net_dec'])
        self.set_requires_grad([self.nets['net_enc']], False)

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.cF = self.nets['net_enc'](self.ci)
        self.sF = self.nets['net_enc'](self.si)
        self.stylized = self.nets['net_dec'](self.cF, self.sF)
        self.visual_items['stylized'] = self.stylized

    def backward_Dec(self):
        self.tF = self.nets['net_enc'](self.stylized)
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """IDENTITY LOSSES"""
        self.Icc = self.nets['net_dec'](self.cF, self.cF)
        self.l_identity1 = self.calc_content_loss(self.Icc, self.ci)
        self.Fcc = self.nets['net_enc'](self.Icc)
        self.l_identity2 = 0
        for layer in self.content_layers:
            self.l_identity2 += self.calc_content_loss(self.Fcc[layer],
                                                       self.cF[layer])
        self.losses['l_identity1'] = self.l_identity1
        self.losses['l_identity2'] = self.l_identity2
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt

        self.loss = self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.l_identity1 * 50 + self.l_identity2 * 1 + self.loss_style_remd * 10 + \
                    self.loss_content_relt * 16
        self.loss.backward()

        return self.loss

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights"""
        self.forward()
        optimizers['optimG'].clear_grad()
        self.backward_Dec()
        self.optimizers['optimG'].step()


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    """
    Laplacian

    return:
       x - upsample(downsample(x))
    """
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])


def make_laplace_pyramid(x, levels):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(
            current,
            (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def fold_laplace_pyramid(pyramid):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current


@MODELS.register()
class LapStyleRevFirstModel(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevFirstModel, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)

        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_dec']], False)

        # define revision-net params
        self.nets['net_rev'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

        self.pyr_ci = make_laplace_pyramid(self.ci, 1)
        self.pyr_si = make_laplace_pyramid(self.si, 1)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[1])
        sF = self.nets['net_enc'](self.pyr_si[1])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev
        self.visual_items['stylized'] = self.stylized

    def backward_G(self):
        self.tF = self.nets['net_enc'](self.stylized)
        self.cF = self.nets['net_enc'](self.pyr_ci[2])
        self.sF = self.nets['net_enc'](self.pyr_si[2])
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(
            self.tF['r31'], self.sF['r31']) + self.calc_style_emd_loss(
                self.tF['r41'], self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r31'], self.cF['r31']) + self.calc_content_relt_loss(
                self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt
        """gan loss"""
        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        self.loss = self.loss_G_GAN + self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.loss_style_remd * 10 + self.loss_content_relt * 16
        self.loss.backward()
        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.nets['netD'](self.pyr_si[2])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

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


@MODELS.register()
class LapStyleRevSecondModel(BaseModel):
    def __init__(self,
                 revnet_generator,
                 revnet_discriminator,
                 draftnet_encode,
                 draftnet_decode,
                 calc_style_emd_loss=None,
                 calc_content_relt_loss=None,
                 calc_content_loss=None,
                 calc_style_loss=None,
                 gan_criterion=None,
                 content_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 style_layers=['r11', 'r21', 'r31', 'r41', 'r51'],
                 content_weight=1.0,
                 style_weight=3.0):

        super(LapStyleRevSecondModel, self).__init__()

        # define draftnet params
        self.nets['net_enc'] = build_generator(draftnet_encode)
        self.nets['net_dec'] = build_generator(draftnet_decode)
        self.set_requires_grad([self.nets['net_enc']], False)
        self.set_requires_grad([self.nets['net_dec']], False)

        # define the first revnet params
        self.nets['net_rev'] = build_generator(revnet_generator)
        self.set_requires_grad([self.nets['net_rev']], False)

        # define the second revnet params
        self.nets['net_rev_2'] = build_generator(revnet_generator)
        init_weights(self.nets['net_rev_2'])
        self.nets['netD'] = build_discriminator(revnet_discriminator)
        init_weights(self.nets['netD'])

        # define loss functions
        self.calc_style_emd_loss = build_criterion(calc_style_emd_loss)
        self.calc_content_relt_loss = build_criterion(calc_content_relt_loss)
        self.calc_content_loss = build_criterion(calc_content_loss)
        self.calc_style_loss = build_criterion(calc_style_loss)
        self.gan_criterion = build_criterion(gan_criterion)

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight

    def setup_input(self, input):
        self.ci = paddle.to_tensor(input['ci'])
        self.visual_items['ci'] = self.ci
        self.si = paddle.to_tensor(input['si'])
        self.visual_items['si'] = self.si
        self.image_paths = input['ci_path']

        self.pyr_ci = make_laplace_pyramid(self.ci, 2)
        self.pyr_si = make_laplace_pyramid(self.si, 2)
        self.pyr_ci.append(self.ci)
        self.pyr_si.append(self.si)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        cF = self.nets['net_enc'](self.pyr_ci[2])
        sF = self.nets['net_enc'](self.pyr_si[2])

        stylized_small = self.nets['net_dec'](cF, sF)
        self.visual_items['stylized_small'] = stylized_small
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[1], stylized_up], axis=1)
        stylized_rev_lap = self.nets['net_rev'](revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        self.visual_items['stylized_rev_first'] = stylized_rev
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[self.pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap_second = self.nets['net_rev_2'](revnet_input)
        stylized_rev_second = fold_laplace_pyramid(
            [stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        self.stylized = stylized_rev_second
        self.visual_items['stylized'] = self.stylized

    def backward_G(self):
        self.tF = self.nets['net_enc'](self.stylized)
        self.cF = self.nets['net_enc'](self.pyr_ci[3])
        self.sF = self.nets['net_enc'](self.pyr_si[3])
        """content loss"""
        self.loss_c = 0
        for layer in self.content_layers:
            self.loss_c += self.calc_content_loss(self.tF[layer],
                                                  self.cF[layer],
                                                  norm=True)
        self.losses['loss_c'] = self.loss_c
        """style loss"""
        self.loss_s = 0
        for layer in self.style_layers:
            self.loss_s += self.calc_style_loss(self.tF[layer], self.sF[layer])
        self.losses['loss_s'] = self.loss_s
        """relative loss"""
        self.loss_style_remd = self.calc_style_emd_loss(self.tF['r41'],
                                                        self.sF['r41'])
        self.loss_content_relt = self.calc_content_relt_loss(
            self.tF['r41'], self.cF['r41'])
        self.losses['loss_style_remd'] = self.loss_style_remd
        self.losses['loss_content_relt'] = self.loss_content_relt
        """gan loss"""
        pred_fake = self.nets['netD'](self.stylized)
        self.loss_G_GAN = self.gan_criterion(pred_fake, True)
        self.losses['loss_gan_G'] = self.loss_G_GAN

        self.loss = self.loss_G_GAN + self.loss_c * self.content_weight + self.loss_s * self.style_weight +\
                    self.loss_style_remd * 10 + self.loss_content_relt * 16
        self.loss.backward()
        return self.loss

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        pred_fake = self.nets['netD'](self.stylized.detach())
        self.loss_D_fake = self.gan_criterion(pred_fake, False)
        pred_real = self.nets['netD'](self.pyr_si[3])
        self.loss_D_real = self.gan_criterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

        self.losses['D_fake_loss'] = self.loss_D_fake
        self.losses['D_real_loss'] = self.loss_D_real

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
