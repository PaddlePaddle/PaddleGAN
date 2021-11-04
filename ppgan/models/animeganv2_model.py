#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn

from .base_model import BaseModel
from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .criterions import build_criterion
from ..modules.caffevgg import CaffeVGG19
from ..modules.init import init_weights
from ..utils.filesystem import load


@MODELS.register()
class AnimeGANV2Model(BaseModel):
    """ This class implements the AnimeGANV2 model.
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_criterion=None,
                 pretrain_ckpt=None,
                 g_adv_weight=300.,
                 d_adv_weight=300.,
                 con_weight=1.5,
                 sty_weight=2.5,
                 color_weight=10.,
                 tv_weight=1.):
        """Initialize the AnimeGANV2 class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            gan_criterion (dict): config of gan criterion.
        """
        super(AnimeGANV2Model, self).__init__()
        self.g_adv_weight = g_adv_weight
        self.d_adv_weight = d_adv_weight
        self.con_weight = con_weight
        self.sty_weight = sty_weight
        self.color_weight = color_weight
        self.tv_weight = tv_weight
        # define networks (both generator and discriminator)
        self.nets['netG'] = build_generator(generator)
        init_weights(self.nets['netG'])

        # define a discriminator
        if self.is_train:
            self.nets['netD'] = build_discriminator(discriminator)
            init_weights(self.nets['netD'])

            self.pretrained = CaffeVGG19()

            self.losses = {}
            # define loss functions
            self.criterionGAN = build_criterion(gan_criterion)
            self.criterionL1 = nn.L1Loss()
            self.criterionHub = nn.SmoothL1Loss()

            if pretrain_ckpt:
                state_dicts = load(pretrain_ckpt)
                self.nets['netG'].set_state_dict(state_dicts['netG'])
                print('Load pretrained generator from', pretrain_ckpt)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        """
        if self.is_train:
            self.real = paddle.to_tensor(input['real'])
            self.anime = paddle.to_tensor(input['anime'])
            self.anime_gray = paddle.to_tensor(input['anime_gray'])
            self.smooth_gray = paddle.to_tensor(input['smooth_gray'])
        else:
            self.real = paddle.to_tensor(input['A'])
            self.image_paths = input['A_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake = self.nets['netG'](self.real)

        # put items to visual dict
        self.visual_items['real'] = self.real
        self.visual_items['fake'] = self.fake

    def test(self):
        self.fake = self.nets['netG'](self.real)

        # put items to visual dict
        self.visual_items['real'] = self.real
        self.visual_items['fake'] = self.fake

    @staticmethod
    def gram(x):
        b, c, h, w = x.shape
        x_tmp = x.reshape((b, c, (h * w)))
        gram = paddle.matmul(x_tmp, x_tmp, transpose_y=True)
        return gram / (c * h * w)

    def style_loss(self, style, fake):
        return self.criterionL1(self.gram(style), self.gram(fake))

    def con_sty_loss(self, real, anime, fake):
        real_feature_map = self.pretrained(real)
        fake_feature_map = self.pretrained(fake)
        anime_feature_map = self.pretrained(anime)

        c_loss = self.criterionL1(real_feature_map, fake_feature_map)
        s_loss = self.style_loss(anime_feature_map, fake_feature_map)

        return c_loss, s_loss

    @staticmethod
    def rgb2yuv(rgb):
        kernel = paddle.to_tensor([[0.299, -0.14714119, 0.61497538],
                                   [0.587, -0.28886916, -0.51496512],
                                   [0.114, 0.43601035, -0.10001026]],
                                  dtype='float32')
        rgb = paddle.transpose(rgb, (0, 2, 3, 1))
        yuv = paddle.matmul(rgb, kernel)
        return yuv

    @staticmethod
    def denormalize(image):
        return image * 0.5 + 0.5

    def color_loss(self, con, fake):
        con = self.rgb2yuv(self.denormalize(con))
        fake = self.rgb2yuv(self.denormalize(fake))
        return (self.criterionL1(con[:, :, :, 0], fake[:, :, :, 0]) +
                self.criterionHub(con[:, :, :, 1], fake[:, :, :, 1]) +
                self.criterionHub(con[:, :, :, 2], fake[:, :, :, 2]))

    @staticmethod
    def variation_loss(image, ksize=1):
        dh = image[:, :, :-ksize, :] - image[:, :, ksize:, :]
        dw = image[:, :, :, :-ksize] - image[:, :, :, ksize:]
        return (paddle.mean(paddle.abs(dh)) + paddle.mean(paddle.abs(dw)))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # use conditional GANs; we need to feed both input and output to the discriminator
        real_logit = self.nets['netD'](self.anime)
        gray_logit = self.nets['netD'](self.anime_gray)
        fake_logit = self.nets['netD'](self.fake.detach())
        smooth_logit = self.nets['netD'](self.smooth_gray)

        d_real_loss = (self.d_adv_weight * 1.2 *
                       self.criterionGAN(real_logit, True))
        d_gray_loss = (self.d_adv_weight * 1.2 *
                       self.criterionGAN(gray_logit, False))
        d_fake_loss = (self.d_adv_weight * 1.2 *
                       self.criterionGAN(fake_logit, False))
        d_blur_loss = (self.d_adv_weight * 0.8 *
                       self.criterionGAN(smooth_logit, False))

        self.loss_D = d_real_loss + d_gray_loss + d_fake_loss + d_blur_loss

        self.loss_D.backward()

        self.losses['d_loss'] = self.loss_D
        self.losses['d_real_loss'] = d_real_loss
        self.losses['d_fake_loss'] = d_fake_loss
        self.losses['d_gray_loss'] = d_gray_loss
        self.losses['d_blur_loss'] = d_blur_loss

    def backward_G(self):
        fake_logit = self.nets['netD'](self.fake)
        c_loss, s_loss = self.con_sty_loss(self.real, self.anime_gray,
                                           self.fake)
        c_loss = self.con_weight * c_loss
        s_loss = self.sty_weight * s_loss
        tv_loss = self.tv_weight * self.variation_loss(self.fake)
        col_loss = self.color_weight * self.color_loss(self.real, self.fake)
        g_loss = (self.g_adv_weight * self.criterionGAN(fake_logit, True))

        self.loss_G = c_loss + s_loss + col_loss + g_loss + tv_loss

        self.loss_G.backward()

        self.losses['g_loss'] = self.loss_G
        self.losses['c_loss'] = c_loss
        self.losses['s_loss'] = s_loss
        self.losses['col_loss'] = col_loss
        self.losses['tv_loss'] = tv_loss

    def train_iter(self, optimizers=None):
        # compute fake images: G(A)
        self.forward()

        # update D
        self.optimizers['optimizer_D'].clear_grad()
        self.backward_D()
        self.optimizers['optimizer_D'].step()

        # update G
        self.optimizers['optimizer_G'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()


@MODELS.register()
class AnimeGANV2PreTrainModel(AnimeGANV2Model):
    def backward_G(self):
        real_feature_map = self.pretrained(self.real)
        fake_feature_map = self.pretrained(self.fake)
        init_c_loss = self.criterionL1(real_feature_map, fake_feature_map)
        loss = self.con_weight * init_c_loss
        loss.backward()
        self.losses['init_c_loss'] = init_c_loss

    def train_iter(self, optimizers=None):
        self.forward()
        # update G
        self.optimizers['optimizer_G'].clear_grad()
        self.backward_G()
        self.optimizers['optimizer_G'].step()
