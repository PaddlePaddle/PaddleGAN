#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
import paddle.nn.functional as F
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .criterions import build_criterion
from .discriminators.builder import build_discriminator

from ..modules.init import init_weights
from ..solver import build_optimizer

# gaussion blur on mask
def gaussian_blur(input, kernel_size, sigma):
    def get_gaussian_kernel(kernel_size: int, sigma: float) -> paddle.Tensor:
        def gauss_fcn(x, window_size, sigma):
            return -(x - window_size // 2)**2 / float(2 * sigma**2)
        gauss = paddle.stack([paddle.exp(paddle.to_tensor(gauss_fcn(x, kernel_size, sigma)))for x in range(kernel_size)])
        return gauss / gauss.sum()


    b, c, h, w = input.shape
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = paddle.matmul(kernel_x, kernel_y, transpose_y=True)
    kernel = kernel_2d.reshape([1, 1, ksize_x, ksize_y])
    kernel = kernel.repeat_interleave(c, 0)
    padding = [(k - 1) // 2 for k in kernel_size]
    return F.conv2d(input, kernel, padding=padding, stride=1, groups=c)

# GAN Loss
class Adversal():
    def __init__(self, ksize=71):
        self.ksize = ksize
        self.loss_fn = nn.MSELoss()

    def __call__(self, netD, fake, real, masks):
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake  = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.shape
        b, c, ht, wt = masks.shape

        # align image shape with mask
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)
        d_fake_label = gaussian_blur(masks, (self.ksize, self.ksize), (10, 10)).detach()
        d_real_label = paddle.zeros_like(d_real)
        g_fake_label = paddle.ones_like(g_fake)

        dis_loss = [self.loss_fn(d_fake, d_fake_label).mean(), self.loss_fn(d_real, d_real_label).mean()]
        gen_loss = (self.loss_fn(g_fake, g_fake_label) * masks / paddle.mean(masks)).mean()

        return dis_loss, gen_loss

@MODELS.register()
class AOTGANModel(BaseModel):
    def __init__(self,
                 generator,
                 discriminator,
                 criterion,
                 l1_weight,
                 perceptual_weight,
                 style_weight,
                 adversal_weight,
                 img_size,
                ):

        super(AOTGANModel, self).__init__()

        # define nets
        self.nets['netG'] = build_generator(generator)
        self.nets['netD'] = build_discriminator(discriminator)
        self.net_vgg = build_criterion(criterion)

        self.adv_loss = Adversal()

        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.adversal_weight = adversal_weight
        self.img_size = img_size

    def setup_input(self, input):
        self.img = input['img']
        self.mask = input['mask']
        self.img_masked = (self.img * (1 - self.mask)) + self.mask
        self.img_paths = input['img_path']

    def forward(self):
        input_x = paddle.concat([self.img_masked, self.mask], 1)
        self.pred_img = self.nets['netG'](input_x)
        self.comp_img = (1 - self.mask) * self.img + self.mask * self.pred_img
        self.visual_items['pred_img'] = self.pred_img.detach()

    def train_iter(self, optimizers=None):
        self.forward()
        l1_loss, perceptual_loss, style_loss = self.net_vgg(self.img, self.pred_img, self.img_size)
        self.losses['l1'] = l1_loss * self.l1_weight
        self.losses['perceptual'] = perceptual_loss * self.perceptual_weight
        self.losses['style'] = style_loss * self.style_weight
        dis_loss, gen_loss = self.adv_loss(self.nets['netD'], self.comp_img, self.img, self.mask)
        self.losses['adv_g'] = gen_loss * self.adversal_weight
        loss_d_fake = dis_loss[0]
        loss_d_real = dis_loss[1]
        self.losses['adv_d'] = loss_d_fake + loss_d_real

        loss_g = self.losses['l1'] + self.losses['perceptual'] + self.losses['style'] + self.losses['adv_g']
        loss_d = self.losses['adv_d']

        self.optimizers['optimG'].clear_grad()
        self.optimizers['optimD'].clear_grad()
        loss_g.backward()
        loss_d.backward()
        self.optimizers['optimG'].step()
        self.optimizers['optimD'].step()

    def test_iter(self, metrics=None):
        self.eval()
        with paddle.no_grad():
            self.forward()
        self.train()

    def setup_optimizers(self, lr, cfg):
        for opt_name, opt_cfg in cfg.items():
            if opt_name == 'lr':
                learning_rate = opt_cfg
                continue
            cfg_ = opt_cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()
            if opt_name == 'optimG':
                lr = learning_rate * 4
            else:
                lr = learning_rate
            self.optimizers[opt_name] = build_optimizer(
                cfg_, lr, parameters)

        return self.optimizers
