#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import math
import warnings
from collections import OrderedDict
from sklearn.cluster import KMeans

import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T

from .base_model import BaseModel
from .builder import MODELS
from .generators.builder import build_generator
from .criterions.builder import build_criterion
from .discriminators.builder import build_discriminator
from ..solver import build_lr_scheduler, build_optimizer

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def pad_shape(shape, pad_size):
    shape[-2] += 2 * pad_size
    shape[-1] += 2 * pad_size
    return shape


def quant(x, num):
    n, c, h, w = x.shape
    kmeans = KMeans(num, random_state=0).fit(
        x.transpose([0, 2, 3, 1]).reshape([-1, c]))
    centers = kmeans.cluster_centers_
    x = centers[kmeans.labels_].reshape([n, h, w, c]).transpose([0, 3, 1, 2])
    return paddle.to_tensor(x, 'float32'), centers


def quant_to_centers(x, centers):
    n, c, h, w = x.shape
    num = centers.shape[0]
    kmeans = KMeans(num, init=centers,
                    n_init=1).fit(x.transpose([0, 2, 3, 1]).reshape([-1, c]))
    x = centers[kmeans.labels_].reshape([n, h, w, c]).transpose([0, 3, 1, 2])
    return paddle.to_tensor(x, 'float32')


@MODELS.register()
class SinGANModel(BaseModel):

    def __init__(self,
                 generator,
                 discriminator,
                 gan_criterion=None,
                 recon_criterion=None,
                 gp_criterion=None,
                 train_image=None,
                 scale_factor=0.75,
                 min_size=25,
                 is_finetune=False,
                 finetune_scale=1,
                 color_num=5,
                 gen_iters=3,
                 disc_iters=3,
                 noise_amp_init=0.1):
        super(SinGANModel, self).__init__()

        # setup config
        self.gen_iters = gen_iters
        self.disc_iters = disc_iters
        self.min_size = min_size
        self.is_finetune = is_finetune
        self.noise_amp_init = noise_amp_init
        self.train_image = T.Compose([T.Transpose(),
                                      T.Normalize(127.5, 127.5)])(cv2.cvtColor(
                                          cv2.imread(train_image,
                                                     cv2.IMREAD_COLOR),
                                          cv2.COLOR_BGR2RGB))
        self.train_image = paddle.to_tensor(self.train_image).unsqueeze(0)
        self.scale_num = math.ceil(
            math.log(self.min_size / min(self.train_image.shape[-2:]),
                     scale_factor)) + 1
        self.scale_factor = math.pow(
            self.min_size / min(self.train_image.shape[-2:]),
            1 / (self.scale_num - 1))
        self.reals = [
            F.interpolate(self.train_image, None, self.scale_factor**i,
                          'bicubic') for i in range(self.scale_num - 1, -1, -1)
        ]

        # build generator
        generator['scale_num'] = self.scale_num
        generator['coarsest_shape'] = self.reals[0].shape
        self.nets['netG'] = build_generator(generator)
        self.niose_pad_size = 0 if generator.get('noise_zero_pad', True) \
                                else self.nets['netG']._pad_size
        self.nets['netG'].scale_factor = paddle.to_tensor(
            self.scale_factor, 'float32')

        # build discriminator
        nfc_init = discriminator.pop('nfc_init', 32)
        min_nfc_init = discriminator.pop('min_nfc_init', 32)
        for i in range(self.scale_num):
            discriminator['nfc'] = min(nfc_init * pow(2, math.floor(i / 4)),
                                       128)
            discriminator['min_nfc'] = min(
                min_nfc_init * pow(2, math.floor(i / 4)), 128)
            self.nets[f'netD{i}'] = build_discriminator(discriminator)

        # build criterion
        self.gan_criterion = build_criterion(gan_criterion)
        self.recon_criterion = build_criterion(recon_criterion)
        self.gp_criterion = build_criterion(gp_criterion)

        if self.is_finetune:
            self.finetune_scale = finetune_scale
            self.quant_real, self.quant_centers = quant(
                self.reals[finetune_scale], color_num)

        # setup training config
        self.lr_schedulers = OrderedDict()
        self.current_scale = (finetune_scale if self.is_finetune else 0) - 1
        self.current_iter = 0

    def set_total_iter(self, total_iter):
        super().set_total_iter(total_iter)
        if self.is_finetune:
            self.scale_iters = total_iter
        else:
            self.scale_iters = math.ceil(total_iter / self.scale_num)

    def setup_lr_schedulers(self, cfg):
        for i in range(self.scale_num):
            self.lr_schedulers[f"lr{i}"] = build_lr_scheduler(cfg)
        return self.lr_schedulers

    def setup_optimizers(self, lr_schedulers, cfg):
        for i in range(self.scale_num):
            self.optimizers[f'optim_netG{i}'] = build_optimizer(
                cfg['optimizer_G'], lr_schedulers[f"lr{i}"],
                self.nets[f'netG'].generators[i].parameters())
            self.optimizers[f'optim_netD{i}'] = build_optimizer(
                cfg['optimizer_D'], lr_schedulers[f"lr{i}"],
                self.nets[f'netD{i}'].parameters())
        return self.optimizers

    def setup_input(self, input):
        pass

    def backward_D(self):
        self.loss_D_real = self.gan_criterion(self.pred_real, True, True)
        self.loss_D_fake = self.gan_criterion(self.pred_fake, False, True)
        self.loss_D_gp = self.gp_criterion(
            self.nets[f'netD{self.current_scale}'], self.real_img,
            self.fake_img)
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp
        self.loss_D.backward()

        self.losses[f'scale{self.current_scale}/D_total_loss'] = self.loss_D
        self.losses[f'scale{self.current_scale}/D_real_loss'] = self.loss_D_real
        self.losses[f'scale{self.current_scale}/D_fake_loss'] = self.loss_D_fake
        self.losses[
            f'scale{self.current_scale}/D_gradient_penalty'] = self.loss_D_gp

    def backward_G(self):
        self.loss_G_gan = self.gan_criterion(self.pred_fake, True, False)
        self.loss_G_recon = self.recon_criterion(self.recon_img, self.real_img)
        self.loss_G = self.loss_G_gan + self.loss_G_recon
        self.loss_G.backward()

        self.losses[f'scale{self.current_scale}/G_adv_loss'] = self.loss_G_gan
        self.losses[
            f'scale{self.current_scale}/G_recon_loss'] = self.loss_G_recon

    def scale_prepare(self):
        self.real_img = self.reals[self.current_scale]
        self.lr_scheduler = self.lr_schedulers[f"lr{self.current_scale}"]
        for i in range(self.current_scale):
            self.optimizers.pop(f'optim_netG{i}', None)
            self.optimizers.pop(f'optim_netD{i}', None)
        self.losses.clear()
        self.visual_items.clear()
        self.visual_items[f'real_img_scale{self.current_scale}'] = self.real_img
        if self.is_finetune:
            self.visual_items['quant_real'] = self.quant_real

        self.recon_prev = paddle.zeros_like(self.reals[0])
        if self.current_scale > 0:
            z_pyramid = []
            for i in range(self.current_scale):
                if i == 0:
                    z = self.nets['netG'].z_fixed
                else:
                    z = paddle.zeros(
                        pad_shape(self.reals[i].shape, self.niose_pad_size))
                z_pyramid.append(z)
            self.recon_prev = self.nets['netG'](z_pyramid, self.recon_prev,
                                                self.current_scale - 1,
                                                0).detach()
            self.recon_prev = F.interpolate(self.recon_prev,
                                            self.real_img.shape[-2:], None,
                                            'bicubic')
            if self.is_finetune:
                self.recon_prev = quant_to_centers(self.recon_prev,
                                                   self.quant_centers)
            self.nets['netG'].sigma[self.current_scale] = F.mse_loss(
                self.real_img, self.recon_prev).sqrt() * self.noise_amp_init

        for i in range(self.scale_num):
            self.set_requires_grad(self.nets['netG'].generators[i],
                                   i == self.current_scale)

    def forward(self):
        if not self.is_finetune:
            self.fake_img = self.nets['netG'](self.z_pyramid,
                                              paddle.zeros(
                                                  pad_shape(
                                                      self.z_pyramid[0].shape,
                                                      -self.niose_pad_size)),
                                              self.current_scale, 0)
        else:
            x_prev = self.nets['netG'](self.z_pyramid[:self.finetune_scale],
                                       paddle.zeros(
                                           pad_shape(self.z_pyramid[0].shape,
                                                     -self.niose_pad_size)),
                                       self.finetune_scale - 1, 0)
            x_prev = F.interpolate(
                x_prev, self.z_pyramid[self.finetune_scale].shape[-2:], None,
                'bicubic')
            x_prev_quant = quant_to_centers(x_prev, self.quant_centers)
            self.fake_img = self.nets['netG'](
                self.z_pyramid[self.finetune_scale:], x_prev_quant,
                self.current_scale, self.finetune_scale)

        self.recon_img = self.nets['netG'](
            [(paddle.randn if self.current_scale == 0 else paddle.zeros)(
                pad_shape(self.real_img.shape, self.niose_pad_size))],
            self.recon_prev, self.current_scale, self.current_scale)

        self.pred_real = self.nets[f'netD{self.current_scale}'](self.real_img)
        self.pred_fake = self.nets[f'netD{self.current_scale}'](
            self.fake_img.detach() if self.update_D else self.fake_img)

        self.visual_items[f'fake_img_scale{self.current_scale}'] = self.fake_img
        self.visual_items[
            f'recon_img_scale{self.current_scale}'] = self.recon_img
        if self.is_finetune:
            self.visual_items[f'prev_img_scale{self.current_scale}'] = x_prev
            self.visual_items[
                f'quant_prev_img_scale{self.current_scale}'] = x_prev_quant

    def train_iter(self, optimizers=None):
        if self.current_iter % self.scale_iters == 0:
            self.current_scale += 1
            self.scale_prepare()

        self.z_pyramid = [
            paddle.randn(pad_shape(self.reals[i].shape, self.niose_pad_size))
            for i in range(self.current_scale + 1)
        ]

        self.update_D = (self.current_iter %
                         (self.disc_iters + self.gen_iters) < self.disc_iters)
        self.set_requires_grad(self.nets[f'netD{self.current_scale}'],
                               self.update_D)
        self.forward()
        if self.update_D:
            optimizers[f'optim_netD{self.current_scale}'].clear_grad()
            self.backward_D()
            optimizers[f'optim_netD{self.current_scale}'].step()
        else:
            optimizers[f'optim_netG{self.current_scale}'].clear_grad()
            self.backward_G()
            optimizers[f'optim_netG{self.current_scale}'].step()

        self.current_iter += 1

    def test_iter(self, metrics=None):
        z_pyramid = [
            paddle.randn(pad_shape(self.reals[i].shape, self.niose_pad_size))
            for i in range(self.scale_num)
        ]
        self.nets['netG'].eval()
        fake_img = self.nets['netG'](z_pyramid,
                                     paddle.zeros(
                                         pad_shape(z_pyramid[0].shape,
                                                   -self.niose_pad_size)),
                                     self.scale_num - 1, 0)
        self.visual_items['fake_img_test'] = fake_img
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(fake_img, self.train_image)
        self.nets['netG'].train()

    class InferGenerator(paddle.nn.Layer):

        def set_config(self, generator, noise_shapes, scale_num):
            self.generator = generator
            self.noise_shapes = noise_shapes
            self.scale_num = scale_num

        def forward(self, x):
            coarsest_shape = self.generator._coarsest_shape
            z_pyramid = [paddle.randn(shp) for shp in self.noise_shapes]
            x_init = paddle.zeros(coarsest_shape)
            out = self.generator(z_pyramid, x_init, self.scale_num - 1, 0)
            return out

    def export_model(self,
                     export_model=None,
                     output_dir=None,
                     inputs_size=None,
                     export_serving_model=False,
                     model_name=None):
        noise_shapes = [
            pad_shape(x.shape, self.niose_pad_size) for x in self.reals
        ]
        infer_generator = self.InferGenerator()
        infer_generator.set_config(self.nets['netG'], noise_shapes,
                                   self.scale_num)
        paddle.jit.save(infer_generator,
                        os.path.join(output_dir, "singan_random_sample"),
                        input_spec=[1])
