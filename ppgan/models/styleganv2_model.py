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
import os
import math
import random
import paddle
import paddle.nn as nn
from .base_model import BaseModel

from .builder import MODELS
from .criterions import build_criterion
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..solver import build_lr_scheduler, build_optimizer



def r1_penalty(real_pred, real_img):
    """
    R1 regularization for discriminator. The core idea is to
    penalize the gradient on real data alone: when the
    generator distribution produces the true data distribution
    and the discriminator is equal to 0 on the data manifold, the
    gradient penalty ensures that the discriminator cannot create
    a non-zero gradient orthogonal to the data manifold without
    suffering a loss in the GAN game.

    Ref:
    Eq. 9 in Which training methods for GANs do actually converge.
    """

    grad_real = paddle.grad(outputs=real_pred.sum(),
                            inputs=real_img,
                            create_graph=True)[0]
    grad_penalty = (grad_real * grad_real).reshape([grad_real.shape[0],
                                                    -1]).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = paddle.randn(fake_img.shape) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = paddle.grad(outputs=(fake_img * noise).sum(),
                       inputs=latents,
                       create_graph=True)[0]
    path_lengths = paddle.sqrt((grad * grad).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() -
                                            mean_path_length)

    path_penalty = ((path_lengths - path_mean) *
                    (path_lengths - path_mean)).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


@MODELS.register()
class StyleGAN2Model(BaseModel):
    """
    This class implements the StyleGANV2 model, for learning image-to-image translation without paired data.

    StyleGAN2 paper: https://arxiv.org/pdf/1912.04958.pdf
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_criterion=None,
                 num_style_feat=512,
                 mixing_prob=0.9,
                 r1_reg_weight=10.,
                 path_reg_weight=2.,
                 path_batch_shrink=2.,
                 params=None,
                 max_eval_steps=50000):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            gan_criterion (dict): config of gan criterion.
        """
        super(StyleGAN2Model, self).__init__(params)
        self.gen_iters = 4 if self.params is None else self.params.get(
            'gen_iters', 4)
        self.disc_iters = 16 if self.params is None else self.params.get(
            'disc_iters', 16)
        self.disc_start_iters = (0 if self.params is None else self.params.get(
            'disc_start_iters', 0))

        self.visual_iters = (500 if self.params is None else self.params.get(
            'visual_iters', 500))

        self.mixing_prob = mixing_prob
        self.num_style_feat = num_style_feat
        self.r1_reg_weight = r1_reg_weight

        self.path_reg_weight = path_reg_weight
        self.path_batch_shrink = path_batch_shrink
        self.mean_path_length = 0

        self.nets['gen'] = build_generator(generator)
        self.max_eval_steps = max_eval_steps

        # define discriminators
        if discriminator:
            self.nets['disc'] = build_discriminator(discriminator)

            self.nets['gen_ema'] = build_generator(generator)
            self.model_ema(0)

            self.nets['gen'].train()
            self.nets['gen_ema'].eval()
            self.nets['disc'].train()
            self.current_iter = 1

        # define loss functions
        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

    def setup_lr_schedulers(self, cfg):
        self.lr_scheduler = dict()
        gen_cfg = cfg.copy()
        net_g_reg_ratio = self.gen_iters / (self.gen_iters + 1)
        gen_cfg['learning_rate'] = cfg['learning_rate'] * net_g_reg_ratio
        self.lr_scheduler['gen'] = build_lr_scheduler(gen_cfg)

        disc_cfg = cfg.copy()
        net_d_reg_ratio = self.disc_iters / (self.disc_iters + 1)
        disc_cfg['learning_rate'] = cfg['learning_rate'] * net_d_reg_ratio
        self.lr_scheduler['disc'] = build_lr_scheduler(disc_cfg)
        return self.lr_scheduler

    def setup_optimizers(self, lr, cfg):
        for opt_name, opt_cfg in cfg.items():
            if opt_name == 'optimG':
                _lr = lr['gen']
            elif opt_name == 'optimD':
                _lr = lr['disc']
            else:
                raise ValueError("opt name must be in ['optimG', optimD]")

            cfg_ = opt_cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()
            self.optimizers[opt_name] = build_optimizer(cfg_, _lr, parameters)

        return self.optimizers

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with DataParallel.
        """
        if isinstance(net, (paddle.DataParallel)):
            net = net._layers
        return net

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.nets['gen'])
        net_g_params = dict(net_g.named_parameters())

        neg_g_ema = self.get_bare_model(self.nets['gen_ema'])
        net_g_ema_params = dict(neg_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].set_value(net_g_ema_params[k] * (decay) +
                                          (net_g_params[k] * (1 - decay)))

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        """
        self.real_img = paddle.to_tensor(input['A'])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = paddle.randn([batch, self.num_style_feat])
        else:
            noises = []
            for _ in range(num_noise):
                noises.append(paddle.randn([batch, self.num_style_feat]))
        return noises

    def mixing_noise(self, batch, prob):
        if random.random() < prob:
            return self.make_noise(batch, 2)
        else:
            return [self.make_noise(batch, 1)]

    def train_iter(self, optimizers=None):
        current_iter = self.current_iter
        self.set_requires_grad(self.nets['disc'], True)
        optimizers['optimD'].clear_grad()
        batch = self.real_img.shape[0]
        noise = self.mixing_noise(batch, self.mixing_prob)

        fake_img, _ = self.nets['gen'](noise)
        self.visual_items['real_img'] = self.real_img
        self.visual_items['fake_img'] = fake_img
        fake_pred = self.nets['disc'](fake_img.detach())

        real_pred = self.nets['disc'](self.real_img)
        # wgan loss with softplus (logistic loss) for discriminator
        l_d_total = 0.
        l_d = self.gan_criterion(real_pred, True,
                                 is_disc=True) + self.gan_criterion(
                                     fake_pred, False, is_disc=True)
        self.losses['l_d'] = l_d
        # In wgan, real_score should be positive and fake_score should be
        # negative
        self.losses['real_score'] = real_pred.detach().mean()
        self.losses['fake_score'] = fake_pred.detach().mean()

        l_d_total += l_d

        if current_iter % self.disc_iters == 0:
            self.real_img.stop_gradient = False
            real_pred = self.nets['disc'](self.real_img)
            l_d_r1 = r1_penalty(real_pred, self.real_img)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.disc_iters +
                      0 * real_pred[0])

            self.losses['l_d_r1'] = l_d_r1.detach().mean()

            l_d_total += l_d_r1
        l_d_total.backward()

        optimizers['optimD'].step()

        self.set_requires_grad(self.nets['disc'], False)
        optimizers['optimG'].clear_grad()

        noise = self.mixing_noise(batch, self.mixing_prob)
        fake_img, _ = self.nets['gen'](noise)
        fake_pred = self.nets['disc'](fake_img)

        # wgan loss with softplus (non-saturating loss) for generator
        l_g_total = 0.
        l_g = self.gan_criterion(fake_pred, True, is_disc=False)
        self.losses['l_g'] = l_g

        l_g_total += l_g
        if current_iter % self.gen_iters == 0:
            path_batch_size = max(1, int(batch // self.path_batch_shrink))
            noise = self.mixing_noise(path_batch_size, self.mixing_prob)
            fake_img, latents = self.nets['gen'](noise, return_latents=True)
            l_g_path, path_lengths, self.mean_path_length = g_path_regularize(
                fake_img, latents, self.mean_path_length)

            l_g_path = (self.path_reg_weight * self.gen_iters * l_g_path +
                        0 * fake_img[0, 0, 0, 0])

            l_g_total += l_g_path
            self.losses['l_g_path'] = l_g_path.detach().mean()
            self.losses['path_length'] = path_lengths
        l_g_total.backward()
        optimizers['optimG'].step()

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

        if self.current_iter % self.visual_iters:
            sample_z = [self.make_noise(1, 1)]
            sample, _ = self.nets['gen_ema'](sample_z)
            self.visual_items['fake_img_ema'] = sample

        self.current_iter += 1

    def test_iter(self, metrics=None):
        self.nets['gen_ema'].eval()
        batch = self.real_img.shape[0]
        noises = [paddle.randn([batch, self.num_style_feat])]
        fake_img, _ = self.nets['gen_ema'](noises)
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(fake_img, self.real_img)
        self.nets['gen_ema'].train()

    class InferGenerator(paddle.nn.Layer):
        def set_generator(self, generator):
            self.generator = generator

        def forward(self, style, truncation):
            truncation_latent = self.generator.get_mean_style()
            out = self.generator(styles=style,
                                 truncation=truncation,
                                 truncation_latent=truncation_latent)
            return out[0]

    def export_model(self,
                     export_model=None,
                     output_dir=None,
                     inputs_size=[[1, 1, 512], [1, 1]],
                     export_serving_model=False):
        infer_generator = self.InferGenerator()
        infer_generator.set_generator(self.nets['gen'])
        style = paddle.rand(shape=inputs_size[0], dtype='float32')
        truncation = paddle.rand(shape=inputs_size[1], dtype='float32')
        if output_dir is None:
            output_dir = 'inference_model'
        paddle.jit.save(infer_generator,
                        os.path.join(output_dir, "stylegan2model_gen"),
                        input_spec=[style, truncation])
