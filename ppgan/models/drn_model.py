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
from .generators.drn import DownBlock
from .sr_model import BaseSRModel
from .builder import MODELS

from .criterions import build_criterion
from ..modules.init import init_weights
from ..utils.visual import tensor2img


@MODELS.register()
class DRN(BaseSRModel):
    """
    This class implements the DRN model.

    DRN paper: https://arxiv.org/pdf/1809.00219.pdf
    """
    def __init__(self,
                 generator,
                 lq_loss_weight=0.1,
                 dual_loss_weight=0.1,
                 discriminator=None,
                 pixel_criterion=None,
                 perceptual_criterion=None,
                 gan_criterion=None,
                 params=None):
        """Initialize the DRN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            pixel_criterion (dict): config of pixel criterion.
            perceptual_criterion (dict): config of perceptual criterion.
            gan_criterion (dict): config of gan criterion.
        """
        super(DRN, self).__init__(generator)
        self.lq_loss_weight = lq_loss_weight
        self.dual_loss_weight = dual_loss_weight
        self.params = params
        self.nets['generator'] = build_generator(generator)
        init_weights(self.nets['generator'])
        negval = generator.negval
        n_feats = generator.n_feats
        n_colors = generator.n_colors
        self.scale = generator.scale

        for i in range(len(self.scale)):
            dual_model = DownBlock(negval, n_feats, n_colors, 2)
            self.nets['dual_model_' + str(i)] = dual_model
            init_weights(self.nets['dual_model_' + str(i)])

        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)

        if pixel_criterion:
            self.pixel_criterion = build_criterion(pixel_criterion)

        if perceptual_criterion:
            self.perceptual_criterion = build_criterion(perceptual_criterion)

        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

    def setup_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])
        self.visual_items['lq'] = self.lq

        if isinstance(self.scale, (list, tuple)) and len(
                self.scale) == 2 and 'lqx2' in input:
            self.lqx2 = input['lqx2']

        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']

    def train_iter(self, optimizers=None):
        lr = [self.lq]

        if hasattr(self, 'lqx2'):
            lr.append(self.lqx2)

        hr = self.gt

        sr = self.nets['generator'](self.lq)

        sr2lr = []

        for i in range(len(self.scale)):
            sr2lr_i = self.nets['dual_model_' + str(i)](sr[i - len(self.scale)])
            sr2lr.append(sr2lr_i)

        # compute primary loss
        loss_primary = self.pixel_criterion(sr[-1], hr)
        for i in range(1, len(sr)):
            if self.lq_loss_weight > 0.0:
                loss_primary += self.pixel_criterion(
                    sr[i - 1 - len(sr)], lr[i - len(sr)]) * self.lq_loss_weight

        # compute dual loss
        loss_dual = self.pixel_criterion(sr2lr[0], lr[0])
        for i in range(1, len(self.scale)):
            if self.dual_loss_weight > 0.0:
                loss_dual += self.pixel_criterion(sr2lr[i],
                                                  lr[i]) * self.dual_loss_weight

        loss_total = loss_primary + loss_dual

        optimizers['optimG'].clear_grad()
        optimizers['optimD'].clear_grad()
        loss_total.backward()
        optimizers['optimG'].step()
        optimizers['optimD'].step()

        self.losses['loss_promary'] = loss_primary
        self.losses['loss_dual'] = loss_dual
        self.losses['loss_total'] = loss_total

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        with paddle.no_grad():
            self.output = self.nets['generator'](self.lq)[-1]
            self.visual_items['output'] = self.output
        self.nets['generator'].train()

        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.gt):
            out_img.append(tensor2img(out_tensor, (0., 255.)))
            gt_img.append(tensor2img(gt_tensor, (0., 255.)))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img)
