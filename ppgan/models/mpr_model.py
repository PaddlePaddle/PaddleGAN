#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn as nn

from .builder import MODELS
from .base_model import BaseModel
from .generators.builder import build_generator
from .criterions.builder import build_criterion
from ..modules.init import reset_parameters, init_weights
from ..utils.visual import tensor2img


@MODELS.register()
class MPRModel(BaseModel):
    """MPR Model.

    Paper: MPR: Multi-Stage Progressive Image Restoration (CVPR 2021).
    https://arxiv.org/abs/2102.02808
    """
    def __init__(self, generator, char_criterion=None, edge_criterion=None):
        """Initialize the MPR class.

        Args:
            generator (dict): config of generator.
            char_criterion (dict): config of char criterion.
            edge_criterion (dict): config of edge criterion.
        """
        super(MPRModel, self).__init__(generator)
        self.current_iter = 1

        self.nets['generator'] = build_generator(generator)
        init_weights(self.nets['generator'])

        if char_criterion:
            self.char_criterion = build_criterion(char_criterion)
        if edge_criterion:
            self.edge_criterion = build_criterion(edge_criterion)

    def setup_input(self, input):
        self.target = input[0]
        self.lq = input[1]

    def train_iter(self, optims=None):
        optims['optim'].clear_gradients()

        restored = self.nets['generator'](self.lq)

        loss_char = []
        loss_edge = []

        for i in range(len(restored)):
            loss_char.append(self.char_criterion(restored[i], self.target))
            loss_edge.append(self.edge_criterion(restored[i], self.target))
        loss_char = paddle.stack(loss_char)
        loss_edge = paddle.stack(loss_edge)
        loss_char = paddle.sum(loss_char)
        loss_edge = paddle.sum(loss_edge)

        loss = (loss_char) + (0.05 * loss_edge)

        loss.backward()
        optims['optim'].step()
        self.losses['loss'] = loss.numpy()

    def forward(self):
        pass

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        with paddle.no_grad():
            self.output = self.nets['generator'](self.lq)[0]
            self.visual_items['output'] = self.output
        self.nets['generator'].train()

        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.target):
            out_img.append(tensor2img(out_tensor, (0., 1.)))
            gt_img.append(tensor2img(gt_tensor, (0., 1.)))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img)
