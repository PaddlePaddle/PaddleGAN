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

from collections import OrderedDict
import paddle
import paddle.nn as nn

from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..solver import build_optimizer
from .base_model import BaseModel
from .losses import GANLoss
from .builder import MODELS

import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from .builder import MODELS


@MODELS.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""
    def __init__(self, cfg):
        super(SRModel, self).__init__(cfg)

        self.model_names = ['G']

        self.netG = build_generator(cfg.model.generator)
        self.visual_names = ['lq', 'output', 'gt']

        self.loss_names = ['l_total']

        self.optimizers = []
        if self.is_train:
            self.criterionL1 = paddle.nn.L1Loss()

            self.build_lr_scheduler()
            self.optimizer_G = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.netG.parameters())
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])
        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])
        self.image_paths = input['lq_path']

    def forward(self):
        pass

    def test(self):
        """Forward function used in test time.
        """
        with paddle.no_grad():
            self.output = self.netG(self.lq)

    def optimize_parameters(self):
        self.optimizer_G.clear_grad()
        self.output = self.netG(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.criterionL1:
            l_pix = self.criterionL1(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.loss_l_total = l_total
        self.optimizer_G.step()
