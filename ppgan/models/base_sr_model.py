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
class BaseSRModel(BaseModel):
    """Base SR model for single image super-resolution."""
    def __init__(self, cfg):
        super(BaseSRModel, self).__init__(cfg)

        self.nets['netG'] = build_generator(cfg.model.generator)
        nn.LayerList
        if self.is_train:
            # self.criterionL1 = paddle.nn.L1Loss()
            # loss_cfgs = self.cfg.model.losses
            self.build_criterions(cfg.model.losses)

            self.build_lr_scheduler()
            self.optimizers['optimizer_G'] = build_optimizer(
                cfg.optimizer,
                self.lr_scheduler,
                parameter_list=self.nets['netG'].parameters())

    def set_input(self, input):
        self.lq = paddle.fluid.dygraph.to_variable(input['lq'])
        self.visual_items['lq'] = self.lq
        if 'gt' in input:
            self.gt = paddle.fluid.dygraph.to_variable(input['gt'])
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']

    def forward(self):
        pass

    def test(self):
        """Forward function used in test time.
        """
        self.nets['netG'].eval()
        with paddle.no_grad():
            self.output = self.nets['netG'](self.lq)
            self.visual_items['output'] = self.output
        self.nets['netG'].train()

    def optimize_parameters(self):
        self.optimizers['optimizer_G'].clear_grad()
        l_total = 0
        self.output = self.nets['netG'](self.lq)
        self.visual_items['output'] = self.output
        # pixel loss
        if self.criterionL1:
            l_pix = self.criterionL1(self.output, self.gt)
            l_total += l_pix
            self.losses['loss_pix'] = l_pix

        l_total.backward()
        self.optimizers['optimizer_G'].step()
