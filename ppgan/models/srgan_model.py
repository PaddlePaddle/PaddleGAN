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
from .base_model import BaseModel
from .losses import GANLoss
from .builder import MODELS


@MODELS.register()
class SRGANModel(BaseModel):
    def __init__(self, cfg):
        super(SRGANModel, self).__init__(cfg)

        # define networks
        self.model_names = ['G']

        self.netG = build_generator(cfg.model.generator)
        self.visual_names = ['LQ', 'GT', 'fake_H']

        # TODO: support srgan train.
        if False:
            # self.netD = build_discriminator(cfg.model.discriminator)
            self.netG.train()
            # self.netD.train()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        # AtoB = self.opt.dataset.train.direction == 'AtoB'
        if 'A' in input:
            self.LQ = paddle.to_tensor(input['A'])
        if 'B' in input:
            self.GT = paddle.to_tensor(input['B'])
        if 'A_paths' in input:
            self.image_paths = input['A_paths']

    def forward(self):
        self.fake_H = self.netG(self.LQ)

    def optimize_parameters(self, step):
        pass
