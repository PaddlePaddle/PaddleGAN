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
from .sr_model import BaseSRModel
from .generators.edvr import ResidualBlockNoBN, DCNPack
from ..modules.init import reset_parameters


@MODELS.register()
class EDVRModel(BaseSRModel):
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    """
    def __init__(self, generator, tsa_iter, pixel_criterion=None):
        """Initialize the EDVR class.

        Args:
            generator (dict): config of generator.
            tsa_iter (dict): config of tsa_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(EDVRModel, self).__init__(generator, pixel_criterion)
        self.tsa_iter = tsa_iter
        self.current_iter = 1
        init_edvr_weight(self.nets['generator'])

    def setup_input(self, input):
        self.lq = input['lq']
        self.visual_items['lq'] = self.lq[:, 2, :, :, :]
        self.visual_items['lq-2'] = self.lq[:, 0, :, :, :]
        self.visual_items['lq-1'] = self.lq[:, 1, :, :, :]
        self.visual_items['lq+1'] = self.lq[:, 3, :, :, :]
        self.visual_items['lq+2'] = self.lq[:, 4, :, :, :]
        if 'gt' in input:
            self.gt = input['gt'][:, 0, :, :, :]
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']

    def train_iter(self, optims=None):
        optims['optim'].clear_grad()
        if self.tsa_iter:
            if self.current_iter == 1:
                print('Only train TSA module for', self.tsa_iter, 'iters.')
                for name, param in self.nets['generator'].named_parameters():
                    if 'TSAModule' not in name:
                        param.trainable = False
            elif self.current_iter == self.tsa_iter + 1:
                print('Train all the parameters.')
                for param in self.nets['generator'].parameters():
                    param.trainable = True
        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output
        # pixel loss
        loss_pixel = self.pixel_criterion(self.output, self.gt)
        self.losses['loss_pixel'] = loss_pixel

        loss_pixel.backward()
        optims['optim'].step()
        self.current_iter += 1


def init_edvr_weight(net):
    def reset_func(m):
        if hasattr(m, 'weight') and (not isinstance(
                m, (nn.BatchNorm, nn.BatchNorm2D))) and (
                    not isinstance(m, ResidualBlockNoBN) and
                    (not isinstance(m, DCNPack))):
            reset_parameters(m)

    net.apply(reset_func)
