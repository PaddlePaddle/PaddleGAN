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


@MODELS.register()
class MPRModel(BaseModel):
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
    """
    def __init__(self, generator, char_criterion=None, edge_criterion=None):
        """Initialize the EDVR class.

        Args:
            generator (dict): config of generator.
            tsa_iter (dict): config of tsa_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(MPRModel, self).__init__(generator)
        self.current_iter = 1
        print('char_criterion:', char_criterion)
        print('edge_criterion:', edge_criterion)
        print('generator:', generator)

        self.nets['generator'] = build_generator(generator)
        init_weights(self.nets['generator'])

        if char_criterion:
            self.char_criterion = build_criterion(char_criterion)
        if edge_criterion:
            self.edge_criterion = build_criterion(edge_criterion)

    def setup_input(self, input):
        self.target = input[0]
        self.input_ = input[1]

        #Debug start
        # import os
        # import cv2
        # import numpy as np
        # def save_img(filename, img):
        #     img = img.numpy()
        #     img = img.transpose(0, 2, 3, 1)
        #     N, H, W, C = img.shape
        #     img = img.reshape((H * N, W, C))
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(filename, img)

        # debug_root = 'output_dir'
        # save_img(os.path.join(debug_root, f'{self.current_iter}_target.jpg'), self.target)
        # save_img(os.path.join(debug_root, f'{self.current_iter}_input.jpg'), self.input_)
        # 1/0
        # Debug end

    def train_iter(self, optims=None):
        optims['optim'].clear_gradients()
        # print('train_iter input_:', self.input_)

        import numpy as np
        input_x = paddle.to_tensor(np.load('/home/ps/share/github/ml/paddle/MPRNet/Deblurring/compare/input_.npy'), dtype='float32')

        restored = self.nets['generator'](self.input_)
        # print('train_iter restored:', restored)

        loss_char = []
        loss_edge = []
        
        for i in range(len(restored)):
            loss_char.append(self.char_criterion(restored[i], self.target))
            loss_edge.append(self.edge_criterion(restored[i], self.target))
        loss_char = paddle.stack(loss_char)
        loss_edge = paddle.stack(loss_edge)
        print('train_iter loss_char:', loss_char)
        print('train_iter loss_edge:', loss_edge)
        loss_char = paddle.sum(loss_char)
        loss_edge = paddle.sum(loss_edge)

        # loss_char = paddle.sum([self.char_criterion(restored[j], self.target) for j in range(len(restored))])
        # loss_edge = paddle.sum([self.edge_criterion(restored[j], self.target) for j in range(len(restored))])
        loss = (loss_char) + (0.05*loss_edge)
       
        loss.backward()
        optims['optim'].step()
        self.losses['loss'] = loss.numpy()
        print('train_iter loss', self.losses['loss'])
        1/0

    def forward(self):
        """Run forward pass; called by both functions <train_iter> and <test_iter>."""
        pass


def init_edvr_weight(net):
    def reset_func(m):
        if hasattr(m,
                   'weight') and (not isinstance(m,
                                                 (nn.BatchNorm, nn.BatchNorm2D))):
            reset_parameters(m)

    net.apply(reset_func)

