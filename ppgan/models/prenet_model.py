#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
from .generators.iconvsr import EDVRFeatureExtractor
from .generators.basicvsr import ResidualBlockNoBN, PixelShufflePack, SPyNet
from ..modules.init import reset_parameters
from ..utils.visual import tensor2img


@MODELS.register()
class PReNetModel(BaseSRModel):
    """PReNet Model.

    Paper: Progressive Image Deraining Networks: A Better and Simpler Baseline, IEEE,2019
    """

    def __init__(self, generator, pixel_criterion=None):
        """Initialize the BasicVSR class.

        Args:
            generator (dict): config of generator.
            fix_iter (dict): config of fix_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(PReNetModel, self).__init__(generator, pixel_criterion)
        self.current_iter = 1
        self.flag = True

    def setup_input(self, input):
        self.lq = input['lq']
        self.visual_items['lq'] = self.lq[0, :, :, :]
        if 'gt' in input:
            self.gt = input['gt']
            self.visual_items['gt'] = self.gt[0, :, :, :]
        self.image_paths = input['lq_path']

    def train_iter(self, optims=None):
        optims['optim'].clear_grad()
        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output[0, :, :, :]
        # pixel loss
        loss_pixel = -self.pixel_criterion(self.output, self.gt)
        loss_pixel.backward()
        optims['optim'].step()

        self.losses['loss_pixel'] = loss_pixel
        self.current_iter += 1

    def test_iter(self, metrics=None):
        self.gt = self.gt.cpu()
        self.nets['generator'].eval()
        with paddle.no_grad():
            output = self.nets['generator'](self.lq)
            self.visual_items['output'] = output[0, :, :, :].cpu()
        self.nets['generator'].train()

        out_img = []
        gt_img = []

        out_tensor = output[0]
        gt_tensor = self.gt[0]
        out_img = tensor2img(out_tensor, (0., 1.))
        gt_img = tensor2img(gt_tensor, (0., 1.))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img, is_seq=True)
