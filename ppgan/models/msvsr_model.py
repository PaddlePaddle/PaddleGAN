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

import paddle
import paddle.nn as nn

from .builder import MODELS
from .sr_model import BaseSRModel
from .generators.basicvsr import ResidualBlockNoBN, PixelShufflePack, SPyNet
from .generators.msvsr import ModifiedSPyNet
from ..modules.init import reset_parameters
from ..utils.visual import tensor2img


@MODELS.register()
class MultiStageVSRModel(BaseSRModel):
    """PP-MSVSR Model.

    Paper:
        PP-MSVSR: Multi-Stage Video Super-Resolution, 2021
    """
    def __init__(self, generator, fix_iter, pixel_criterion=None):
        """Initialize the PP-MSVSR class.

        Args:
            generator (dict): config of generator.
            fix_iter (dict): config of fix_iter.
            pixel_criterion (dict): config of pixel criterion.
        """
        super(MultiStageVSRModel, self).__init__(generator, pixel_criterion)
        self.fix_iter = fix_iter
        self.current_iter = 1
        self.flag = True
        init_basicvsr_weight(self.nets['generator'])
        if not self.fix_iter:
            print('init train all parameters!!!')
            for name, param in self.nets['generator'].named_parameters():
                param.trainable = True
                if 'spynet' in name:
                    param.optimize_attr['learning_rate'] = 0.25

    def setup_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])
        self.visual_items['lq'] = self.lq[:, 0, :, :, :]
        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])
            self.visual_items['gt'] = self.gt[:, 0, :, :, :]
        self.image_paths = input['lq_path']

    def train_iter(self, optims=None):
        optims['optim'].clear_grad()
        if self.fix_iter:
            if self.current_iter == 1:
                print('Train MSVSR with fixed spynet for', self.fix_iter,
                      'iters.')
                for name, param in self.nets['generator'].named_parameters():
                    if 'spynet' in name:
                        param.trainable = False
            elif self.current_iter >= self.fix_iter + 1 and self.flag:
                print('Train all the parameters.')
                for name, param in self.nets['generator'].named_parameters():
                    param.trainable = True
                    if 'spynet' in name:
                        param.optimize_attr['learning_rate'] = 0.25
                self.flag = False
                for net in self.nets.values():
                    net.find_unused_parameters = False

        output = self.nets['generator'](self.lq)
        if isinstance(output, (list, tuple)):
            out_stage2, output = output
            loss_pix_stage2 = self.pixel_criterion(out_stage2, self.gt)
            self.losses['loss_pix_stage2'] = loss_pix_stage2
        self.visual_items['output'] = output[:, 0, :, :, :]
        # pixel loss
        loss_pix = self.pixel_criterion(output, self.gt)
        self.losses['loss_pix'] = loss_pix

        self.loss = sum(_value for _key, _value in self.losses.items()
                        if 'loss_pix' in _key)
        self.losses['loss'] = self.loss

        self.loss.backward()
        optims['optim'].step()

        self.current_iter += 1

    def test_iter(self, metrics=None):
        self.gt = self.gt.cpu()
        self.nets['generator'].eval()
        with paddle.no_grad():
            output = self.nets['generator'](self.lq)
            if isinstance(output, (list, tuple)):
                out_stage1, output = output
        self.nets['generator'].train()

        out_img = []
        gt_img = []

        _, t, _, _, _ = self.gt.shape
        for i in range(t):
            out_tensor = output[0, i]
            gt_tensor = self.gt[0, i]
            out_img.append(tensor2img(out_tensor, (0., 1.)))
            gt_img.append(tensor2img(gt_tensor, (0., 1.)))

        if metrics is not None:
            for metric in metrics.values():
                metric.update(out_img, gt_img, is_seq=True)


def init_basicvsr_weight(net):
    for m in net.children():
        if hasattr(m,
                   'weight') and not isinstance(m,
                                                (nn.BatchNorm, nn.BatchNorm2D)):
            reset_parameters(m)
            continue

        if (not isinstance(
                m,
            (ResidualBlockNoBN, PixelShufflePack, SPyNet, ModifiedSPyNet))):
            init_basicvsr_weight(m)
