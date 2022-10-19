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

import os

import paddle
import paddle.nn as nn

from .builder import MODELS
from .base_model import BaseModel
from .generators.builder import build_generator
from .criterions.builder import build_criterion
from ..utils.visual import tensor2img


@MODELS.register()
class NAFNetModel(BaseModel):
    """NAFNet Model.

    Paper: Simple Baselines for Image Restoration
    https://arxiv.org/pdf/2204.04676
    """

    def __init__(self, generator, psnr_criterion=None):
        """Initialize the MPR class.

        Args:
            generator (dict): config of generator.
            psnr_criterion (dict): config of psnr criterion.
        """
        super(NAFNetModel, self).__init__(generator)
        self.current_iter = 1

        self.nets['generator'] = build_generator(generator)

        if psnr_criterion:
            self.psnr_criterion = build_criterion(psnr_criterion)

    def setup_input(self, input):
        self.target = input[0]
        self.lq = input[1]

    def train_iter(self, optims=None):
        optims['optim'].clear_gradients()

        restored = self.nets['generator'](self.lq)

        loss = self.psnr_criterion(restored, self.target)

        loss.backward()
        optims['optim'].step()
        self.losses['loss'] = loss.numpy()

    def forward(self):
        pass

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        with paddle.no_grad():
            self.output = self.nets['generator'](self.lq)
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

    def export_model(self,
                     export_model=None,
                     output_dir=None,
                     inputs_size=None,
                     export_serving_model=False,
                     model_name=None):
        shape = inputs_size[0]
        new_model = self.nets['generator']
        new_model.eval()
        input_spec = [paddle.static.InputSpec(shape=shape, dtype="float32")]

        static_model = paddle.jit.to_static(new_model, input_spec=input_spec)

        if output_dir is None:
            output_dir = 'inference_model'
        if model_name is None:
            model_name = '{}_{}'.format(self.__class__.__name__.lower(),
                                        export_model[0]['name'])

        paddle.jit.save(static_model, os.path.join(output_dir, model_name))
