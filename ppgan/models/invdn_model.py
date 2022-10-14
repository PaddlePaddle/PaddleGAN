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
from ppgan.utils.visual import tensor2img
from ..solver import build_lr_scheduler, build_optimizer


@MODELS.register()
class InvDNModel(BaseModel):
    """InvDN Model.
    Invertible Denoising Network: A Light Solution for Real Noise Removal (CVPR 2021)
    Originally Written by Liu, Yang and Qin, Zhenyue.
    """
    def __init__(self, generator):
        """Initialize the the class.

        Args:
            generator (dict): config of generator.
        """
        super(InvDNModel, self).__init__(generator)
        self.current_iter = 1

        self.nets['generator'] = build_generator(generator)

        self.generator_cfg = generator

    def setup_input(self, input):
        self.noisy = input[0]
        self.gt = input[1]
        self.lq = input[2]

    def train_iter(self, optims=None):
        optims['optim'].clear_gradients()

        noise_channel = 3 * 4**(self.generator_cfg.down_num) - 3
        noise = paddle.randn((self.noisy.shape[0], noise_channel,
                              self.noisy.shape[2], self.noisy.shape[3]))
        output_hq, output_lq = self.nets['generator'](self.noisy, noise)
        output_hq = output_hq[:, :3, :, :]
        output_lq = output_lq[:, :3, :, :]

        self.lq = self.lq.detach()
        l_forw_fit = 16.0 * paddle.mean(
            paddle.sum((output_lq - self.lq)**2, (1, 2, 3)))
        l_back_rec = paddle.mean(
            paddle.sum(
                paddle.sqrt((self.gt - output_hq) * (self.gt - output_hq) +
                            1e-3), (1, 2, 3)))

        l_total = l_forw_fit + l_back_rec

        l_total.backward()
        optims['optim'].step()
        self.losses['loss'] = l_total.numpy()

    def setup_optimizers(self, lr, cfg):
        if cfg.get('name', None):
            cfg_ = cfg.copy()
            net_names = cfg_.pop('net_names')
            parameters = []
            for net_name in net_names:
                parameters += self.nets[net_name].parameters()

            cfg_['grad_clip'] = nn.ClipGradByNorm(cfg_['clip_grad_norm'])
            cfg_.pop('clip_grad_norm')

            self.optimizers['optim'] = build_optimizer(cfg_, lr, parameters)
        else:
            for opt_name, opt_cfg in cfg.items():
                cfg_ = opt_cfg.copy()
                net_names = cfg_.pop('net_names')
                parameters = []
                for net_name in net_names:
                    parameters += self.nets[net_name].parameters()
                self.optimizers[opt_name] = build_optimizer(
                    cfg_, lr, parameters)

        return self.optimizers

    def forward(self):
        pass

    def test_iter(self, metrics=None):
        self.nets['generator'].eval()
        with paddle.no_grad():

            noise_channel = 3 * 4**(self.generator_cfg.down_num) - 3
            noise = paddle.randn((self.noisy.shape[0], noise_channel,
                                  self.noisy.shape[2], self.noisy.shape[3]))
            output_hq, _ = self.nets['generator'](self.noisy, noise)
            output_hq = output_hq[:, :3, :, :]

            self.output = output_hq
            self.visual_items['output'] = self.output

        self.nets['generator'].train()

        out_img = []
        gt_img = []
        for out_tensor, gt_tensor in zip(self.output, self.gt):
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

        noise_channel = 3 * 4**(self.generator_cfg.down_num) - 3
        noise_shape = (shape[0], noise_channel, shape[2], shape[3])
        input_spec = [
            paddle.static.InputSpec(shape=shape, dtype="float32"),
            paddle.static.InputSpec(shape=noise_shape, dtype="float32")
        ]

        static_model = paddle.jit.to_static(new_model, input_spec=input_spec)

        if output_dir is None:
            output_dir = 'inference_model'
        if model_name is None:
            model_name = '{}_{}'.format(self.__class__.__name__.lower(),
                                        export_model[0]['name'])

        paddle.jit.save(static_model, os.path.join(output_dir, model_name))
