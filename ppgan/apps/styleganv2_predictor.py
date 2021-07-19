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

import os
import random
import numpy as np
import paddle
from .base_predictor import BasePredictor
from ppgan.models.generators import StyleGANv2Generator
from ppgan.utils.download import get_path_from_url
from ppgan.utils.visual import make_grid, tensor2img, save_image

model_cfgs = {
    'ffhq-config-f': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/stylegan2-ffhq-config-f.pdparams',
        'size': 1024,
        'style_dim': 512,
        'n_mlp': 8,
        'channel_multiplier': 2
    },
    'animeface-512': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/stylegan2-animeface-512.pdparams',
        'size': 512,
        'style_dim': 512,
        'n_mlp': 8,
        'channel_multiplier': 2
    }
}


@paddle.no_grad()
def get_mean_style(generator):
    mean_style = None

    for i in range(10):
        style = generator.mean_latent(1024)

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


@paddle.no_grad()
def sample(generator, mean_style, n_sample):
    image = generator(
        [paddle.randn([n_sample, generator.style_dim])],
        truncation=0.7,
        truncation_latent=mean_style,
    )[0]

    return image


@paddle.no_grad()
def style_mixing(generator, mean_style, n_source, n_target):
    source_code = paddle.randn([n_source, generator.style_dim])
    target_code = paddle.randn([n_target, generator.style_dim])

    resolution = 2**((generator.n_latent + 2) // 2)

    images = [paddle.ones([1, 3, resolution, resolution]) * -1]

    source_image = generator([source_code],
                             truncation_latent=mean_style,
                             truncation=0.7)[0]
    target_image = generator([target_code],
                             truncation_latent=mean_style,
                             truncation=0.7)[0]

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).tile([n_source, 1]), source_code],
            truncation_latent=mean_style,
            truncation=0.7,
        )[0]
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = paddle.concat(images, 0)

    return images


class StyleGANv2Predictor(BasePredictor):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 model_type=None,
                 seed=None,
                 size=1024,
                 style_dim=512,
                 n_mlp=8,
                 channel_multiplier=2):
        self.output_path = output_path

        if weight_path is None:
            if model_type in model_cfgs.keys():
                weight_path = get_path_from_url(
                    model_cfgs[model_type]['model_urls'])
                size = model_cfgs[model_type].get('size', size)
                style_dim = model_cfgs[model_type].get('style_dim', style_dim)
                n_mlp = model_cfgs[model_type].get('n_mlp', n_mlp)
                channel_multiplier = model_cfgs[model_type].get(
                    'channel_multiplier', channel_multiplier)
                checkpoint = paddle.load(weight_path)
            else:
                raise ValueError(
                    'Predictor need a weight path or a pretrained model type')
        else:
            checkpoint = paddle.load(weight_path)

        self.generator = StyleGANv2Generator(size, style_dim, n_mlp,
                                             channel_multiplier)
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

        if seed is not None:
            paddle.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def run(self, n_row=3, n_col=5):
        os.makedirs(self.output_path, exist_ok=True)
        mean_style = get_mean_style(self.generator)

        img = sample(self.generator, mean_style, n_row * n_col)
        save_image(tensor2img(make_grid(img, nrow=n_col)),
                   f'{self.output_path}/sample.png')

        for j in range(2):
            img = style_mixing(self.generator, mean_style, n_col, n_row)
            save_image(tensor2img(make_grid(img, nrow=n_col + 1)),
                       f'{self.output_path}/sample_mixing_{j}.png')
