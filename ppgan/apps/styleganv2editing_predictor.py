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
import cv2
import numpy as np
import paddle

from ppgan.utils.download import get_path_from_url
from .styleganv2_predictor import StyleGANv2Predictor

model_cfgs = {
    'ffhq-config-f': {
        'direction_urls':
        'https://paddlegan.bj.bcebos.com/models/stylegan2-ffhq-config-f-directions.pdparams'
    }
}


def make_image(tensor):
    return (((tensor.detach() + 1) / 2 * 255).clip(min=0, max=255).transpose(
        (0, 2, 3, 1)).numpy().astype('uint8'))


class StyleGANv2EditingPredictor(StyleGANv2Predictor):
    def __init__(self, model_type=None, direction_path=None, **kwargs):
        super().__init__(model_type=model_type, **kwargs)

        if direction_path is None and model_type is not None:
            assert model_type in model_cfgs, f'There is not any pretrained direction file for {model_type} model.'
            direction_path = get_path_from_url(
                model_cfgs[model_type]['direction_urls'])
        self.directions = paddle.load(direction_path)

    @paddle.no_grad()
    def run(self, latent, direction, offset):

        latent = paddle.to_tensor(
            np.load(latent)).unsqueeze(0).astype('float32')
        direction = self.directions[direction].unsqueeze(0).astype('float32')

        latent_n = paddle.concat([latent, latent + offset * direction], 0)
        generator = self.generator
        img_gen, _ = generator([latent_n],
                               input_is_latent=True,
                               randomize_noise=False)
        imgs = make_image(img_gen)
        src_img = imgs[0]
        dst_img = imgs[1]

        dst_latent = (latent + offset * direction)[0].numpy().astype('float32')

        os.makedirs(self.output_path, exist_ok=True)
        save_src_path = os.path.join(self.output_path, 'src.editing.png')
        cv2.imwrite(save_src_path, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
        save_dst_path = os.path.join(self.output_path, 'dst.editing.png')
        cv2.imwrite(save_dst_path, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))
        save_npy_path = os.path.join(self.output_path, 'dst.editing.npy')
        np.save(save_npy_path, dst_latent)

        return src_img, dst_img, dst_latent
