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
from .styleganv2_predictor import StyleGANv2Predictor


def make_image(tensor):
    return (((tensor.detach() + 1) / 2 * 255).clip(min=0, max=255).transpose(
        (0, 2, 3, 1)).numpy().astype('uint8'))


class StyleGANv2MixingPredictor(StyleGANv2Predictor):
    @paddle.no_grad()
    def run(self, latent1, latent2, weights=[0.5] * 18):

        latent1 = paddle.to_tensor(np.load(latent1)).unsqueeze(0)
        latent2 = paddle.to_tensor(np.load(latent2)).unsqueeze(0)
        assert latent1.shape[1] == latent2.shape[1] == len(
            weights
        ), 'latents and their weights should have the same level nums.'
        mix_latent = []
        for i, weight in enumerate(weights):
            mix_latent.append(latent1[:, i:i + 1] * weight +
                              latent2[:, i:i + 1] * (1 - weight))
        mix_latent = paddle.concat(mix_latent, 1)
        latent_n = paddle.concat([latent1, latent2, mix_latent], 0)
        generator = self.generator
        img_gen, _ = generator([latent_n],
                               input_is_latent=True,
                               randomize_noise=False)
        imgs = make_image(img_gen)
        src_img1 = imgs[0]
        src_img2 = imgs[1]
        dst_img = imgs[2]

        os.makedirs(self.output_path, exist_ok=True)
        save_src_path = os.path.join(self.output_path, 'src1.mixing.png')
        cv2.imwrite(save_src_path, cv2.cvtColor(src_img1, cv2.COLOR_RGB2BGR))
        save_src_path = os.path.join(self.output_path, 'src2.mixing.png')
        cv2.imwrite(save_src_path, cv2.cvtColor(src_img2, cv2.COLOR_RGB2BGR))
        save_dst_path = os.path.join(self.output_path, 'dst.mixing.png')
        cv2.imwrite(save_dst_path, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))

        return src_img1, src_img2, dst_img
