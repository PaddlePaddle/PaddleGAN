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

import paddle
import paddle.nn as nn
import math
import cv2
import numpy as np
from ppgan.models.generators import StyleGANv2Generator
from ppgan.models.discriminators.discriminator_styleganv2 import ConvLayer
from ppgan.modules.equalized import EqualLinear
from ppgan.faceutils.face_detection.detection.blazeface.utils import *
from ppgan.utils.download import get_path_from_url

GPEN_weights = 'https://paddlegan.bj.bcebos.com/models/GPEN-512.pdparams'

class GPEN(nn.Layer):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super(GPEN, self).__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.log_size = int(math.log(size, 2))
        self.generator = StyleGANv2Generator(size, 
                                             style_dim, 
                                             n_mlp, 
                                             channel_multiplier=channel_multiplier, 
                                             blur_kernel=blur_kernel, 
                                             lr_mlp=lr_mlp,
                                             isconcat=True)
        
        conv = [ConvLayer(3, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            #conv = [ResBlock(in_channel, out_channel, blur_kernel)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu'))

    def forward(self,
        inputs,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        noise = []
        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
            #print(inputs.shape)
        inputs = inputs.reshape([inputs.shape[0], -1])
        outs = self.final_linear(inputs)
        #print(outs.shape)
        outs = self.generator([outs], return_latents, inject_index, truncation,
                              truncation_latent, input_is_latent, 
                              noise=noise[::-1])
        return outs


class FaceEnhancement(object):
    def __init__(self,
                 path_to_enhance=None,
                 size = 512,
                 batch_size=1
                 ):
        super(FaceEnhancement, self).__init__()

        # Initialise the face detector
        if path_to_enhance is None:
            model_weights_path = get_path_from_url(GPEN_weights)
            model_weights = paddle.load(model_weights_path)
        else:
            model_weights = paddle.load(path_to_enhance)
            
        self.face_enhance = GPEN(size=512, style_dim=512, n_mlp=8)
        self.face_enhance.load_dict(model_weights)
        self.face_enhance.eval()
        self.size = size
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = paddle.tile(paddle.to_tensor(self.mask).unsqueeze(0).unsqueeze(-1), repeat_times=[batch_size,1,1,3]).numpy()
        

    def enhance_from_image(self, img):
        if isinstance(img, np.ndarray):
            img, _ = resize_and_crop_image(img, 512)
            img = paddle.to_tensor(img).transpose([2, 0, 1])
        else:
            assert img.shape == [3, 512, 512]
        return self.enhance_from_batch(img.unsqueeze(0))[0]

    def enhance_from_batch(self, img):
        if isinstance(img, np.ndarray):
            img_ori, _ = resize_and_crop_batch(img, 512)
            img = paddle.to_tensor(img_ori).transpose([0, 3, 1, 2])
        else:
            assert img.shape[1:] == [3, 512, 512]
        img_t = (img/255. - 0.5) / 0.5
        
        with paddle.no_grad():
            out, __ = self.face_enhance(img_t)
        
        image_tensor = out * 0.5 + 0.5
        image_tensor = image_tensor.transpose([0, 2, 3, 1]) # RGB
        image_numpy = paddle.clip(image_tensor, 0, 1) * 255.0
        
        out = image_numpy.astype(np.uint8).cpu().numpy()
        return out * self.mask + (1-self.mask) * img_ori 
