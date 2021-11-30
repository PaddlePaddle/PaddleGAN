# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import cv2 as cv
import numpy as np
import urllib.request
from PIL import Image

import paddle
import paddle.nn.functional as F
from paddle.vision.transforms import functional

from ppgan.utils.download import get_path_from_url
from ppgan.utils.visual import tensor2img
from ppgan.models.generators import DecoderNet, Encoder, RevisionNet
from .base_predictor import BasePredictor

LapStyle_circuit_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/lapstyle_circuit.pdparams'
LapStyle_ocean_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/lapstyle_ocean.pdparams'
LapStyle_starrynew_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/lapstyle_starrynew.pdparams'
LapStyle_stars_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/lapstyle_stars.pdparams'


def img(img):
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    # HWC to CHW
    return img


def img_read(content_img_path, style_image_path):
    content_img = cv.imread(content_img_path)
    if content_img.ndim == 2:
        content_img = cv.cvtColor(content_img, cv.COLOR_GRAY2RGB)
    else:
        content_img = cv.cvtColor(content_img, cv.COLOR_BGR2RGB)
    h, w, c = content_img.shape
    content_img = Image.fromarray(content_img)
    content_img = content_img.resize((512, 512), Image.BILINEAR)
    content_img = np.array(content_img)
    content_img = img(content_img)
    content_img = functional.to_tensor(content_img)

    style_img = cv.imread(style_image_path)
    style_img = cv.cvtColor(style_img, cv.COLOR_BGR2RGB)
    style_img = Image.fromarray(style_img)
    style_img = style_img.resize((512, 512), Image.BILINEAR)
    style_img = np.array(style_img)
    style_img = img(style_img)
    style_img = functional.to_tensor(style_img)

    content_img = paddle.unsqueeze(content_img, axis=0)
    style_img = paddle.unsqueeze(style_img, axis=0)
    return content_img, style_img, h, w


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):
    """
    Laplacian

    return:
       x - upsample(downsample(x))
    """
    return x - tensor_resample(
        tensor_resample(x, [x.shape[2] // 2, x.shape[3] // 2]),
        [x.shape[2], x.shape[3]])


def make_laplace_pyramid(x, levels):
    """
    Make Laplacian Pyramid
    """
    pyramid = []
    current = x
    for i in range(levels):
        pyramid.append(laplacian(current))
        current = tensor_resample(
            current,
            (max(current.shape[2] // 2, 1), max(current.shape[3] // 2, 1)))
    pyramid.append(current)
    return pyramid


def fold_laplace_pyramid(pyramid):
    """
    Fold Laplacian Pyramid
    """
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):  # iterate from len-2 to 0
        up_h, up_w = pyramid[i].shape[2], pyramid[i].shape[3]
        current = pyramid[i] + tensor_resample(current, (up_h, up_w))
    return current


class LapStylePredictor(BasePredictor):
    def __init__(self,
                 output='output_dir',
                 style='starrynew',
                 weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'LapStyle')
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.net_enc = Encoder()
        self.net_dec = DecoderNet()
        self.net_rev = RevisionNet()
        self.net_rev_2 = RevisionNet()

        if weight_path is None:
            if style == 'starrynew':
                weight_path = get_path_from_url(LapStyle_starrynew_WEIGHT_URL)
            elif style == 'circuit':
                weight_path = get_path_from_url(LapStyle_circuit_WEIGHT_URL)
            elif style == 'ocean':
                weight_path = get_path_from_url(LapStyle_ocean_WEIGHT_URL)
            elif style == 'stars':
                weight_path = get_path_from_url(LapStyle_stars_WEIGHT_URL)
            else:
                raise Exception(f'has not implemented {style}.')
        self.net_enc.set_dict(paddle.load(weight_path)['net_enc'])
        self.net_enc.eval()
        self.net_dec.set_dict(paddle.load(weight_path)['net_dec'])
        self.net_dec.eval()
        self.net_rev.set_dict(paddle.load(weight_path)['net_rev'])
        self.net_rev.eval()
        self.net_rev_2.set_dict(paddle.load(weight_path)['net_rev_2'])
        self.net_rev_2.eval()

    def run(self, content_img_path, style_image_path):
        content_img, style_img, h, w = img_read(content_img_path,
                                                style_image_path)
        content_img_visual = tensor2img(content_img, min_max=(0., 1.))
        content_img_visual = cv.cvtColor(content_img_visual, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(self.output, 'content.png'), content_img_visual)
        style_img_visual = tensor2img(style_img, min_max=(0., 1.))
        style_img_visual = cv.cvtColor(style_img_visual, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(self.output, 'style.png'), style_img_visual)
        pyr_ci = make_laplace_pyramid(content_img, 2)
        pyr_si = make_laplace_pyramid(style_img, 2)
        pyr_ci.append(content_img)
        pyr_si.append(style_img)
        cF = self.net_enc(pyr_ci[2])
        sF = self.net_enc(pyr_si[2])
        stylized_small = self.net_dec(cF, sF)
        stylized_small_visual = tensor2img(stylized_small, min_max=(0., 1.))
        stylized_small_visual = cv.cvtColor(stylized_small_visual,
                                            cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(self.output, 'stylized_small.png'),
                   stylized_small_visual)
        stylized_up = F.interpolate(stylized_small, scale_factor=2)

        revnet_input = paddle.concat(x=[pyr_ci[1], stylized_up], axis=1)
        stylized_rev_lap = self.net_rev(revnet_input)
        stylized_rev = fold_laplace_pyramid([stylized_rev_lap, stylized_small])
        stylized_rev_visual = tensor2img(stylized_rev, min_max=(0., 1.))
        stylized_rev_visual = cv.cvtColor(stylized_rev_visual, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(self.output, 'stylized_rev_first.png'),
                   stylized_rev_visual)
        stylized_up = F.interpolate(stylized_rev, scale_factor=2)

        revnet_input = paddle.concat(x=[pyr_ci[0], stylized_up], axis=1)
        stylized_rev_lap_second = self.net_rev_2(revnet_input)
        stylized_rev_second = fold_laplace_pyramid(
            [stylized_rev_lap_second, stylized_rev_lap, stylized_small])

        stylized = stylized_rev_second
        stylized_visual = tensor2img(stylized, min_max=(0., 1.))
        stylized_visual = cv.cvtColor(stylized_visual, cv.COLOR_RGB2BGR)
        stylized_visual = cv.resize(stylized_visual, (w, h))
        cv.imwrite(os.path.join(self.output, 'stylized.png'), stylized_visual)

        print('Model LapStyle output images path:', self.output)

        return stylized
