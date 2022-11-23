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
import math
import os
import random
import numpy as np
import paddle
import sys

sys.path.append(".")
from .base_predictor import BasePredictor
from ppgan.datasets.gpen_dataset import GFPGAN_degradation
from ppgan.models.generators import GPENGenerator
from ppgan.metrics.fid import FID
from ppgan.utils.download import get_path_from_url
import cv2

import warnings

model_cfgs = {
    'gpen-ffhq-256': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/gpen-ffhq-256-generator.pdparams',
        'size': 256,
        'style_dim': 512,
        'n_mlp': 8,
        'channel_multiplier': 1,
        'narrow': 0.5
    }
}


def psnr(pred, gt):
    pred = paddle.clip(pred, min=0, max=1)
    gt = paddle.clip(gt, min=0, max=1)
    imdff = np.asarray(pred - gt)
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


def data_loader(path, size=256):
    degrader = GFPGAN_degradation()

    img_gt = cv2.imread(path, cv2.IMREAD_COLOR)

    img_gt = cv2.resize(img_gt, (size, size), interpolation=cv2.INTER_NEAREST)

    img_gt = img_gt.astype(np.float32) / 255.
    img_gt, img_lq = degrader.degrade_process(img_gt)

    img_gt = (paddle.to_tensor(img_gt) - 0.5) / 0.5
    img_lq = (paddle.to_tensor(img_lq) - 0.5) / 0.5

    img_gt = img_gt.transpose([2, 0, 1]).flip(0).unsqueeze(0)
    img_lq = img_lq.transpose([2, 0, 1]).flip(0).unsqueeze(0)

    return np.array(img_lq).astype('float32'), np.array(img_gt).astype(
        'float32')


class GPENPredictor(BasePredictor):

    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 model_type=None,
                 seed=100,
                 size=256,
                 style_dim=512,
                 n_mlp=8,
                 channel_multiplier=1,
                 narrow=0.5):
        self.output_path = output_path
        self.size = size
        if weight_path is None:
            if model_type in model_cfgs.keys():
                weight_path = get_path_from_url(
                    model_cfgs[model_type]['model_urls'])
                size = model_cfgs[model_type].get('size', size)
                style_dim = model_cfgs[model_type].get('style_dim', style_dim)
                n_mlp = model_cfgs[model_type].get('n_mlp', n_mlp)
                channel_multiplier = model_cfgs[model_type].get(
                    'channel_multiplier', channel_multiplier)
                narrow = model_cfgs[model_type].get('narrow', narrow)
                checkpoint = paddle.load(weight_path)
            else:
                raise ValueError(
                    'Predictor need a weight path or a pretrained model type')
        else:
            checkpoint = paddle.load(weight_path)

        warnings.filterwarnings("always")
        self.generator = GPENGenerator(size, style_dim, n_mlp, channel_multiplier,
                              narrow)
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

        if seed is not None:
            paddle.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def run(self, img_path):
        os.makedirs(self.output_path, exist_ok=True)
        input_array, target_array = data_loader(img_path, self.size)
        input_tensor = paddle.to_tensor(input_array)
        target_tensor = paddle.to_tensor(target_array)

        FID_model = FID(use_GPU=True)

        with paddle.no_grad():
            output, _ = self.generator(input_tensor)
            psnr_score = psnr(target_tensor, output)
            FID_model.update(output, target_tensor)
            fid_score = FID_model.accumulate()

        input_tensor = input_tensor.transpose([0, 2, 3, 1])
        target_tensor = target_tensor.transpose([0, 2, 3, 1])
        output = output.transpose([0, 2, 3, 1])
        sample_result = paddle.concat(
            (input_tensor[0], output[0], target_tensor[0]), 1)
        sample = cv2.cvtColor((sample_result.numpy() + 1) / 2 * 255,
                              cv2.COLOR_RGB2BGR)
        file_name = self.output_path + '/gpen_predict.png'
        cv2.imwrite(file_name, sample)
        print(f"result saved in : {file_name}")
        print(f"\tFID: {fid_score}\n\tPSNR:{psnr_score}")
