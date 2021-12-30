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
from natsort import natsorted
from glob import glob
import numpy as np
import cv2
from PIL import Image
import paddle
from .base_predictor import BasePredictor
from ppgan.models.generators import MPRNet
from ppgan.utils.download import get_path_from_url
from ppgan.utils.visual import make_grid, tensor2img, save_image
from ppgan.datasets.mpr_dataset import to_tensor
from paddle.vision.transforms import Pad
from tqdm import tqdm

model_cfgs = {
    'Deblurring': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/MPR_Deblurring.pdparams',
        'n_feat': 96,
        'scale_unetfeats': 48,
        'scale_orsnetfeats': 32,
    },
    'Denoising': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/MPR_Denoising.pdparams',
        'n_feat': 80,
        'scale_unetfeats': 48,
        'scale_orsnetfeats': 32,
    },
    'Deraining': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/MPR_Deraining.pdparams',
        'n_feat': 40,
        'scale_unetfeats': 20,
        'scale_orsnetfeats': 16,
    }
}


class MPRPredictor(BasePredictor):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 seed=None,
                 task=None):
        self.output_path = output_path
        self.task = task
        self.max_size = 640
        self.img_multiple_of = 8

        if weight_path is None:
            if task in model_cfgs.keys():
                weight_path = get_path_from_url(model_cfgs[task]['model_urls'])
                checkpoint = paddle.load(weight_path)
            else:
                raise ValueError(
                    'Predictor need a weight path or a pretrained model type')
        else:
            checkpoint = paddle.load(weight_path)

        self.generator = MPRNet(
            n_feat=model_cfgs[task]['n_feat'],
            scale_unetfeats=model_cfgs[task]['scale_unetfeats'],
            scale_orsnetfeats=model_cfgs[task]['scale_orsnetfeats'])
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

        if seed is not None:
            paddle.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def get_images(self, images_path):
        if os.path.isdir(images_path):
            return natsorted(
                glob(os.path.join(images_path, '*.jpg')) +
                glob(os.path.join(images_path, '*.JPG')) +
                glob(os.path.join(images_path, '*.png')) +
                glob(os.path.join(images_path, '*.PNG')))
        else:
            return [images_path]

    def read_image(self, image_file):
        img = Image.open(image_file).convert('RGB')
        max_length = max(img.width, img.height)
        if max_length > self.max_size:
            ratio = max_length / self.max_size
            dw = int(img.width / ratio)
            dh = int(img.height / ratio)
            img = img.resize((dw, dh))
        return img

    def run(self, images_path=None):
        os.makedirs(self.output_path, exist_ok=True)
        task_path = os.path.join(self.output_path, self.task)
        os.makedirs(task_path, exist_ok=True)
        image_files = self.get_images(images_path)
        for image_file in tqdm(image_files):
            img = self.read_image(image_file)
            image_name = os.path.basename(image_file)
            img.save(os.path.join(task_path, image_name))
            tmps = image_name.split('.')
            assert len(
                tmps) == 2, f'Invalid image name: {image_name}, too much "."'
            restoration_save_path = os.path.join(
                task_path, f'{tmps[0]}_restoration.{tmps[1]}')
            input_ = to_tensor(img)

            # Pad the input if not_multiple_of 8
            h, w = input_.shape[1], input_.shape[2]

            H, W = ((h + self.img_multiple_of) //
                    self.img_multiple_of) * self.img_multiple_of, (
                        (w + self.img_multiple_of) //
                        self.img_multiple_of) * self.img_multiple_of
            padh = H - h if h % self.img_multiple_of != 0 else 0
            padw = W - w if w % self.img_multiple_of != 0 else 0
            input_ = paddle.to_tensor(input_)
            transform = Pad((0, 0, padw, padh), padding_mode='reflect')
            input_ = transform(input_)

            input_ = paddle.to_tensor(np.expand_dims(input_.numpy(), 0))

            with paddle.no_grad():
                restored = self.generator(input_)
            restored = restored[0]
            restored = paddle.clip(restored, 0, 1)

            # Unpad the output
            restored = restored[:, :, :h, :w]

            restored = restored.numpy()
            restored = restored.transpose(0, 2, 3, 1)
            restored = restored[0]
            restored = restored * 255
            restored = restored.astype(np.uint8)

            cv2.imwrite(restoration_save_path,
                        cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))

        print('Done, output path is:', task_path)
