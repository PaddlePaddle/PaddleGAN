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

import os.path as osp
import cv2
import numpy as np
import paddle
from paddle.utils.download import get_path_from_url

from .fcn import FCN
from .hrnet import HRNet_W18


BISENET_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/faceseg_FCN-HRNetW18.pdparams'


class FaceSeg:
    def __init__(self):
        save_pth = get_path_from_url(BISENET_WEIGHT_URL, osp.split(osp.realpath(__file__))[0])

        self.net = FCN(num_classes=2, backbone=HRNet_W18())
        state_dict = paddle.load(save_pth)
        self.net.set_state_dict(state_dict)
        self.net.eval()

    def __call__(self, image):
        image_input = self.input_transform(image)  # RGB image

        with paddle.no_grad():
            logits = self.net(image_input)
            pred = paddle.argmax(logits[0], axis=1)
        pred = pred.numpy()
        mask = np.squeeze(pred).astype(np.uint8)

        mask = self.output_transform(mask, shape=image.shape[:2])
        return mask

    def input_transform(self, image):
        image_input = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
        image_input = (image_input / 255.)[np.newaxis, :, :, :]
        image_input = np.transpose(image_input, (0, 3, 1, 2)).astype(np.float32)
        image_input = paddle.to_tensor(image_input)
        return image_input

    @staticmethod
    def output_transform(output, shape):
        output = cv2.resize(output, (shape[1], shape[0]))
        image_output = np.clip((output * 255), 0, 255).astype(np.uint8)
        return image_output
