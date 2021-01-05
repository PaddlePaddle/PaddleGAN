# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from PIL import Image
import numpy as np

import paddle
from paddle.utils.download import get_path_from_url
from ppgan.faceutils.dlibutils import align_crop
from ppgan.faceutils.face_segmentation import FaceSeg
from ppgan.models.generators import ResnetUGATITP2CGenerator

from .base_predictor import BasePredictor


P2C_WEIGHT_URL = "https://paddlegan.bj.bcebos.com/models/photo2cartoon_genA2B_weight.pdparams"


class Photo2CartoonPredictor(BasePredictor):
    def __init__(self, output_path='output', weight_path=None):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if weight_path is None:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            weight_path = get_path_from_url(P2C_WEIGHT_URL, cur_path)

        self.genA2B = ResnetUGATITP2CGenerator()
        params = paddle.load(weight_path)
        self.genA2B.set_state_dict(params)
        self.genA2B.eval()

        self.faceseg = FaceSeg()

    def run(self, image_path):
        image = Image.open(image_path)
        face_image = align_crop(image)
        face_mask = self.faceseg(face_image)

        face_image = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_AREA)
        face_mask = cv2.resize(face_mask, (256, 256))[:, :, np.newaxis] / 255.
        face = (face_image * face_mask + (1 - face_mask) * 255) / 127.5 - 1

        face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
        face = paddle.to_tensor(face)

        # inference
        with paddle.no_grad():
            cartoon = self.genA2B(face)[0][0]

        # post-process
        cartoon = np.transpose(cartoon.numpy(), (1, 2, 0))
        cartoon = (cartoon + 1) * 127.5
        cartoon = (cartoon * face_mask + (1 - face_mask) * 255).astype(np.uint8)

        pnoto_save_path = os.path.join(self.output_path, 'p2c_photo.png')
        cv2.imwrite(pnoto_save_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        cartoon_save_path = os.path.join(self.output_path, 'p2c_cartoon.png')
        cv2.imwrite(cartoon_save_path, cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR))

        print("Cartoon image has been saved at '{}'.".format(cartoon_save_path))
        return cartoon
