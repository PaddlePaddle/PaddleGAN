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
import sys
import argparse

from PIL import Image
import numpy as np
import cv2

import ppgan.faceutils as futils
from ppgan.utils.preprocess import *
from ppgan.utils.visual import mask2image
from .base_predictor import BasePredictor


class FaceParsePredictor(BasePredictor):
    def __init__(self, output_path='output'):
        self.output_path = output_path
        self.input_size = (512, 512)
        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85
        self.face_parser = futils.mask.FaceParser()

    def run(self, image):
        image = Image.open(image).convert("RGB")
        face = futils.dlib.detect(image)

        if not face:
            return
        face_on_image = face[0]
        image, face, crop_face = futils.dlib.crop(image, face_on_image,
                                                  self.up_ratio,
                                                  self.down_ratio,
                                                  self.width_ratio)
        np_image = np.array(image)
        mask = self.face_parser.parse(
            np.float32(cv2.resize(np_image, self.input_size)))
        mask = cv2.resize(mask.numpy(), (256, 256))
        mask = mask.astype(np.uint8)
        mask = mask2image(mask)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        save_path = os.path.join(self.output_path, 'face_parse.png')
        cv2.imwrite(save_path, mask)
        return mask
