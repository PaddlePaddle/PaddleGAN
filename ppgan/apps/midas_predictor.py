#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import numpy as np
import cv2

import paddle
from paddle.vision.transforms import Compose

from ppgan.utils.download import get_path_from_url
from .base_predictor import BasePredictor
from .midas.transforms import Resize, NormalizeImage, PrepareForNet
from .midas.midas_net import MidasNet
from .midas.utils import write_depth


class MiDaSPredictor(BasePredictor):
    def __init__(self, output=None, weight_path=None):
        """
        output (str|None): output path, if None, do not write
            depth map to pfm and png file.
        weight_path (str|None): weight path, if None, load default
            MiDaSv2.1 model.
        """
        self.output_path = os.path.join(output, 'MiDaS') if output else None

        self.net_h, self.net_w = 384, 384
        if weight_path is None:
            midasv2_weight_url = 'https://paddlegan.bj.bcebos.com/applications/midas.pdparams'
            weight_path = get_path_from_url(midasv2_weight_url)
        self.weight_path = weight_path

        self.model = self.load_checkpoints()

        self.transform = Compose([
            Resize(
                self.net_w,
                self.net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def load_checkpoints(self):
        model = MidasNet(self.weight_path, non_negative=True)
        model.eval()
        return model

    def run(self, img):
        """
        img (str|np.ndarray|Image.Image): input image, it can be
            images directory, Numpy.array or Image.Image.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        img_input = self.transform({"image": img})["image"]

        with paddle.no_grad():
            sample = paddle.to_tensor(img_input).unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (paddle.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy())

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            img_name = img if isinstance(img, str) else 'depth'
            filename = os.path.join(
                self.output_path,
                os.path.splitext(os.path.basename(img_name))[0])
            pfm_f, png_f = write_depth(filename, prediction, bits=2)
            return prediction, pfm_f, png_f
        return prediction
