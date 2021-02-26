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
from pathlib import Path

from PIL import Image
import numpy as np

import paddle
import paddle.vision.transforms as T
from paddle.utils.download import get_weights_path_from_url
import ppgan.faceutils as futils
from ppgan.utils.options import parse_args
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.utils.filesystem import load
from ppgan.models.builder import build_model
from ppgan.utils.preprocess import *
from .base_predictor import BasePredictor


def toImage(net_output):
    img = net_output.squeeze(0).transpose(
        (1, 2, 0)).numpy()  # [1,c,h,w]->[h,w,c]
    img = (img * 255.0).clip(0, 255)
    img = np.uint8(img)
    img = Image.fromarray(img, mode='RGB')
    return img


PS_WEIGHT_URL = "https://paddlegan.bj.bcebos.com/models/psgan_weight.pdparams"


class PreProcess:
    def __init__(self, config, need_parser=True):
        self.img_size = 256
        self.transform = transform = T.Compose([
            T.Resize(size=256),
            T.ToTensor(),
        ])
        self.norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        if need_parser:
            self.face_parser = futils.mask.FaceParser()
        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85

    def __call__(self, image):
        face = futils.dlib.detect(image)

        if not face:
            return
        face_on_image = face[0]
        image, face, crop_face = futils.dlib.crop(image, face_on_image,
                                                  self.up_ratio,
                                                  self.down_ratio,
                                                  self.width_ratio)
        np_image = np.array(image)
        image_trans = self.transform(np_image)
        mask = self.face_parser.parse(
            np.float32(cv2.resize(np_image, (512, 512))))
        mask = cv2.resize(mask.numpy(), (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        mask_tensor = paddle.to_tensor(mask)

        lms = futils.dlib.landmarks(
            image, face) / image_trans.shape[:2] * self.img_size
        lms = lms.round()

        P_np = generate_P_from_lmks(lms, self.img_size, self.img_size,
                                    self.img_size)

        mask_aug = generate_mask_aug(mask, lms)

        return [
            self.norm(image_trans).unsqueeze(0),
            np.float32(mask_aug),
            np.float32(P_np),
            np.float32(mask)
        ], face_on_image, crop_face


class PostProcess:
    def __init__(self, config):
        self.denoise = True
        self.img_size = 256

    def __call__(self, source: Image, result: Image):
        # TODO: Refract -> name, resize
        source = np.array(source)
        result = np.array(result)

        height, width = source.shape[:2]
        small_source = cv2.resize(source, (self.img_size, self.img_size))
        laplacian_diff = source.astype(np.float) - cv2.resize(
            small_source, (width, height)).astype(np.float)
        result = (cv2.resize(result,
                             (width, height)) + laplacian_diff).round().clip(
                                 0, 255).astype(np.uint8)
        if self.denoise:
            result = cv2.fastNlMeansDenoisingColored(result)
        result = Image.fromarray(result).convert('RGB')
        return result


class Inference:
    def __init__(self, config, model_path=''):
        self.model = build_model(config.model)
        self.preprocess = PreProcess(config)
        self.model_path = model_path

    def transfer(self, source, reference, with_face=False):
        source_input, face, crop_face = self.preprocess(source)
        reference_input, face, crop_face = self.preprocess(reference)

        consis_mask = np.float32(
            calculate_consis_mask(source_input[1], reference_input[1]))
        consis_mask = paddle.to_tensor(np.expand_dims(consis_mask, 0))

        if not (source_input and reference_input):
            if with_face:
                return None, None
            return

        for i in range(1, len(source_input) - 1):
            source_input[i] = paddle.to_tensor(
                np.expand_dims(source_input[i], 0))

        for i in range(1, len(reference_input) - 1):
            reference_input[i] = paddle.to_tensor(
                np.expand_dims(reference_input[i], 0))

        input_data = {
            'image_A': source_input[0],
            'image_B': reference_input[0],
            'mask_A_aug': source_input[1],
            'mask_B_aug': reference_input[1],
            'P_A': source_input[2],
            'P_B': reference_input[2],
            'consis_mask': consis_mask
        }

        state_dicts = load(self.model_path)
        for net_name, net in self.model.nets.items():
            net.set_state_dict(state_dicts[net_name])
        result, _ = self.model.test(input_data)
        min_, max_ = result.min(), result.max()
        result += -min_
        result = paddle.divide(result, max_ - min_ + 1e-5)
        img = toImage(result)

        if with_face:
            return img, crop_face

        return img


class PSGANPredictor(BasePredictor):
    def __init__(self, args, cfg, output_path='output'):
        self.args = args
        self.cfg = cfg
        self.weight_path = self.args.model_path
        if self.weight_path is None:
            self.weight_path = get_weights_path_from_url(PS_WEIGHT_URL)
        self.output_path = output_path

    def run(self):
        setup(self.args, self.cfg)
        inference = Inference(self.cfg, self.weight_path)
        postprocess = PostProcess(self.cfg)

        try:
            source = Image.open(self.args.source_path).convert("RGB")
        except IOError:
            print("Error: {} is not exist".format(self.args.source_path))
            sys.exit()

        reference_paths = list(Path(self.args.reference_dir).glob("*"))
        if len(reference_paths) == 0:
            print("Error: Can't find image file in {}.".format(
                self.args.reference_dir))
            sys.exit()
        np.random.shuffle(reference_paths)
        for reference_path in reference_paths:
            if not reference_path.is_file():
                print(reference_path, "is not a valid file.")
                continue
            reference = Image.open(reference_path).convert("RGB")

            # Transfer the psgan from reference to source.
            image, face = inference.transfer(source, reference, with_face=True)
            source_crop = source.crop(
                (face.left(), face.top(), face.right(), face.bottom()))
            image = postprocess(source_crop, image)

            ref_img_name = os.path.split(reference_path)[1]
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            save_path = os.path.join(self.output_path,
                                     'transfered_ref_' + ref_img_name)
            image.save(save_path)
            print('Transfered image {} has been saved!'.format(save_path))
        print('done!!!')
