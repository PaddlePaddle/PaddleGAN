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
from fire import Fire
import numpy as np

import paddle
import paddle.vision.transforms as T
import ppgan.faceutils as futils
from ppgan.utils.options import parse_args
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.utils.filesystem import load
from ppgan.engine.trainer import Trainer
from ppgan.models.builder import build_model
from ppgan.utils.preprocess import *


def toImage(net_output):
    img = net_output.squeeze(0).transpose(
        (1, 2, 0)).numpy()  # [1,c,h,w]->[h,w,c]
    img = (img * 255.0).clip(0, 255)
    img = np.uint8(img)
    img = Image.fromarray(img, mode='RGB')
    return img


def mask2image(mask: np.array, format="HWC"):
    H, W = mask.shape

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(int(mask.max())):
        color = np.random.rand(1, 1, 3) * 255
        canvas += (mask == i)[:, :, None] * color.astype(np.uint8)
    return canvas


class PreProcess:
    def __init__(self, config, need_parser=True):
        self.img_size = 256
        self.transform = transform = T.Compose([
            T.Resize(size=256),
            T.Permute(to_rgb=False),
        ])
        self.norm = T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
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
        mask = self.face_parser.parse(
            np.float32(cv2.resize(np_image, (512, 512))))
        mask = cv2.resize(mask.numpy(), (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        mask_color = mask2image(mask)
        cv2.imwrite('mask_temp.png', mask_color)
        mask_tensor = paddle.to_tensor(mask)

        lms = futils.dlib.landmarks(image, face) * self.img_size / image.width
        lms = lms.round()

        P_np = generate_P_from_lmks(lms, self.img_size, self.img_size,
                                    self.img_size)

        mask_aug = generate_mask_aug(mask, lms)

        image = self.transform(np_image)

        return [
            self.norm(image),
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
        self.model = build_model(config)
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
        for i in range(len(source_input) - 1):
            source_input[i] = paddle.to_tensor(
                np.expand_dims(source_input[i], 0))

        for i in range(len(reference_input) - 1):
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
        net = getattr(self.model, 'netG')
        net.set_dict(state_dicts['netG'])
        result, _ = self.model.test(input_data)
        print('result shape: ', result.shape)
        min_, max_ = result.min(), result.max()
        result += -min_
        result = paddle.divide(result, max_ - min_ + 1e-5)
        img = toImage(result)

        if with_face:
            return img, crop_face
        img.save('before.png')

        return img


def main(args, cfg, save_path='transferred_image.png'):

    setup(args, cfg)

    inference = Inference(cfg, args.model_path)
    postprocess = PostProcess(cfg)

    source = Image.open(args.source_path).convert("RGB")
    reference_paths = list(Path(args.reference_dir).glob("*"))
    np.random.shuffle(reference_paths)
    for reference_path in reference_paths:
        if not reference_path.is_file():
            print(reference_path, "is not a valid file.")
            continue

        reference = Image.open(reference_path).convert("RGB")

        # Transfer the psgan from reference to source.
        image, face = inference.transfer(source, reference, with_face=True)
        image.save('before.png')
        source_crop = source.crop(
            (face.left(), face.top(), face.right(), face.bottom()))
        image = postprocess(source_crop, image)
        image.save(save_path)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.config_file)
    main(args, cfg)
