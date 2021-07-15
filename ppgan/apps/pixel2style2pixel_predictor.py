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
import cv2
import scipy
import random
import numpy as np
import paddle
import paddle.vision.transforms as T
import ppgan.faceutils as futils
from .base_predictor import BasePredictor
from ppgan.models.generators import Pixel2Style2Pixel
from ppgan.utils.download import get_path_from_url
from PIL import Image

model_cfgs = {
    'ffhq-inversion': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/pSp-ffhq-inversion.pdparams',
        'transform':
        T.Compose([
            T.Resize((256, 256)),
            T.Transpose(),
            T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
        ]),
        'size':
        1024,
        'style_dim':
        512,
        'n_mlp':
        8,
        'channel_multiplier':
        2
    },
    'ffhq-toonify': {
        'model_urls':
        'https://paddlegan.bj.bcebos.com/models/pSp-ffhq-toonify.pdparams',
        'transform':
        T.Compose([
            T.Resize((256, 256)),
            T.Transpose(),
            T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
        ]),
        'size':
        1024,
        'style_dim':
        512,
        'n_mlp':
        8,
        'channel_multiplier':
        2
    },
    'default': {
        'transform':
        T.Compose([
            T.Resize((256, 256)),
            T.Transpose(),
            T.Normalize([127.5, 127.5, 127.5], [127.5, 127.5, 127.5])
        ])
    }
}


def run_alignment(image_path):
    img = Image.open(image_path).convert("RGB")
    face = futils.dlib.detect(img)
    if not face:
        raise Exception('Could not find a face in the given image.')
    face_on_image = face[0]
    lm = futils.dlib.landmarks(img, face_on_image)
    lm = np.array(lm)[:, ::-1]
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img),
                     ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)

    return img


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Pixel2Style2PixelPredictor(BasePredictor):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 model_type=None,
                 seed=None,
                 size=1024,
                 style_dim=512,
                 n_mlp=8,
                 channel_multiplier=2):
        self.output_path = output_path

        if weight_path is None and model_type != 'default':
            if model_type in model_cfgs.keys():
                weight_path = get_path_from_url(
                    model_cfgs[model_type]['model_urls'])
                size = model_cfgs[model_type].get('size', size)
                style_dim = model_cfgs[model_type].get('style_dim', style_dim)
                n_mlp = model_cfgs[model_type].get('n_mlp', n_mlp)
                channel_multiplier = model_cfgs[model_type].get(
                    'channel_multiplier', channel_multiplier)
                checkpoint = paddle.load(weight_path)
            else:
                raise ValueError(
                    'Predictor need a weight path or a pretrained model type')
        else:
            checkpoint = paddle.load(weight_path)

        opts = checkpoint.pop('opts')
        opts = AttrDict(opts)
        opts['size'] = size
        opts['style_dim'] = style_dim
        opts['n_mlp'] = n_mlp
        opts['channel_multiplier'] = channel_multiplier

        self.generator = Pixel2Style2Pixel(opts)
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()

        if seed is not None:
            paddle.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.model_type = 'default' if model_type is None else model_type

    def run(self, image):
        src_img = run_alignment(image)
        src_img = np.asarray(src_img)
        transformed_image = model_cfgs[self.model_type]['transform'](src_img)
        dst_img, latents = self.generator(paddle.to_tensor(
            transformed_image[None, ...]),
                                          resize=False,
                                          return_latents=True)
        dst_img = (dst_img * 0.5 + 0.5)[0].numpy() * 255
        dst_img = dst_img.transpose((1, 2, 0))
        dst_npy = latents[0].numpy()

        os.makedirs(self.output_path, exist_ok=True)
        save_src_path = os.path.join(self.output_path, 'src.png')
        cv2.imwrite(save_src_path, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))
        save_dst_path = os.path.join(self.output_path, 'dst.png')
        cv2.imwrite(save_dst_path, cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR))
        save_npy_path = os.path.join(self.output_path, 'dst.npy')
        np.save(save_npy_path, dst_npy)

        return src_img, dst_img, dst_npy
