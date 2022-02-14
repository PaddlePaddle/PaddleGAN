#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
import math
import skimage
import imageio

import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T

from .base_predictor import BasePredictor
from ..models.singan_model import pad_shape
from ppgan.models.generators import SinGANGenerator
from ppgan.utils.download import get_path_from_url
from ppgan.utils.visual import tensor2img, save_image, make_grid

pretrained_weights_url = {
    'trees': 'https://paddlegan.bj.bcebos.com/models/singan_universal_trees.pdparams',
    'stone': 'https://paddlegan.bj.bcebos.com/models/singan_universal_stone.pdparams',
    'mountains': 'https://paddlegan.bj.bcebos.com/models/singan_universal_mountains.pdparams',
    'birds': 'https://paddlegan.bj.bcebos.com/models/singan_universal_birds.pdparams',
    'lightning': 'https://paddlegan.bj.bcebos.com/models/singan_universal_lightning.pdparams'
}


def imread(path):
    return cv2.cvtColor(
        cv2.imread(
            path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def imgpath2tensor(path):
    return paddle.to_tensor(T.Compose([
        T.Transpose(),
        T.Normalize(127.5, 127.5)
    ])(imread(path))).unsqueeze(0)
    
def dilate_mask(mask, mode):
    if mode == "harmonization":
        element = skimage.morphology.disk(radius=7)
    elif mode == "editing":
        element = skimage.morphology.disk(radius=20)
    else:
        raise NotImplementedError('mode %s is not implemented' % mode)
    mask = skimage.morphology.binary_dilation(mask, selem=element)
    mask = skimage.filters.gaussian(mask, sigma=5)
    return mask

class SinGANPredictor(BasePredictor):
    def __init__(self,
                 output_path='output_dir',
                 weight_path=None,
                 pretrained_model=None,
                 seed=None):
        self.output_path = output_path
        if weight_path is None:
            if pretrained_model in pretrained_weights_url.keys():
                weight_path = get_path_from_url(
                    pretrained_weights_url[pretrained_model])
            else:
                raise ValueError(
                    'Predictor need a weight path or a pretrained model.')
        checkpoint = paddle.load(weight_path)

        self.scale_num = checkpoint['scale_num'].item()
        self.coarsest_shape = checkpoint['coarsest_shape'].tolist()
        self.nfc_init = checkpoint['nfc_init'].item()
        self.min_nfc_init = checkpoint['min_nfc_init'].item()
        self.num_layers = checkpoint['num_layers'].item()
        self.ker_size = checkpoint['ker_size'].item()
        self.noise_zero_pad = checkpoint['noise_zero_pad'].item()
        self.generator = SinGANGenerator(self.scale_num, 
                                         self.coarsest_shape, 
                                         self.nfc_init, 
                                         self.min_nfc_init, 
                                         3, 
                                         self.num_layers, 
                                         self.ker_size, 
                                         self.noise_zero_pad)
        self.generator.set_state_dict(checkpoint)
        self.generator.eval()
        self.scale_factor = self.generator.scale_factor.item()
        self.niose_pad_size = 0 if self.noise_zero_pad \
                                else self.generator._pad_size
        if seed is not None:
            paddle.seed(seed)

    def noise_like(self, x):
        return paddle.randn(pad_shape(x.shape, self.niose_pad_size))

    def run(self, 
            mode='random_sample', 
            generate_start_scale=0, 
            scale_h=1.0, 
            scale_v=1.0, 
            ref_image=None,
            mask_image=None,
            sr_factor=4,
            animation_alpha=0.9,
            animation_beta=0.9,
            animation_frames=20,
            animation_duration=0.1,
            n_row=5,
            n_col=3):

        # check config
        if mode not in ['random_sample', 
                        'sr', 'animation', 
                        'harmonization', 
                        'editing', 'paint2image']:
            raise ValueError(
                'Only random_sample, sr, animation, harmonization, \
                 editing and paint2image is implemented.')
        if mode in ['sr', 'harmonization', 'editing', 'paint2image'] and \
           ref_image is None:
            raise ValueError(
                'When mode is sr, harmonization, editing, or \
                 paint2image, a reference image must be privided.')
        if mode in ['harmonization', 'editing'] and mask_image is None:
            raise ValueError(
                'When mode is harmonization or editing, \
                 a mask image must be privided.')

        if mode == 'animation':
            batch_size = animation_frames
        elif mode == 'random_sample':
            batch_size = n_row * n_col
        else:
            batch_size = 1

        # prepare input
        if mode == 'harmonization' or mode == 'editing' or mode == 'paint2image':
            ref = imgpath2tensor(ref_image)
            x_init = F.interpolate(
                ref, None, 
                self.scale_factor ** (self.scale_num - generate_start_scale), 
                'bicubic')
            x_init = F.interpolate(
                x_init, None, 1 / self.scale_factor, 'bicubic')
        elif mode == 'sr':
            ref = imgpath2tensor(ref_image)
            sr_iters = math.ceil(math.log(sr_factor, 1 / self.scale_factor))
            sr_scale_factor = sr_factor ** (1 / sr_factor)
            x_init = F.interpolate(ref, None, sr_scale_factor, 'bicubic')
        else:
            x_init = paddle.zeros([
                batch_size,
                self.coarsest_shape[1],
                int(self.coarsest_shape[2] * scale_v),
                int(self.coarsest_shape[3] * scale_h)])

        # forward
        if mode == 'sr':
            for _ in range(sr_iters):
                out = self.generator([self.noise_like(x_init)], x_init, -1, -1)
                x_init = F.interpolate(out, None, sr_scale_factor, 'bicubic')
        else:
            z_pyramid = [
                self.noise_like(
                    F.interpolate(
                        x_init, None, 1 / self.scale_factor ** i)) 
                for i in range(self.scale_num - generate_start_scale)]

            if mode == 'animation':
                a = animation_alpha
                b = animation_beta
                for i in range(len(z_pyramid)):
                    z = paddle.chunk(z_pyramid[i], batch_size)
                    if i == 0 and generate_start_scale == 0:
                        z_0 = F.interpolate(
                            self.generator.z_fixed, 
                            pad_shape(x_init.shape[-2:], self.niose_pad_size), 
                            None, 'bicubic')
                    else:
                        z_0 = 0
                    z_1 = z_0
                    z_2 = 0.95 * z_1 + 0.05 * z[0]
                    for j in range(len(z)):
                        z[j] = a * z_0 + (1 - a) * (z_2 + b * (z_2 - z_1) + (1 - b) * z[j])
                        z_1 = z_2
                        z_2 = z[j]
                    z = paddle.concat(z)
                    z_pyramid[i] = z

            out = self.generator(z_pyramid, x_init, self.scale_num - 1, generate_start_scale)
        
        # postprocess and save    
        os.makedirs(self.output_path, exist_ok=True)
        if mode == 'animation':
            frames = [tensor2img(x) for x in out.chunk(animation_frames)]
            imageio.mimsave(
                os.path.join(self.output_path, 'animation.gif'), 
                frames, 'GIF', duration=animation_duration)
        else:
            if mode == 'harmonization' or mode == 'editing':
                mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)
                mask = paddle.to_tensor(dilate_mask(mask, mode), 'float32')
                out = F.interpolate(out, mask.shape, None, 'bicubic')
                out = (1 - mask) * ref + mask * out
            elif mode == 'sr':
                out = F.interpolate(
                    out, 
                    [ref.shape[-2] * sr_factor, ref.shape[-1] * sr_factor],
                    None, 'bicubic')
            elif mode == 'paint2image':
                out = F.interpolate(out, ref.shape[-2:], None, 'bicubic')
            elif mode == 'random_sample':
                out = make_grid(out, n_row)

            save_image(tensor2img(out), os.path.join(self.output_path, mode + '.png'))
