# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from PIL import Image, ImageOps
import cv2
import numpy as np
import os

import paddle
from paddle.vision.transforms import Resize

from .base_predictor import BasePredictor
from ppgan.models.generators import InpaintGenerator
from ..utils.filesystem import load


class AOTGANPredictor(BasePredictor):
    def __init__(self,
                 output_path,
                 weight_path,
                 gen_cfg):

        # initialize model
        gen = InpaintGenerator(
                 gen_cfg.rates,
                 gen_cfg.block_num,
                 )
        gen.eval()
        para = load(weight_path)
        if 'net_gen' in para:
            gen.set_state_dict(para['net_gen'])
        else:
            gen.set_state_dict(para)

        self.gen = gen
        self.output_path = output_path
        self.gen_cfg = gen_cfg


    def run(self, input_image_path, input_mask_path):
        img = Image.open(input_image_path)
        mask = Image.open(input_mask_path)
        img = Resize([self.gen_cfg.img_size, self.gen_cfg.img_size], interpolation='bilinear')(img)
        mask = Resize([self.gen_cfg.img_size, self.gen_cfg.img_size], interpolation='nearest')(mask)
        img = img.convert('RGB')
        mask = mask.convert('L')
        img = np.array(img)
        mask = np.array(mask)

        # normalize image data to (-1, +1)，image tensor shape:[n=1, c=3, h=512, w=512]
        img = (img.astype('float32') / 255.) * 2. - 1.
        img = np.transpose(img, (2, 0, 1))
        img = paddle.to_tensor(np.expand_dims(img, 0))
        # mask tensor shape:[n=1, c=3, h=512, w=512], value 0 denotes known pixels and 1 denotes missing regions
        mask = np.expand_dims(mask.astype('float32') / 255., 0)
        mask = paddle.to_tensor(np.expand_dims(mask, 0))

        # predict
        img_masked = (img * (1 - mask)) + mask # put the mask onto the image
        input_data = paddle.concat((img_masked, mask), axis=1) # concatenate
        pred_img = self.gen(input_data) # predict by masked image
        comp_img = (1 - mask) * img + mask * pred_img # compound the inpainted image
        img_save = ((comp_img.numpy()[0].transpose((1,2,0)) + 1.) / 2. * 255).astype('uint8')

        pic = cv2.cvtColor(img_save,cv2.COLOR_BGR2RGB)
        path, _ = os.path.split(self.output_path)
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(self.output_path, pic)
        print('Predicted pictures are saved: '+self.output_path+' 。')
