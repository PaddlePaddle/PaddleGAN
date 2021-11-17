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

import numpy as np
from paddle.io import Dataset, DataLoader
import paddle
import paddle.nn as nn
import math
import functools
from paddle.nn import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, Linear

# 处理图片数据：裁切、水平翻转、调整图片数据形状、归一化数据
def data_transform(img, resize_w, resize_h, load_size=286, pos=[0, 0, 256, 256], flip=True, is_image=True):
    if is_image:
        resized = img.resize((resize_w, resize_h), Image.BICUBIC)
    else:
        resized = img.resize((resize_w, resize_h), Image.NEAREST)
    croped = resized.crop((pos[0], pos[1], pos[2], pos[3]))
    fliped = ImageOps.mirror(croped) if flip else croped
    fliped = np.array(fliped) # transform to numpy array
    expanded = np.expand_dims(fliped, 2) if len(fliped.shape) < 3 else fliped
    transposed = np.transpose(expanded, (2, 0, 1)).astype('float32')
    if is_image:
        normalized = transposed / 255. * 2. - 1.
    else:
        normalized = transposed
    return normalized

# 定义CoCo数据集对象
class COCODateset(Dataset):
    def __init__(self, opt):
        super(COCODateset, self).__init__()
        inst_dir = opt.dataroot+'train_inst/'
        _, _, inst_list = next(os.walk(inst_dir))
        self.inst_list = np.sort(inst_list)
        self.opt = opt

    def __getitem__(self, idx):
        ins = Image.open(self.opt.dataroot+'train_inst/'+self.inst_list[idx])
        img = Image.open(self.opt.dataroot+'train_img/'+self.inst_list[idx].replace(".png", ".jpg"))
        img = img.convert('RGB')

        w, h = img.size
        resize_w, resize_h = 0, 0
        if w < h:
            resize_w, resize_h = self.opt.load_size, int(h * self.opt.load_size / w)
        else:
            resize_w, resize_h = int(w * self.opt.load_size / h), self.opt.load_size
        left = random.randint(0, resize_w - self.opt.crop_size)
        top = random.randint(0, resize_h - self.opt.crop_size)
        flip = False
        
        img = data_transform(img, resize_w, resize_h, load_size=opt.load_size, 
            pos=[left, top, left + self.opt.crop_size, top + self.opt.crop_size], flip=flip, is_image=True)
        ins = data_transform(ins, resize_w, resize_h, load_size=opt.load_size, 
            pos=[left, top, left + self.opt.crop_size, top + self.opt.crop_size], flip=flip, is_image=False)

        return img, ins, self.inst_list[idx]

    def __len__(self):
        return len(self.inst_list)


def data_onehot_pro(instance, opt):
    shape = instance.shape
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc
    shape[1] = nc
    semantics = paddle.nn.functional.one_hot(instance.astype('int64'). \
        reshape([opt.batchSize, opt.crop_size, opt.crop_size]), nc). \
        transpose((0, 3, 1, 2))

    # edge
    edge = np.zeros(instance.shape, 'int64')
    t = instance.numpy()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge = paddle.to_tensor(edge).astype('float32')

    semantics = paddle.concat([semantics, edge], 1)
    return semantics

# 设置除 spade 以外的归一化层
def build_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Args:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we do not use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2D,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'syncbatch':
        norm_layer = functools.partial(
            nn.SyncBatchNorm,
            weight_attr=False,
            bias_attr=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2D,)
    elif norm_type == 'spectral':
        norm_layer = functools.partial(Spectralnorm)
    elif norm_type == 'none':

        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
                                  norm_type)
    return norm_layer

def simam(x, e_lambda=1e-4):
    b, c, h, w = x.shape
    n = w * h - 1
    x_minus_mu_square = (x - x.mean(axis=[2, 3], keepdim=True)) ** 2
    y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(axis=[2, 3], keepdim=True) / n + e_lambda)) + 0.5
    return x * nn.functional.sigmoid(y)

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

