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

import paddle
import paddle.nn.functional as F

import os
import sys
import cv2
import random
import datetime
import math
import argparse
import numpy as np

import scipy.io as sio
import zipfile
from .net_s3fd import s3fd
from .bbox import *


def detect(net, img):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1, ) + img.shape)

    img = paddle.to_tensor(img).astype('float32')
    BB, CC, HH, WW = img.shape
    with paddle.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], axis=1)
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.shape  # feature map size
        stride = 2**(i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls.numpy()[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls.numpy()[0, 1, hindex, windex]
            loc = oreg.numpy()[0, :, hindex, windex].reshape(1, 4)
            priors = paddle.to_tensor(
                [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(paddle.to_tensor(loc), priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist


def batch_detect(net, imgs):
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)

    imgs = paddle.to_tensor(imgs).astype('float32')
    BB, CC, HH, WW = imgs.shape
    with paddle.no_grad():
        olist = net(imgs)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], axis=1)
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.shape  # feature map size
        stride = 2**(i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls.numpy()[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls.numpy()[:, 1, hindex, windex]
            loc = oreg.numpy()[:, :, hindex, windex].reshape(BB, 1, 4)
            priors = paddle.to_tensor(
                [[axc / 1.0, ayc / 1.0, stride * 4 / 1.0,
                  stride * 4 / 1.0]]).reshape([1, 1, 4])
            variances = [0.1, 0.2]
            box = batch_decode(paddle.to_tensor(loc), priors, variances)
            box = box[:, 0] * 1.0
            bboxlist.append(
                paddle.concat([box, paddle.to_tensor(score).unsqueeze(1)],
                              1).numpy())
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    return bboxlist
