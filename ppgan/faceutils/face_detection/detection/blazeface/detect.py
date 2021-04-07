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

import cv2
import numpy as np

from .utils import *


def detect(net, img, device):
    H, W, C = img.shape
    orig_size = min(H, W)
    img, (xshift, yshift) = resize_and_crop_image(img, 128)
    preds = net.predict_on_image(img.astype('float32')).numpy()

    if 0 == len(preds):
        return [[]]

    shift = np.array([xshift, yshift] * 2)
    scores = preds[:, -1:]

    locs = np.concatenate(
        (preds[:, 1:2], preds[:, 0:1], preds[:, 3:4], preds[:, 2:3]), axis=1)
    return [np.concatenate((locs * orig_size + shift, scores), axis=1)]


def batch_detect(net, img_batch):
    """
    Inputs:
        - img_batch: a numpy array or tensor of shape (Batch size, Channels, Height, Width)
    Outputs:
        - list of 2-dim numpy arrays with shape (faces_on_this_image, 5): x1, y1, x2, y2, confidence
          (x1, y1) - top left corner, (x2, y2) - bottom right corner
    """
    B, H, W, C = img_batch.shape
    orig_size = min(H, W)

    if isinstance(img_batch, paddle.Tensor):
        img_batch = img_batch.numpy()

    imgs, (xshift, yshift) = resize_and_crop_batch(img_batch, 128)
    preds = net.predict_on_batch(imgs.astype('float32'))
    bboxlists = []
    for pred in preds:
        pred = pred.numpy()
        shift = np.array([xshift, yshift] * 2)
        scores = pred[:, -1:]
        xmin = pred[:, 1:2]
        ymin = pred[:, 0:1]
        xmax = pred[:, 3:4]
        ymax = pred[:, 2:3]
        locs = np.concatenate((xmin, ymin, xmax, ymax), axis=1)
        bboxlists.append(
            np.concatenate((locs * orig_size + shift, scores), axis=1))

    return bboxlists


def flip_detect(net, img):
    img = cv2.flip(img, 1)
    b = detect(net, img)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
