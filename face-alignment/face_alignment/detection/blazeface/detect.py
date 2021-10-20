import torch
import torch.nn.functional as F

import cv2
import numpy as np

from .utils import *


def detect(net, img, device):
    H, W, C = img.shape
    orig_size = min(H, W)
    img, (xshift, yshift) = resize_and_crop_image(img, 128)
    preds = net.predict_on_image(img)

    if 0 == len(preds):
        return [[]]

    shift = np.array([xshift, yshift] * 2)
    scores = preds[:, -1:]

    # TODO: ugly
    # reverses, x and y to adapt with face-alignment code
    locs = np.concatenate((preds[:, 1:2], preds[:, 0:1], preds[:, 3:4], preds[:, 2:3]), axis=1)
    return [np.concatenate((locs * orig_size + shift, scores), axis=1)]


def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a numpy array or tensor of shape (Batch size, Channels, Height, Width)
    Outputs:
        - list of 2-dim numpy arrays with shape (faces_on_this_image, 5): x1, y1, x2, y2, confidence
          (x1, y1) - top left corner, (x2, y2) - bottom right corner
    """
    B, C, H, W = img_batch.shape
    orig_size = min(H, W)

    if isinstance(img_batch, torch.Tensor):
        img_batch = img_batch.cpu().numpy()

    img_batch = img_batch.transpose((0, 2, 3, 1))

    imgs, (xshift, yshift) = resize_and_crop_batch(img_batch, 128)
    preds = net.predict_on_batch(imgs)
    bboxlists = []
    for pred in preds:
        shift = np.array([xshift, yshift] * 2)
        scores = pred[:, -1:]
        locs = np.concatenate((pred[:, 1:2], pred[:, 0:1], pred[:, 3:4], pred[:, 2:3]), axis=1)
        bboxlists.append(np.concatenate((locs * orig_size + shift, scores), axis=1))

    return bboxlists


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

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
