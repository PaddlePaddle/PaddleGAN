# code was heavily based on https://github.com/cszn/KAIR
# MIT License
# Copyright (c) 2019 Kai Zhang

import os
import random
import numpy as np
import cv2

import paddle
from paddle.io import Dataset

from .builder import DATASETS


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return paddle.Tensor(np.ascontiguousarray(img, dtype=np.float32)).transpose(
        [2, 0, 1]) / 255.


def uint2single(img):

    return np.float32(img / 255.)


# convert single (HxWxC) to 3-dimensional paddle tensor
def single2tensor3(img):
    return paddle.Tensor(np.ascontiguousarray(img, dtype=np.float32)).transpose(
        [2, 0, 1])


@DATASETS.register()
class SwinIRDataset(Dataset):
    """ Get L/H for denosing on AWGN with fixed sigma.
    Ref:
        DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    Args:
        opt (dict): A dictionary defining dataset-related parameters.
    """

    def __init__(self, opt=None):
        super(SwinIRDataset, self).__init__()

        print(
            'Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.'
        )
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma
        self.paths_H = get_image_paths(opt['dataroot_H'])

    def __len__(self):
        return len(self.paths_H)

    def __getitem__(self, index):
        # get H image
        H_path = self.paths_H[index]

        img_H = imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            # get L/H patch pairs
            H, W, _ = img_H.shape

            # randomly crop the patch
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size,
                            rnd_w:rnd_w + self.patch_size, :]

            # augmentation - flip, rotate
            mode = random.randint(0, 7)
            patch_H = augment_img(patch_H, mode=mode)
            img_H = uint2tensor3(patch_H)
            img_L = img_H.clone()

            # add noise
            noise = paddle.randn(img_L.shape) * self.sigma / 255.0
            img_L = img_L + noise

        else:
            # get L/H image pairs
            img_H = uint2single(img_H)
            img_L = np.copy(img_H)

            # add noise
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)

            # HWC to CHW, numpy to tensor
            img_L = single2tensor3(img_L)
            img_H = single2tensor3(img_H)

        filename = os.path.splitext(os.path.split(H_path)[-1])[0]

        return img_H, img_L, filename
