# code was heavily based on https://github.com/cszn/KAIR
# MIT License
# Copyright (c) 2019 Kai Zhang

import os
import os.path as osp
import pickle
import random
import numpy as np
import cv2
import math

import paddle
from paddle.io import Dataset

from .builder import DATASETS

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp',
    '.BMP'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    '''get image path list'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


def read_img(env, path, size=None):
    '''read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=False) for img in img_list]
        return y_list
        # return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if isinstance(img, list):
            if hflip:
                img = [image[:, ::-1, :] for image in img]
            if vflip:
                img = [image[::-1, :, :] for image in img]
            if rot90:
                img = [image.transpose(1, 0, 2) for image in img]
        else:
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


@DATASETS.register()
class InvDNDataset(Dataset):
    '''
    Read LQ (Low Quality, here is LR), GT and noisy image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    def __init__(self, opt=None):
        super(InvDNDataset, self).__init__()
        self.opt = opt
        self.is_train = True if self.opt['phase'] == 'train' else False

        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.sizes_LQ, self.sizes_GT, self.sizes_Noisy = None, None, None
        self.LQ_env, self.GT_env, self.Noisy_env = None, None, None

        self.data_type = "img"

        if self.is_train:
            dataroot_gt = osp.join(opt["train_dir"], "GT")
            dataroot_noisy = osp.join(opt["train_dir"], "Noisy")
            dataroot_lq = osp.join(opt["train_dir"], "LQ")
        else:
            dataroot_gt = osp.join(opt["val_dir"], "GT")
            dataroot_noisy = osp.join(opt["val_dir"], "Noisy")
            dataroot_lq = None

        self.paths_GT, self.sizes_GT = get_image_paths(self.data_type,
                                                       dataroot_gt)
        self.paths_Noisy, self.sizes_Noisy = get_image_paths(
            self.data_type, dataroot_noisy)
        self.paths_LQ, self.sizes_LQ = get_image_paths(self.data_type,
                                                       dataroot_lq)

        assert self.paths_GT, 'Error: GT path is empty.'
        assert self.paths_Noisy, 'Error: Noisy path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def __getitem__(self, index):
        GT_path, Noisy_path, LQ_path = None, None, None

        scale = self.opt["scale"]

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = None
        img_GT = read_img(self.GT_env, GT_path, resolution)

        # modcrop in the validation / test phase
        if not self.is_train:
            img_GT = modcrop(img_GT, scale)

        # change color space if necessary
        img_GT = channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]

        # get Noisy image
        Noisy_path = self.paths_Noisy[index]
        resolution = None
        img_Noisy = read_img(self.Noisy_env, Noisy_path, resolution)

        # modcrop in the validation / test phase
        if not self.is_train:
            img_Noisy = modcrop(img_Noisy, scale)

        # change color space if necessary
        img_Noisy = channel_convert(img_Noisy.shape[2], "RGB", [img_Noisy])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = None
            img_LQ = read_img(self.LQ_env, LQ_path, resolution)

        if self.is_train:
            GT_size = self.opt["crop_size"]

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w +
                            LQ_size, :]  # (128, 128, 3) --> (36, 36, 3)
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT +
                            GT_size, :]  # (512, 512, 3) --> (144, 144, 3)
            img_Noisy = img_Noisy[rnd_h_GT:rnd_h_GT + GT_size,
                                  rnd_w_GT:rnd_w_GT + GT_size, :]
            # augmentation - flip, rotate
            img_LQ, img_GT, img_Noisy = augment([img_LQ, img_GT, img_Noisy],
                                                True, True)

            # change color space if necessary
            C = img_LQ.shape[0]
            img_LQ = channel_convert(C, "RGB", [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_Noisy = img_Noisy[:, :, [2, 1, 0]]
            if self.is_train:
                img_LQ = img_LQ[:, :, [2, 1, 0]]

        img_GT = paddle.to_tensor(np.ascontiguousarray(
            np.transpose(img_GT, (2, 0, 1))),
                                  dtype="float32")
        img_Noisy = paddle.to_tensor(np.ascontiguousarray(
            np.transpose(img_Noisy, (2, 0, 1))),
                                     dtype="float32")
        if self.is_train:
            img_LQ = paddle.to_tensor(np.ascontiguousarray(
                np.transpose(img_LQ, (2, 0, 1))),
                                      dtype="float32")

        if self.is_train:
            return img_Noisy, img_GT, img_LQ
        return img_Noisy, img_GT, img_GT

    def __len__(self):
        return len(self.paths_GT)  #32000 for train, 1280 for valid
