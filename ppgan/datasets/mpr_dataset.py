# code was heavily based on https://github.com/swz30/MPRNet
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/swz30/MPRNet/blob/main/LICENSE.md

import os
import random
import numpy as np
import cv2
import paddle
from PIL import Image, ImageEnhance
import numpy as np
import random
import numbers
from paddle.io import Dataset
from .builder import DATASETS
from paddle.vision.transforms.functional import to_tensor, adjust_brightness, adjust_saturation, rotate, hflip, hflip, vflip, center_crop


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


@DATASETS.register()
class MPRTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(MPRTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [
            os.path.join(rgb_dir, 'input', x) for x in inp_files
            if is_image_file(x)
        ]
        self.tar_filenames = [
            os.path.join(rgb_dir, 'target', x) for x in tar_files
            if is_image_file(x)
        ]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = np.pad(inp_img, (0, 0, padw, padh),
                             padding_mode='reflect')
            tar_img = np.pad(tar_img, (0, 0, padw, padh),
                             padding_mode='reflect')

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = adjust_brightness(inp_img, 1)
            tar_img = adjust_brightness(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = adjust_saturation(inp_img, sat_factor)
            tar_img = adjust_saturation(tar_img, sat_factor)

        # Data Augmentations
        if aug == 1:
            inp_img = vflip(inp_img)
            tar_img = vflip(tar_img)
        elif aug == 2:
            inp_img = hflip(inp_img)
            tar_img = hflip(tar_img)
        elif aug == 3:
            inp_img = rotate(inp_img, 90)
            tar_img = rotate(tar_img, 90)
        elif aug == 4:
            inp_img = rotate(inp_img, 90 * 2)
            tar_img = rotate(tar_img, 90 * 2)
        elif aug == 5:
            inp_img = rotate(inp_img, 90 * 3)
            tar_img = rotate(tar_img, 90 * 3)
        elif aug == 6:
            inp_img = rotate(vflip(inp_img), 90)
            tar_img = rotate(vflip(tar_img), 90)
        elif aug == 7:
            inp_img = rotate(hflip(inp_img), 90)
            tar_img = rotate(hflip(tar_img), 90)

        inp_img = to_tensor(inp_img)
        tar_img = to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


@DATASETS.register()
class MPRVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(MPRVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [
            os.path.join(rgb_dir, 'input', x) for x in inp_files
            if is_image_file(x)
        ]
        self.tar_filenames = [
            os.path.join(rgb_dir, 'target', x) for x in tar_files
            if is_image_file(x)
        ]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = center_crop(inp_img, (ps, ps))
            tar_img = center_crop(tar_img, (ps, ps))

        inp_img = to_tensor(inp_img)
        tar_img = to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


@DATASETS.register()
class MPRTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(MPRTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [
            os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)
        ]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = to_tensor(inp)
        return inp, filename
