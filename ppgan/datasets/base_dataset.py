# code was heavily based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import random
import numpy as np

from paddle.io import Dataset
from PIL import Image
import cv2

import paddle.incubate.hapi.vision.transforms as transforms
from .transforms import transforms as T
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    """

    def __init__(self, cfg):
        """Initialize the class; save the options in the class

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        self.cfg = cfg
        self.root = cfg.dataroot

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(cfg, size):
    w, h = size
    new_h = h
    new_w = w
    if cfg.preprocess == 'resize_and_crop':
        new_h = new_w = cfg.load_size
    elif cfg.preprocess == 'scale_width_and_crop':
        new_w = cfg.load_size
        new_h = cfg.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - cfg.crop_size))
    y = random.randint(0, np.maximum(0, new_h - cfg.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}



def get_transform(cfg, params=None, grayscale=False, method=cv2.INTER_CUBIC, convert=True):
    transform_list = []
    if grayscale:
        print('grayscale not support for now!!!')
        # transform_list.append(transforms.Grayscale(1))
    if 'resize' in cfg.preprocess:
        osize = (cfg.load_size, cfg.load_size)
        # print('os size:', osize)
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in cfg.preprocess:
        print('scale_width not support for now!!!')
        # transform_list.append(transforms.Lambda(lambda img: __scale_width(img, cfg.load_size, cfg.crop_size, method)))

    if 'crop' in cfg.preprocess:
        # print('crop not support for now!!!', cfg.crop_size)
        # transform_list.append(T.RandomCrop(cfg.crop_size))
        if params is None:
            transform_list.append(T.RandomCrop(cfg.crop_size))
        else:
            # print('crop not support for now!!!')
            transform_list.append(T.Crop(params['crop_pos'], cfg.crop_size))

    if cfg.preprocess == 'none':
        print('preprocess not support for now!!!')
        # transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not cfg.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.RandomHorizontalFlip(1.0))
    
    if convert:
        transform_list += [transforms.Permute(to_rgb=True)]
        transform_list += [transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
