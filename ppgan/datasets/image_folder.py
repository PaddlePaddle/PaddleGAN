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
"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import cv2
from PIL import Image
import os
import os.path
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset
from .transforms.builder import build_transforms


IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
    '.tif',
    '.TIF',
    '.tiff',
    '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:min(float(max_dataset_size), len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


@DATASETS.register()
class ImageFolder(BaseDataset):
    def __init__(self, cfg):
        BaseDataset.__init__(self, cfg)
        root = self.cfg.root
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = build_transforms(self.cfg.transforms)
        self.return_paths = self.cfg.return_paths

    def __getitem__(self, index):
        path = self.imgs[index]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return_dict = {'img': img}

        if self.return_paths:
            return_dict['path'] = path

        return return_dict

    def __len__(self):
        return len(self.imgs)


@DATASETS.register()
class ImageFolderWithClassification(ImageFolder):
    def __init__(self, cfg):
        super(ImageFolderWithClassification, self).__init__(cfg)

        root = self.root
        folders = [os.path.join(root, f) for f in sorted(os.listdir(root))]
        folders = [f for f in folders if os.path.isdir(f)]
        class_ids = []
        for path in self.imgs:
            for class_id, folder in enumerate(folders):
                if folder in path:
                    class_ids.append(class_id)
                    break
        self.class_ids = class_ids

    def __getitem__(self, index):
        return_dict = super(ImageFolderWithClassification, self).__getitem__(index)

        class_id = np.asarray(self.class_ids[index])
        return_dict['class_id'] = class_id
        
        return return_dict