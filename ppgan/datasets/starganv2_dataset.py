# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license

import paddle
from .base_dataset import BaseDataset
from .builder import DATASETS
import os
from itertools import chain
from pathlib import Path
import traceback
import random
import numpy as np
from PIL import Image

from paddle.io import Dataset, WeightedRandomSampler


def listdir(dname):
    fnames = list(
        chain(*[
            list(Path(dname).rglob('*.' + ext))
            for ext in ['png', 'jpg', 'jpeg', 'JPG']
        ]))
    return fnames


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class ImageFolder(Dataset):
    def __init__(self, root, use_sampler=False):
        self.samples, self.targets = self._make_dataset(root)
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.sampler = _make_balanced_sampler(self.targets)
            self.iter_sampler = iter(self.sampler)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        return fnames, labels

    def __getitem__(self, i):
        if self.use_sampler:
            try:
                index = next(self.iter_sampler)
            except StopIteration:
                self.iter_sampler = iter(self.sampler)
                index = next(self.iter_sampler)
        else:
            index = i
        fname = self.samples[index]
        label = self.targets[index]
        return fname, label

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(Dataset):
    def __init__(self, root, use_sampler=None):
        self.samples, self.targets = self._make_dataset(root)
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.sampler = _make_balanced_sampler(self.targets)
            self.iter_sampler = iter(self.sampler)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, i):
        if self.use_sampler:
            try:
                index = next(self.iter_sampler)
            except StopIteration:
                self.iter_sampler = iter(self.sampler)
                index = next(self.iter_sampler)
        else:
            index = i
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        return fname, fname2, label

    def __len__(self):
        return len(self.targets)


@DATASETS.register()
class StarGANv2Dataset(BaseDataset):
    """
    """
    def __init__(self, dataroot, is_train, preprocess, test_count=0):
        """Initialize single dataset class.

        Args:
            dataroot (str): Directory of dataset.
            preprocess (list[dict]): A sequence of data preprocess config.
        """
        super(StarGANv2Dataset, self).__init__(preprocess)

        self.dataroot = dataroot
        self.is_train = is_train
        if self.is_train:
            self.src_loader = ImageFolder(self.dataroot, use_sampler=True)
            self.ref_loader = ReferenceDataset(self.dataroot, use_sampler=True)
            self.counts = len(self.src_loader)
        else:
            files = os.listdir(self.dataroot)
            if 'src' in files and 'ref' in files:
                self.src_loader = ImageFolder(os.path.join(
                    self.dataroot, 'src'))
                self.ref_loader = ImageFolder(os.path.join(
                    self.dataroot, 'ref'))
            else:
                self.src_loader = ImageFolder(self.dataroot)
                self.ref_loader = ImageFolder(self.dataroot)
            self.counts = min(test_count, len(self.src_loader))
            self.counts = min(self.counts, len(self.ref_loader))

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter_src)
        except (AttributeError, StopIteration):
            self.iter_src = iter(self.src_loader)
            x, y = next(self.iter_src)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.ref_loader)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __getitem__(self, idx):
        if self.is_train:
            x, y = self._fetch_inputs()
            x_ref, x_ref2, y_ref = self._fetch_refs()
            datas = {
                'src_path': x,
                'src_cls': y,
                'ref_path': x_ref,
                'ref2_path': x_ref2,
                'ref_cls': y_ref,
            }
        else:
            x, y = self.src_loader[idx]
            x_ref, y_ref = self.ref_loader[idx]
            datas = {
                'src_path': x,
                'src_cls': y,
                'ref_path': x_ref,
                'ref_cls': y_ref,
            }

        if hasattr(self, 'preprocess') and self.preprocess:
            datas = self.preprocess(datas)

        return datas

    def __len__(self):
        return self.counts

    def prepare_data_infos(self, dataroot):
        pass
