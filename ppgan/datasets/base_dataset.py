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

import os
import copy
from pathlib import Path
from paddle.io import Dataset
from abc import ABCMeta, abstractmethod

from .preprocess import build_preprocess

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path,
                                        suffix=suffix,
                                        recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets should subclass it.
    All subclasses should overwrite:

        ``prepare_data_infos``, supporting to load information and generate
        image lists.

    Args:
        preprocess (list[dict]): A sequence of data preprocess config.

    """
    def __init__(self, preprocess=None):
        super(BaseDataset, self).__init__()

        if preprocess:
            self.preprocess = build_preprocess(preprocess)

    @abstractmethod
    def prepare_data_infos(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        should set self.annotations in this fucntion
        data_infos should be as list of dict:
        [{key_path: file_path}, {key_path: file_path}, {key_path: file_path}]
        """
        self.data_infos = None

    @staticmethod
    def scan_folder(path):
        """Obtain sample path list (including sub-folders) from a given folder.

        Args:
            path (str|pathlib.Path): Folder path.

        Returns:
            list[str]: sample list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        samples = list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True))
        samples = [os.path.join(path, v) for v in samples]
        assert samples, '{} has no valid image file.'.format(path)
        return samples

    def __getitem__(self, idx):
        datas = copy.deepcopy(self.data_infos[idx])

        if hasattr(self, 'preprocess') and self.preprocess:
            datas = self.preprocess(datas)

        return datas

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)
