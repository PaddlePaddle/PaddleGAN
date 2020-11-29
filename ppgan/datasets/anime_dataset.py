import cv2
import numpy as np
import os.path
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset, ImageFolder

from .builder import DATASETS
from .transforms.builder import build_transforms


@DATASETS.register()
class AnimeV2Dataset(BaseDataset):
    """
    """
    def __init__(self, cfg):
        """Initialize this dataset class.

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, cfg)
        self.style = cfg.style

        self.transform_real = build_transforms(self.cfg.transform_real)
        self.transform_anime = build_transforms(self.cfg.transform_anime)
        self.transform_gray = build_transforms(self.cfg.transform_gray)

        self.real_root = os.path.join(self.root, 'train_photo')
        self.anime_root = os.path.join(self.root, f'{self.style}', 'style')
        self.smooth_root = os.path.join(self.root, f'{self.style}', 'smooth')

        self.real = ImageFolder(self.real_root,
                                transform=self.transform_real,
                                loader=self.loader)
        self.anime = ImageFolder(self.anime_root,
                                 transform=self.transform_anime,
                                 loader=self.loader)
        self.anime_gray = ImageFolder(self.anime_root,
                                      transform=self.transform_gray,
                                      loader=self.loader)
        self.smooth_gray = ImageFolder(self.smooth_root,
                                       transform=self.transform_gray,
                                       loader=self.loader)
        self.sizes = [
            len(fold) for fold in [self.real, self.anime, self.smooth_gray]
        ]
        self.size = max(self.sizes)
        self.reshuffle()

    @staticmethod
    def loader(path):
        return cv2.cvtColor(cv2.imread(path, flags=cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)

    def reshuffle(self):
        indexs = []
        for cur_size in self.sizes:
            x = np.arange(0, cur_size)
            np.random.shuffle(x)
            if cur_size != self.size:
                pad_num = self.size - cur_size
                pad = np.random.choice(cur_size, pad_num, replace=True)
                x = np.concatenate((x, pad))
                np.random.shuffle(x)
            indexs.append(x.tolist())
        self.indexs = list(zip(*indexs))

    def __getitem__(self, index):
        try:
            index = self.indexs.pop()
        except IndexError as e:
            self.reshuffle()
            index = self.indexs.pop()

        real_idx, anime_idx, smooth_idx = index

        return {
            'real': self.real[real_idx],
            'anime': self.anime[anime_idx],
            'anime_gray': self.anime_gray[anime_idx],
            'smooth_gray': self.smooth_gray[smooth_idx]
        }

    def __len__(self):
        return self.size
