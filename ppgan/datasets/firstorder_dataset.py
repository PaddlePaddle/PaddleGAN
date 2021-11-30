# code was heavily based on https://github.com/AliaksandrSiarohin/first-order-model
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/AliaksandrSiarohin/first-order-model/blob/master/LICENSE.md

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm
from imageio import imread, mimread, imwrite
import cv2
from paddle.io import Dataset
from .builder import DATASETS
from .preprocess.builder import build_transforms
import glob, os

POOL_SIZE = 64  # If POOL_SIZE>0 use multiprocessing to extract frames from gif file


@DATASETS.register()
class FirstOrderDataset(Dataset):
    def __init__(self, **cfg):
        """Initialize FirstOrder dataset class.

        Args:
            dataroot (str): Directory of dataset.
            phase (str): train or test
            num_repeats (int): Number for datasets to repeat
            time_flip (bool): whether to exchange the driving image and source image randomly
            batch_size (int): dataset batch size
            id_sampling (bool): whether to sample person's id
            frame_shape (list): image shape
            create_frames_folder (bool): if the format of your input datasets is '.mp4', \
                                         you can choose whether to save it with images
            num_workers (int): dataset
        """
        super(FirstOrderDataset, self).__init__()
        self.cfg = cfg
        self.frameDataset = FramesDataset(self.cfg)

        # create frames folder before 'DatasetRepeater'
        if self.cfg['create_frames_folder']:
            file_idx_set = [
                idx for idx, path in enumerate(self.frameDataset.videos)
                if not self.frameDataset.root_dir.joinpath(path).is_dir()
            ]
            file_idx_set = list(file_idx_set)
            if len(file_idx_set) != 0:
                if POOL_SIZE == 0:
                    for idx in tqdm.tqdm(file_idx_set,
                                         desc='Extracting frames'):
                        _ = self.frameDataset[idx]
                else:
                    # multiprocessing
                    bar = tqdm.tqdm(total=len(file_idx_set),
                                    desc='Extracting frames')
                    with Pool(POOL_SIZE) as pl:
                        _p = 0
                        while _p <= len(file_idx_set) - 1:
                            _ = pl.map(self.frameDataset.__getitem__,
                                       file_idx_set[_p:_p + POOL_SIZE * 2])
                            _p = _p + POOL_SIZE * 2
                            bar.update(POOL_SIZE * 2)
                    bar.close()

                # rewrite video path
                self.frameDataset.videos = [
                    i.with_suffix('') for i in self.frameDataset.videos
                ]

        if self.cfg['phase'] == 'train':
            self.outDataset = DatasetRepeater(self.frameDataset,
                                              self.cfg['num_repeats'])
        else:
            self.outDataset = self.frameDataset

    def __len__(self):
        return len(self.outDataset)

    def __getitem__(self, idx):
        return self.outDataset[idx]


def read_video(name: Path, frame_shape=tuple([256, 256, 3]), saveto='folder'):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """
    if name.is_dir():
        frames = sorted(name.iterdir(),
                        key=lambda x: int(x.with_suffix('').name))
        video_array = np.array([imread(path) for path in frames],
                               dtype='float32')
        return video_array
    elif name.suffix.lower() in ['.gif', '.mp4', '.mov']:
        try:
            video = mimread(name, memtest=False)
        except Exception as err:
            logging.error('DataLoading File:%s Msg:%s' % (str(name), str(err)))
            return None

        # convert to 3-channel image
        if video[0].shape[-1] == 4:
            video = [i[..., :3] for i in video]
        elif video[0].shape[-1] == 1:
            video = [np.tile(i, (1, 1, 3)) for i in video]
        elif len(video[0].shape) == 2:
            video = [np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video]
        video_array = np.asarray(video)
        video_array_reshape = []
        for idx, img in enumerate(video_array):
            img = cv2.resize(img, (frame_shape[0], frame_shape[1]))
            video_array_reshape.append(img.astype(np.uint8))
        video_array_reshape = np.asarray(video_array_reshape)

        if saveto == 'folder':
            sub_dir = name.with_suffix('')
            try:
                sub_dir.mkdir()
            except FileExistsError:
                pass
            for idx, img in enumerate(video_array_reshape):
                cv2.imwrite(str(sub_dir.joinpath('%i.png' % idx)), img[:,:,[2,1,0]])
            name.unlink()
        return video_array_reshape
    else:
        raise Exception("Unknown dataset file extensions  %s" % name)


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    FramesDataset[i]: obtain sample from i-th video in self.videos
    """
    def __init__(self, cfg):
        self.root_dir = Path(cfg['dataroot'])
        self.videos = None
        self.frame_shape = tuple(cfg['frame_shape'])
        self.id_sampling = cfg['id_sampling']
        self.time_flip = cfg['time_flip']
        self.is_train = True if cfg['phase'] == 'train' else False
        self.pairs_list = cfg.setdefault('pairs_list', None)
        self.create_frames_folder = cfg['create_frames_folder']
        self.transform = None
        random_seed = 0
        assert self.root_dir.joinpath('train').exists()
        assert self.root_dir.joinpath('test').exists()
        logging.info("Use predefined train-test split.")
        if self.id_sampling:
            train_videos = {
                video.name.split('#')[0]
                for video in self.root_dir.joinpath('train').iterdir()
            }
            train_videos = list(train_videos)
        else:
            train_videos = list(self.root_dir.joinpath('train').iterdir())
        test_videos = list(self.root_dir.joinpath('test').iterdir())
        self.root_dir = self.root_dir.joinpath(
            'train' if self.is_train else 'test')

        if self.is_train:
            self.videos = train_videos
            self.transform = build_transforms(cfg['transforms'])
        else:
            self.videos = test_videos
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = Path(
                np.random.choice(
                    glob.glob(os.path.join(self.root_dir, name + '*.mp4'))))
        else:
            path = self.videos[idx]
        video_name = path.name
        if self.is_train and path.is_dir():
            frames = sorted(path.iterdir(),
                            key=lambda x: int(x.with_suffix('').name))
            num_frames = len(frames)
            frame_idx = np.sort(
                np.random.choice(num_frames, replace=True, size=2))
            video_array = [imread(str(frames[idx])) for idx in frame_idx]
        else:
            if self.create_frames_folder:
                video_array = read_video(path,
                                         frame_shape=self.frame_shape,
                                         saveto='folder')
                self.videos[idx] = path.with_suffix(
                    '')  # rename /xx/xx/xx.gif -> /xx/xx/xx
            else:
                video_array = read_video(path,
                                         frame_shape=self.frame_shape,
                                         saveto=None)
            num_frames = len(video_array)
            frame_idx = np.sort(
                np.random.choice(
                    num_frames, replace=True,
                    size=2)) if self.is_train else range(num_frames)
            video_array = [video_array[i] for i in frame_idx]
        # convert to 3-channel image
        if video_array[0].shape[-1] == 4:
            video_array = [i[..., :3] for i in video_array]
        elif video_array[0].shape[-1] == 1:
            video_array = [np.tile(i, (1, 1, 3)) for i in video_array]
        elif len(video_array[0].shape) == 2:
            video_array = [
                np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video_array
            ]
        out = {}
        if self.is_train:
            if self.transform is not None:  #modify
                t = self.transform(tuple(video_array))
                out['driving'] = t[0].transpose(2, 0, 1).astype(
                    np.float32) / 255.0
                out['source'] = t[1].transpose(2, 0, 1).astype(
                    np.float32) / 255.0
            else:
                source = np.array(video_array[0],
                                  dtype='float32') / 255.0  # shape is [H, W, C]
                driving = np.array(
                    video_array[1],
                    dtype='float32') / 255.0  # shape is [H, W, C]
                out['driving'] = driving.transpose(2, 0, 1)
                out['source'] = source.transpose(2, 0, 1)
            if self.time_flip and np.random.rand() < 0.5:  #modify
                buf = out['driving']
                out['driving'] = out['source']
                out['source'] = buf
        else:
            video = np.stack(video_array, axis=0).astype(np.float32) / 255.0
            out['video'] = video.transpose(3, 0, 1, 2)
        out['name'] = video_name
        return out

    def get_sample(self, idx):
        return self.__getitem__(idx)


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """
    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
