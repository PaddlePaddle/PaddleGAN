import glob
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
from imageio import mimread, imwrite
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from paddle.io import Dataset
from skimage import io
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split

from .builder import DATASETS


@DATASETS.register()
class FirstOrderDataset(Dataset):
    def __init__(self, cfg):
        super(FirstOrderDataset, self).__init__()
        self.cfg = cfg
        self.frameDataset = FramesDataset(cfg)
        if cfg['phase'] == 'train':
            self.outDataset = DatasetRepeater(self.frameDataset, cfg['num_repeats'])
        else:
            self.outDataset = self.frameDataset

    def __len__(self):
        return len(self.outDataset)

    def __getitem__(self, idx):
        return self.outDataset[idx]


def read_video(name, frame_shape, saveto='folder'):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
      - saveto: 若为'folder',则会在数据集同目录下创建与视频文件同名的文件夹，并向其写入jpg图像序列，最后删除源视频文件
    """
    Name = name
    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [io.imread(os.path.join(name, frames[idx]))[:, :, :3] for idx in range(num_frames)])
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        try:
            video = mimread(name)
        except Exception as err:
            logging.error('DataLoading File:%s Msg:%s' % (str(name), str(err)))
            return None
        if len(video[0].shape) == 3:
            if video[0].shape[-1] == 1:
                video = [gray2rgb(frame) for frame in video]
        if video[0].shape[-1] == 4:
            video = [i[..., :3] for i in video]
        video_array = np.array(video)
        if saveto == 'folder':
            sub_dir = pathlib.Path(name).with_suffix('')
            try:
                sub_dir.mkdir()
            except FileExistsError:
                pass
            for idx, img in enumerate(video_array):
                imwrite(sub_dir.joinpath('%i.png' % idx), img)
            pathlib.Path(Name).unlink()
    else:
        raise Exception("Unknown dataset file extensions  %s" % name)
    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    FramesDataset[i]: obtain sample from i-th video in self.videos
    """
    
    def __init__(self, cfg, augmentation_params=None):
        self.root_dir = cfg['dataroot']
        self.videos = os.listdir(self.root_dir)
        self.frame_shape = tuple(cfg['frame_shape'])
        self.id_sampling = cfg['id_sampling']
        phase = cfg['phase']
        random_seed = 0
        self.is_train = True if phase == 'train' else False
        self.pairs_list = cfg['pairs_list']
        self.process_time = cfg['process_time']
        self.create_frames_folder = cfg['create_frames_folder']
        if os.path.exists(os.path.join(self.root_dir, 'train')):
            assert os.path.exists(os.path.join(self.root_dir, 'test'))
            logging.info("Use predefined train-test split.")
            if self.id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(self.root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(self.root_dir, 'train'))
            test_videos = os.listdir(os.path.join(self.root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if self.is_train else 'test')
        else:
            logging.info("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        
        if self.is_train:
            self.videos = train_videos
            self.transform = augmentation_params
        else:
            self.videos = test_videos
            self.transform = None
        self.buffed = [None] * len(self.videos)

    def __len__(self):
        return len(self.videos)
    
    def colorize(self, image, hue):
        """Hue disturbance
        input range: [-1, 1]
        """
        res = rgb_to_hsv(image)
        res[:, :, 0] = res[:, :, 0] + hue
        res[:, :, 0][res[:, :, 0] < 0] = res[:, :, 0][res[:, :, 0] < 0] + 1
        res[:, :, 0][res[:, :, 0] > 1] = res[:, :, 0][res[:, :, 0] > 1] - 1
        res = hsv_to_rgb(res)
        return res
    
    def preload(self, idx):
        """return the $idx$-th video
        """
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
        video_array = read_video(path, frame_shape=self.frame_shape)
        return video_array
    
    def __getitem__(self, idx):
        if self.process_time:
            a0 = time.process_time()
        if self.is_train and self.id_sampling:
            # id_sampling=True is not tested, because id_sampling in mgif/bair/fashion are False
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
        video_name = os.path.basename(path)
        
        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [io.imread(os.path.join(path, frames[idx])) for idx in frame_idx]
            
            # convert to 3-channel image
            if video_array[0].shape[-1] == 4:
                video_array = [i[..., :3] for i in video_array]
            elif video_array[0].shape[-1] == 1:
                video_array = [np.tile(i, (1, 1, 3)) for i in video_array]
            elif len(video_array[0].shape) == 2:
                video_array = [np.tile(i[..., np.newaxis], (1, 1, 3)) for i in video_array]
        else:
            if self.buffed[idx] is None:
                if self.create_frames_folder:
                    video_array = read_video(path, frame_shape=self.frame_shape, saveto='folder')
                    self.videos[idx] = name.split('.')[0]
                else:
                    video_array = read_video(path, frame_shape=self.frame_shape, saveto=None)
            else:
                video_array = self.buffed[idx]
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = [video_array[i] for i in frame_idx]
        if self.process_time:
            a1 = time.process_time()
            print('Load T:%1.5f' % (a1 - a0))
        out = {}
        
        # Dataset enhancement
        if self.is_train:
            source = np.array(video_array[0], dtype='float32') / 255.0  # shape is [H, W, C]
            driving = np.array(video_array[1], dtype='float32') / 255.0  # shape is [H, W, C]
            if len(driving.shape) == 2:
                driving = driving[..., np.newaxis]
                driving = np.tile(driving, (1, 1, 3))
            if len(source.shape) == 2:
                source = source[..., np.newaxis]
                source = np.tile(source, (1, 1, 3))
            if self.process_time:
                a11 = time.process_time()
            # random_flip_left_right
            if 'flip_param' in self.transform.keys() and self.transform['flip_param']['horizontal_flip']:
                if np.random.random() >= 0.5:
                    driving = driving[:, ::-1, :]
                    source = source[:, ::-1, :]
            if self.process_time:
                a12 = time.process_time()
                print('A11-12 T:%1.5f' % (a12 - a11))
            # time_flip
            if 'flip_param' in self.transform.keys() and self.transform['flip_param']['time_flip']:
                if np.random.random() >= 0.5:
                    buf = driving
                    driving = source
                    source = buf
            if self.process_time:
                a13 = time.process_time()
                print('A12-13 T:%1.5f' % (a13 - a12))
            # jitter_param 只写了hue
            if 'jitter_param' in self.transform.keys():
                if 'hue' in self.transform['jitter_param'].keys():
                    jitter_value = (np.random.random() * 2 - 1) * self.transform['jitter_param']['hue']
                    driving = self.colorize(driving, jitter_value)
                    source = self.colorize(source, jitter_value)
            if self.process_time:
                a14 = time.process_time()
                print('A13-14 T:%1.5f' % (a14 - a13))
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.stack(video_array, axis=0).astype(np.float32) / 255.0
            out['video'] = video.transpose((3, 0, 1, 2))
            return out['video']
        if self.process_time:
            a2 = time.process_time()
            print('Trans T:%1.5f' % (a2 - a14))
        out['name'] = video_name
        return out
    
    def getSample(self, idx):
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


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """
    
    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list
        
        np.random.seed(seed)
        
        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]
            
            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]  # 'driving'
        second = self.initial_dataset[pair[1]]  # 'source':[channel, frame, h, w]
        return first, second[:, 0, :, :]