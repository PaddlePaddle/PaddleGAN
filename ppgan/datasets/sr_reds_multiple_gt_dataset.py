# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import logging
import os
import random
import numpy as np
import cv2
from paddle.io import Dataset

from .builder import DATASETS

logger = logging.getLogger(__name__)


@DATASETS.register()
class SRREDSMultipleGTDataset(Dataset):
    """REDS dataset for video super resolution for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """
    def __init__(self,
                 mode,
                 lq_folder,
                 gt_folder,
                 crop_size=256,
                 interval_list=[1],
                 random_reverse=False,
                 number_frames=15,
                 use_flip=False,
                 use_rot=False,
                 scale=4,
                 val_partition='REDS4',
                 batch_size=4,
                 num_clips=270):
        super(SRREDSMultipleGTDataset, self).__init__()
        self.mode = mode
        self.fileroot = str(lq_folder)
        self.gtroot = str(gt_folder)
        self.crop_size = crop_size
        self.interval_list = interval_list
        self.random_reverse = random_reverse
        self.number_frames = number_frames
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.scale = scale
        self.val_partition = val_partition
        self.batch_size = batch_size
        self.num_clips = num_clips  # training num of LQ and GT pairs
        self.data_infos = self.load_annotations()

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        item = self.data_infos[idx]
        idt = random.randint(0, 100 - self.number_frames)
        item = item + '_' + f'{idt:03d}'
        img_LQs, img_GTs = self.get_sample_data(
            item, self.number_frames, self.interval_list, self.random_reverse,
            self.gtroot, self.fileroot, self.crop_size, self.scale,
            self.use_flip, self.use_rot, self.mode)
        return {'lq': img_LQs, 'gt': img_GTs, 'lq_path': self.data_infos[idx]}

    def load_annotations(self):
        """Load annoations for REDS dataset.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        # generate keys
        keys = [f'{i:03d}' for i in range(0, self.num_clips)]

        if self.val_partition == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif self.val_partition == 'official':
            val_partition = [f'{i:03d}' for i in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {self.val_partition}.'
                             f'Supported ones are ["official", "REDS4"]')

        if self.mode == 'train':
            keys = [v for v in keys if v not in val_partition]
        else:
            keys = [v for v in keys if v in val_partition]

        data_infos = []
        for key in keys:
            data_infos.append(key)

        return data_infos

    def get_sample_data(self,
                        item,
                        number_frames,
                        interval_list,
                        random_reverse,
                        gtroot,
                        fileroot,
                        crop_size,
                        scale,
                        use_flip,
                        use_rot,
                        mode='train'):
        video_name = item.split('_')[0]
        frame_name = item.split('_')[1]
        frame_idxs = self.get_neighbor_frames(frame_name,
                                              number_frames=number_frames,
                                              interval_list=interval_list,
                                              random_reverse=random_reverse)

        frame_list = []
        gt_list = []
        for frame_idx in frame_idxs:
            frame_idx_name = "%08d" % frame_idx
            img = self.read_img(
                os.path.join(fileroot, video_name, frame_idx_name + '.png'))
            frame_list.append(img)
            gt_img = self.read_img(
                os.path.join(gtroot, video_name, frame_idx_name + '.png'))
            gt_list.append(gt_img)
        H, W, C = frame_list[0].shape
        # add random crop
        if (mode == 'train') or (mode == 'valid'):
            LQ_size = crop_size // scale
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            frame_list = [
                v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                for v in frame_list
            ]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            gt_list = [
                v[rnd_h_HR:rnd_h_HR + crop_size,
                  rnd_w_HR:rnd_w_HR + crop_size, :] for v in gt_list
            ]

        # add random flip and rotation
        for v in gt_list:
            frame_list.append(v)
        if (mode == 'train') or (mode == 'valid'):
            rlt = self.img_augment(frame_list, use_flip, use_rot)
        else:
            rlt = frame_list
        frame_list = rlt[0:number_frames]
        gt_list = rlt[number_frames:]

        # stack LQ images to NHWC, N is the frame number
        frame_list = [
            v.transpose(2, 0, 1).astype('float32') for v in frame_list
        ]
        gt_list = [v.transpose(2, 0, 1).astype('float32') for v in gt_list]

        img_LQs = np.stack(frame_list, axis=0)
        img_GTs = np.stack(gt_list, axis=0)

        return img_LQs, img_GTs

    def get_neighbor_frames(self, frame_name, number_frames, interval_list,
                            random_reverse):
        frame_idx = int(frame_name)
        interval = random.choice(interval_list)
        neighbor_list = list(
            range(frame_idx, frame_idx + number_frames, interval))
        if random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == number_frames, \
            "frames slected have length({}), but it should be ({})".format(len(neighbor_list), number_frames)

        return neighbor_list

    def read_img(self, path, size=None):
        """read image by cv2

        return: Numpy float32, HWC, BGR, [0,1]
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def img_augment(self, img_list, hflip=True, rot=True):
        """horizontal flip OR rotate (0, 90, 180, 270 degrees)
        """
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in img_list]

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)
