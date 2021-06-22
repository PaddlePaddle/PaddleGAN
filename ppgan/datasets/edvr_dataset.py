# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import scipy.io as scio
import cv2
import paddle
from paddle.io import Dataset, DataLoader
from .builder import DATASETS

logger = logging.getLogger(__name__)


@DATASETS.register()
class REDSDataset(Dataset):
    """
    REDS dataset for EDVR model
    """
    def __init__(self,
                 mode,
                 lq_folder,
                 gt_folder,
                 img_format="png",
                 crop_size=256,
                 interval_list=[1],
                 random_reverse=False,
                 number_frames=5,
                 batch_size=32,
                 use_flip=False,
                 use_rot=False,
                 buf_size=1024,
                 scale=4,
                 fix_random_seed=False):
        super(REDSDataset, self).__init__()
        self.format = img_format
        self.mode = mode
        self.crop_size = crop_size
        self.interval_list = interval_list
        self.random_reverse = random_reverse
        self.number_frames = number_frames
        self.batch_size = batch_size
        self.fileroot = lq_folder
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.buf_size = buf_size
        self.fix_random_seed = fix_random_seed

        if self.mode != 'infer':
            self.gtroot = gt_folder
            self.scale = scale
            self.LR_input = (self.scale > 1)
        if self.fix_random_seed:
            random.seed(10)
            np.random.seed(10)
            self.num_reader_threads = 1

        self._init_()

    def _init_(self):
        logger.info('initialize reader ... ')
        print("initialize reader")
        self.filelist = []
        for video_name in os.listdir(self.fileroot):
            if (self.mode == 'train') and (video_name in [
                    '000', '011', '015', '020'
            ]):  #These four videos are used as val
                continue
            for frame_name in os.listdir(os.path.join(self.fileroot,
                                                      video_name)):
                frame_idx = frame_name.split('.')[0]
                video_frame_idx = video_name + '_' + str(frame_idx)
                # for each item in self.filelist is like '010_00000015', '260_00000090'
                self.filelist.append(video_frame_idx)
        if self.mode == 'test':
            self.filelist.sort()
        print(len(self.filelist))

    def __getitem__(self, index):
        """Get training sample

        return: lq:[5,3,W,H],
                gt:[3,W,H],
                lq_path:str
        """
        item = self.filelist[index]
        img_LQs, img_GT = self.get_sample_data(
            item, self.number_frames, self.interval_list, self.random_reverse,
            self.gtroot, self.fileroot, self.LR_input, self.crop_size,
            self.scale, self.use_flip, self.use_rot, self.mode)
        return {'lq': img_LQs, 'gt': img_GT, 'lq_path': self.filelist[index]}

    def get_sample_data(self,
                        item,
                        number_frames,
                        interval_list,
                        random_reverse,
                        gtroot,
                        fileroot,
                        LR_input,
                        crop_size,
                        scale,
                        use_flip,
                        use_rot,
                        mode='train'):
        video_name = item.split('_')[0]
        frame_name = item.split('_')[1]
        if (mode == 'train') or (mode == 'valid'):
            ngb_frames, name_b = self.get_neighbor_frames(frame_name, \
                                                     number_frames=number_frames, \
                                                     interval_list=interval_list, \
                                                     random_reverse=random_reverse)
        elif mode == 'test':
            ngb_frames, name_b = self.get_test_neighbor_frames(
                int(frame_name), number_frames)
        else:
            raise NotImplementedError('mode {} not implemented'.format(mode))
        frame_name = name_b
        img_GT = self.read_img(
            os.path.join(gtroot, video_name, frame_name + '.png'))
        frame_list = []
        for ngb_frm in ngb_frames:
            ngb_name = "%08d" % ngb_frm
            img = self.read_img(
                os.path.join(fileroot, video_name, ngb_name + '.png'))
            frame_list.append(img)
        H, W, C = frame_list[0].shape
        # add random crop
        if (mode == 'train') or (mode == 'valid'):
            if LR_input:
                LQ_size = crop_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                frame_list = [
                    v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                    for v in frame_list
                ]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + crop_size,
                                rnd_w_HR:rnd_w_HR + crop_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - crop_size))
                rnd_w = random.randint(0, max(0, W - crop_size))
                frame_list = [
                    v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]
                    for v in frame_list
                ]
                img_GT = img_GT[rnd_h:rnd_h + crop_size,
                                rnd_w:rnd_w + crop_size, :]

        # add random flip and rotation
        frame_list.append(img_GT)
        if (mode == 'train') or (mode == 'valid'):
            rlt = self.img_augment(frame_list, use_flip, use_rot)
        else:
            rlt = frame_list
        frame_list = rlt[0:-1]
        img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(frame_list, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = np.transpose(img_GT, (2, 0, 1)).astype('float32')
        img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')

        return img_LQs, img_GT

    def get_neighbor_frames(self,
                            frame_name,
                            number_frames,
                            interval_list,
                            random_reverse,
                            max_frame=99,
                            bordermode=False):
        center_frame_idx = int(frame_name)
        half_N_frames = number_frames // 2
        interval = random.choice(interval_list)
        if bordermode:
            direction = 1
            if random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (number_frames - 1) > max_frame:
                direction = 0
            elif center_frame_idx - interval * (number_frames - 1) < 0:
                direction = 1
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx,
                          center_frame_idx + interval * number_frames,
                          interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx,
                          center_frame_idx - interval * number_frames,
                          -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + half_N_frames * interval > max_frame) or (
                    center_frame_idx - half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, max_frame)
            neighbor_list = list(
                range(center_frame_idx - half_N_frames * interval,
                      center_frame_idx + half_N_frames * interval + 1,
                      interval))
            if random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[half_N_frames])
        assert len(neighbor_list) == number_frames, \
            "frames slected have length({}), but it should be ({})".format(len(neighbor_list), number_frames)

        return neighbor_list, name_b

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
        return img

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

    def get_test_neighbor_frames(self, crt_i, N, max_n=100, padding='new_info'):
        """Generate an index list for reading N frames from a sequence of images
        Args:
            crt_i (int): current center index
            max_n (int): max number of the sequence of images (calculated from 1)
            N (int): reading N frames
            padding (str): padding mode, one of replicate | reflection | new_info | circle
                Example: crt_i = 0, N = 5
                replicate: [0, 0, 0, 1, 2]
                reflection: [2, 1, 0, 1, 2]
                new_info: [4, 3, 0, 1, 2]
                circle: [3, 4, 0, 1, 2]

        Returns:
            return_l (list [int]): a list of indexes
        """
        max_n = max_n - 1
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = N + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i > max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - N
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            return_l.append(add_idx)
        name_b = '{:08d}'.format(crt_i)
        return return_l, name_b

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return len(self.filelist)
