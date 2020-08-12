#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import cv2
import math
import random
import multiprocessing
import functools
import numpy as np
import paddle
import cv2
import logging
from .reader_utils import DataReader

logger = logging.getLogger(__name__)
python_ver = sys.version_info

random.seed(0)
np.random.seed(0)

class EDVRReader(DataReader):
    """
    Data reader for video super resolution task fit for EDVR model.
    This is specified for REDS dataset.
    """
    def __init__(self, name, mode, cfg):
        super(EDVRReader, self).__init__(name, mode, cfg)
        self.format = cfg.MODEL.format
        self.crop_size = self.get_config_from_sec(mode, 'crop_size')
        self.interval_list = self.get_config_from_sec(mode, 'interval_list')
        self.random_reverse = self.get_config_from_sec(mode, 'random_reverse')
        self.number_frames = self.get_config_from_sec(mode, 'number_frames')
        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.fileroot = cfg[mode.upper()]['file_root']
        self.use_flip = self.get_config_from_sec(mode, 'use_flip', False)
        self.use_rot = self.get_config_from_sec(mode, 'use_rot', False)

        self.num_reader_threads = self.get_config_from_sec(mode, 'num_reader_threads', 1)
        self.buf_size = self.get_config_from_sec(mode, 'buf_size', 1024)
        self.fix_random_seed = self.get_config_from_sec(mode, 'fix_random_seed', False)

        if self.mode != 'infer':
            self.gtroot = self.get_config_from_sec(mode, 'gt_root')
            self.scale = self.get_config_from_sec(mode, 'scale', 1)
            self.LR_input = (self.scale > 1)
        if self.fix_random_seed:
            random.seed(0)
            np.random.seed(0)
            self.num_reader_threads = 1

    def create_reader(self):
        logger.info('initialize reader ... ')
        self.filelist = []
        for video_name in os.listdir(self.fileroot):
            if (self.mode == 'train') and (video_name in ['000', '011', '015', '020']):
                continue
            for frame_name in os.listdir(os.path.join(self.fileroot, video_name)):
                frame_idx = frame_name.split('.')[0]
                video_frame_idx = video_name + '_' + frame_idx
                # for each item in self.filelist is like '010_00000015', '260_00000090'
                self.filelist.append(video_frame_idx)
        if self.mode == 'test' or self.mode == 'infer':
            self.filelist.sort()

        if self.num_reader_threads == 1:
            reader_func = make_reader
        else:
            reader_func = make_multi_reader

        if self.mode != 'infer':
            return reader_func(filelist = self.filelist,
                               num_threads = self.num_reader_threads,
                               batch_size = self.batch_size,
                               is_training = (self.mode == 'train'),
                               number_frames = self.number_frames,
                               interval_list = self.interval_list,
                               random_reverse = self.random_reverse,
                               fileroot = self.fileroot,
                               crop_size = self.crop_size,
                               use_flip = self.use_flip,
                               use_rot = self.use_rot,
                               gtroot = self.gtroot,
                               LR_input = self.LR_input,
                               scale = self.scale,
                               mode = self.mode)
        else:
            return reader_func(filelist = self.filelist,
                               num_threads = self.num_reader_threads,
                               batch_size = self.batch_size,
                               is_training = (self.mode == 'train'),
                               number_frames = self.number_frames,
                               interval_list = self.interval_list,
                               random_reverse = self.random_reverse,
                               fileroot = self.fileroot,
                               crop_size = self.crop_size,
                               use_flip = self.use_flip,
                               use_rot = self.use_rot,
                               gtroot = '',
                               LR_input = True,
                               scale = 4,
                               mode = self.mode)


def get_sample_data(item, number_frames, interval_list, random_reverse, fileroot, 
                    crop_size, use_flip, use_rot, gtroot, LR_input, scale, mode='train'):
    video_name = item.split('_')[0]
    frame_name = item.split('_')[1]
    if (mode == 'train') or (mode == 'valid'):
        ngb_frames, name_b = get_neighbor_frames(frame_name, \
                          number_frames = number_frames, \
                          interval_list = interval_list, \
                          random_reverse = random_reverse)
    elif (mode == 'test') or (mode == 'infer'):
        ngb_frames, name_b = get_test_neighbor_frames(int(frame_name), number_frames)
    else:
        raise NotImplementedError('mode {} not implemented'.format(mode))
    frame_name = name_b
    print('key2', ngb_frames, name_b)
    if mode != 'infer':
        img_GT = read_img(os.path.join(gtroot, video_name, frame_name + '.png'), is_gt=True)
    #print('gt_mean', np.mean(img_GT))
    frame_list = []
    for ngb_frm in ngb_frames:
        ngb_name = "%04d"%ngb_frm
        #img = read_img(os.path.join(fileroot, video_name, frame_name + '.png'))
        img = read_img(os.path.join(fileroot, video_name, ngb_name + '.png'))
        frame_list.append(img)
        #print('img_mean', np.mean(img))

    H, W, C = frame_list[0].shape
    # add random crop
    if (mode == 'train') or (mode == 'valid'):
        if LR_input:
            LQ_size = crop_size // scale
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            #print('rnd_h {}, rnd_w {}', rnd_h, rnd_w)
            frame_list = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in frame_list]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + crop_size, rnd_w_HR:rnd_w_HR + crop_size, :]
        else:
            rnd_h = random.randint(0, max(0, H - crop_size))
            rnd_w = random.randint(0, max(0, W - crop_size))
            frame_list = [v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for v in frame_list]
            img_GT = img_GT[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :]

    # add random flip and rotation
    if mode != 'infer': 
        frame_list.append(img_GT)
    if (mode == 'train') or (mode == 'valid'):
        rlt = img_augment(frame_list, use_flip, use_rot)
    else:
        rlt = frame_list
    if mode != 'infer':
        frame_list = rlt[0:-1]
        img_GT = rlt[-1]
    else:
        frame_list = rlt

    # stack LQ images to NHWC, N is the frame number
    img_LQs = np.stack(frame_list, axis=0)
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_LQs = img_LQs[:, :, :, [2, 1, 0]]
    img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')
    if mode != 'infer':
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_GT = np.transpose(img_GT, (2, 0, 1)).astype('float32')

        return img_LQs, img_GT
    else:
        return img_LQs

def get_test_neighbor_frames(crt_i, N, max_n=100, padding='new_info'):
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


def get_neighbor_frames(frame_name, number_frames, interval_list, random_reverse, max_frame=99, bordermode=False):
    center_frame_idx = int(frame_name)
    half_N_frames = number_frames // 2
    #### determine the neighbor frames
    interval = random.choice(interval_list)
    if bordermode:
        direction = 1  # 1: forward; 0: backward
        if random_reverse and random.random() < 0.5:
            direction = random.choice([0, 1])
        if center_frame_idx + interval * (number_frames - 1) > max_frame:
            direction = 0
        elif center_frame_idx - interval * (number_frames - 1) < 0:
            direction = 1
        # get the neighbor list
        if direction == 1:
            neighbor_list = list(
                range(center_frame_idx, center_frame_idx + interval * number_frames, interval))
        else:
            neighbor_list = list(
                range(center_frame_idx, center_frame_idx - interval * number_frames, -interval))
        name_b = '{:08d}'.format(neighbor_list[0])
    else:
        # ensure not exceeding the borders
        while (center_frame_idx + half_N_frames * interval >
           max_frame) or (center_frame_idx - half_N_frames * interval < 0):
            center_frame_idx = random.randint(0, max_frame)
        # get the neighbor list
        neighbor_list = list(
            range(center_frame_idx - half_N_frames * interval,
                  center_frame_idx + half_N_frames * interval + 1, interval))
        if random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
        name_b = '{:08d}'.format(neighbor_list[half_N_frames])
    assert len(neighbor_list) == number_frames, \
              "frames slected have length({}), but it should be ({})".format(len(neighbor_list), number_frames)

    return neighbor_list, name_b


def read_img(path, size=None, is_gt=False):
    """read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #if not is_gt:
    #    #print(path)
    #    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3] 
    return img 


def img_augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
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


def make_reader(filelist,
                num_threads,
                batch_size,
                is_training,
                number_frames,
                interval_list,
                random_reverse,
                fileroot,
                crop_size,
                use_flip,
                use_rot,
                gtroot,
                LR_input,
                scale,
                mode='train'):
    fl = filelist
    def reader_():
        if is_training:
            random.shuffle(fl)
        batch_out = []
        for item in fl:
            if mode != 'infer':
                img_LQs, img_GT = get_sample_data(item,
                                   number_frames, interval_list, random_reverse, fileroot,
                                   crop_size,use_flip, use_rot, gtroot, LR_input, scale, mode)
            else:
                img_LQs = get_sample_data(item,
                                   number_frames, interval_list, random_reverse, fileroot,
                                   crop_size,use_flip, use_rot, gtroot, LR_input, scale, mode)
            videoname = item.split('_')[0]
            framename = item.split('_')[1]
            if (mode == 'train') or (mode == 'valid'):
                batch_out.append((img_LQs, img_GT))
            elif mode == 'test':
                batch_out.append((img_LQs, img_GT, videoname, framename))
            elif mode == 'infer':
                batch_out.append((img_LQs, videoname, framename))
            else:
                raise NotImplementedError("mode {} not implemented".format(mode))
            if len(batch_out) == batch_size:
                yield batch_out
                batch_out = []
    return reader_


def make_multi_reader(filelist,
                      num_threads,
                      batch_size,
                      is_training,
                      number_frames,
                      interval_list,
                      random_reverse,
                      fileroot,
                      crop_size,
                      use_flip,
                      use_rot,
                      gtroot,
                      LR_input,
                      scale,
                      mode='train'):
    def read_into_queue(flq, queue):
        batch_out = []
        for item in flq:
            if mode != 'infer':
                img_LQs, img_GT = get_sample_data(item,
                                   number_frames, interval_list, random_reverse, fileroot,
                                   crop_size,use_flip, use_rot, gtroot, LR_input, scale, mode)
            else:
                img_LQs = get_sample_data(item,
                                   number_frames, interval_list, random_reverse, fileroot,
                                   crop_size,use_flip, use_rot, gtroot, LR_input, scale, mode)
            videoname = item.split('_')[0]
            framename = item.split('_')[1]
            if (mode == 'train') or (mode == 'valid'):
                batch_out.append((img_LQs, img_GT))
            elif mode == 'test':
                batch_out.append((img_LQs, img_GT, videoname, framename))
            elif mode == 'infer':
                batch_out.append((img_LQs, videoname, framename))
            else:
                raise NotImplementedError("mode {} not implemented".format(mode))
            if len(batch_out) == batch_size:
                queue.put(batch_out)
                batch_out = []
        queue.put(None)


    def queue_reader():
        fl = filelist
        if is_training:
            random.shuffle(fl)

        n = num_threads
        queue_size = 20
        reader_lists = [None] * n
        file_num = int(len(fl) // n)
        for i in range(n):
            if i < len(reader_lists) - 1:
                tmp_list = fl[i * file_num:(i + 1) * file_num]
            else:
                tmp_list = fl[i * file_num:]
            reader_lists[i] = tmp_list

        queue = multiprocessing.Queue(queue_size)
        p_list = [None] * len(reader_lists)
        # for reader_list in reader_lists:
        for i in range(len(reader_lists)):
            reader_list = reader_lists[i]
            p_list[i] = multiprocessing.Process(
                target=read_into_queue, args=(reader_list, queue))
            p_list[i].start()
        reader_num = len(reader_lists)
        finish_num = 0
        while finish_num < reader_num:
            sample = queue.get()
            if sample is None:
                finish_num += 1
            else:
                yield sample
        for i in range(len(p_list)):
            if p_list[i].is_alive():
                p_list[i].join()

    return queue_reader
