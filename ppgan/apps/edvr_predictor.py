#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
import time
import glob
import numpy as np
from tqdm import tqdm

import paddle
from paddle.io import Dataset, DataLoader

from ppgan.utils.download import get_path_from_url
from ppgan.utils.video import frames2video, video2frames
from ppgan.models.generators import EDVRNet
from .base_predictor import BasePredictor

EDVR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/EDVR_L_w_tsa_SRx4.pdparams'


def get_img(pred):
    pred = pred.squeeze()
    pred = np.clip(pred, a_min=0., a_max=1.0)
    pred = pred * 255
    pred = pred.round()
    pred = pred.astype('uint8')
    pred = np.transpose(pred, (1, 2, 0))  # chw -> hwc
    pred = pred[:, :, ::-1]  # rgb -> bgr
    return pred


def save_img(img, framename):
    dirname = os.path.dirname(framename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cv2.imwrite(framename, img)


def read_img(path, size=None, is_gt=False):
    """read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def get_test_neighbor_frames(crt_i, N, max_n, padding='new_info'):
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

    return return_l


class EDVRDataset(Dataset):
    def __init__(self, frame_paths):
        self.frames = frame_paths

    def __getitem__(self, index):
        indexs = get_test_neighbor_frames(index, 5, len(self.frames))
        frame_list = []
        for i in indexs:
            img = read_img(self.frames[i])
            frame_list.append(img)

        img_LQs = np.stack(frame_list, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')

        return img_LQs, self.frames[index]

    def __len__(self):
        return len(self.frames)


class EDVRPredictor(BasePredictor):
    def __init__(self, output='output', weight_path=None, bs=1):
        self.input = input
        self.output = os.path.join(output, 'EDVR')
        self.bs = bs
        self.model = EDVRNet(nf=128, back_RBs=40)
        if weight_path is None:
            weight_path = get_path_from_url(EDVR_WEIGHT_URL)
        self.model.set_dict(paddle.load(weight_path)['generator'])
        self.model.eval()

    def run(self, video_path):
        vid = video_path
        base_name = os.path.basename(vid).split('.')[0]
        output_path = os.path.join(self.output, base_name)
        pred_frame_path = os.path.join(output_path, 'frames_pred')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(pred_frame_path):
            os.makedirs(pred_frame_path)

        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = video2frames(vid, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        test_dataset = EDVRDataset(frames)
        dataset = DataLoader(test_dataset, batch_size=self.bs, num_workers=2)

        periods = []
        cur_time = time.time()
        for infer_iter, data in enumerate(tqdm(dataset)):
            data_feed_in = paddle.to_tensor(data[0])
            with paddle.no_grad():
                outs = self.model(data_feed_in).numpy()
            infer_result_list = [outs[i, :, :, :] for i in range(self.bs)]
            frame_path = data[1]
            for i in range(self.bs):
                img_i = get_img(infer_result_list[i])
                save_img(
                    img_i,
                    os.path.join(pred_frame_path,
                                 os.path.basename(frame_path[i])))

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')
        vid_out_path = os.path.join(self.output,
                                    '{}_edvr_out.mp4'.format(base_name))
        frames2video(frame_pattern_combined, vid_out_path, str(int(fps)))

        return frame_pattern_combined, vid_out_path
