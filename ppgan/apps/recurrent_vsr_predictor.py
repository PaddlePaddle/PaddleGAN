#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from ppgan.models.generators import BasicVSRNet, IconVSR, BasicVSRPlusPlus, MSVSR
from .base_predictor import BasePredictor
from .edvr_predictor import get_img, read_img, save_img

BasicVSR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/BasicVSR_reds_x4.pdparams'
IconVSR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/IconVSR_reds_x4.pdparams'
BasicVSR_PP_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/BasicVSR%2B%2B_reds_x4.pdparams'
PP_MSVSR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/PP-MSVSR_reds_x4.pdparams'
PP_MSVSR_BD_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/PP-MSVSR_vimeo90k_x4.pdparams'
PP_MSVSR_L_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/models/PP-MSVSR-L_reds_x4.pdparams'


class RecurrentDataset(Dataset):
    def __init__(self, frames_path, num_frames=30):
        self.frames_path = frames_path

        if num_frames is not None:
            self.num_frames = num_frames
        else:
            self.num_frames = len(self.frames_path)

        if len(frames_path) % self.num_frames == 0:
            self.size = len(frames_path) // self.num_frames
        else:
            self.size = len(frames_path) // self.num_frames + 1

    def __getitem__(self, index):
        indexs = list(
            range(index * self.num_frames, (index + 1) * self.num_frames))
        frame_list = []
        frames_path = []
        for i in indexs:
            if i >= len(self.frames_path):
                break

            frames_path.append(self.frames_path[i])
            img = read_img(self.frames_path[i])
            frame_list.append(img)

        img_LQs = np.stack(frame_list, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')

        return img_LQs, frames_path

    def __len__(self):
        return self.size


class BasicVSRPredictor(BasePredictor):
    def __init__(self, output='output', weight_path=None, num_frames=10):
        self.input = input
        self.name = 'BasiVSR'
        self.num_frames = num_frames
        self.output = os.path.join(output, self.name)
        self.model = BasicVSRNet()
        if weight_path is None:
            weight_path = get_path_from_url(BasicVSR_WEIGHT_URL)
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

        test_dataset = RecurrentDataset(frames, num_frames=self.num_frames)
        dataset = DataLoader(test_dataset, batch_size=1, num_workers=2)

        periods = []
        cur_time = time.time()
        for infer_iter, data in enumerate(tqdm(dataset)):
            data_feed_in = paddle.to_tensor(data[0])
            with paddle.no_grad():
                outs = self.model(data_feed_in)

                if isinstance(outs, (list, tuple)):
                    outs = outs[-1]

                outs = outs[0].numpy()

            infer_result_list = [outs[i, :, :, :] for i in range(outs.shape[0])]

            frames_path = data[1]

            for i in range(len(infer_result_list)):
                img_i = get_img(infer_result_list[i])
                save_img(
                    img_i,
                    os.path.join(pred_frame_path,
                                 os.path.basename(frames_path[i][0])))

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')
        vid_out_path = os.path.join(
            self.output, '{}_{}_out.mp4'.format(base_name, self.name))
        frames2video(frame_pattern_combined, vid_out_path, str(int(fps)))

        return frame_pattern_combined, vid_out_path


class IconVSRPredictor(BasicVSRPredictor):
    def __init__(self, output='output', weight_path=None, num_frames=10):
        self.input = input
        self.name = 'IconVSR'
        self.output = os.path.join(output, self.name)
        self.num_frames = num_frames
        self.model = IconVSR()
        if weight_path is None:
            weight_path = get_path_from_url(IconVSR_WEIGHT_URL)
        self.model.set_dict(paddle.load(weight_path)['generator'])
        self.model.eval()


class BasiVSRPlusPlusPredictor(BasicVSRPredictor):
    def __init__(self, output='output', weight_path=None, num_frames=10):
        self.input = input
        self.name = 'BasiVSR_PP'
        self.output = os.path.join(output, self.name)
        self.num_frames = num_frames
        self.model = BasicVSRPlusPlus()
        if weight_path is None:
            weight_path = get_path_from_url(BasicVSR_PP_WEIGHT_URL)
        self.model.set_dict(paddle.load(weight_path)['generator'])
        self.model.eval()


class PPMSVSRPredictor(BasicVSRPredictor):
    def __init__(self, output='output', weight_path=None, num_frames=10):
        self.input = input
        self.name = 'PPMSVSR'
        self.output = os.path.join(output, self.name)
        self.num_frames = num_frames
        self.model = MSVSR()
        if weight_path is None:
            weight_path = get_path_from_url(PP_MSVSR_WEIGHT_URL)
        self.model.set_dict(paddle.load(weight_path)['generator'])
        self.model.eval()


class PPMSVSRLargePredictor(BasicVSRPredictor):
    def __init__(self, output='output', weight_path=None, num_frames=10):
        self.input = input
        self.name = 'PPMSVSR-L'
        self.output = os.path.join(output, self.name)
        self.num_frames = num_frames
        self.model = MSVSR(mid_channels=64,
                           num_init_blocks=5,
                           num_blocks=7,
                           num_reconstruction_blocks=5,
                           only_last=False,
                           use_tiny_spynet=False,
                           deform_groups=8,
                           aux_reconstruction_blocks=2)
        if weight_path is None:
            weight_path = get_path_from_url(PP_MSVSR_L_WEIGHT_URL)
        self.model.set_dict(paddle.load(weight_path)['generator'])
        self.model.eval()
