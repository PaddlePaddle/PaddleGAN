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
import glob
import shutil
import numpy as np
from tqdm import tqdm
from imageio import imread, imsave

import paddle
import paddle.fluid as fluid
from paddle.utils.download import get_path_from_url
from ppgan.utils.video import video2frames, frames2video

from .base_predictor import BasePredictor

DAIN_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/DAIN_weight.tar'


class DAINPredictor(BasePredictor):
    def __init__(self,
                 output_path='output',
                 weight_path=None,
                 time_step=None,
                 use_gpu=True,
                 key_frame_thread=0.,
                 remove_duplicates=False):
        self.output_path = os.path.join(output_path, 'DAIN')
        if weight_path is None:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            weight_path = get_path_from_url(DAIN_WEIGHT_URL, cur_path)

        self.weight_path = weight_path
        self.time_step = time_step
        self.key_frame_thread = key_frame_thread
        self.remove_duplicates = remove_duplicates

        self.build_inference_model()

    def run(self, video_path):
        frame_path_input = os.path.join(self.output_path, 'frames-input')
        frame_path_interpolated = os.path.join(self.output_path,
                                               'frames-interpolated')
        frame_path_combined = os.path.join(self.output_path, 'frames-combined')
        video_path_output = os.path.join(self.output_path, 'videos-output')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(frame_path_input):
            os.makedirs(frame_path_input)
        if not os.path.exists(frame_path_interpolated):
            os.makedirs(frame_path_interpolated)
        if not os.path.exists(frame_path_combined):
            os.makedirs(frame_path_combined)
        if not os.path.exists(video_path_output):
            os.makedirs(video_path_output)

        timestep = self.time_step
        num_frames = int(1.0 / timestep) - 1

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Old fps (frame rate): ", fps)

        times_interp = int(1.0 / timestep)
        r2 = str(int(fps) * times_interp)
        print("New fps (frame rate): ", r2)

        out_path = video2frames(video_path, frame_path_input)

        vidname = video_path.split('/')[-1].split('.')[0]

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))
        orig_frames = len(frames)
        need_frames = orig_frames * times_interp

        if self.remove_duplicates:
            frames = self.remove_duplicate_frames(out_path)
            left_frames = len(frames)
            timestep = left_frames / need_frames
            num_frames = int(1.0 / timestep) - 1

        img = imread(frames[0])

        int_width = img.shape[1]
        int_height = img.shape[0]
        channel = img.shape[2]
        if not channel == 3:
            return

        if int_width != ((int_width >> 7) << 7):
            int_width_pad = (((int_width >> 7) + 1) << 7)  # more than necessary
            padding_left = int((int_width_pad - int_width) / 2)
            padding_right = int_width_pad - int_width - padding_left
        else:
            int_width_pad = int_width
            padding_left = 32
            padding_right = 32

        if int_height != ((int_height >> 7) << 7):
            int_height_pad = (
                ((int_height >> 7) + 1) << 7)  # more than necessary
            padding_top = int((int_height_pad - int_height) / 2)
            padding_bottom = int_height_pad - int_height - padding_top
        else:
            int_height_pad = int_height
            padding_top = 32
            padding_bottom = 32

        frame_num = len(frames)

        if not os.path.exists(os.path.join(frame_path_interpolated, vidname)):
            os.makedirs(os.path.join(frame_path_interpolated, vidname))
        if not os.path.exists(os.path.join(frame_path_combined, vidname)):
            os.makedirs(os.path.join(frame_path_combined, vidname))

        for i in tqdm(range(frame_num - 1)):
            first = frames[i]
            second = frames[i + 1]

            img_first = imread(first)
            img_second = imread(second)
            '''--------------Frame change test------------------------'''
            img_first_gray = np.dot(img_first[..., :3], [0.299, 0.587, 0.114])
            img_second_gray = np.dot(img_second[..., :3], [0.299, 0.587, 0.114])

            img_first_gray = img_first_gray.flatten(order='C')
            img_second_gray = img_second_gray.flatten(order='C')
            corr = np.corrcoef(img_first_gray, img_second_gray)[0, 1]
            key_frame = False
            if corr < self.key_frame_thread:
                key_frame = True
            '''-------------------------------------------------------'''

            X0 = img_first.astype('float32').transpose((2, 0, 1)) / 255
            X1 = img_second.astype('float32').transpose((2, 0, 1)) / 255

            assert (X0.shape[1] == X1.shape[1])
            assert (X0.shape[2] == X1.shape[2])

            X0 = np.pad(X0, ((0,0), (padding_top, padding_bottom), \
                (padding_left, padding_right)), mode='edge')
            X1 = np.pad(X1, ((0,0), (padding_top, padding_bottom), \
                (padding_left, padding_right)), mode='edge')

            X0 = np.expand_dims(X0, axis=0)
            X1 = np.expand_dims(X1, axis=0)

            X0 = np.expand_dims(X0, axis=0)
            X1 = np.expand_dims(X1, axis=0)

            X = np.concatenate((X0, X1), axis=0)

            o = self.base_forward(X)

            y_ = o[0]

            y_ = [
                np.transpose(
                    255.0 * item.clip(
                        0, 1.0)[0, :, padding_top:padding_top + int_height,
                                padding_left:padding_left + int_width],
                    (1, 2, 0)) for item in y_
            ]
            time_offsets = [kk * timestep for kk in range(1, 1 + num_frames, 1)]

            count = 1
            for item, time_offset in zip(y_, time_offsets):
                out_dir = os.path.join(frame_path_interpolated, vidname,
                                       "{:0>6d}_{:0>4d}.png".format(i, count))
                count = count + 1
                imsave(out_dir, np.round(item).astype(np.uint8))

        num_frames = int(1.0 / timestep) - 1

        input_dir = os.path.join(frame_path_input, vidname)
        interpolated_dir = os.path.join(frame_path_interpolated, vidname)
        combined_dir = os.path.join(frame_path_combined, vidname)
        self.combine_frames(input_dir, interpolated_dir, combined_dir,
                            num_frames)

        frame_pattern_combined = os.path.join(frame_path_combined, vidname,
                                              '%08d.png')
        video_pattern_output = os.path.join(video_path_output, vidname + '.mp4')
        if os.path.exists(video_pattern_output):
            os.remove(video_pattern_output)
        frames2video(frame_pattern_combined, video_pattern_output, r2)

        return frame_pattern_combined, video_pattern_output

    def combine_frames(self, input, interpolated, combined, num_frames):
        frames1 = sorted(glob.glob(os.path.join(input, '*.png')))
        frames2 = sorted(glob.glob(os.path.join(interpolated, '*.png')))
        num1 = len(frames1)
        num2 = len(frames2)

        for i in range(num1):
            src = frames1[i]
            imgname = int(src.split('/')[-1].split('.')[-2])
            assert i == imgname
            dst = os.path.join(combined,
                               '{:08d}.png'.format(i * (num_frames + 1)))
            shutil.copy2(src, dst)
            if i < num1 - 1:
                try:
                    for k in range(num_frames):
                        src = frames2[i * num_frames + k]
                        dst = os.path.join(
                            combined,
                            '{:08d}.png'.format(i * (num_frames + 1) + k + 1))
                        shutil.copy2(src, dst)
                except Exception as e:
                    print(e)

    def remove_duplicate_frames(self, paths):
        def dhash(image, hash_size=8):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size + 1, hash_size))
            diff = resized[:, 1:] > resized[:, :-1]
            return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

        hashes = {}
        image_paths = sorted(glob.glob(os.path.join(paths, '*.png')))
        for image_path in image_paths:
            image = cv2.imread(image_path)
            h = dhash(image)
            p = hashes.get(h, [])
            p.append(image_path)
            hashes[h] = p

        for (h, hashed_paths) in hashes.items():
            if len(hashed_paths) > 1:
                for p in hashed_paths[1:]:
                    os.remove(p)

        frames = sorted(glob.glob(os.path.join(paths, '*.png')))
        for fid, frame in enumerate(frames):
            new_name = '{:08d}'.format(fid) + '.png'
            new_name = os.path.join(paths, new_name)
            os.rename(frame, new_name)

        frames = sorted(glob.glob(os.path.join(paths, '*.png')))
        return frames
