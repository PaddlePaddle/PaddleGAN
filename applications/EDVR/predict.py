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
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

import time
import argparse
import ast
import glob
import numpy as np

import paddle.fluid as fluid
import cv2

from tqdm import tqdm
from data import EDVRDataset
from paddle.utils.download import get_path_from_url

EDVR_weight_url = 'https://paddlegan.bj.bcebos.com/applications/edvr_infer_model.tar'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default=None,
                        help='input video path')
    parser.add_argument('--output',
                        type=str,
                        default='output',
                        help='output path')
    parser.add_argument('--weight_path',
                        type=str,
                        default=None,
                        help='weight path')
    args = parser.parse_args()
    return args


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


def dump_frames_ffmpeg(vid_path, outpath, r=None, ss=None, t=None):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(outpath, 'frames_input')

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = out_full_path + '/%08d.png'

    if ss is not None and t is not None and r is not None:
        cmd = ffmpeg + [
            ' -ss ', ss, ' -t ', t, ' -i ', vid_path, ' -r ', r, ' -qscale:v ',
            ' 0.1 ', ' -start_number ', ' 0 ', outformat
        ]
    else:
        cmd = ffmpeg + [' -i ', vid_path, ' -start_number ', ' 0 ', outformat]

    cmd = ''.join(cmd)
    print(cmd)
    if os.system(cmd) == 0:
        print('Video: {} done'.format(vid_name))
    else:
        print('Video: {} error'.format(vid_name))
    print('')
    sys.stdout.flush()
    return out_full_path


def frames_to_video_ffmpeg(framepath, videopath, r):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    cmd = ffmpeg + [
        ' -r ', r, ' -f ', ' image2 ', ' -i ', framepath, ' -vcodec ',
        ' libx264 ', ' -pix_fmt ', ' yuv420p ', ' -crf ', ' 16 ', videopath
    ]
    cmd = ''.join(cmd)
    print(cmd)

    if os.system(cmd) == 0:
        print('Video: {} done'.format(videopath))
    else:
        print('Video: {} error'.format(videopath))
    print('')
    sys.stdout.flush()


class EDVRPredictor:
    def __init__(self, input, output, weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'EDVR')

        place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        self.exe = fluid.Executor(place)

        if weight_path is None:
            weight_path = get_path_from_url(EDVR_weight_url, cur_path)

        print(weight_path)

        model_filename = 'EDVR_model.pdmodel'
        params_filename = 'EDVR_params.pdparams'

        out = fluid.io.load_inference_model(dirname=weight_path,
                                            model_filename=model_filename,
                                            params_filename=params_filename,
                                            executor=self.exe)
        self.infer_prog, self.feed_list, self.fetch_list = out

    def run(self):
        vid = self.input
        base_name = os.path.basename(vid).split('.')[0]
        output_path = os.path.join(self.output, base_name)
        pred_frame_path = os.path.join(output_path, 'frames_pred')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(pred_frame_path):
            os.makedirs(pred_frame_path)

        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = dump_frames_ffmpeg(vid, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        dataset = EDVRDataset(frames)

        periods = []
        cur_time = time.time()
        for infer_iter, data in enumerate(tqdm(dataset)):
            data_feed_in = [data[0]]

            infer_outs = self.exe.run(
                self.infer_prog,
                fetch_list=self.fetch_list,
                feed={self.feed_list[0]: np.array(data_feed_in)})
            infer_result_list = [item for item in infer_outs]

            frame_path = data[1]

            img_i = get_img(infer_result_list[0])
            save_img(
                img_i,
                os.path.join(pred_frame_path, os.path.basename(frame_path)))

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

            # print('Processed {} samples'.format(infer_iter + 1))
        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')
        vid_out_path = os.path.join(self.output,
                                    '{}_edvr_out.mp4'.format(base_name))
        frames_to_video_ffmpeg(frame_pattern_combined, vid_out_path,
                               str(int(fps)))

        return frame_pattern_combined, vid_out_path


if __name__ == "__main__":
    args = parse_args()
    predictor = EDVRPredictor(args.input, args.output, args.weight_path)
    predictor.run()
