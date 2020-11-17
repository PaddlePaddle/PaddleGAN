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
import numpy as np
from PIL import Image
from tqdm import tqdm

import paddle
from ppgan.utils.download import get_path_from_url
from ppgan.utils.video import frames2video, video2frames
from ppgan.models.generators.deoldify import build_model
from ppgan.utils.logger import get_logger

from .base_predictor import BasePredictor

DEOLDIFY_STABLE_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/DeOldify_stable.pdparams'
DEOLDIFY_ART_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/DeOldify_art.pdparams'


class DeOldifyPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 artistic=False,
                 render_factor=32):
        self.output = os.path.join(output, 'DeOldify')
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.render_factor = render_factor
        self.model = build_model(
            model_type='artistic' if artistic else 'stable')
        if weight_path is None:
            if artistic:
                weight_path = get_path_from_url(DEOLDIFY_ART_WEIGHT_URL)
            else:
                weight_path = get_path_from_url(DEOLDIFY_STABLE_WEIGHT_URL)

        state_dict = paddle.load(weight_path)
        self.model.load_dict(state_dict)
        self.model.eval()

    def norm(self, img, render_factor=32, render_base=16):
        target_size = render_factor * render_base
        img = img.resize((target_size, target_size), resample=Image.BILINEAR)

        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 255.0

        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        img -= img_mean
        img /= img_std
        return img.astype('float32')

    def denorm(self, img):
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        img *= img_std
        img += img_mean
        img = img.transpose((1, 2, 0))

        return (img * 255).clip(0, 255).astype('uint8')

    def post_process(self, raw_color, orig):
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        final = Image.fromarray(final)
        return final

    def run_image(self, img):
        if isinstance(img, str):
            ori_img = Image.open(img).convert('LA').convert('RGB')
        elif isinstance(img, np.ndarray):
            ori_img = Image.fromarray(img).convert('LA').convert('RGB')
        elif isinstance(img, Image.Image):
            ori_img = img

        img = self.norm(ori_img, self.render_factor)
        x = paddle.to_tensor(img[np.newaxis, ...])
        out = self.model(x)

        pred_img = self.denorm(out.numpy()[0])
        pred_img = Image.fromarray(pred_img)
        pred_img = pred_img.resize(ori_img.size, resample=Image.BILINEAR)
        pred_img = self.post_process(pred_img, ori_img)
        return pred_img

    def run_video(self, video):
        base_name = os.path.basename(video).split('.')[0]
        output_path = os.path.join(self.output, base_name)
        pred_frame_path = os.path.join(output_path, 'frames_pred')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(pred_frame_path):
            os.makedirs(pred_frame_path)

        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = video2frames(video, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        for frame in tqdm(frames):
            pred_img = self.run_image(frame)

            frame_name = os.path.basename(frame)
            pred_img.save(os.path.join(pred_frame_path, frame_name))

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')

        vid_out_path = os.path.join(output_path,
                                    '{}_deoldify_out.mp4'.format(base_name))
        frames2video(frame_pattern_combined, vid_out_path, str(int(fps)))

        return frame_pattern_combined, vid_out_path

    def run(self, input):
        if not self.is_image(input):
            return self.run_video(input)
        else:
            pred_img = self.run_image(input)

            out_path = None
            if self.output:
                try:
                    base_name = os.path.splitext(os.path.basename(input))[0]
                except:
                    base_name = 'result'
                out_path = os.path.join(self.output, base_name + '.png')
                pred_img.save(out_path)
                logger = get_logger()
                logger.info('Image saved to {}'.format(out_path))

            return pred_img, out_path
