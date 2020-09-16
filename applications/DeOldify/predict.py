import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

import cv2
import glob
import argparse
import numpy as np
import paddle
import pickle

from PIL import Image
from tqdm import tqdm
from paddle import fluid
from paddle.utils.download import get_path_from_url
from ppgan.utils.video import frames2video, video2frames
from ppgan.models.generators.deoldify import build_model


parser = argparse.ArgumentParser(description='DeOldify')
parser.add_argument('--input', type=str, default='none', help='Input video')
parser.add_argument('--output', type=str, default='output', help='output dir')
parser.add_argument('--render_factor',
                    type=int,
                    default=32,
                    help='model inputsize=render_factor*16')
parser.add_argument('--weight_path',
                    type=str,
                    default=None,
                    help='Path to the reference image directory')

DeOldify_weight_url = 'https://paddlegan.bj.bcebos.com/applications/DeOldify_stable.pdparams'


class DeOldifyPredictor():
    def __init__(self,
                 input,
                 output,
                 batch_size=1,
                 weight_path=None,
                 render_factor=32):
        self.input = input
        self.output = os.path.join(output, 'DeOldify')
        self.render_factor = render_factor
        self.model = build_model()
        if weight_path is None:
            weight_path = get_path_from_url(DeOldify_weight_url, cur_path)

        state_dict, _ = paddle.load(weight_path)
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

    def run_single(self, img_path):
        ori_img = Image.open(img_path).convert('LA').convert('RGB')
        img = self.norm(ori_img, self.render_factor)
        x = paddle.to_tensor(img[np.newaxis, ...])
        out = self.model(x)

        pred_img = self.denorm(out.numpy()[0])
        pred_img = Image.fromarray(pred_img)
        pred_img = pred_img.resize(ori_img.size, resample=Image.BILINEAR)
        pred_img = self.post_process(pred_img, ori_img)
        return pred_img

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

        out_path = video2frames(vid, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        for frame in tqdm(frames):
            pred_img = self.run_single(frame)

            frame_name = os.path.basename(frame)
            pred_img.save(os.path.join(pred_frame_path, frame_name))

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')

        vid_out_path = os.path.join(output_path,
                                    '{}_deoldify_out.mp4'.format(base_name))
        frames2video(frame_pattern_combined, vid_out_path,
                               str(int(fps)))

        return frame_pattern_combined, vid_out_path


if __name__ == '__main__':
    paddle.disable_static()
    args = parser.parse_args()

    predictor = DeOldifyPredictor(args.input,
                                  args.output,
                                  weight_path=args.weight_path,
                                  render_factor=args.render_factor)
    frames_path, temp_video_path = predictor.run()

    print('output video path:', temp_video_path)
