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
from sr_model import RRDBNet
from paddle.utils.download import get_path_from_url

parser = argparse.ArgumentParser(description='RealSR')
parser.add_argument('--input', type=str, default='none', help='Input video')
parser.add_argument('--output', type=str, default='output', help='output dir')
parser.add_argument('--weight_path',
                    type=str,
                    default=None,
                    help='Path to the reference image directory')

RealSR_weight_url = 'https://paddlegan.bj.bcebos.com/applications/DF2K_JPEG.pdparams'


def frames_to_video_ffmpeg(framepath, videopath, r):
    ffmpeg = ['ffmpeg ', ' -loglevel ', ' error ']
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


class RealSRPredictor():
    def __init__(self, input, output, batch_size=1, weight_path=None):
        self.input = input
        self.output = os.path.join(output, 'RealSR')
        self.model = RRDBNet(3, 3, 64, 23)
        if weight_path is None:
            weight_path = get_path_from_url(RealSR_weight_url, cur_path)

        state_dict, _ = paddle.load(weight_path)
        self.model.load_dict(state_dict)
        self.model.eval()

    def norm(self, img):
        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 255.0
        return img.astype('float32')

    def denorm(self, img):
        img = img.transpose((1, 2, 0))
        return (img * 255).clip(0, 255).astype('uint8')

    def run_single(self, img_path):
        ori_img = Image.open(img_path).convert('RGB')
        img = self.norm(ori_img)
        x = paddle.to_tensor(img[np.newaxis, ...])
        out = self.model(x)

        pred_img = self.denorm(out.numpy()[0])
        pred_img = Image.fromarray(pred_img)
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

        out_path = dump_frames_ffmpeg(vid, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        for frame in tqdm(frames):
            pred_img = self.run_single(frame)

            frame_name = os.path.basename(frame)
            pred_img.save(os.path.join(pred_frame_path, frame_name))

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')

        vid_out_path = os.path.join(output_path,
                                    '{}_realsr_out.mp4'.format(base_name))
        frames_to_video_ffmpeg(frame_pattern_combined, vid_out_path,
                               str(int(fps)))

        return frame_pattern_combined, vid_out_path


def dump_frames_ffmpeg(vid_path, outpath, r=None, ss=None, t=None):
    ffmpeg = ['ffmpeg ', ' -loglevel ', ' error ']
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


if __name__ == '__main__':
    paddle.disable_static()
    args = parser.parse_args()

    predictor = RealSRPredictor(args.input,
                                args.output,
                                weight_path=args.weight_path)
    frames_path, temp_video_path = predictor.run()

    print('output video path:', temp_video_path)
