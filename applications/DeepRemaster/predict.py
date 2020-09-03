import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

import paddle
import paddle.nn as nn

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
import utils
from remasternet import NetworkR, NetworkC
from paddle.utils.download import get_path_from_url

DeepRemaster_weight_url = 'https://paddlegan.bj.bcebos.com/applications/deep_remaster.pdparams'

parser = argparse.ArgumentParser(description='Remastering')
parser.add_argument('--input', type=str, default=None, help='Input video')
parser.add_argument('--output', type=str, default='output', help='output dir')
parser.add_argument('--reference_dir',
                    type=str,
                    default=None,
                    help='Path to the reference image directory')
parser.add_argument('--colorization',
                    action='store_true',
                    default=False,
                    help='Remaster without colorization')
parser.add_argument('--mindim',
                    type=int,
                    default='360',
                    help='Length of minimum image edges')


class DeepReasterPredictor:
    def __init__(self,
                 input,
                 output,
                 weight_path=None,
                 colorization=False,
                 reference_dir=None,
                 mindim=360):
        self.input = input
        self.output = os.path.join(output, 'DeepRemaster')
        self.colorization = colorization
        self.reference_dir = reference_dir
        self.mindim = mindim

        if weight_path is None:
            weight_path = get_path_from_url(DeepRemaster_weight_url, cur_path)

        state_dict, _ = paddle.load(weight_path)

        self.modelR = NetworkR()
        self.modelR.load_dict(state_dict['modelR'])
        self.modelR.eval()
        if colorization:
            self.modelC = NetworkC()
            self.modelC.load_dict(state_dict['modelC'])
            self.modelC.eval()

    def run(self):
        outputdir = self.output
        outputdir_in = os.path.join(outputdir, 'input/')
        os.makedirs(outputdir_in, exist_ok=True)
        outputdir_out = os.path.join(outputdir, 'output/')
        os.makedirs(outputdir_out, exist_ok=True)

        # Prepare reference images
        if self.colorization:
            if self.reference_dir is not None:
                import glob
                ext_list = ['png', 'jpg', 'bmp']
                reference_files = []
                for ext in ext_list:
                    reference_files += glob.glob(self.reference_dir + '/*.' +
                                                 ext,
                                                 recursive=True)
                aspect_mean = 0
                minedge_dim = 256
                refs = []
                for v in reference_files:
                    refimg = Image.open(v).convert('RGB')
                    w, h = refimg.size
                    aspect_mean += w / h
                    refs.append(refimg)
                aspect_mean /= len(reference_files)
                target_w = int(256 * aspect_mean) if aspect_mean > 1 else 256
                target_h = 256 if aspect_mean >= 1 else int(256 / aspect_mean)

                refimgs = []
                for i, v in enumerate(refs):
                    refimg = utils.addMergin(v,
                                             target_w=target_w,
                                             target_h=target_h)
                    refimg = np.array(refimg).astype('float32').transpose(
                        2, 0, 1) / 255.0
                    refimgs.append(refimg)
                refimgs = paddle.to_tensor(np.array(refimgs).astype('float32'))

                refimgs = paddle.unsqueeze(refimgs, 0)

        # Load video
        cap = cv2.VideoCapture(self.input)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        v_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        minwh = min(v_w, v_h)
        scale = 1
        if minwh != self.mindim:
            scale = self.mindim / minwh

        t_w = round(v_w * scale / 16.) * 16
        t_h = round(v_h * scale / 16.) * 16
        fps = cap.get(cv2.CAP_PROP_FPS)
        pbar = tqdm(total=nframes)
        block = 5

        # Process
        with paddle.no_grad():
            it = 0
            while True:
                frame_pos = it * block
                if frame_pos >= nframes:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                if block >= nframes - frame_pos:
                    proc_g = nframes - frame_pos
                else:
                    proc_g = block

                input = None
                gtC = None
                for i in range(proc_g):
                    index = frame_pos + i
                    _, frame = cap.read()
                    frame = cv2.resize(frame, (t_w, t_h))
                    nchannels = frame.shape[2]
                    if nchannels == 1 or self.colorization:
                        frame_l = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        cv2.imwrite(outputdir_in + '%07d.png' % index, frame_l)
                        frame_l = paddle.to_tensor(frame_l.astype('float32'))
                        frame_l = paddle.reshape(
                            frame_l, [frame_l.shape[0], frame_l.shape[1], 1])
                        frame_l = paddle.transpose(frame_l, [2, 0, 1])
                        frame_l /= 255.

                        frame_l = paddle.reshape(frame_l, [
                            1, frame_l.shape[0], 1, frame_l.shape[1],
                            frame_l.shape[2]
                        ])
                    elif nchannels == 3:
                        cv2.imwrite(outputdir_in + '%07d.png' % index, frame)
                        frame = frame[:, :, ::-1]  ## BGR -> RGB
                        frame_l, frame_ab = utils.convertRGB2LABTensor(frame)
                        frame_l = frame_l.transpose([2, 0, 1])
                        frame_ab = frame_ab.transpose([2, 0, 1])
                        frame_l = frame_l.reshape([
                            1, frame_l.shape[0], 1, frame_l.shape[1],
                            frame_l.shape[2]
                        ])
                        frame_ab = frame_ab.reshape([
                            1, frame_ab.shape[0], 1, frame_ab.shape[1],
                            frame_ab.shape[2]
                        ])

                    if input is not None:
                        paddle.concat((input, frame_l), 2)

                    input = frame_l if i == 0 else paddle.concat(
                        (input, frame_l), 2)
                    if nchannels == 3 and not self.colorization:
                        gtC = frame_ab if i == 0 else paddle.concat(
                            (gtC, frame_ab), 2)

                input = paddle.to_tensor(input)

                output_l = self.modelR(input)  # [B, C, T, H, W]

                # Save restoration output without colorization when using the option [--disable_colorization]
                if not self.colorization:
                    for i in range(proc_g):
                        index = frame_pos + i
                        if nchannels == 3:
                            out_l = output_l.detach()[0, :, i]
                            out_ab = gtC[0, :, i]

                            out = paddle.concat(
                                (out_l, out_ab),
                                axis=0).detach().numpy().transpose((1, 2, 0))
                            out = Image.fromarray(
                                np.uint8(utils.convertLAB2RGB(out) * 255))
                            out.save(outputdir_out + '%07d.png' % (index))
                        else:
                            raise ValueError('channels of imag3 must be 3!')

                # Perform colorization
                else:
                    if self.reference_dir is None:
                        output_ab = self.modelC(output_l)
                    else:
                        output_ab = self.modelC(output_l, refimgs)
                    output_l = output_l.detach()
                    output_ab = output_ab.detach()

                    for i in range(proc_g):
                        index = frame_pos + i
                        out_l = output_l[0, :, i, :, :]
                        out_c = output_ab[0, :, i, :, :]
                        output = paddle.concat(
                            (out_l, out_c), axis=0).numpy().transpose((1, 2, 0))
                        output = Image.fromarray(
                            np.uint8(utils.convertLAB2RGB(output) * 255))
                        output.save(outputdir_out + '%07d.png' % index)

                it = it + 1
                pbar.update(proc_g)

            # Save result videos
            outfile = os.path.join(outputdir,
                                   self.input.split('/')[-1].split('.')[0])
            cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_in.mp4' % (
                fps, outputdir_in, fps, outfile)
            subprocess.call(cmd, shell=True)
            cmd = 'ffmpeg -y -r %d -i %s%%07d.png -vcodec libx264 -pix_fmt yuv420p -r %d %s_out.mp4' % (
                fps, outputdir_out, fps, outfile)
            subprocess.call(cmd, shell=True)
            cmd = 'ffmpeg -y -i %s_in.mp4 -vf "[in] pad=2.01*iw:ih [left];movie=%s_out.mp4[right];[left][right] overlay=main_w/2:0,scale=2*iw/2:2*ih/2[out]" %s_comp.mp4' % (
                outfile, outfile, outfile)
            subprocess.call(cmd, shell=True)

        cap.release()
        pbar.close()
        return outputdir_out, '%s_out.mp4' % outfile


if __name__ == "__main__":
    args = parser.parse_args()
    paddle.disable_static()
    predictor = DeepReasterPredictor(args.input,
                                     args.output,
                                     colorization=args.colorization,
                                     reference_dir=args.reference_dir,
                                     mindim=args.mindim)
    predictor.run()
