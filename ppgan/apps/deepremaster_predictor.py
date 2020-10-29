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
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage import color

import paddle
from ppgan.models.generators.remaster import NetworkR, NetworkC
from ppgan.utils.download import get_path_from_url
from .base_predictor import BasePredictor

DEEPREMASTER_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/deep_remaster.pdparams'


def convertLAB2RGB(lab):
    lab[:, :, 0:1] = lab[:, :, 0:1] * 100  # [0, 1] -> [0, 100]
    lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100,
                             100)  # [0, 1] -> [-128, 128]
    rgb = color.lab2rgb(lab.astype(np.float64))
    return rgb


def convertRGB2LABTensor(rgb):
    lab = color.rgb2lab(
        np.asarray(rgb))  # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
    ab = np.clip(lab[:, :, 1:3] + 128, 0, 255)  # AB --> [0, 255]
    ab = paddle.to_tensor(ab.astype('float32')) / 255.
    L = lab[:, :, 0] * 2.55  # L --> [0, 255]
    L = Image.fromarray(np.uint8(L))

    L = paddle.to_tensor(np.array(L).astype('float32')[..., np.newaxis] / 255.0)
    return L, ab


def addMergin(img, target_w, target_h, background_color=(0, 0, 0)):
    width, height = img.size
    if width == target_w and height == target_h:
        return img
    scale = max(target_w, target_h) / max(width, height)
    width = int(width * scale / 16.) * 16
    height = int(height * scale / 16.) * 16

    img = img.resize((width, height), Image.BICUBIC)
    xp = (target_w - width) // 2
    yp = (target_h - height) // 2
    result = Image.new(img.mode, (target_w, target_h), background_color)
    result.paste(img, (xp, yp))
    return result


class DeepRemasterPredictor(BasePredictor):
    def __init__(self,
                 output='output',
                 weight_path=None,
                 colorization=False,
                 reference_dir=None,
                 mindim=360):
        self.output = os.path.join(output, 'DeepRemaster')
        self.colorization = colorization
        self.reference_dir = reference_dir
        self.mindim = mindim

        if weight_path is None:
            weight_path = get_path_from_url(DEEPREMASTER_WEIGHT_URL)

        self.weight_path = weight_path

        state_dict = paddle.load(weight_path)

        self.modelR = NetworkR()
        self.modelR.load_dict(state_dict['modelR'])
        self.modelR.eval()
        if colorization:
            self.modelC = NetworkC()
            self.modelC.load_dict(state_dict['modelC'])
            self.modelC.eval()

    def run(self, video_path):
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
                    refimg = addMergin(v, target_w=target_w, target_h=target_h)
                    refimg = np.array(refimg).astype('float32').transpose(
                        2, 0, 1) / 255.0
                    refimgs.append(refimg)
                refimgs = paddle.to_tensor(np.array(refimgs).astype('float32'))

                refimgs = paddle.unsqueeze(refimgs, 0)

        # Load video
        cap = cv2.VideoCapture(video_path)
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
                        frame_l, frame_ab = convertRGB2LABTensor(frame)
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
                                np.uint8(convertLAB2RGB(out) * 255))
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
                            np.uint8(convertLAB2RGB(output) * 255))
                        output.save(outputdir_out + '%07d.png' % index)

                it = it + 1
                pbar.update(proc_g)

            # Save result videos
            outfile = os.path.join(outputdir,
                                   video_path.split('/')[-1].split('.')[0])
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
