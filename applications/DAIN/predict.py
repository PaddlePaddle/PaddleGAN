import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)

import time
import glob
import numpy as np
from imageio import imread, imsave
import cv2

import paddle.fluid as fluid
from paddle.incubate.hapi.download import get_path_from_url

import networks
from util import *
from my_args import parser

DAIN_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/DAIN_weight.tar'

def infer_engine(model_dir,
                 run_mode='fluid',
                 batch_size=1,
                 use_gpu=False,
                 min_subgraph_size=3):
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    precision_map = {
        'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
        'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
    }
    config = fluid.core.AnalysisConfig(os.path.join(model_dir, 'model'),
                                       os.path.join(model_dir, 'params'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(workspace_size=1 << 10,
                                      max_batch_size=batch_size,
                                      min_subgraph_size=min_subgraph_size,
                                      precision_mode=precision_map[run_mode],
                                      use_static=False,
                                      use_calib_mode=False)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


def executor(model_dir, use_gpu=False):
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        dirname=model_dir,
        executor=exe,
        model_filename='model',
        params_filename='params')
    return exe, program, fetch_targets


class VideoFrameInterp(object):
    def __init__(self,
                 time_step,
                 model_path,
                 video_path,
                 use_gpu=True,
                 key_frame_thread=0.,
                 output_path='output'):
        self.video_path = video_path
        self.output_path = os.path.join(output_path, 'DAIN')
        if model_path is None:
            model_path = get_path_from_url(DAIN_WEIGHT_URL, cur_path)

        self.model_path = model_path
        self.time_step = time_step
        self.key_frame_thread = key_frame_thread

        self.exe, self.program, self.fetch_targets = executor(model_path,
                                                              use_gpu=use_gpu)


    def run(self):
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

        if self.video_path.endswith('.mp4'):
            videos = [self.video_path]
        else:
            videos = sorted(glob.glob(os.path.join(self.video_path, '*.mp4')))

        for cnt, vid in enumerate(videos):
            print("Interpolating video:", vid)
            cap = cv2.VideoCapture(vid)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("Old fps (frame rate): ", fps)

            times_interp = int(1.0 / timestep)
            r2 = str(int(fps) * times_interp)
            print("New fps (frame rate): ", r2)

            out_path = dump_frames_ffmpeg(vid, frame_path_input)

            vidname = vid.split('/')[-1].split('.')[0]

            tot_timer = AverageMeter()
            proc_timer = AverageMeter()
            end = time.time()

            frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

            img = imread(frames[0])

            int_width = img.shape[1]
            int_height = img.shape[0]
            channel = img.shape[2]
            if not channel == 3:
                continue

            if int_width != ((int_width >> 7) << 7):
                int_width_pad = (
                    ((int_width >> 7) + 1) << 7)  # more than necessary
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
            print('processing {} frames, from video: {}'.format(frame_num, vid))

            if not os.path.exists(os.path.join(frame_path_interpolated,
                                               vidname)):
                os.makedirs(os.path.join(frame_path_interpolated, vidname))
            if not os.path.exists(os.path.join(frame_path_combined, vidname)):
                os.makedirs(os.path.join(frame_path_combined, vidname))

            for i in range(frame_num - 1):
                print(frames[i])
                first = frames[i]
                second = frames[i + 1]

                img_first = imread(first)
                img_second = imread(second)
                '''--------------Frame change test------------------------'''
                img_first_gray = np.dot(img_first[..., :3],
                                        [0.299, 0.587, 0.114])
                img_second_gray = np.dot(img_second[..., :3],
                                         [0.299, 0.587, 0.114])

                img_first_gray = img_first_gray.flatten(order='C')
                img_second_gray = img_second_gray.flatten(order='C')
                corr = np.corrcoef(img_first_gray, img_second_gray)[0, 1]
                key_frame = False
                if corr < self.key_frame_thread:
                    key_frame = True
                '''-------------------------------------------------------'''

                X0 = img_first.astype('float32').transpose((2, 0, 1)) / 255
                X1 = img_second.astype('float32').transpose((2, 0, 1)) / 255

                if key_frame:
                    y_ = [
                        np.transpose(255.0 * X0.clip(0, 1.0), (1, 2, 0))
                        for i in range(num_frames)
                    ]
                else:
                    assert (X0.shape[1] == X1.shape[1])
                    assert (X0.shape[2] == X1.shape[2])

                    print("size before padding ", X0.shape)
                    X0 = np.pad(X0, ((0,0), (padding_top, padding_bottom), \
                        (padding_left, padding_right)), mode='edge')
                    X1 = np.pad(X1, ((0,0), (padding_top, padding_bottom), \
                        (padding_left, padding_right)), mode='edge')
                    print("size after padding ", X0.shape)

                    X0 = np.expand_dims(X0, axis=0)
                    X1 = np.expand_dims(X1, axis=0)

                    X0 = np.expand_dims(X0, axis=0)
                    X1 = np.expand_dims(X1, axis=0)

                    X = np.concatenate((X0, X1), axis=0)

                    proc_end = time.time()
                    o = self.exe.run(self.program,
                                     fetch_list=self.fetch_targets,
                                     feed={"image": X})

                    y_ = o[0]

                    proc_timer.update(time.time() - proc_end)
                    tot_timer.update(time.time() - end)
                    end = time.time()
                    print("*********** current image process time \t " +
                          str(time.time() - proc_end) + "s *********")

                    y_ = [
                        np.transpose(
                            255.0 * item.clip(
                                0, 1.0)[0, :,
                                        padding_top:padding_top + int_height,
                                        padding_left:padding_left + int_width],
                            (1, 2, 0)) for item in y_
                    ]
                    time_offsets = [
                        kk * timestep for kk in range(1, 1 + num_frames, 1)
                    ]

                    count = 1
                    for item, time_offset in zip(y_, time_offsets):
                        out_dir = os.path.join(
                            frame_path_interpolated, vidname,
                            "{:0>6d}_{:0>4d}.png".format(i, count))
                        count = count + 1
                        imsave(out_dir, np.round(item).astype(np.uint8))

            num_frames = int(1.0 / timestep) - 1

            input_dir = os.path.join(frame_path_input, vidname)
            interpolated_dir = os.path.join(frame_path_interpolated, vidname)
            combined_dir = os.path.join(frame_path_combined, vidname)
            combine_frames(input_dir, interpolated_dir, combined_dir,
                           num_frames)

            frame_pattern_combined = os.path.join(frame_path_combined, vidname,
                                                  '%08d.png')
            video_pattern_output = os.path.join(video_path_output,
                                                vidname + '.mp4')
            if os.path.exists(video_pattern_output):
                os.remove(video_pattern_output)
            frames_to_video_ffmpeg(frame_pattern_combined, video_pattern_output,
                                   r2)
            
        return frame_pattern_combined, video_pattern_output


if __name__ == '__main__':
    args = parser.parse_args()
    predictor = VideoFrameInterp(args.time_step, args.saved_model,
                                 args.video_path, args.output_path)
    predictor.run()
