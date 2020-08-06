import os, sys
import math
import random
import time
import glob
import shutil
import numpy as np
from imageio import imread, imsave
import cv2

import paddle.fluid as fluid

import networks
from util import *
from my_args import args

if __name__ == '__main__':

    DO_MiddleBurryOther = True

    video_path = args.video_path
    output_path = args.output_path
    frame_path_input = os.path.join(output_path, 'frames-input')
    frame_path_interpolated = os.path.join(output_path, 'frames-interpolated')
    frame_path_combined = os.path.join(output_path, 'frames-combined')
    video_path_input = os.path.join(output_path, 'videos-input')
    video_path_output = os.path.join(output_path, 'videos-output')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(frame_path_input):
        os.makedirs(frame_path_input)
    if not os.path.exists(frame_path_interpolated):
        os.makedirs(frame_path_interpolated)
    if not os.path.exists(frame_path_combined):
        os.makedirs(frame_path_combined)
    if not os.path.exists(video_path_input):
        os.makedirs(video_path_input)
    if not os.path.exists(video_path_output):
        os.makedirs(video_path_output)

    args.KEY_FRAME_THREAD = 0.
    saved_model = args.saved_model

    timestep = args.time_step
    num_frames = int(1.0 / timestep) - 1

    image = fluid.data(name='image',
                       shape=[2, 1, args.channels, -1, -1],
                       dtype='float32')
    DAIN = networks.__dict__["DAIN_slowmotion"](channel=args.channels,
                                                filter_size=args.filter_size,
                                                timestep=args.time_step,
                                                training=False)
    out = DAIN(image)
    out = out[0][1]

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    fetch_list = [out.name]

    inference_program = fluid.default_main_program().clone(for_test=True)
    inference_program = fluid.io.load_persistables(exe, saved_model,
                                                   inference_program)

    if not DO_MiddleBurryOther:
        sys.exit()

    if video_path.endswith('.mp4'):
        videos = [video_path]
    else:
        videos = sorted(glob.glob(os.path.join(video_path, '*.mp4')))
    for cnt, vid in enumerate(videos):
        print("Interpolating video:", vid)
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Old fps (frame rate): ", fps)

        timestep = args.time_step
        times_interp = int(1.0 / timestep)
        r2 = str(int(fps) * times_interp)

        print("New fps (frame rate): ", r2)

        # set start and end of video
        #ss = 0
        #t = 10
        #ss = time.strftime('%H:%M:%S', time.gmtime(ss))
        #t = time.strftime('%H:%M:%S', time.gmtime(t))
        #print(r, ss, t)
        r = None
        ss = None
        t = None

        out_path = dump_frames_ffmpeg(vid, frame_path_input, r, ss, t)

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
        print(os.path.join(frame_path_input, vidname, '*.png'))
        print('processing {} frames, from video: {}'.format(frame_num, vid))

        if not os.path.exists(os.path.join(frame_path_interpolated, vidname)):
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
            img_first_gray = np.dot(img_first[..., :3], [0.299, 0.587, 0.114])
            img_second_gray = np.dot(img_second[..., :3], [0.299, 0.587, 0.114])

            img_first_gray = img_first_gray.flatten(order='C')
            img_second_gray = img_second_gray.flatten(order='C')
            corr = np.corrcoef(img_first_gray, img_second_gray)[0, 1]
            key_frame = False
            if corr < args.KEY_FRAME_THREAD:
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
                o = exe.run(inference_program,
                            fetch_list=fetch_list,
                            feed={"image": X})
                y_ = o[0]

                proc_timer.update(time.time() - proc_end)
                tot_timer.update(time.time() - end)
                end = time.time()
                print("*******current image process time \t " +
                      str(time.time() - proc_end) + "s ******")

                y_ = [
                    np.transpose(
                        255.0 * item.clip(
                            0, 1.0)[0, :, padding_top:padding_top + int_height,
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
                        "{:0>4d}_{:0>4d}.png".format(i, count))
                    count = count + 1
                    imsave(out_dir, np.round(item).astype(np.uint8))

        timestep = args.time_step
        num_frames = int(1.0 / timestep) - 1

        input_dir = os.path.join(frame_path_input, vidname)
        interpolated_dir = os.path.join(frame_path_interpolated, vidname)
        combined_dir = os.path.join(frame_path_combined, vidname)
        combine_frames(input_dir, interpolated_dir, combined_dir, num_frames)

        frame_pattern_combined = os.path.join(frame_path_combined, vidname,
                                              '%08d.png')
        video_pattern_output = os.path.join(video_path_output, vidname + '.mp4')
        if os.path.exists(video_pattern_output):
            os.remove(video_pattern_output)
        frames_to_video_ffmpeg(frame_pattern_combined, video_pattern_output, r2)
