#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys


def video2frames(video_path, outpath, **kargs):
    def _dict2str(kargs):
        cmd_str = ''
        for k, v in kargs.items():
            cmd_str += (' ' + str(k) + ' ' + str(v))
        return cmd_str

    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = os.path.basename(video_path).split('.')[0]
    out_full_path = os.path.join(outpath, vid_name)

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = os.path.join(out_full_path, '%08d.png')

    cmd = ffmpeg
    cmd = ffmpeg + [' -i ', video_path, ' -start_number ', ' 0 ', outformat]

    cmd = ''.join(cmd) + _dict2str(kargs)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(vid_name))

    sys.stdout.flush()
    return out_full_path


def frames2video(frame_path, video_path, r):
    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    cmd = ffmpeg + [
        ' -r ', r, ' -f ', ' image2 ', ' -i ', frame_path, ' -vcodec ',
        ' libx264 ', ' -pix_fmt ', ' yuv420p ', ' -crf ', ' 16 ', video_path
    ]
    cmd = ''.join(cmd)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))

    sys.stdout.flush()
