#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import argparse

import paddle
from ppgan.apps.wav2lip_predictor import Wav2LipPredictor

parser = argparse.ArgumentParser(
    description=
    'Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='Name of saved checkpoint to load weights from',
                    default=None)
parser.add_argument(
    '--audio',
    type=str,
    help='Filepath of video/audio file to use as raw audio source',
    required=True)
parser.add_argument('--face',
                    type=str,
                    help='Filepath of video/image that contains faces to use',
                    required=True)
parser.add_argument('--outfile',
                    type=str,
                    help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')

parser.add_argument(
    '--static',
    type=bool,
    help='If True, then use only first video frame for inference',
    default=False)
parser.add_argument(
    '--fps',
    type=float,
    help='Can be specified only if input is a static image (default: 25)',
    default=25.,
    required=False)

parser.add_argument(
    '--pads',
    nargs='+',
    type=int,
    default=[0, 10, 0, 0],
    help=
    'Padding (top, bottom, left, right). Please adjust to include chin at least'
)

parser.add_argument('--face_det_batch_size',
                    type=int,
                    help='Batch size for face detection',
                    default=16)
parser.add_argument('--wav2lip_batch_size',
                    type=int,
                    help='Batch size for Wav2Lip model(s)',
                    default=128)

parser.add_argument(
    '--resize_factor',
    default=1,
    type=int,
    help=
    'Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p'
)

parser.add_argument(
    '--crop',
    nargs='+',
    type=int,
    default=[0, -1, 0, -1],
    help=
    'Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width'
)

parser.add_argument(
    '--box',
    nargs='+',
    type=int,
    default=[-1, -1, -1, -1],
    help=
    'Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).'
)

parser.add_argument(
    '--rotate',
    default=False,
    action='store_true',
    help=
    'Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument(
    '--nosmooth',
    default=False,
    action='store_true',
    help='Prevent smoothing face detections over a short temporal window')
parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
parser.add_argument(
    "--face_detector",
    dest="face_detector",
    type=str,
    default='sfd',
    help="face detector to be used, can choose s3fd or blazeface")
parser.add_argument("--face_enhancement",
                    dest="face_enhancement",
                    action="store_true",
                    help="use face enhance for face")
parser.set_defaults(face_enhancement=False)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        paddle.set_device('cpu')

    predictor = Wav2LipPredictor(checkpoint_path=args.checkpoint_path,
                                 static=args.static,
                                 fps=args.fps,
                                 pads=args.pads,
                                 face_det_batch_size=args.face_det_batch_size,
                                 wav2lip_batch_size=args.wav2lip_batch_size,
                                 resize_factor=args.resize_factor,
                                 crop=args.crop,
                                 box=args.box,
                                 rotate=args.rotate,
                                 nosmooth=args.nosmooth,
                                 face_detector=args.face_detector,
                                 face_enhancement=args.face_enhancement)
    predictor.run(args.face, args.audio, args.outfile)
