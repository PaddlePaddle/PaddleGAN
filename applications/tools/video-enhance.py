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

import argparse
import paddle

from ppgan.apps import DAINPredictor
from ppgan.apps import DeepRemasterPredictor
from ppgan.apps import DeOldifyPredictor
from ppgan.apps import RealSRPredictor
from ppgan.apps import EDVRPredictor
from ppgan.apps import PPMSVSRPredictor, BasicVSRPredictor, \
                       BasiVSRPlusPlusPredictor, IconVSRPredictor, \
                       PPMSVSRLargePredictor

parser = argparse.ArgumentParser(description='Fix video')
parser.add_argument('--input', type=str, default=None, help='Input video')
parser.add_argument('--output', type=str, default='output', help='output dir')
parser.add_argument('--DAIN_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--DeepRemaster_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--DeOldify_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--RealSR_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--EDVR_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--PPMSVSR_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--PPMSVSRLarge_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--BasicVSR_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--IconVSR_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
parser.add_argument('--BasiVSRPlusPlus_weight',
                    type=str,
                    default=None,
                    help='Path to model weight')
# DAIN args
parser.add_argument('--time_step',
                    type=float,
                    default=0.5,
                    help='choose the time steps')
parser.add_argument('--remove_duplicates',
                    action='store_true',
                    default=False,
                    help='whether to remove duplicated frames')
# DeepRemaster args
parser.add_argument('--reference_dir',
                    type=str,
                    default=None,
                    help='Path to the reference image directory')
parser.add_argument('--colorization',
                    action='store_true',
                    default=False,
                    help='Remaster with colorization')
parser.add_argument('--mindim',
                    type=int,
                    default=360,
                    help='Length of minimum image edges')
# DeOldify args
parser.add_argument('--artistic',
                    action='store_true',
                    default=False,
                    help='whether to use artistic DeOldify Model')
parser.add_argument('--render_factor',
                    type=int,
                    default=32,
                    help='model inputsize=render_factor*16')
#vsr input number frames
parser.add_argument('--num_frames',
                    type=int,
                    default=10,
                    help='num frames for recurrent vsr model')
#process order support model name:[DAIN, DeepRemaster, DeOldify, RealSR, EDVR]
parser.add_argument('--process_order',
                    type=str,
                    default='none',
                    nargs='+',
                    help='Process order')

if __name__ == "__main__":
    args = parser.parse_args()

    orders = args.process_order
    temp_video_path = None

    for order in orders:
        print('Model {} process start..'.format(order))
        if temp_video_path is None:
            temp_video_path = args.input
        if order == 'DAIN':
            paddle.enable_static()
            predictor = DAINPredictor(args.output,
                                      weight_path=args.DAIN_weight,
                                      time_step=args.time_step,
                                      remove_duplicates=args.remove_duplicates)
            frames_path, temp_video_path = predictor.run(temp_video_path)
            paddle.disable_static()
        elif order == 'DeepRemaster':
            predictor = DeepRemasterPredictor(
                args.output,
                weight_path=args.DeepRemaster_weight,
                colorization=args.colorization,
                reference_dir=args.reference_dir,
                mindim=args.mindim)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'DeOldify':
            predictor = DeOldifyPredictor(args.output,
                                          weight_path=args.DeOldify_weight,
                                          artistic=args.artistic,
                                          render_factor=args.render_factor)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'RealSR':
            predictor = RealSRPredictor(args.output,
                                        weight_path=args.RealSR_weight)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'EDVR':
            predictor = EDVRPredictor(args.output, weight_path=args.EDVR_weight)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'PPMSVSR':
            predictor = PPMSVSRPredictor(args.output,
                                         weight_path=args.PPMSVSR_weight,
                                         num_frames=args.num_frames)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'PPMSVSRLarge':
            predictor = PPMSVSRLargePredictor(
                args.output,
                weight_path=args.PPMSVSRLarge_weight,
                num_frames=args.num_frames)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'BasicVSR':
            predictor = BasicVSRPredictor(args.output,
                                          weight_path=args.BasicVSR_weight,
                                          num_frames=args.num_frames)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'IconVSR':
            predictor = IconVSRPredictor(args.output,
                                         weight_path=args.IconVSR_weight,
                                         num_frames=args.num_frames)
            frames_path, temp_video_path = predictor.run(temp_video_path)
        elif order == 'BasiVSRPlusPlus':
            predictor = BasiVSRPlusPlusPredictor(
                args.output,
                weight_path=args.BasiVSRPlusPlus_weight,
                num_frames=args.num_frames)
            frames_path, temp_video_path = predictor.run(temp_video_path)

        print('Model {} output frames path:'.format(order), frames_path)
        print('Model {} output video path:'.format(order), temp_video_path)
        print('Model {} process done!'.format(order))
