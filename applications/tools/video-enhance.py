import sys
sys.path.append('.')

import argparse
import paddle

from DAIN.predict import VideoFrameInterp
from DeepRemaster.predict import DeepReasterPredictor
from DeOldify.predict import DeOldifyPredictor
from RealSR.predict import RealSRPredictor
from EDVR.predict import EDVRPredictor

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
# DAIN args
parser.add_argument('--time_step',
                    type=float,
                    default=0.5,
                    help='choose the time steps')
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
#process order support model name:[DAIN, DeepRemaster, DeOldify, RealSR, EDVR]
parser.add_argument('--proccess_order',
                    type=str,
                    default='none',
                    nargs='+',
                    help='Process order')

if __name__ == "__main__":
    args = parser.parse_args()

    orders = args.proccess_order
    temp_video_path = None

    for order in orders:
        if temp_video_path is None:
            temp_video_path = args.input
        if order == 'DAIN':
            predictor = VideoFrameInterp(args.time_step,
                                         args.DAIN_weight,
                                         temp_video_path,
                                         output_path=args.output)
            frames_path, temp_video_path = predictor.run()
        elif order == 'DeepRemaster':
            paddle.disable_static()
            predictor = DeepReasterPredictor(
                temp_video_path,
                args.output,
                weight_path=args.DeepRemaster_weight,
                colorization=args.colorization,
                reference_dir=args.reference_dir,
                mindim=args.mindim)
            frames_path, temp_video_path = predictor.run()
            paddle.enable_static()
        elif order == 'DeOldify':
            paddle.disable_static()
            predictor = DeOldifyPredictor(temp_video_path,
                                          args.output,
                                          weight_path=args.DeOldify_weight)
            frames_path, temp_video_path = predictor.run()
            paddle.enable_static()
        elif order == 'RealSR':
            paddle.disable_static()
            predictor = RealSRPredictor(temp_video_path,
                                        args.output,
                                        weight_path=args.RealSR_weight)
            frames_path, temp_video_path = predictor.run()
            paddle.enable_static()
        elif order == 'EDVR':
            predictor = EDVRPredictor(temp_video_path,
                                      args.output,
                                      weight_path=args.EDVR_weight)
            frames_path, temp_video_path = predictor.run()

        print('Model {} output frames path:'.format(order), frames_path)
        print('Model {} output video path:'.format(order), temp_video_path)
