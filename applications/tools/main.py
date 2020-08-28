import sys
sys.path.append('.')

import argparse
import paddle

from DAIN.predict import VideoFrameInterp
from DeOldify.predict import DeOldifyPredictor
from EDVR.predict import EDVRPredictor

parser = argparse.ArgumentParser(description='Fix video')
parser.add_argument('--input',   type=str,   default=None, help='Input video')
parser.add_argument('--output',   type=str,   default='output', help='output dir')
parser.add_argument('--DAIN_weight',  type=str, default=None, help='Path to the reference image directory')
parser.add_argument('--DeOldify_weight',  type=str, default=None, help='Path to the reference image directory')
parser.add_argument('--EDVR_weight',  type=str, default=None, help='Path to the reference image directory')
# DAIN args
parser.add_argument('--time_step', type=float, default=0.5, help='choose the time steps')
parser.add_argument('--proccess_order',  type=str, default='none', nargs='+', help='Process order')


if __name__ == "__main__":
    args = parser.parse_args()
    print('args...', args)
    orders = args.proccess_order
    temp_video_path = None

    for order in orders:
        if temp_video_path is None:
            temp_video_path = args.input
        if order == 'DAIN':
            predictor = VideoFrameInterp(args.time_step, args.DAIN_weight,
                                        temp_video_path, output_path=args.output)
            frames_path, temp_video_path = predictor.run()
        elif order == 'DeOldify':
            print('frames:', frames_path)
            print('video_path:', temp_video_path)
            
            paddle.disable_static()
            predictor = DeOldifyPredictor(temp_video_path, args.output, weight_path=args.DeOldify_weight)
            frames_path, temp_video_path = predictor.run()
            print('frames:', frames_path)
            print('video_path:', temp_video_path)
            paddle.enable_static()
        elif order == 'EDVR':
            predictor = EDVRPredictor(temp_video_path, args.output, weight_path=args.EDVR_weight)
            frames_path, temp_video_path = predictor.run()
            print('frames:', frames_path)
            print('video_path:', temp_video_path)

