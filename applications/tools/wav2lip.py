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

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cpu:
        paddle.set_device('cpu')

    predictor = Wav2LipPredictor(args)
    predictor.run()
