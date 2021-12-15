import paddle
import os
import sys

sys.path.insert(0, os.getcwd())
from ppgan.apps import MPRPredictor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="path to image")

    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model weight path")

    parser.add_argument(
        "--task",
        type=str,
        default='Deblurring',
        help="task can be chosen in 'Deblurring', 'Denoising', 'Deraining'")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = MPRPredictor(output_path=args.output_path,
                             task=args.task,
                             weight_path=args.weight_path)
    predictor.run(args.input_image)
