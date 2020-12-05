import paddle
import os
import sys
sys.path.insert(0, os.getcwd())
from ppgan.apps import AnimeGANPredictor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="path to source image")

    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model checkpoint path")

    parser.add_argument("--use_adjust_brightness",
                        action="store_false",
                        help="adjust brightness mode.")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = AnimeGANPredictor(args.output_path, args.weight_path,
                                  args.use_adjust_brightness)
    predictor.run(args.input_image)
