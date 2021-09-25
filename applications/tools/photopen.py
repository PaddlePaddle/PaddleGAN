import paddle
import os
import sys

sys.path.insert(0, os.getcwd())
from ppgan.apps import PhotoPenPredictor
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--semantic_label_path",
                        type=str,
                        default=None,
                        help="path to input semantic label")

    parser.add_argument("--output_path",
                        type=str,
                        default=None,
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model weight")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = PhotoPenPredictor(output_path=args.output_path,
                        weight_path=args.weight_path)
    predictor.run(semantic_label_path=args.semantic_label_path)
    