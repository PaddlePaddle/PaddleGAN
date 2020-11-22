import argparse

import paddle
from ppgan.apps.face_parse_predictor import FaceParsePredictor

parser = argparse.ArgumentParser()
parser.add_argument("--input_image", type=str, help="path to source image")

parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = FaceParsePredictor()
    predictor.run(args.input_image)
