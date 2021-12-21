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
