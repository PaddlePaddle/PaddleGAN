#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import paddle
from ppgan.apps import StyleGANv2MixingPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent1",
                        type=str,
                        help="path to first image latent codes")

    parser.add_argument("--latent2",
                        type=str,
                        help="path to second image latent codes")

    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.5] * 18,
        help="different weights at each level of two latent codes")

    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model checkpoint path")

    parser.add_argument("--model_type",
                        type=str,
                        default=None,
                        help="type of model for loading pretrained model")

    parser.add_argument("--size",
                        type=int,
                        default=1024,
                        help="resolution of output image")

    parser.add_argument("--style_dim",
                        type=int,
                        default=512,
                        help="number of style dimension")

    parser.add_argument("--n_mlp",
                        type=int,
                        default=8,
                        help="number of mlp layer depth")

    parser.add_argument("--channel_multiplier",
                        type=int,
                        default=2,
                        help="number of channel multiplier")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = StyleGANv2MixingPredictor(
        output_path=args.output_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        seed=None,
        size=args.size,
        style_dim=args.style_dim,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier)
    predictor.run(args.latent1, args.latent2, args.weights)
