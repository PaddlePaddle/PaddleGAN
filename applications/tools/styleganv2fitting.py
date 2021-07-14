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
from ppgan.apps import StyleGANv2FittingPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, help="path to source image")

    parser.add_argument("--need_align",
                        action="store_true",
                        help="whether to align input image")

    parser.add_argument("--start_lr",
                        type=float,
                        default=0.1,
                        help="learning rate at the begin of training")

    parser.add_argument("--final_lr",
                        type=float,
                        default=0.025,
                        help="learning rate at the end of training")

    parser.add_argument("--latent_level",
                        type=int,
                        nargs="+",
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help="indices of latent code for training")

    parser.add_argument("--step",
                        type=int,
                        default=100,
                        help="optimize iterations")

    parser.add_argument("--mse_weight",
                        type=float,
                        default=1,
                        help="weight of the mse loss")

    parser.add_argument("--pre_latent",
                        type=str,
                        default=None,
                        help="path to pre-prepared latent codes")

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

    predictor = StyleGANv2FittingPredictor(
        output_path=args.output_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        seed=None,
        size=args.size,
        style_dim=args.style_dim,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier)
    predictor.run(args.input_image,
                  need_align=args.need_align,
                  start_lr=args.start_lr,
                  final_lr=args.final_lr,
                  latent_level=args.latent_level,
                  step=args.step,
                  mse_weight=args.mse_weight,
                  pre_latent=args.pre_latent)
