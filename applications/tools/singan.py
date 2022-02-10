#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from ppgan.apps import SinGANPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        type=str,
                        default='output_dir',
                        help="path to output image dir")

    parser.add_argument("--weight_path",
                        type=str,
                        default=None,
                        help="path to model checkpoint path")

    parser.add_argument("--pretrained_model",
                        type=str,
                        default=None,
                        help="a pretianed model, only trees, stone, mountains, birds, and lightning are implemented.")

    parser.add_argument("--mode",
                        type=str,
                        default="random_sample",
                        help="type of model for loading pretrained model")

    parser.add_argument("--generate_start_scale",
                        type=int,
                        default=0,
                        help="sample random seed for model's image generation")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="sample random seed for model's image generation")

    parser.add_argument("--scale_h",
                        type=float,
                        default=1.0,
                        help="horizontal scale")

    parser.add_argument("--scale_v",
                        type=float,
                        default=1.0,
                        help="vertical scale")

    parser.add_argument("--ref_image",
                        type=str,
                        default=None,
                        help="reference image for harmonization, editing and paint2image")

    parser.add_argument("--mask_image",
                        type=str,
                        default=None,
                        help="mask image for harmonization and editing")

    parser.add_argument("--sr_factor",
                        type=float,
                        default=4.0,
                        help="scale for super resolution")

    parser.add_argument("--animation_alpha",
                        type=float,
                        default=0.9,
                        help="a parameter determines how close the frames of the sequence remain to the training image")

    parser.add_argument("--animation_beta",
                        type=float,
                        default=0.9,
                        help="a parameter controls the smoothness and rate of change in the generated clip")

    parser.add_argument("--animation_frames",
                        type=int,
                        default=20,
                        help="frame number of output animation when mode is animation")

    parser.add_argument("--animation_duration",
                        type=float,
                        default=0.1,
                        help="duration of each frame in animation")

    parser.add_argument("--n_row",
                        type=int,
                        default=5,
                        help="row number of output image grid")

    parser.add_argument("--n_col",
                        type=int,
                        default=3,
                        help="column number of output image grid")

    parser.add_argument("--cpu",
                        dest="cpu",
                        action="store_true",
                        help="cpu mode.")

    args = parser.parse_args()

    if args.cpu:
        paddle.set_device('cpu')

    predictor = SinGANPredictor(args.output_path,
                                args.weight_path,
                                args.pretrained_model,
                                args.seed)
    predictor.run(args.mode,
                  args.generate_start_scale,
                  args.scale_h,
                  args.scale_v,
                  args.ref_image,
                  args.mask_image,
                  args.sr_factor,
                  args.animation_alpha,
                  args.animation_beta,
                  args.animation_frames,
                  args.animation_duration,
                  args.n_row,
                  args.n_col)
