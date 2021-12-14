# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import sys
import argparse

import ppgan
from ppgan.utils.config import get_config
from ppgan.utils.setup import setup
from ppgan.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config-file',
                        metavar="FILE",
                        required=True,
                        help="config file path")
    parser.add_argument("--load",
                        type=str,
                        default=None,
                        required=True,
                        help="put the path to resuming file if needed")
    # config options
    parser.add_argument("-o",
                        "--opt",
                        nargs="+",
                        help="set configuration options")
    parser.add_argument("-s",
                        "--inputs_size",
                        type=str,
                        default=None,
                        required=True,
                        help="the inputs size")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--export_serving_model",
        default=False,
        type=bool,
        help="export serving model.",
    )
    args = parser.parse_args()
    return args


def main(args, cfg):
    inputs_size = [[int(size) for size in input_size.split(',')]
                   for input_size in args.inputs_size.split(';')]
    model = ppgan.models.builder.build_model(cfg.model)
    model.setup_train_mode(is_train=False)
    state_dicts = ppgan.utils.filesystem.load(args.load)
    for net_name, net in model.nets.items():
        if net_name in state_dicts:
            net.set_state_dict(state_dicts[net_name])
    model.export_model(cfg.export_model, args.output_dir, inputs_size, args.export_serving_model)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config(args.config_file, args.opt)
    main(args, cfg)
