#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from compute_fid import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_path1',
                        type=str,
                        default='./real',
                        help='path of image data')
    parser.add_argument('--image_data_path2',
                        type=str,
                        default='./fake',
                        help='path of image data')
    parser.add_argument('--inference_model',
                        type=str,
                        default='./pretrained/params_inceptionV3',
                        help='path of inference_model.')
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='default use gpu.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='sample number in a batch for inference.')
    parser.add_argument(
        '--style',
        type=str,
        help='calculation style: stargan or default (gan-compression style)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path1 = args.image_data_path1
    path2 = args.image_data_path2
    paths = (path1, path2)
    inference_model_path = args.inference_model
    batch_size = args.batch_size

    fid_value = calculate_fid_given_paths(paths,
                                          inference_model_path,
                                          batch_size,
                                          args.use_gpu,
                                          2048,
                                          style=args.style)
    print('FID: ', fid_value)


if __name__ == "__main__":
    main()
