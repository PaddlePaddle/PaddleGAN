#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import cv2
import numpy as np
from PIL import Image
import paddle


class BasePredictor(object):
    def __init__(self):
        pass

    def build_inference_model(self):
        if paddle.in_dynamic_mode():
            # todo self.model = build_model(self.cfg)
            pass
        else:
            place = paddle.get_device()
            self.exe = paddle.static.Executor(place)
            file_names = os.listdir(self.weight_path)
            for file_name in file_names:
                if file_name.find('model') > -1:
                    model_file = file_name
                elif file_name.find('param') > -1:
                    param_file = file_name

            self.program, self.feed_names, self.fetch_targets = paddle.static.load_inference_model(
                self.weight_path,
                executor=self.exe,
                model_filename=model_file,
                params_filename=param_file)

    def base_forward(self, inputs):
        if paddle.in_dynamic_mode():
            out = self.model(inputs)
        else:
            feed_dict = {}
            if isinstance(inputs, dict):
                feed_dict = inputs
            elif isinstance(inputs, (list, tuple)):
                for i, feed_name in enumerate(self.feed_names):
                    feed_dict[feed_name] = inputs[i]
            else:
                feed_dict[self.feed_names[0]] = inputs

            out = self.exe.run(self.program,
                               fetch_list=self.fetch_targets,
                               feed=feed_dict)

        return out

    def is_image(self, input):
        try:
            if isinstance(input, (np.ndarray, Image.Image)):
                return True
            elif isinstance(input, str):
                if not os.path.isfile(input):
                    raise ValueError('input must be a file')
                img = Image.open(input)
                _ = img.size
                return True
            else:
                return False
        except:
            return False

    def run(self):
        raise NotImplementedError
