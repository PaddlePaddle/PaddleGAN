# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import numpy as np
from paddle_serving_client import Client
from paddle_serving_app.reader import *
import cv2
import os
import imageio

def get_img(pred):
    pred = pred.squeeze()
    pred = np.clip(pred, a_min=0., a_max=1.0)
    pred = pred * 255
    pred = pred.round()
    pred = pred.astype('uint8')
    pred = np.transpose(pred, (1, 2, 0))  # chw -> hwc
    return pred

preprocess = Sequential([
    BGR2RGB(), Resize(
        (320, 180)), Div(255.0), Transpose(
            (2, 0, 1))
])

client = Client()

client.load_client_config("serving_client/serving_client_conf.prototxt")
client.connect(['127.0.0.1:9393'])

frame_num = int(sys.argv[2])

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
success, frame = cap.read()
read_end = False
res_frames = []
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while success:
    frames = []
    for i in range(frame_num):
        if success: 
            frames.append(preprocess(frame))
            success, frame = cap.read()
        else:
            read_end = True
    if read_end: break

    frames = np.stack(frames, axis=0)
    fetch_map = client.predict(
        feed={
            "lqs": frames,
        },
        fetch=["stack_19.tmp_0"],
        batch=False)
    res_frames.extend([fetch_map["stack_19.tmp_0"][0][i] for i in range(frame_num)])

imageio.mimsave("output/output.mp4",
                        [get_img(frame) for frame in res_frames],
                        fps=fps)

