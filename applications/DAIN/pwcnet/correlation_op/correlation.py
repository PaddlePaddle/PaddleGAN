# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import os
file_dir = os.path.dirname(os.path.abspath(__file__))
fluid.load_op_library(os.path.join(file_dir, 'correlation_lib.so'))

from paddle.fluid.layer_helper import LayerHelper


def correlation(input1,
                input2,
                pad_size,
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                corr_type_multiply=1):
    helper = LayerHelper("correlation", **locals())
    output = helper.create_variable_for_type_inference(dtype=input1.dtype)
    helper.append_op(type="correlation",
                     inputs={
                         "Input1": input1,
                         "Input2": input2
                     },
                     attrs={
                         "pad_size": pad_size,
                         "kernel_size": kernel_size,
                         "max_displacement": max_displacement,
                         "stride1": stride1,
                         "stride2": stride2,
                         "corr_type_multiply": corr_type_multiply
                     },
                     outputs={"Output": output})
    return output
