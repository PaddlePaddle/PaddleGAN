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

from .resnet import ResnetGenerator
from .unet import UnetGenerator
from .rrdb_net import RRDBNet
from .makeup import GeneratorPSGANAttention
from .deep_conv import DeepConvGenerator, ConditionalDeepConvGenerator
from .resnet_ugatit import ResnetUGATITGenerator
from .dcgenerator import DCGenerator
from .generater_animegan import AnimeGenerator, AnimeGeneratorLite
from .wav2lip import Wav2Lip
from .resnet_ugatit_p2c import ResnetUGATITP2CGenerator
