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
from .lesrcnn import LESRCNNGenerator
from .resnet_ugatit_p2c import ResnetUGATITP2CGenerator
from .generator_styleganv2 import StyleGANv2Generator
from .generator_pixel2style2pixel import Pixel2Style2Pixel
from .drn import DRNGenerator
from .generator_starganv2 import StarGANv2Generator, StarGANv2Style, StarGANv2Mapping, FAN
from .edvr import EDVRNet
from .generator_firstorder import FirstOrderGenerator
from .generater_lapstyle import DecoderNet, Encoder, RevisionNet
from .basicvsr import BasicVSRNet
from .mpr import MPRNet
from .iconvsr import IconVSR
from .gpen import GPEN
from .pan import PAN
from .generater_photopen import SPADEGenerator
from .basicvsr_plus_plus import BasicVSRPlusPlus
from .msvsr import MSVSR
