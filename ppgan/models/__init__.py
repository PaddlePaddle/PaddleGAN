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

from .base_model import BaseModel
from .gan_model import GANModel
from .cycle_gan_model import CycleGANModel
from .pix2pix_model import Pix2PixModel
from .sr_model import BaseSRModel
from .makeup_model import MakeupModel
from .esrgan_model import ESRGAN
from .ugatit_model import UGATITModel
from .dc_gan_model import DCGANModel
from .drn_model import DRN
from .animeganv2_model import AnimeGANV2Model, AnimeGANV2PreTrainModel
from .styleganv2_model import StyleGAN2Model
from .wav2lip_model import Wav2LipModel
from .wav2lip_hq_model import Wav2LipModelHq
from .starganv2_model import StarGANv2Model
from .edvr_model import EDVRModel
from .firstorder_model import FirstOrderModel
from .lapstyle_model import LapStyleDraModel, LapStyleRevFirstModel, LapStyleRevSecondModel
from .basicvsr_model import BasicVSRModel
from .mpr_model import MPRModel
from .photopen_model import PhotoPenModel
from .msvsr_model import MultiStageVSRModel
