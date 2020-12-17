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

from .unpaired_dataset import UnpairedDataset
from .single_dataset import SingleDataset
from .paired_dataset import PairedDataset
from .base_sr_dataset import SRDataset
from .makeup_dataset import MakeupDataset
from .common_vision_dataset import CommonVisionDataset
from .animeganv2_dataset import AnimeGANV2Dataset
from .wav2lip_dataset import Wav2LipDataset
