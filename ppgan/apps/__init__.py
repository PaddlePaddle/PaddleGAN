#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .dain_predictor import DAINPredictor
from .deepremaster_predictor import DeepRemasterPredictor
from .deoldify_predictor import DeOldifyPredictor
from .realsr_predictor import RealSRPredictor
from .edvr_predictor import EDVRPredictor
from .first_order_predictor import FirstOrderPredictor
from .face_parse_predictor import FaceParsePredictor
from .animegan_predictor import AnimeGANPredictor
from .midas_predictor import MiDaSPredictor
from .photo2cartoon_predictor import Photo2CartoonPredictor
from .styleganv2_predictor import StyleGANv2Predictor
from .styleganv2fitting_predictor import StyleGANv2FittingPredictor
from .styleganv2mixing_predictor import StyleGANv2MixingPredictor
from .styleganv2editing_predictor import StyleGANv2EditingPredictor
from .pixel2style2pixel_predictor import Pixel2Style2PixelPredictor
from .wav2lip_predictor import Wav2LipPredictor
from .mpr_predictor import MPRPredictor
from .lapstyle_predictor import LapStylePredictor
from .photopen_predictor import PhotoPenPredictor
from .recurrent_vsr_predictor import (PPMSVSRPredictor, BasicVSRPredictor, \
                                     BasiVSRPlusPlusPredictor, IconVSRPredictor, \
                                     PPMSVSRLargePredictor)
