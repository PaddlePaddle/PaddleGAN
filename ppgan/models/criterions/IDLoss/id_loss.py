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

import os
import paddle
from .model_irse import Backbone
from paddle.vision.transforms import Resize
from ..builder import CRITERIONS
from ppgan.utils.download import get_path_from_url

model_cfgs = {
    'model_urls':
    'https://paddlegan.bj.bcebos.com/models/model_ir_se50.pdparams',
}


@CRITERIONS.register()
class IDLoss(paddle.nn.Layer):

    def __init__(self, base_dir='./'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112,
                                num_layers=50,
                                drop_ratio=0.6,
                                mode='ir_se')

        facenet_weights_path = os.path.join(base_dir, 'data/gpen/weights',
                                            'model_ir_se50.pdparams')

        if not os.path.isfile(facenet_weights_path):
            facenet_weights_path = get_path_from_url(model_cfgs['model_urls'])

        self.facenet.load_dict(paddle.load(facenet_weights_path))

        self.face_pool = paddle.nn.AdaptiveAvgPool2D((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h == w
        ss = h // 256
        x = x[:, :, 35 * ss:-33 * ss, 32 * ss:-36 * ss]
        transform = Resize(size=(112, 112))

        for num in range(x.shape[0]):
            mid_feats = transform(x[num]).unsqueeze(0)
            if num == 0:
                x_feats = mid_feats
            else:
                x_feats = paddle.concat([x_feats, mid_feats], axis=0)

        x_feats = self.facenet(x_feats)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
