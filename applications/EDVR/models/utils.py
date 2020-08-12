#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import wget
import tarfile

__all__ = ['decompress', 'download', 'AttrDict']


def decompress(path):
    t = tarfile.open(path)
    t.extractall(path=os.path.split(path)[0])
    t.close()
    os.remove(path)


def download(url, path):
    weight_dir = os.path.split(path)[0]
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    path = path + ".tar.gz"
    wget.download(url, path)
    decompress(path)


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value
