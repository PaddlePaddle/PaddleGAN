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

import time


class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._cnt = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._cnt += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._cnt == 0:
            return 0
        return self._total_time / float(self._cnt)

    def get_ips_average(self):
        if not self._total_samples or self._cnt == 0:
            return 0
        return float(self._total_samples) / self._total_time
