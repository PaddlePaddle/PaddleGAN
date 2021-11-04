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

import random
import paddle


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.

    Args:
        pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
    """
    def __init__(self, pool_size, prob=0.5):
        self.pool_size = pool_size
        self.prob = prob

        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Args:
            images (paddle.Tensor): the latest generated images from the generator

        Returns images from the buffer.
        """
        # if the buffer size is 0, do nothing
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = paddle.unsqueeze(image, 0)
            # if the buffer is not full; keep inserting current images to the buffer
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                if p > self.prob:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        # collect all the images and return
        return_images = paddle.concat(return_images, 0)
        return return_images
