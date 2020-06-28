import random


class RandomCrop(object):

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def _get_params(self, img):
        h, w, _ = img.shape
        th, tw = self.output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self._get_params(img)
        cropped_img = img[i:i + h, j:j + w]
        return cropped_img


class Crop():
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

    def __call__(self, img):
        oh, ow, _ = img.shape
        x, y = self.pos
        th = tw = self.size
        if (ow > tw or oh > th):
            return img[y: y + th, x: x + tw]

        return img