import numpy as np
import cv2
from io import BytesIO


def load_image(path):
    with path.open("rb") as reader:
        data = np.fromstring(reader.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = img[..., ::-1]
    return img


def resize_by_max(image, max_side=512, force=False):
    h, w = image.shape[:2]
    if max(h, w) < max_side and not force:
        return image
    ratio = max(h, w) / max_side

    w = int(w / ratio + 0.5)
    h = int(h / ratio + 0.5)
    return cv2.resize(image, (w, h))


def image2buffer(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        return None
    return BytesIO(buffer)
