from PIL import Image
import numpy as np
from ppgan.faceutils.mask.face_parser import FaceParser
import cv2


def cutout(image, mask):
    empty = Image.new("RGBA", (image.size), 0)
    cutout = Image.composite(image, empty, mask)

    return cutout


def decode_prediction_mask(mask):
    LABEL_CONTOURS = [(0, 0, 0),  # 0=background
                  # 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=neck_l
                  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                  # 16=cloth, 17=hair, 18=hat
                  (0, 64, 0), (128, 64, 0), (0, 192, 0)]

    mask_shape = mask.shape
    mask_color = np.zeros(shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)

    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]

    return mask_color


face_parcer = FaceParser()
img = Image.open("/home/alexey/Documents/PaddleGAN/data/ppm2.jpg")
original_size = img.size
img = img.resize((512, 512), Image.LANCZOS)
mask = face_parcer.parse(np.array(img).astype(np.float32))
mask = np.array(mask).astype('uint8')

mask_color = decode_prediction_mask(mask)
mask_color = Image.fromarray(mask_color).convert("RGB")
mask_color = mask_color.resize(original_size, Image.LANCZOS)
mask_color.save("/home/alexey/Documents/PaddleGAN/data/ppm2_mask.png", "PNG")
# mask_color.show()