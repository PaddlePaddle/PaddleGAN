import paddle.vision.transforms as T
import cv2


def get_makeup_transform(cfg, pic="image"):
    if pic == "image":
        transform = T.Compose([
            T.Resize(size=cfg.trans_size),
            T.Permute(to_rgb=False),
        ])
    else:
        transform = T.Resize(size=cfg.trans_size,
                             interpolation=cv2.INTER_NEAREST)

    return transform
