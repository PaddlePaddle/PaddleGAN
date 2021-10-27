import sys
sys.path.insert(0, '/home/user/paddle/PaddleGAN/')
import os 
from ppgan.apps.first_order_predictor import FirstOrderPredictor
args = {
    "output": "output",
    "filename": "result.mp4",
    "weight_path": None,
    "relative": True,
    "adapt_scale": True,
    "find_best_frame": False,
    "best_frame": None,
    "ratio": 0.4,
    "face_detector": "sfd",
    "multi_person": False,
    "image_size": 256,
    "batch_size": 1,
    "face_enhancement": True,
    "gfpgan_model_path": "/home/user/paddle/PaddleGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth",
    "mobile_net": False,
    "detection_func": "Union"
}
resources = {
    "source_image": ["/home/user/paddle/PaddleGAN/data/selfie2.JPEG"],
                    # "/home/user/paddle/PaddleGAN/data/getty_517194189_373099.jpg"],
    "driving_video": "/home/user/paddle/PaddleGAN/data/video_6.mp4"
}
if __name__ == '__main__':
    predictor = FirstOrderPredictor(**args)
    for img_path in resources["source_image"]:
        basename = os.path.basename(img_path) 
        name, ext = os.path.splitext(basename)

        predictor.run(img_path, resources["driving_video"], name + '.mp4')
