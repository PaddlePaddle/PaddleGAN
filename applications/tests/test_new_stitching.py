import sys
sys.path.insert(0, '/home/anastasia/paddleGan/PaddleGAN/')
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
    "multi_person": True,
    "image_size": 256,
    "batch_size": 1,
    "face_enhancement": True,
    "mobile_net": False,
    "detection_func": "Union"
}
resources = {
    "source_image": ["/home/anastasia/paddleGan/PaddleGAN/data/selfie2.JPEG"],
    # "/home/anastasia/paddleGan/PaddleGAN/data/front-slide-6.jpg",
    # "/home/anastasia/paddleGan/PaddleGAN/data/300x450.jpeg",
    # "/home/anastasia/paddleGan/PaddleGAN/data/people-persons-peoples.jpg" ],
    "driving_video": "/home/anastasia/paddleGan/PaddleGAN/data/video.mp4"
}
if __name__ == '__main__':
    predictor = FirstOrderPredictor(**args)
    for img_path in resources["source_image"]:
        basename = os.path.basename(img_path) 
        name, ext = os.path.splitext(basename)
        predictor.run(img_path, resources["driving_video"], name + '.mp4')
