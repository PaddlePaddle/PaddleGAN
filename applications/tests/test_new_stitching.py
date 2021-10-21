import paddle
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
    "face_detector": "blazeface",
    "multi_person": True,
    "image_size": 256,
    "batch_size": 1,
    "face_enhancement": True,
    "mobile_net": False
}
resources = {
    "source_image": "/home/anastasia/paddleGan/PaddleGAN/data/selfie2.JPEG",
    "driving_video": "/home/anastasia/paddleGan/PaddleGAN/data/mayiyahei1.mp4"
}
if __name__ == '__main__':
    predictor = FirstOrderPredictor(**args)
    predictor.run(resources["source_image"], resources["driving_video"])
