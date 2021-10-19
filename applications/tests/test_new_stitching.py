import paddle
from ppgan.apps.first_order_predictor import FirstOrderPredictor
args = {
    "output": "output",
    "filename": "result.mp4",
    "weight_path": None,
    "relative": True,
    "adapt_scale": True,
    "find_best_frame": True,
    "best_frame": None,
    "ratio": 0.4,
    "face_detector": "sfd",
    "multi_person": False,
    "image_size": 512,
    "batch_size": 1,
    "face_enhancement": True,
    "mobile_net": False
}
resources = {
    "source_image": "/home/anastasia/paddleGan/PaddleGAN/data/selfie2.JPEG",
    "driving_video": "/home/anastasia/paddleGan/PaddleGAN/data/mayiyahei.MP4"
}
if __name__ == '__main__':
    predictor = FirstOrderPredictor(**args)
    predictor.run(args.source_image, args.driving_video)
