import cv2
import numpy as np
from tqdm import tqdm, trange
import sys
sys.path.insert(0, '/home/anastasia/paddleGan/PaddleGAN/')
from ppgan.faceutils.dlibutils import face_align
import imageio

class VideoPreprocessor:
    @staticmethod
    def video_preprocessing(raw_video):
        preprocessed_video = [face_align.align_crop(frame) for frame in raw_video]
        width = max([frame.shape[1] for frame in preprocessed_video])
        height = max([frame.shape[0] for frame in preprocessed_video])
        return [cv2.resize(frame, (width, height)) for frame in preprocessed_video]

if __name__ == '__main__':
    reader = imageio.get_reader("/home/anastasia/paddleGan/PaddleGAN/data/Jingle_Bells.mp4")
    raw_driving_video = [im for im in reader]
    video = VideoPreprocessor.video_preprocessing(raw_driving_video)
    fps = reader.get_meta_data()['fps']
    imageio.mimsave("test.mp4",
                    [frame for frame in video],
                    fps=fps)