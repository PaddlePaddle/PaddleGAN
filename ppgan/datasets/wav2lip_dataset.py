# code was heavily based on https://github.com/Rudrabha/Wav2Lip
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/Rudrabha/Wav2Lip#license-and-citation

import cv2
import random
import os.path
import numpy as np
from PIL import Image
from glob import glob
from os.path import dirname, join, basename, isfile
from ppgan.utils import audio
from ppgan.utils.audio_config import get_audio_config
import numpy as np

import paddle
from .builder import DATASETS


def get_image_list(data_root, split):
    filelist = []

    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            video_path = os.path.join(data_root, line)
            assert os.path.exists(video_path), '{} is not found'.format(
                video_path)
            filelist.append(video_path)

    return filelist


syncnet_T = 5
syncnet_mel_step_size = 16
audio_cfg = get_audio_config()


@DATASETS.register()
class Wav2LipDataset(paddle.io.Dataset):
    def __init__(self, dataroot, img_size, filelists_path, split):
        """Initialize Wav2Lip dataset class.

        Args:
            dataroot (str): Directory of dataset.
        """
        self.image_path = dataroot
        self.img_size = img_size
        self.split = split
        self.all_videos = get_image_list(self.image_path, self.split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (self.img_size, self.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(
                start_frame)  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(audio_cfg["fps"])))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx:end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(
            start_frame) + 1  # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue

            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, audio_cfg["sample_rate"])

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2] // 2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = np.float32(x)
            mel = np.transpose(mel)
            mel = np.expand_dims(mel, 0)
            indiv_mels = np.expand_dims(indiv_mels, 1)

            return {
                'x': x,
                'indiv_mels': np.float32(indiv_mels),
                'mel': np.float32(mel),
                'y': np.float32(y)
            }

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return len(self.all_videos)
