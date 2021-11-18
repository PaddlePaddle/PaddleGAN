# code was reference to mmcv
import os
import cv2
import numpy as np
from .builder import PREPROCESS


@PREPROCESS.register()
class LoadImageFromFile(object):
    """Load image from file.

    Args:
        key (str): Keys in datas to find corresponding path. Default: 'image'.
        flag (str): Loading flag for images. Default: -1.
        to_rgb (str): Convert img to 'rgb' format. Default: True.
        backend (str): io backend where images are store. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `datas` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """
    def __init__(self,
                 key='image',
                 flag=-1,
                 to_rgb=True,
                 save_original_img=False,
                 backend=None,
                 **kwargs):
        self.key = key
        self.flag = flag
        self.to_rgb = to_rgb
        self.backend = backend
        self.save_original_img = save_original_img
        self.kwargs = kwargs

    def __call__(self, datas):
        """Call function.

        Args:
            datas (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        filepath = str(datas[f'{self.key}_path'])
        #TODO: use file client to manage io backend
        # such as opencv, pil, imdb
        img = cv2.imread(filepath, self.flag)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        datas[self.key] = img
        datas[f'{self.key}_path'] = filepath
        datas[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            datas[f'ori_{self.key}'] = img.copy()

        return datas


@PREPROCESS.register()
class ReadImageSequence(LoadImageFromFile):
    """Read image sequence.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        key (str): Keys in datas to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        to_rgb (str): Convert img to 'rgb' format. Default: True.
        save_original_img (bool): If True, maintain a copy of the image in
            `datas` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """
    def __call__(self, datas):
        """Call function.

        Args:
            datas (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        filepaths = datas[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []
        for filepath in filepaths:
            img = cv2.imread(filepath, self.flag)

            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            imgs.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())

        datas[self.key] = imgs
        datas[f'{self.key}_path'] = filepaths
        datas[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            datas[f'ori_{self.key}'] = ori_imgs

        return datas


@PREPROCESS.register()
class GetNeighboringFramesIdx:
    """Get neighboring frame indices for a video. It also performs temporal
    augmention with random interval.

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """
    def __init__(self, interval_list, start_idx=0, filename_tmpl='{:08d}.png'):
        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    def __call__(self, datas):
        """Call function.

        Args:
            datas (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        clip_name = datas['key']
        interval = np.random.choice(self.interval_list)

        self.sequence_length = datas['sequence_length']
        num_frames = datas.get('num_frames', self.sequence_length)

        if self.sequence_length - num_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_frames].')
        start_frame_idx = np.random.randint(
            0, self.sequence_length - num_frames * interval + 1)
        end_frame_idx = start_frame_idx + num_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        neighbor_list = [v + self.start_idx for v in neighbor_list]

        lq_path_root = datas['lq_path']
        gt_path_root = datas['gt_path']

        lq_path = [
            os.path.join(lq_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]
        gt_path = [
            os.path.join(gt_path_root, clip_name, self.filename_tmpl.format(v))
            for v in neighbor_list
        ]

        datas['lq_path'] = lq_path
        datas['gt_path'] = gt_path
        datas['interval'] = interval

        return datas
