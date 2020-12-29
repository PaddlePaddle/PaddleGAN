import cv2

from .builder import PREPROCESS


@PREPROCESS.register()
class LoadImageFromFile(object):
    """Load image from file.

    Args:
        key (str): Keys in results to find corresponding path. Default: 'image'.
        flag (str): Loading flag for images. Default: -1.
        to_rgb (str): Convert img to 'rgb' format. Default: True.
        backend (str): io backend where images are store. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
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

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        filepath = str(results[f'{self.key}_path'])
        #TODO: use file client to manage io backend
        # such as opencv, pil, imdb
        img = cv2.imread(filepath, self.flag)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results
