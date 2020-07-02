import cv2
import paddle
import os.path
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset

from .builder import DATASETS


@DATASETS.register()
class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Args:
            cfg (dict) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.cfg.transform.load_size >= self.cfg.transform.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.cfg.output_nc if self.cfg.direction == 'BtoA' else self.cfg.input_nc
        self.output_nc = self.cfg.input_nc if self.cfg.direction == 'BtoA' else self.cfg.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = cv2.imread(AB_path)

        # split AB image into A and B
        h, w = AB.shape[:2]
        # w, h = AB.size
        w2 = int(w / 2)

        A = AB[:h, :w2, :]
        B = AB[:h, w2:, :]


        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        transform_params = get_params(self.cfg.transform, (w2, h))

        A_transform = get_transform(self.cfg.transform, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.cfg.transform, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
        # return A, B, index #{'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def get_path_by_indexs(self, indexs):
        if isinstance(indexs, paddle.Variable):
            indexs = indexs.numpy()
        current_paths = []
        for index in indexs:
            current_paths.append(self.AB_paths[index])
        return current_paths
