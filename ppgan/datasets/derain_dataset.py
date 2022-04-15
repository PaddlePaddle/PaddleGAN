import os
import copy

from pathlib import Path
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register()
class DerainDataset(BaseDataset):
    """Base super resulotion dataset for image restoration."""

    def __init__(self, lq_folder, gt_folder, preprocess, filename_tmpl='{}'):
        super(DerainDataset, self).__init__(preprocess)
        self.lq_folder = lq_folder
        self.gt_folder = gt_folder
        self.filename_tmpl = filename_tmpl

        self.prepare_data_infos()

    def prepare_data_infos(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        self.data_infos = []
        lq_paths = self.scan_folder(self.lq_folder)
        gt_paths = self.scan_folder(self.gt_folder)
        for gt_path in gt_paths:
            basename, ext = os.path.splitext(os.path.basename(gt_path))
            if basename.split('-')[0] == 'norain':
                basename_lq = 'rain-' + basename.split('-')[-1]
                lq_path = os.path.join(
                    self.lq_folder, (f'{self.filename_tmpl.format(basename_lq)}'
                                     f'{ext}'))
                assert lq_path in lq_paths, f'{lq_path} is not in lq_paths.'
                self.data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
        return self.data_infos
