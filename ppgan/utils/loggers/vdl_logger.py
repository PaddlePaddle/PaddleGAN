import paddle
from visualdl import LogWriter

from ..visual import tensor2img
from .base_logger import BaseLogger


class VDLLogger(BaseLogger):
    def __init__(self, save_dir):
        super().__init__(save_dir)
        self.vdl_writer = LogWriter(logdir=save_dir)

    def log_metrics(self, metrics, prefix=None, step=None):
        if prefix:
            updated_metrics = {
                prefix.lower() + "/" + k: v.item() for k, v in metrics.items() if isinstance(v, paddle.Tensor)
            }
        else:
            updated_metrics = {k: v.item() for k, v in metrics.items()}
        for k, v in updated_metrics.items():
            self.vdl_writer.add_scalar(tag=k, value=v, step=step)
    
    def log_model(self, file_path, aliases=None, metadata=None):
        pass
    
    def log_images(self, results, image_num, min_max, results_dir, dataformats, step=None):
        for label, image in results.items():
            image_numpy = tensor2img(image, min_max, image_num)
            self.vdl_writer.add_image(
                results_dir + "/" + label,
                image_numpy,
                step,
                dataformats=dataformats
            )

    def close(self):
        self.vdl_writer.close() 
