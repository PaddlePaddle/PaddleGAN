import os

import paddle

from ..visual import tensor2img
from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, 
        project=None, 
        name=None, 
        id=None, 
        entity=None, 
        save_dir=None, 
        config=None,
        log_model=False,
        **kwargs):
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install wandb using `pip install wandb`"
                )

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self.model_logging = log_model
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                print(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def log_metrics(self, metrics, prefix=None, step=None):
        if prefix:
            updated_metrics = {
                prefix.lower() + "/" + k: v.item() for k, v in metrics.items() if isinstance(v, paddle.Tensor)
            }
        else:
            updated_metrics = {k: v.item() for k, v in metrics.items()}
        self.run.log(updated_metrics, step=step)

    def log_model(self, file_path, aliases=None, metadata=None):
        if self.model_logging == False:
            return
        artifact = self.wandb.Artifact('model-{}'.format(self.run.id), type='model', metadata=metadata)
        artifact.add_file(file_path, name="model_ckpt.pkl")

        self.run.log_artifact(artifact, aliases=aliases)

    def log_images(self, results, image_num, min_max, results_dir, dataformats, step=None):
        reqd = dict()
        for label, image in results.items():
            image_numpy = tensor2img(image, min_max, image_num)

            images = []
            if dataformats == 'HWC':
                images.append(self.wandb.Image(image_numpy))
            elif dataformats == 'NCHW':
                for img in image_numpy:
                    images.append(self.wandb.Image(img.transpose(1, 2, 0)))
            
            reqd.update({
                results_dir + "/" + label: images
            })
        
        self.run.log(reqd)

    def close(self):
        self.run.finish()
