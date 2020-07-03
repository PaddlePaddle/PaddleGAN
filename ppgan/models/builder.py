import paddle

from ..utils.registry import Registry


MODELS = Registry("MODEL")


def build_model(cfg):
    # dataset = MODELS.get(cfg.MODEL.name)(cfg.MODEL)
    # place = paddle.CUDAPlace(0)
    # dataloader = paddle.io.DataLoader(dataset,
    #                                 batch_size=1, #opt.batch_size,
    #                                 places=place,
    #                                 shuffle=True, #not opt.serial_batches,
    #                                 num_workers=0)#int(opt.num_threads))
    model = MODELS.get(cfg.model.name)(cfg)
    return model
    # pass