import paddle


def build_lr_scheduler(cfg):
    name = cfg.pop('name')

    # TODO: add more learning rate scheduler
    if name == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 - cfg.start_epoch) / float(cfg.decay_epochs + 1)
            return lr_l

        scheduler = paddle.optimizer.lr.LambdaLR(cfg.learning_rate,
                                                 lr_lambda=lambda_rule)
        return scheduler
    else:
        raise NotImplementedError
