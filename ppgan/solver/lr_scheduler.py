import paddle


def build_lr_scheduler(cfg):
    name = cfg.pop('name')

    # TODO: add more learning rate scheduler
    if name == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 - cfg.start_epoch) / float(cfg.decay_epochs + 1)
            return lr_l

        scheduler = paddle.optimizer.lr_scheduler.LambdaLR(
            cfg.learning_rate, lr_lambda=lambda_rule)
        return scheduler
    else:
        raise NotImplementedError


# paddle.optimizer.lr_scheduler
class LinearDecay(paddle.optimizer.lr_scheduler._LRScheduler):
    def __init__(self, learning_rate, step_per_epoch, start_epoch,
                 decay_epochs):
        super(LinearDecay, self).__init__()
        self.learning_rate = learning_rate
        self.start_epoch = start_epoch
        self.decay_epochs = decay_epochs
        self.step_per_epoch = step_per_epoch

    def step(self):
        cur_epoch = int(self.step_num // self.step_per_epoch)
        decay_rate = 1.0 - max(
            0, cur_epoch + 1 - self.start_epoch) / float(self.decay_epochs + 1)
        return self.create_lr_var(decay_rate * self.learning_rate)
