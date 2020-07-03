import paddle


def build_lr_scheduler(cfg):
    name = cfg.pop('name')

    # TODO: add more learning rate scheduler
    if name == 'linear':
        return LinearDecay(**cfg)
    else:
        raise NotImplementedError


class LinearDecay(paddle.fluid.dygraph.learning_rate_scheduler.LearningRateDecay):
    def __init__(self, learning_rate, step_per_epoch, start_epoch, decay_epochs):
        super(LinearDecay, self).__init__()
        self.learning_rate = learning_rate
        self.start_epoch = start_epoch
        self.decay_epochs = decay_epochs
        self.step_per_epoch = step_per_epoch

    def step(self):
        cur_epoch = int(self.step_num // self.step_per_epoch)
        decay_rate = 1.0 - max(0, cur_epoch + 1 - self.start_epoch) / float(self.decay_epochs + 1)
        return self.create_lr_var(decay_rate * self.learning_rate)