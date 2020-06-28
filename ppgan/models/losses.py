import paddle
import paddle.nn as nn
import numpy as np

from ..modules.nn import BCEWithLogitsLoss

class GANLoss(paddle.fluid.dygraph.Layer):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.real_label = paddle.fluid.dygraph.to_variable(np.array(target_real_label))
        self.fake_label = paddle.fluid.dygraph.to_variable(np.array(target_fake_label))
        # self.real_label.stop_gradients = True
        # self.fake_label.stop_gradients = True

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = BCEWithLogitsLoss()#nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = paddle.fill_constant(shape=paddle.shape(prediction), value=1.0, dtype='float32')#self.real_label
        else:
            target_tensor = paddle.fill_constant(shape=paddle.shape(prediction), value=0.0, dtype='float32')#self.fake_label

        # target_tensor = paddle.cast(target_tensor, prediction.dtype)
        # target_tensor = paddle.expand_as(target_tensor, prediction)
        # target_tensor.stop_gradient = True
        return target_tensor#paddle.expand_as(target_tensor, prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss