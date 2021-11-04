# Copyright (c) MMEditing Authors.
import paddle
import paddle.nn as nn
import paddle.vision.models.vgg as vgg

from ppgan.utils.download import get_path_from_url
from .builder import CRITERIONS


class PerceptualVGG(nn.Layer):
    """VGG network used in calculating perceptual loss.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_tyep (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained_url (str): Path for pretrained weights. Default:
    """
    def __init__(
            self,
            layer_name_list,
            vgg_type='vgg19',
            use_input_norm=True,
            pretrained_url='https://paddlegan.bj.bcebos.com/models/vgg19.pdparams'
    ):
        super(PerceptualVGG, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        _vgg = getattr(vgg, vgg_type)()

        if pretrained_url:
            weight_path = get_path_from_url(pretrained_url)
            state_dict = paddle.load(weight_path)
            _vgg.load_dict(state_dict)
            print('PerceptualVGG loaded pretrained weight.')

        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers

        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = nn.Sequential(
            *list(_vgg.features.children())[:num_layers])

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

        for v in self.vgg_layers.parameters():
            v.trainable = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output


@CRITERIONS.register()
class PerceptualLoss(nn.Layer):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplified by the
            weight. Default: 1.0.
        style_weight (flaot): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplified by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward fucntion of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:

    """
    def __init__(
            self,
            layer_weights,
            vgg_type='vgg19',
            use_input_norm=True,
            perceptual_weight=1.0,
            style_weight=1.0,
            norm_img=True,
            pretrained='https://paddlegan.bj.bcebos.com/models/vgg19.pdparams',
            criterion='l1'):
        super(PerceptualLoss, self).__init__()
        # when loss weight less than zero return None
        if perceptual_weight <= 0 and style_weight <= 0:
            return None

        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(layer_name_list=list(layer_weights.keys()),
                                 vgg_type=vgg_type,
                                 use_input_norm=use_input_norm,
                                 pretrained_url=pretrained)

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate preceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(self._gram_mat(
                    x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (paddle.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            paddle.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.shape
        features = x.reshape([n, c, w * h])
        features_t = features.transpose([1, 2])
        gram = features.bmm(features_t) / (c * h * w)
        return gram
