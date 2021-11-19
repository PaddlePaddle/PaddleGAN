# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license

from paddle.fluid.layers.nn import soft_relu
from .base_model import BaseModel

from paddle import nn
import paddle
import paddle.nn.functional as F
from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from ..modules.init import kaiming_normal_, constant_
from ppgan.utils.visual import make_grid, tensor2img

import numpy as np


def translate_using_reference(nets, w_hpf, x_src, x_ref, y_ref):
    N, C, H, W = x_src.shape
    wb = paddle.to_tensor(np.ones((1, C, H, W))).astype('float32')
    x_src_with_wb = paddle.concat([wb, x_src], axis=0)

    masks = nets['fan'].get_heatmap(x_src) if w_hpf > 0 else None
    s_ref = nets['style_encoder'](x_ref, y_ref)
    s_ref_list = paddle.unsqueeze(s_ref, axis=[1])
    s_ref_lists = []
    for _ in range(N):
        s_ref_lists.append(s_ref_list)
    s_ref_list = paddle.stack(s_ref_lists, axis=1)
    s_ref_list = paddle.reshape(
        s_ref_list,
        (s_ref_list.shape[0], s_ref_list.shape[1], s_ref_list.shape[3]))
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets['generator'](x_src, s_ref, masks=masks)
        x_fake_with_ref = paddle.concat([x_ref[i:i + 1], x_fake], axis=0)
        x_concat += [x_fake_with_ref]

    x_concat = paddle.concat(x_concat, axis=0)
    img = tensor2img(make_grid(x_concat, nrow=N + 1, range=(0, 1)))
    del x_concat
    return img


def compute_d_loss(nets,
                   lambda_reg,
                   x_real,
                   y_org,
                   y_trg,
                   z_trg=None,
                   x_ref=None,
                   masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.stop_gradient = False
    out = nets['discriminator'](x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with paddle.no_grad():
        if z_trg is not None:
            s_trg = nets['mapping_network'](z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets['style_encoder'](x_ref, y_trg)

        x_fake = nets['generator'](x_real, s_trg, masks=masks)
    out = nets['discriminator'](x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + lambda_reg * loss_reg
    return loss, {
        'real': loss_real.numpy(),
        'fake': loss_fake.numpy(),
        'reg': loss_reg.numpy()
    }


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = paddle.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    grad_dout = paddle.grad(outputs=d_out.sum(),
                            inputs=x_in,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.shape == x_in.shape)
    reg = 0.5 * paddle.reshape(grad_dout2, (batch_size, -1)).sum(1).mean(0)
    return reg


def soft_update(source, target, beta=1.0):
    assert 0.0 <= beta <= 1.0

    if isinstance(source, paddle.DataParallel):
        source = source._layers

    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(beta * source_param +
                               (1.0 - beta) * target_param)


def dump_model(model):
    params = {}
    for k in model.state_dict().keys():
        if k.endswith('.scale'):
            params[k] = model.state_dict()[k].shape
    return params


def compute_g_loss(nets,
                   w_hpf,
                   lambda_sty,
                   lambda_ds,
                   lambda_cyc,
                   x_real,
                   y_org,
                   y_trg,
                   z_trgs=None,
                   x_refs=None,
                   masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets['mapping_network'](z_trg, y_trg)
    else:
        s_trg = nets['style_encoder'](x_ref, y_trg)

    x_fake = nets['generator'](x_real, s_trg, masks=masks)
    out = nets['discriminator'](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets['style_encoder'](x_fake, y_trg)
    loss_sty = paddle.mean(paddle.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets['mapping_network'](z_trg2, y_trg)
    else:
        s_trg2 = nets['style_encoder'](x_ref2, y_trg)
    x_fake2 = nets['generator'](x_real, s_trg2, masks=masks)
    loss_ds = paddle.mean(paddle.abs(x_fake - x_fake2))

    # cycle-consistency loss
    if w_hpf > 0:
        if isinstance(nets['fan'], paddle.DataParallel):
            masks = nets['fan']._layers.get_heatmap(x_fake)
        else:
            masks = nets['fan'].get_heatmap(x_fake)
    else:
        masks = None

    s_org = nets['style_encoder'](x_real, y_org)
    x_rec = nets['generator'](x_fake, s_org, masks=masks)
    loss_cyc = paddle.mean(paddle.abs(x_rec - x_real))

    loss = loss_adv + lambda_sty * loss_sty \
        - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    return loss, {
        'adv': loss_adv.numpy(),
        'sty': loss_sty.numpy(),
        'ds:': loss_ds.numpy(),
        'cyc': loss_cyc.numpy()
    }


def he_init(module):
    if isinstance(module, nn.Conv2D):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)


@MODELS.register()
class StarGANv2Model(BaseModel):
    def __init__(
        self,
        generator,
        style=None,
        mapping=None,
        discriminator=None,
        fan=None,
        latent_dim=16,
        lambda_reg=1,
        lambda_sty=1,
        lambda_ds=1,
        lambda_cyc=1,
    ):
        super(StarGANv2Model, self).__init__()
        self.w_hpf = generator['w_hpf']
        self.nets_ema = {}
        self.nets['generator'] = build_generator(generator)
        self.nets_ema['generator'] = build_generator(generator)
        self.nets['style_encoder'] = build_generator(style)
        self.nets_ema['style_encoder'] = build_generator(style)
        self.nets['mapping_network'] = build_generator(mapping)
        self.nets_ema['mapping_network'] = build_generator(mapping)
        if discriminator:
            self.nets['discriminator'] = build_discriminator(discriminator)
        if self.w_hpf > 0:
            fan_model = build_generator(fan)
            fan_model.eval()
            self.nets['fan'] = fan_model
            self.nets_ema['fan'] = fan_model
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_cyc = lambda_cyc

        self.nets['generator'].apply(he_init)
        self.nets['style_encoder'].apply(he_init)
        self.nets['mapping_network'].apply(he_init)
        self.nets['discriminator'].apply(he_init)

        # remember the initial value of ds weight
        self.initial_lambda_ds = self.lambda_ds

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        pass
        self.input = input
        self.input['z_trg'] = paddle.randn(
            (input['src'].shape[0], self.latent_dim))
        self.input['z_trg2'] = paddle.randn(
            (input['src'].shape[0], self.latent_dim))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def _reset_grad(self, optims):
        for optim in optims.values():
            optim.clear_gradients()

    def train_iter(self, optimizers=None):
        #TODO
        x_real, y_org = self.input['src'], self.input['src_cls']
        x_ref, x_ref2, y_trg = self.input['ref'], self.input[
            'ref2'], self.input['ref_cls']
        z_trg, z_trg2 = self.input['z_trg'], self.input['z_trg2']

        if self.w_hpf > 0:
            if isinstance(self.nets['fan'], paddle.DataParallel):
                masks = self.nets['fan']._layers.get_heatmap(x_real)
            else:
                masks = self.nets['fan'].get_heatmap(x_real)
        else:
            masks = None

        # train the discriminator
        d_loss, d_losses_latent = compute_d_loss(self.nets,
                                                 self.lambda_reg,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trg=z_trg,
                                                 masks=masks)
        self._reset_grad(optimizers)
        d_loss.backward()
        optimizers['discriminator'].minimize(d_loss)

        d_loss, d_losses_ref = compute_d_loss(self.nets,
                                              self.lambda_reg,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_ref=x_ref,
                                              masks=masks)
        self._reset_grad(optimizers)
        d_loss.backward()
        optimizers['discriminator'].step()

        # train the generator
        g_loss, g_losses_latent = compute_g_loss(self.nets,
                                                 self.w_hpf,
                                                 self.lambda_sty,
                                                 self.lambda_ds,
                                                 self.lambda_cyc,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trgs=[z_trg, z_trg2],
                                                 masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()
        optimizers['mapping_network'].step()
        optimizers['style_encoder'].step()

        g_loss, g_losses_ref = compute_g_loss(self.nets,
                                              self.w_hpf,
                                              self.lambda_sty,
                                              self.lambda_ds,
                                              self.lambda_cyc,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_refs=[x_ref, x_ref2],
                                              masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()

        # compute moving average of network parameters
        soft_update(self.nets['generator'],
                    self.nets_ema['generator'],
                    beta=0.999)
        soft_update(self.nets['mapping_network'],
                    self.nets_ema['mapping_network'],
                    beta=0.999)
        soft_update(self.nets['style_encoder'],
                    self.nets_ema['style_encoder'],
                    beta=0.999)

        # decay weight for diversity sensitive loss
        if self.lambda_ds > 0:
            self.lambda_ds -= (self.initial_lambda_ds / self.total_iter)

        for loss, prefix in zip(
            [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
            for key, value in loss.items():
                self.losses[prefix + key] = value
        self.losses['G/lambda_ds'] = self.lambda_ds
        self.losses['Total iter'] = int(self.total_iter)

    def test_iter(self, metrics=None):
        #TODO
        self.nets_ema['generator'].eval()
        self.nets_ema['style_encoder'].eval()
        soft_update(self.nets['generator'],
                    self.nets_ema['generator'],
                    beta=0.999)
        soft_update(self.nets['mapping_network'],
                    self.nets_ema['mapping_network'],
                    beta=0.999)
        soft_update(self.nets['style_encoder'],
                    self.nets_ema['style_encoder'],
                    beta=0.999)
        src_img = self.input['src']
        ref_img = self.input['ref']
        ref_label = self.input['ref_cls']
        with paddle.no_grad():
            img = translate_using_reference(
                self.nets_ema, self.w_hpf,
                paddle.to_tensor(src_img).astype('float32'),
                paddle.to_tensor(ref_img).astype('float32'),
                paddle.to_tensor(ref_label).astype('float32'))
        self.visual_items['reference'] = img
        self.nets_ema['generator'].train()
        self.nets_ema['style_encoder'].train()
