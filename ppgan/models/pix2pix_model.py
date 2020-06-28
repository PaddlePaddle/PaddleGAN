# import torch
# import paddle
# from .base_model import BaseModel
# from . import networks
import paddle
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss
# from ..modules.nn import L1Loss
from ..solver import build_optimizer
from ..utils.image_pool import ImagePool


@MODELS.register()
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = build_generator(opt.model.generator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = build_discriminator(opt.model.discriminator)
            # self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
            #                               opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss(opt.model.gan_mode, [[[[1.0]]]], [[[[0.0]]]])#.to(self.device)
            self.criterionL1 = paddle.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # FIXME: step per epoch
            # lr_scheduler_g = self.build_lr_scheduler(opt.lr, step_per_epoch=2975)
            # lr_scheduler_d = self.build_lr_scheduler(opt.lr, step_per_epoch=2975)
            # lr_scheduler = self.build_lr_scheduler()
            self.optimizer_G = build_optimizer(opt.optimizer, parameter_list=self.netG.parameters()) #paddle.optimizer.Adam(learning_rate=lr_scheduler_g, parameter_list=self.netG.parameters(), beta1=opt.beta1)
            self.optimizer_D = build_optimizer(opt.optimizer, parameter_list=self.netD.parameters()) #paddle.optimizer.Adam(learning_rate=lr_scheduler_d, parameter_list=self.netD.parameters(), beta1=opt.beta1)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizer_names.extend(['optimizer_G', 'optimizer_D'])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        AtoB = self.opt.dataset.train.direction == 'AtoB'
        self.real_A = paddle.imperative.to_variable(input[0] if AtoB else input[1])
        self.real_B = paddle.imperative.to_variable(input[1] if AtoB else input[0])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def forward_test(self, input):
        input = paddle.imperative.to_variable(input)
        return self.netG(input)

    def test(self, input):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with paddle.imperative.no_grad():
            return self.forward_test(input)
            
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = paddle.concat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = paddle.concat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = paddle.concat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.clear_gradients()   # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.minimize(self.loss_D)        # update D's weights
        # self.netD.clear_gradients()
        # self.optimizer_D.clear_gradients()
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.clear_gradients()       # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.minimize(self.loss_G)           # udpate G's weights
