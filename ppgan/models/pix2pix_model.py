import paddle
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..utils.image_pool import ImagePool


@MODELS.register()
class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires 'paired' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (from PatchGAN),
    and a vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (config dict)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = build_generator(opt.model.generator)


        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netD = build_discriminator(opt.model.discriminator)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss(opt.model.gan_mode)
            self.criterionL1 = paddle.nn.L1Loss()

            # build optimizers
            self.optimizer_G = build_optimizer(opt.optimizer, parameter_list=self.netG.parameters())
            self.optimizer_D = build_optimizer(opt.optimizer, parameter_list=self.netD.parameters()) 

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizer_names.extend(['optimizer_G', 'optimizer_D'])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """

        AtoB = self.opt.dataset.train.direction == 'AtoB'
        self.real_A = paddle.imperative.to_variable(input['A' if AtoB else 'B'])
        self.real_B = paddle.imperative.to_variable(input['B' if AtoB else 'A'])
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
 

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def forward_test(self, input):
        input = paddle.imperative.to_variable(input)
        return self.netG(input)
            
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = paddle.concat((self.real_A, self.fake_B), 1)
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
        # compute fake images: G(A)
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.clear_gradients() 
        self.backward_D()
        self.optimizer_D.minimize(self.loss_D) 
       
        # update G
        self.set_requires_grad(self.netD, False) 
        self.optimizer_G.clear_gradients()
        self.backward_G()
        self.optimizer_G.minimize(self.loss_G)
