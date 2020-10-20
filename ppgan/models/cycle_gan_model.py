import paddle
from paddle.distributed import ParallelEnv
from .base_model import BaseModel

from .builder import MODELS
from .generators.builder import build_generator
from .discriminators.builder import build_discriminator
from .losses import GANLoss

from ..solver import build_optimizer
from ..modules.init import init_weights
from ..utils.image_pool import ImagePool


@MODELS.register()
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (config)-- stores all the experiment flags; needs to be a subclass of Dict
        """
        BaseModel.__init__(self, opt)
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = build_generator(opt.model.generator)
        self.netG_B = build_generator(opt.model.generator)
        init_weights(self.netG_A)
        init_weights(self.netG_B)

        if self.isTrain:  # define discriminators
            self.netD_A = build_discriminator(opt.model.discriminator)
            self.netD_B = build_discriminator(opt.model.discriminator)
            init_weights(self.netD_A)
            init_weights(self.netD_B)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (
                    opt.dataset.train.input_nc == opt.dataset.train.output_nc)
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.dataset.train.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.dataset.train.pool_size)
            # define loss functions
            self.criterionGAN = GANLoss(opt.model.gan_mode)
            self.criterionCycle = paddle.nn.L1Loss()
            self.criterionIdt = paddle.nn.L1Loss()

            self.build_lr_scheduler()
            self.optimizer_G = build_optimizer(
                opt.optimizer,
                self.lr_scheduler,
                parameter_list=self.netG_A.parameters() +
                self.netG_B.parameters())
            self.optimizer_D = build_optimizer(
                opt.optimizer,
                self.lr_scheduler,
                parameter_list=self.netD_A.parameters() +
                self.netD_B.parameters())

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.optimizer_names.extend(['optimizer_G', 'optimizer_D'])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        mode = 'train' if self.isTrain else 'test'
        AtoB = self.opt.dataset[mode].direction == 'AtoB'

        if AtoB:
            if 'A' in input:
                self.real_A = paddle.to_tensor(input['A'])
            if 'B' in input:
                self.real_B = paddle.to_tensor(input['B'])
        else:
            if 'B' in input:
                self.real_A = paddle.to_tensor(input['B'])
            if 'A' in input:
                self.real_B = paddle.to_tensor(input['A'])

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'real_A'):
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))

        if hasattr(self, 'real_B'):
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        if ParallelEnv().nranks > 1:
            loss_D = netD.scale_loss(loss_D)
            loss_D.backward()
            netD.apply_collective_grads()
        else:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.losses['D_A_loss'] = self.loss_D_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.losses['D_B_loss'] = self.loss_D_B

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A,
                                                self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B,
                                                self.real_B) * lambda_B

        self.losses['G_idt_A_loss'] = self.loss_idt_A
        self.losses['G_idt_B_loss'] = self.loss_idt_B
        self.losses['G_A_adv_loss'] = self.loss_G_A
        self.losses['G_B_adv_loss'] = self.loss_G_B
        self.losses['G_A_cycle_loss'] = self.loss_cycle_A
        self.losses['G_B_cycle_loss'] = self.loss_cycle_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        if ParallelEnv().nranks > 1:
            self.loss_G = self.netG_A.scale_loss(self.loss_G)
            self.loss_G.backward()
            self.netG_A.apply_collective_grads()
            self.netG_B.apply_collective_grads()
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # compute fake images and reconstruction images.
        self.forward()
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        # set G_A and G_B's gradients to zero
        self.optimizer_G.clear_gradients()
        # calculate gradients for G_A and G_B
        self.backward_G()
        # update G_A and G_B's weights
        self.optimizer_G.minimize(self.loss_G)
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)

        # set D_A and D_B's gradients to zero
        self.optimizer_D.clear_gradients()
        # calculate gradients for D_A
        self.backward_D_A()
        # calculate graidents for D_B
        self.backward_D_B()
        # update D_A and D_B's weights
        self.optimizer_D.minimize(self.loss_D_A + self.loss_D_B)
