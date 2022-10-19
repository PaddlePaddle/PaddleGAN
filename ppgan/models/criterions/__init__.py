from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss
from .pixel_loss import L1Loss, MSELoss, CharbonnierLoss, \
                        CalcStyleEmdLoss, CalcContentReltLoss, \
                        CalcContentLoss, CalcStyleLoss, EdgeLoss, PSNRLoss
from .photopen_perceptual_loss import PhotoPenPerceptualLoss
from .gradient_penalty import GradientPenalty

from .builder import build_criterion

from .ssim import SSIM
from .id_loss import IDLoss
from .gfpgan_loss import GFPGANGANLoss, GFPGANL1Loss, GFPGANPerceptualLoss
from .aotgan_perceptual_loss import AOTGANCriterionLoss
