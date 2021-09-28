from .gan_loss import GANLoss
from .perceptual_loss import PerceptualLoss
from .pixel_loss import L1Loss, MSELoss, CharbonnierLoss, \
                        CalcStyleEmdLoss, CalcContentReltLoss, \
                        CalcContentLoss, CalcStyleLoss, EdgeLoss
from .photopen_perceptual_loss import PhotoPenPerceptualLoss

from .builder import build_criterion
