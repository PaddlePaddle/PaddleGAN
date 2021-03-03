import paddle
import paddle.nn as nn

from .builder import MODELS
from .sr_model import BaseSRModel
from .generators.builder import build_generator
from .criterions.builder import build_criterion
from .generators.edvr import ResidualBlockNoBN
from ..utils.filesystem import load
from ..modules.init import reset_parameters

@MODELS.register()
class EDVRModel(BaseSRModel):
    def __init__(self, generator, tsa_iter, pixel_criterion=None):
        super(EDVRModel, self).__init__(generator, pixel_criterion)

        init_edvr_weight(self.nets['generator'])
        self.tsa_iter = tsa_iter
        self.current_iter = 1
    def setup_input(self, input):
        self.lq = paddle.to_tensor(input['lq'])
        self.visual_items['lq'] = self.lq[:,2,:,:,:]
        self.visual_items['lq-2'] = self.lq[:,0,:,:,:]
        self.visual_items['lq-1'] = self.lq[:,1,:,:,:]
        self.visual_items['lq+1'] = self.lq[:,3,:,:,:]
        self.visual_items['lq+2'] = self.lq[:,4,:,:,:]
        if 'gt' in input:
            self.gt = paddle.to_tensor(input['gt'])
            self.visual_items['gt'] = self.gt
        self.image_paths = input['lq_path']
    def train_iter(self, optims=None):
        optims['optim'].clear_grad()
        if self.tsa_iter:
            if self.current_iter==1:
                print('Only train TSA module for', self.tsa_iter, 'iters.')
                for name, param in self.nets['generator'].named_parameters():
                    if 'TSAModule' not in name:
                        param.trainable = False
            elif self.current_iter==self.tsa_iter+1:
                print('Train all the parameters.')
                for param in self.nets['generator'].parameters():
                    param.trainable = True
        self.output = self.nets['generator'](self.lq)
        self.visual_items['output'] = self.output
        # pixel loss
        loss_pixel = self.pixel_criterion(self.output, self.gt)
        self.losses['loss_pixel'] = loss_pixel

        loss_pixel.backward()
        optims['optim'].step()
        self.current_iter += 1

def init_edvr_weight(net):
    def reset_func(m):
        if hasattr(m, 'weight') and (not isinstance(m, (nn.BatchNorm, nn.BatchNorm2D))) and (not isinstance(m, ResidualBlockNoBN)):
            reset_parameters(m)
   
    net.apply(reset_func)
   

