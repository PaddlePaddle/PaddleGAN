import paddle
from skimage import color
import numpy as np
from PIL import Image

def convertLAB2RGB( lab ):
   lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
   lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
   rgb = color.lab2rgb( lab.astype(np.float64) )
   return rgb

def convertRGB2LABTensor( rgb ):
   lab = color.rgb2lab( np.asarray( rgb ) ) # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
   ab = np.clip(lab[:, :, 1:3] + 128, 0, 255) # AB --> [0, 255]
   ab = paddle.to_tensor(ab.astype('float32')) / 255.
   L = lab[:, :, 0] * 2.55 # L --> [0, 255]
   L = Image.fromarray( np.uint8( L ) )

   L = paddle.to_tensor(np.array(L).astype('float32')[..., np.newaxis] / 255.0)
   return L, ab

def addMergin(img, target_w, target_h, background_color=(0,0,0)):
   width, height = img.size
   if width==target_w and height==target_h:
      return img
   scale = max(target_w,target_h)/max(width, height)
   width = int(width*scale/16.)*16
   height = int(height*scale/16.)*16

   img = img.resize((width, height), Image.BICUBIC)
   xp = (target_w-width)//2
   yp = (target_h-height)//2
   result = Image.new(img.mode, (target_w, target_h), background_color)
   result.paste(img, (xp, yp))
   return result
