"""Compute depth maps for images in the input folder.
"""
import os
import glob
import utils
import cv2
import argparse
import pickle
import numpy as np

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import paddle
from paddle.vision.transforms import Compose
from midas.midas_net import MidasNet
#from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


def run(input_path, output_path, model_path, model_type="large"):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # load network
    if model_type == "large":
        model = MidasNet(model_path, non_negative=True)
        #model = MidasNet(non_negative=True)
        net_w, net_h = 384, 384
#    elif model_type == "small":
#        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
#        net_w, net_h = 256, 256
    else:
        print(
            f"model_type '{model_type}' not implemented, use: --model_type large"
        )
        assert False


#    torch_params = pickle.load(open('midas.pkl','rb'))
#    weight_list = []
#    for k,v in torch_params.items():
#        #print(k)
#        weight_list.append(k)
#    print('torch len ', len(weight_list))
#    print('pretrained.layer1.0.weight ', np.sum(np.abs(torch_params['pretrained.layer1.0.weight'])))
#
#
#    print('===============================')
#    paddle_list = []
#    for k, v in model.named_parameters():
##        print(k, v.name)
#        paddle_list.append(k)
#        if '_mean' in k:
#            k = k.replace('_mean', 'running_mean')
#        if '_variance' in k:
#            k = k.replace('_variance', 'running_var')
#        assert v.shape == list(torch_params[k].shape)
##            print(v.shape, torch_params[k].shape)
#        v.set_value(torch_params[k])
#    print('paddle len ', len(paddle_list))
#
#    print('pretrained.layer1.0.weight--paddle ', np.sum(np.abs(model.state_dict()['pretrained.layer1.0.weight'].numpy())))

    transform = Compose([
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # compute
        print('Input ', np.sum(np.abs(img_input)))
        with paddle.no_grad():
            sample = paddle.to_tensor(img_input).unsqueeze(0)
            prediction = model.forward(sample)
            print('Predict ', np.sum(np.abs(prediction.numpy())))
            prediction = (paddle.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy())

        # output
        filename = os.path.join(output_path,
                                os.path.splitext(os.path.basename(img_name))[0])
        utils.write_depth(filename, prediction, bits=2)

        vmax = np.percentile(prediction, 95)
        normalizer = mpl.colors.Normalize(vmin=prediction.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(prediction)[:, :, :3] * 255).astype(
            np.uint8)
        im = pil.fromarray(colormapped_im)
        name_dest_im = os.path.join(
            'output',
            "{}_disp.jpeg".format(img_name.split('.')[0].split('/')[-1]))
        im.save(name_dest_im)

    paddle.save(model.state_dict(), 'midas.pdparams')

    print("finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--input_path',
                        default='input',
                        help='folder with input images')
    parser.add_argument('-o',
                        '--output_path',
                        default='output',
                        help='folder for output images')
    parser.add_argument('-m',
                        '--model_weights',
                        default='midas.pdparams',
                        help='path to the trained weights of model')
    parser.add_argument('-t',
                        '--model_type',
                        default='large',
                        help='model type: large or small')
    args = parser.parse_args()

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type)
