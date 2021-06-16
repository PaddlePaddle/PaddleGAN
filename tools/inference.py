import paddle
import argparse
import numpy as np

from ppgan.utils.config import get_config
from ppgan.datasets.builder import build_dataloader
from ppgan.engine.trainer import IterLoader
from ppgan.utils.visual import save_image
from ppgan.utils.visual import tensor2img
from ppgan.utils.filesystem import makedirs

MODEL_CLASSES = ["pix2pix", "cyclegan", "wav2lip", "esrgan", "edvr"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES))
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument('-c',
                        '--config-file',
                        metavar="FILE",
                        help='config file path')
    # config options
    parser.add_argument("-o",
                        "--opt",
                        nargs='+',
                        help="set configuration options")
    args = parser.parse_args()
    return args


def create_predictor(model_path, device="gpu"):
    config = paddle.inference.Config(model_path + ".pdmodel",
                                     model_path + ".pdiparams")
    if device == "gpu":
        config.enable_use_gpu(100, 0)
    elif device == "cpu":
        config.disable_gpu()
    elif device == "xpu":
        config.enable_xpu(100)
    else:
        config.disable_gpu()

    predictor = paddle.inference.create_predictor(config)
    return predictor


def main():
    args = parse_args()
    cfg = get_config(args.config_file, args.opt)
    predictor = create_predictor(args.model_path, args.device)
    input_handles = [predictor.get_input_handle(
        name) for name in predictor.get_input_names()]
    output_handle = predictor.get_output_handle(
        predictor.get_output_names()[0])
    test_dataloader = build_dataloader(
        cfg.dataset.test, is_train=False, distributed=False)

    max_eval_steps = len(test_dataloader)
    iter_loader = IterLoader(test_dataloader)

    min_max = cfg.get('min_max', None)
    if min_max is None:
        min_max = (-1., 1.)

    model_type = args.model_type
    makedirs("infer_output/" + model_type)
    for i in range(max_eval_steps):
        data = next(iter_loader)
        if model_type == "pix2pix":
            real_A = data['B'].numpy()
            input_handles[0].copy_from_cpu(real_A)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            save_image(image_numpy, "infer_output/pix2pix/{}.png".format(i))
        elif model_type == "cyclegan":
            real_A = data['A'].numpy()
            input_handles[0].copy_from_cpu(real_A)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            save_image(image_numpy, "infer_output/cyclegan/{}.png".format(i))
        elif model_type == "wav2lip":
            indiv_mels, x = data['indiv_mels'].numpy()[0], data['x'].numpy()[0]
            x = x.transpose([1, 0, 2, 3])
            input_handles[0].copy_from_cpu(indiv_mels)
            input_handles[1].copy_from_cpu(x)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            for j in range(prediction.shape[0]):
                prediction[j] = prediction[j][::-1, :, :]
                image_numpy = paddle.to_tensor(prediction[j])
                image_numpy = tensor2img(image_numpy, (0, 1))
                save_image(
                    image_numpy, "infer_output/wav2lip/{}_{}.png".format(i, j))
        elif model_type == "esrgan":
            lq = data['lq'].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            save_image(image_numpy, "infer_output/esrgan/{}.png".format(i))
        elif model_type == "edvr":
            lq = data['lq'].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            save_image(image_numpy, "infer_output/edvr/{}.png".format(i))


if __name__ == '__main__':
    main()
