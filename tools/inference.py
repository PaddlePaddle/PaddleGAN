import paddle
import argparse
import numpy as np
import random
import os
from collections import OrderedDict

from ppgan.utils.config import get_config
from ppgan.datasets.builder import build_dataloader
from ppgan.engine.trainer import IterLoader
from ppgan.utils.visual import save_image
from ppgan.utils.visual import tensor2img
from ppgan.utils.filesystem import makedirs
from ppgan.metrics import build_metric


MODEL_CLASSES = ["pix2pix", "cyclegan", "wav2lip", "esrgan", "edvr", "fom", "stylegan2", "basicvsr"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES))
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
    parser.add_argument("--output_path",
                        type=str,
                        default="infer_output",
                        help="output_path")
    # config options
    parser.add_argument("-o",
                        "--opt",
                        nargs='+',
                        help="set configuration options")
    # fix random numbers by setting seed
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='fix random numbers by setting seed\".'
    )
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

def setup_metrics(cfg):
    metrics = OrderedDict()
    if isinstance(list(cfg.values())[0], dict):
        for metric_name, cfg_ in cfg.items():
            metrics[metric_name] = build_metric(cfg_)
    else:
        metric = build_metric(cfg)
        metrics[metric.__class__.__name__] = metric

    return metrics

def main():
    args = parse_args()
    if args.seed:
        paddle.seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)    
    cfg = get_config(args.config_file, args.opt)
    predictor = create_predictor(args.model_path, args.device)
    input_handles = [
        predictor.get_input_handle(name)
        for name in predictor.get_input_names()
    ]
    output_handle = predictor.get_output_handle(predictor.get_output_names()[0])
    test_dataloader = build_dataloader(cfg.dataset.test,
                                       is_train=False,
                                       distributed=False)

    max_eval_steps = len(test_dataloader)
    iter_loader = IterLoader(test_dataloader)

    min_max = cfg.get('min_max', None)
    if min_max is None:
        min_max = (-1., 1.)

    model_type = args.model_type
    makedirs(os.path.join(args.output_path, model_type))

    validate_cfg = cfg.get('validate', None)
    metrics = None
    if validate_cfg and 'metrics' in validate_cfg:
        metrics = setup_metrics(validate_cfg['metrics'])
        for metric in metrics.values():
            metric.reset()

    for i in range(max_eval_steps):
        data = next(iter_loader)
        if model_type == "pix2pix":
            real_A = data['B'].numpy()
            input_handles[0].copy_from_cpu(real_A)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(image_numpy, os.path.join(args.output_path, "pix2pix/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "pix2pix/metric.txt")
            real_B = paddle.to_tensor(data['A'])
            for metric in metrics.values():
                metric.update(prediction, real_B)

        elif model_type == "cyclegan":
            real_A = data['A'].numpy()
            input_handles[0].copy_from_cpu(real_A)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(image_numpy, os.path.join(args.output_path, "cyclegan/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "cyclegan/metric.txt")
            real_B = paddle.to_tensor(data['B'])
            for metric in metrics.values():
                metric.update(prediction, real_B)

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
                save_image(image_numpy,
                           "infer_output/wav2lip/{}_{}.png".format(i, j))

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
        elif model_type == "stylegan2":
            noise = paddle.randn([1, 1, 512]).cpu().numpy()
            input_handles[0].copy_from_cpu(noise)
            input_handles[1].copy_from_cpu(np.array([0.7]).astype('float32'))
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(image_numpy, os.path.join(args.output_path, "stylegan2/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "stylegan2/metric.txt")
            real_img = paddle.to_tensor(data['A'])
            for metric in metrics.values():
                metric.update(prediction, real_img)
        elif model_type == "basicvsr":
            lq = data['lq'].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            _, t, _, _, _ = prediction.shape
            out_img = []
            gt_img = []
            for ti in range(t):
                out_tensor = prediction[0, ti]
                gt_tensor = data['gt'][0, ti]
                out_img.append(tensor2img(out_tensor, (0.,1.)))
                gt_img.append(tensor2img(gt_tensor, (0.,1.)))
                
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(image_numpy, os.path.join(args.output_path, "basicvsr/{}.png".format(i)))

            metric_file = os.path.join(args.output_path, "basicvsr/metric.txt")
            for metric in metrics.values():
                metric.update(out_img, gt_img, is_seq=True)

    if metrics:
        log_file = open(metric_file, 'a')
        for metric_name, metric in metrics.items():
            loss_string = "Metric {}: {:.4f}".format(
                    metric_name, metric.accumulate())
            print(loss_string, file=log_file)
        log_file.close()

if __name__ == '__main__':
    main()
