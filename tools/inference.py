import paddle
import argparse
import numpy as np
import random
import os
from collections import OrderedDict
import sys
import cv2

sys.path.append(".")

from ppgan.utils.config import get_config
from ppgan.datasets.builder import build_dataloader
from ppgan.engine.trainer import IterLoader
from ppgan.utils.visual import save_image
from ppgan.utils.visual import tensor2img
from ppgan.utils.filesystem import makedirs
from ppgan.metrics import build_metric
from ppgan.utils.logger import get_logger


MODEL_CLASSES = ["pix2pix", "cyclegan", "wav2lip", "esrgan", \
                 "edvr", "fom", "stylegan2", "basicvsr", "msvsr", \
                 "singan", "swinir", "invdn",  "aotgan", "nafnet"]


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
        choices=["cpu", "gpu", "xpu", "npu"],
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
                        help='fix random numbers by setting seed\".')
    # for tensorRT
    parser.add_argument("--run_mode",
                        default="fluid",
                        type=str,
                        choices=["fluid", "trt_fp32", "trt_fp16"],
                        help="mode of running(fluid/trt_fp32/trt_fp16)")
    parser.add_argument("--trt_min_shape",
                        default=1,
                        type=int,
                        help="trt_min_shape for tensorRT")
    parser.add_argument("--trt_max_shape",
                        default=1280,
                        type=int,
                        help="trt_max_shape for tensorRT")
    parser.add_argument("--trt_opt_shape",
                        default=640,
                        type=int,
                        help="trt_opt_shape for tensorRT")
    parser.add_argument("--min_subgraph_size",
                        default=3,
                        type=int,
                        help="trt_opt_shape for tensorRT")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="batch_size for tensorRT")
    parser.add_argument("--use_dynamic_shape",
                        dest="use_dynamic_shape",
                        action="store_true",
                        help="use_dynamic_shape for tensorRT")
    parser.add_argument("--trt_calib_mode",
                        dest="trt_calib_mode",
                        action="store_true",
                        help="trt_calib_mode for tensorRT")
    args = parser.parse_args()
    return args


def create_predictor(model_path,
                     device="gpu",
                     run_mode='fluid',
                     batch_size=1,
                     min_subgraph_size=3,
                     use_dynamic_shape=False,
                     trt_min_shape=1,
                     trt_max_shape=1280,
                     trt_opt_shape=640,
                     trt_calib_mode=False):
    config = paddle.inference.Config(model_path + ".pdmodel",
                                     model_path + ".pdiparams")
    if device == "gpu":
        config.enable_use_gpu(100, 0)
    elif device == "cpu":
        config.disable_gpu()
    elif device == "npu":
        config.enable_custom_device('npu')
    elif device == "xpu":
        config.enable_xpu()
    else:
        config.disable_gpu()

    precision_map = {
        'trt_int8': paddle.inference.Config.Precision.Int8,
        'trt_fp32': paddle.inference.Config.Precision.Float32,
        'trt_fp16': paddle.inference.Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(workspace_size=1 << 25,
                                      max_batch_size=batch_size,
                                      min_subgraph_size=min_subgraph_size,
                                      precision_mode=precision_map[run_mode],
                                      use_static=False,
                                      use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    predictor = paddle.inference.create_predictor(config)
    return predictor, config


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
    cfg = get_config(args.config_file, args.opt, show=True)
    predictor, config = create_predictor(
        args.model_path, args.device, args.run_mode, args.batch_size,
        args.min_subgraph_size, args.use_dynamic_shape, args.trt_min_shape,
        args.trt_max_shape, args.trt_opt_shape, args.trt_calib_mode)
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
            save_image(
                image_numpy,
                os.path.join(args.output_path, "pix2pix/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "pix2pix/metric.txt")
            real_B = paddle.to_tensor(data['A'])
            for metric in metrics.values():
                metric.update(prediction, real_B)

        elif model_type == "cyclegan":
            import auto_log
            logger = get_logger(name='ppgan')

            size = data['A'].shape
            pid = os.getpid()
            auto_logger = auto_log.AutoLogger(
                model_name=args.model_type,
                model_precision=args.run_mode,
                batch_size=args.batch_size,
                data_shape=size,
                save_path=args.output_path + 'auto_log.lpg',
                inference_config=config,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0)
            auto_logger.times.start()
            real_A = data['A'].numpy()
            input_handles[0].copy_from_cpu(real_A)
            auto_logger.times.stamp()
            predictor.run()
            auto_logger.times.stamp()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "cyclegan/{}.png".format(i)))
            logger.info(
                "Inference succeeded! The inference result has been saved in {}"
                .format(
                    os.path.join(args.output_path,
                                 "cyclegan/{}.png".format(i))))
            auto_logger.times.end(stamp=True)
            auto_logger.report()
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
            gt_numpy = tensor2img(data['gt'][0], min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "esrgan/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(image_numpy, gt_numpy)
            break
        elif model_type == "edvr":
            lq = data['lq'].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            gt_numpy = tensor2img(data['gt'][0, 0], min_max)
            save_image(image_numpy,
                       os.path.join(args.output_path, "edvr/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(image_numpy, gt_numpy)
            break
        elif model_type == "stylegan2":
            noise = paddle.randn([1, 1, 512]).cpu().numpy()
            input_handles[0].copy_from_cpu(noise)
            input_handles[1].copy_from_cpu(np.array([0.7]).astype('float32'))
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction[0], min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "stylegan2/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "stylegan2/metric.txt")
            real_img = paddle.to_tensor(data['A'])
            for metric in metrics.values():
                metric.update(prediction, real_img)
        elif model_type in ["basicvsr", "msvsr"]:
            lq = data['lq'].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            if len(predictor.get_output_names()) > 1:
                output_handle = predictor.get_output_handle(
                    predictor.get_output_names()[-1])
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            _, t, _, _, _ = prediction.shape

            out_img = []
            gt_img = []
            for ti in range(t):
                out_tensor = prediction[0, ti]
                gt_tensor = data['gt'][0, ti]
                out_img.append(tensor2img(out_tensor, (0., 1.)))
                gt_img.append(tensor2img(gt_tensor, (0., 1.)))

            image_numpy = tensor2img(prediction[0], min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, model_type, "{}.png".format(i)))

            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(out_img, gt_img, is_seq=True)
        elif model_type == "singan":
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction, min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "singan/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, "singan/metric.txt")
            for metric in metrics.values():
                metric.update(prediction, data['A'])
        elif model_type == 'gfpgan':
            input_handles[0].copy_from_cpu(data['lq'].numpy())
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction, min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "gfpgan/{}.png".format(i)))
        elif model_type == "swinir":
            lq = data[1].numpy()
            _, _, h_old, w_old = lq.shape
            window_size = 8
            tile = 128
            tile_overlap = 32
            # after feed data to model, shape of feature map is change
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            lq = np.concatenate([lq, np.flip(lq, 2)],
                                axis=2)[:, :, :h_old + h_pad, :]
            lq = np.concatenate([lq, np.flip(lq, 3)],
                                axis=3)[:, :, :, :w_old + w_pad]
            lq = lq.astype("float32")

            b, c, h, w = lq.shape
            tile = min(tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            sf = 1  # scale
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = np.zeros([b, c, h * sf, w * sf], dtype=np.float32)
            W = np.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    input_handles[0].copy_from_cpu(in_patch)
                    predictor.run()
                    out_patch = output_handle.copy_to_cpu()
                    out_patch_mask = np.ones_like(out_patch)

                    E[..., h_idx * sf:(h_idx + tile) * sf,
                      w_idx * sf:(w_idx + tile) * sf] += out_patch
                    W[..., h_idx * sf:(h_idx + tile) * sf,
                      w_idx * sf:(w_idx + tile) * sf] += out_patch_mask

            output = np.true_divide(E, W)
            prediction = output[..., :h_old * sf, :w_old * sf]

            prediction = paddle.to_tensor(prediction)
            target = tensor2img(data[0], (0., 1.))
            prediction = tensor2img(prediction, (0., 1.))

            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(prediction, target)

            lq = tensor2img(data[1], (0., 1.))

            sample_result = np.concatenate((lq, prediction, target), 1)
            sample = cv2.cvtColor(sample_result, cv2.COLOR_RGB2BGR)
            file_name = os.path.join(args.output_path, model_type,
                                     "{}.png".format(i))
            cv2.imwrite(file_name, sample)
        elif model_type == "invdn":
            noisy = data[0].numpy()
            noise_channel = 3 * 4**(cfg.model.generator.down_num) - 3
            input_handles[0].copy_from_cpu(noisy)
            input_handles[1].copy_from_cpu(
                np.random.randn(noisy.shape[0], noise_channel, noisy.shape[2],
                                noisy.shape[3]).astype(np.float32))
            predictor.run()
            output_handles = [
                predictor.get_output_handle(name)
                for name in predictor.get_output_names()
            ]
            prediction = output_handles[0].copy_to_cpu()
            prediction = paddle.to_tensor(prediction[0])
            image_numpy = tensor2img(prediction, min_max)
            gt_numpy = tensor2img(data[1], min_max)
            save_image(image_numpy,
                       os.path.join(args.output_path, "invdn/{}.png".format(i)))
            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(image_numpy, gt_numpy)
            break

        elif model_type == "nafnet":
            lq = data[1].numpy()
            input_handles[0].copy_from_cpu(lq)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            target = tensor2img(data[0], (0., 1.))
            prediction = tensor2img(prediction, (0., 1.))

            metric_file = os.path.join(args.output_path, model_type,
                                       "metric.txt")
            for metric in metrics.values():
                metric.update(prediction, target)

            lq = tensor2img(data[1], (0., 1.))

            sample_result = np.concatenate((lq, prediction, target), 1)
            sample = cv2.cvtColor(sample_result, cv2.COLOR_RGB2BGR)
            file_name = os.path.join(args.output_path, model_type,
                                     "{}.png".format(i))
            cv2.imwrite(file_name, sample)
        elif model_type == 'aotgan':
            input_data = paddle.concat((data['img'], data['mask']),
                                       axis=1).numpy()
            input_handles[0].copy_from_cpu(input_data)
            predictor.run()
            prediction = output_handle.copy_to_cpu()
            prediction = paddle.to_tensor(prediction)
            image_numpy = tensor2img(prediction, min_max)
            save_image(
                image_numpy,
                os.path.join(args.output_path, "aotgan/{}.png".format(i)))

    if metrics:
        log_file = open(metric_file, 'a')
        for metric_name, metric in metrics.items():
            loss_string = "Metric {}: {:.4f}".format(metric_name,
                                                     metric.accumulate())
            print(loss_string, file=log_file)
        log_file.close()


if __name__ == '__main__':
    main()
