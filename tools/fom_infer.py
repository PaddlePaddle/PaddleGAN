import paddle.inference as paddle_infer
import argparse
import numpy as np
import cv2
import imageio
import time
from tqdm import tqdm
import paddle.fluid as fluid
import os
from functools import reduce
import paddle
from ppgan.utils.filesystem import makedirs
from pathlib import Path


def read_img(path):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # som images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_video(path):
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        print("Read driving video error!")
        pass
    reader.close()
    return driving_video, fps


def face_detection(img_ori, weight_path):
    config = paddle_infer.Config(os.path.join(weight_path, '__model__'),
                                 os.path.join(weight_path, '__params__'))
    config.disable_gpu()
    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = paddle_infer.create_predictor(config)

    img = img_ori.astype(np.float32)
    mean = np.array([123, 117, 104])[np.newaxis, np.newaxis, :]
    std = np.array([127.502231, 127.502231, 127.502231])[np.newaxis,
                                                         np.newaxis, :]
    img -= mean
    img /= std
    img = img[:, :, [2, 1, 0]]
    img = img[np.newaxis].transpose([0, 3, 1, 2])

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.copy_from_cpu(img)
    predictor.run()
    output_names = predictor.get_output_names()
    boxes_tensor = predictor.get_output_handle(output_names[0])
    np_boxes = boxes_tensor.copy_to_cpu()
    if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
        print('[WARNNING] No object detected.')
        exit()
    w, h = img.shape[2:]
    np_boxes[:, 2] *= h
    np_boxes[:, 3] *= w
    np_boxes[:, 4] *= h
    np_boxes[:, 5] *= w
    expect_boxes = (np_boxes[:, 1] > 0.5) & (np_boxes[:, 0] > -1)
    rect = np_boxes[expect_boxes, :][0][2:]
    bh = rect[3] - rect[1]
    bw = rect[2] - rect[0]
    cy = rect[1] + int(bh / 2)
    cx = rect[0] + int(bw / 2)
    margin = max(bh, bw)
    y1 = max(0, cy - margin)
    x1 = max(0, cx - int(0.8 * margin))
    y2 = min(h, cy + margin)
    x2 = min(w, cx + int(0.8 * margin))
    return int(y1), int(y2), int(x1), int(x2)


def main():
    args = parse_args()

    source_path = args.source_path
    driving_path = Path(args.driving_path)
    makedirs(args.output_path)
    if driving_path.is_dir():
        driving_paths = list(driving_path.iterdir())
    else:
        driving_paths = [driving_path]

    # 创建 config
    kp_detector_config = paddle_infer.Config(os.path.join(
        args.model_path, "kp_detector.pdmodel"),
        os.path.join(args.model_path, "kp_detector.pdiparams"))
    generator_config = paddle_infer.Config(os.path.join(
        args.model_path, "generator.pdmodel"),
        os.path.join(args.model_path, "generator.pdiparams"))
    if args.device == "gpu":
        kp_detector_config.enable_use_gpu(100, 0)
        generator_config.enable_use_gpu(100, 0)
    else:
        kp_detector_config.set_mkldnn_cache_capacity(10)
        kp_detector_config.enable_mkldnn()
        generator_config.set_mkldnn_cache_capacity(10)
        generator_config.enable_mkldnn()
        kp_detector_config.disable_gpu()
        kp_detector_config.set_cpu_math_library_num_threads(6)
        generator_config.disable_gpu()
        generator_config.set_cpu_math_library_num_threads(6)
    # 根据 config 创建 predictor
    kp_detector_predictor = paddle_infer.create_predictor(kp_detector_config)
    generator_predictor = paddle_infer.create_predictor(generator_config)
    test_loss = []
    for k in range(len(driving_paths)):
        driving_path = driving_paths[k]
        driving_video, fps = read_video(driving_path)
        driving_video = [
            cv2.resize(frame, (256, 256)) / 255.0 for frame in driving_video
        ]
        driving_len = len(driving_video)
        driving_video = np.array(driving_video).astype(np.float32).transpose(
            [0, 3, 1, 2])

        if source_path == None:
            source = driving_video[0:1]
        else:
            source_img = read_img(source_path)
            #Todo：add blazeface static model
            #left, right, up, bottom = face_detection(source_img, "/workspace/PaddleDetection/static/inference_model/blazeface/")
            source = source_img  #[left:right, up:bottom]
            source = cv2.resize(source, (256, 256)) / 255.0
            source = source[np.newaxis].astype(np.float32).transpose(
                [0, 3, 1, 2])

        # 获取输入的名称
        kp_detector_input_names = kp_detector_predictor.get_input_names()
        kp_detector_input_handle = kp_detector_predictor.get_input_handle(
            kp_detector_input_names[0])

        kp_detector_input_handle.reshape([args.batch_size, 3, 256, 256])
        kp_detector_input_handle.copy_from_cpu(source)
        kp_detector_predictor.run()
        kp_detector_output_names = kp_detector_predictor.get_output_names()
        kp_detector_output_handle = kp_detector_predictor.get_output_handle(
            kp_detector_output_names[0])
        source_j = kp_detector_output_handle.copy_to_cpu()
        kp_detector_output_handle = kp_detector_predictor.get_output_handle(
            kp_detector_output_names[1])
        source_v = kp_detector_output_handle.copy_to_cpu()

        kp_detector_input_handle.reshape([args.batch_size, 3, 256, 256])
        kp_detector_input_handle.copy_from_cpu(driving_video[0:1])
        kp_detector_predictor.run()
        kp_detector_output_names = kp_detector_predictor.get_output_names()
        kp_detector_output_handle = kp_detector_predictor.get_output_handle(
            kp_detector_output_names[0])
        driving_init_j = kp_detector_output_handle.copy_to_cpu()
        kp_detector_output_handle = kp_detector_predictor.get_output_handle(
            kp_detector_output_names[1])
        driving_init_v = kp_detector_output_handle.copy_to_cpu()
        start_time = time.time()
        results = []
        for i in tqdm(range(0, driving_len)):
            kp_detector_input_handle.copy_from_cpu(driving_video[i:i + 1])
            kp_detector_predictor.run()
            kp_detector_output_names = kp_detector_predictor.get_output_names()
            kp_detector_output_handle = kp_detector_predictor.get_output_handle(
                kp_detector_output_names[0])
            driving_j = kp_detector_output_handle.copy_to_cpu()
            kp_detector_output_handle = kp_detector_predictor.get_output_handle(
                kp_detector_output_names[1])
            driving_v = kp_detector_output_handle.copy_to_cpu()
            generator_inputs = [
                source, source_j, source_v, driving_j, driving_v,
                driving_init_j, driving_init_v
            ]
            generator_input_names = generator_predictor.get_input_names()
            for i in range(len(generator_input_names)):
                generator_input_handle = generator_predictor.get_input_handle(
                    generator_input_names[i])
                generator_input_handle.copy_from_cpu(generator_inputs[i])
            generator_predictor.run()
            generator_output_names = generator_predictor.get_output_names()
            generator_output_handle = generator_predictor.get_output_handle(
                generator_output_names[0])
            output_data = generator_output_handle.copy_to_cpu()
            loss = paddle.abs(paddle.to_tensor(output_data) -
                              paddle.to_tensor(driving_video[i])).mean().cpu().numpy()
            test_loss.append(loss)
            output_data = np.transpose(output_data, [0, 2, 3, 1])[0] * 255.0
            
            #Todo：add blazeface static model
            #frame = source_img.copy()
            #frame[left:right, up:bottom] = cv2.resize(output_data.astype(np.uint8), (bottom - up, right - left), cv2.INTER_AREA)
            results.append(output_data.astype(np.uint8))
        print(time.time() - start_time)
        imageio.mimsave(os.path.join(args.output_path,
                                     "result_" + str(k) + ".mp4"),
                        [frame for frame in results],
                        fps=fps)
    metric_file = os.path.join(args.output_path, "metric.txt")
    log_file = open(metric_file, 'a')
    loss_string = "Metric {}: {:.4f}".format(
                  "l1 loss", np.mean(test_loss))
    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model filename profix")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--source_path",
                        type=str,
                        default=None,
                        help="source_path")
    parser.add_argument("--driving_path",
                        type=str,
                        default=None,
                        help="driving_path")
    parser.add_argument("--output_path",
                        type=str,
                        default="infer_output/fom/",
                        help="output_path")
    parser.add_argument("--device", type=str, default="gpu", help="device")

    return parser.parse_args()


if __name__ == "__main__":
    main()
