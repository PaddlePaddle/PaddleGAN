#include <glog/logging.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <algorithm>

#include "include/vsr.h"
#include <gflags/gflags.h>


DEFINE_string(model_path, "", "Path of inference model");
DEFINE_string(param_path, "", "Path of inference param");
DEFINE_int32(frame_num, 2, "frame_num");
DEFINE_string(video_path, "", "Path of input video, `video_file` or `camera_id` has a highest priority.");
DEFINE_string(device, "CPU", "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");


void main_predict(const std::string& video_path,
                  PaddleGAN::VSR* vsr,
                  const std::string& output_dir = "output") {

  // Open video
  cv::VideoCapture capture;
  std::string video_out_name = "output.mp4";
  capture.open(video_path.c_str());
  if (!capture.isOpened()) {
    printf("can not open video : %s\n", video_path.c_str());
    return;
  }

  // Get Video info :fps, frame count
  int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
  int video_frame_count = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_COUNT));
  // Set fixed size for output frame, only for msvsr model
  int out_width = 1280;
  int out_height = 720;
  printf("fps: %d, frame_count: %d\n", video_fps, video_frame_count);

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path(output_dir);
  video_out_path += video_out_name;

  video_out.open(video_out_path,
                 0x00000021,
                 video_fps,
                 cv::Size(out_width, out_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  bool reach_end = false;
  while (capture.isOpened()) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < FLAGS_frame_num; i++) {
        capture.read(frame);
        if (!frame.empty()) {
          imgs.push_back(frame);
        }
        else {
          reach_end = true;
        }
    }
    if (reach_end) {
      break;
    }

    std::vector<cv::Mat> result;
    vsr->Run(imgs, &result);
    for (auto& item : result) {
      cv::Mat temp = cv::Mat::zeros(item.size(), CV_8UC3);
      item.convertTo(temp, CV_8UC3, 255);
      video_out.write(temp);
      printf("Processing frame: %d\n", frame_id);
      // auto im_nm = std::to_string(frame_id) + "test.jpg";
      // cv::imwrite(FLAGS_output_dir + im_nm, temp);
      frame_id += 1;
    } 
  }
  printf("inference finished, output video saved at %s", video_out_path.c_str());
  capture.release();
  video_out.release();
} 

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Load model and create a vsr
  PaddleGAN::VSR vsr(FLAGS_model_path, FLAGS_param_path, FLAGS_device, FLAGS_gpu_id, FLAGS_use_mkldnn,
                        FLAGS_cpu_threads);
                        
  // Do inference on input video or image
  main_predict(FLAGS_video_path, &vsr, FLAGS_output_dir);
  return 0;
}