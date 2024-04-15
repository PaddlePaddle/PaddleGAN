//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <sstream>
// for setprecision
#include <iomanip>
#include <chrono>
#include "include/vsr.h"

using namespace paddle_infer;

namespace PaddleGAN {

// Load Model and create model predictor
void VSR::LoadModel(const std::string& model_dir,
                               const int batch_size,
                               const std::string& run_mode) {
  paddle_infer::Config config;
  std::string prog_file = model_dir + ".pdmodel";
  std::string params_file = model_dir + ".pdiparams";
  config.SetModel(prog_file, params_file);
  if (this->device_ == "GPU") {
    config.EnableUseGpu(200, this->gpu_id_);
    config.SwitchIrOptim(true);
    // use tensorrt
    if (run_mode != "fluid") {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (run_mode == "trt_fp32") {
        precision = paddle_infer::Config::Precision::kFloat32;
      }
      else if (run_mode == "trt_fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      else if (run_mode == "trt_int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      } else {
          printf("run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'");
      }
      // set tensorrt
      config.EnableTensorRtEngine(
          1 << 30,
          batch_size,
          this->min_subgraph_size_,
          precision,
          false,
          this->trt_calib_mode_);

      // set use dynamic shape
      if (this->use_dynamic_shape_) {
        // set DynamicShsape for image tensor
        const std::vector<int> min_input_shape = {1, 3, this->trt_min_shape_, this->trt_min_shape_};
        const std::vector<int> max_input_shape = {1, 3, this->trt_max_shape_, this->trt_max_shape_};
        const std::vector<int> opt_input_shape = {1, 3, this->trt_opt_shape_, this->trt_opt_shape_};
        const std::map<std::string, std::vector<int>> map_min_input_shape = {{"image", min_input_shape}};
        const std::map<std::string, std::vector<int>> map_max_input_shape = {{"image", max_input_shape}};
        const std::map<std::string, std::vector<int>> map_opt_input_shape = {{"image", opt_input_shape}};

        config.SetTRTDynamicShapeInfo(map_min_input_shape,
                                      map_max_input_shape,
                                      map_opt_input_shape);
        std::cout << "TensorRT dynamic shape enabled" << std::endl;
      }
    }

  } else if (this->device_ == "XPU"){
    config.EnableXpu(10*1024*1024);
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchIrOptim(true);
  config.DisableGlogInfo();
  // Memory optimization
  config.EnableMemoryOptim();
  predictor_ = std::move(CreatePredictor(config));
}

void VSR::Preprocess(const cv::Mat& ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  preprocessor_.Run(&im, &inputs_);
}


void VSR::Predict(const std::vector<cv::Mat> imgs,
      const int warmup,
      const int repeats,
      std::vector<cv::Mat>* result,
      std::vector<double>* times) {
  auto preprocess_start = std::chrono::steady_clock::now();
  int frames_num = imgs.size();
  // in_data_batch
  std::vector<float> in_data_all;
  std::vector<float> im_shape_all(frames_num * 2);
  std::vector<float> scale_factor_all(frames_num * 2);
  std::vector<const float *> output_data_list_;
  std::vector<int> out_bbox_num_data_;
  
  // Preprocess image
  for (int bs_idx = 0; bs_idx < frames_num; bs_idx++) {
    cv::Mat im = imgs.at(bs_idx);    
    Preprocess(im);
    im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
    im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

    scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
    scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

    // TODO: reduce cost time
    in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());
  }
  auto preprocess_end = std::chrono::steady_clock::now();
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputHandle(tensor_name);
    int rh = inputs_.in_net_shape_[0];
    int rw = inputs_.in_net_shape_[1];
    in_tensor->Reshape({1, frames_num, 3, rw, rh});
    in_tensor->CopyFromCpu(in_data_all.data());
  }
  // warmup
  for (int i = 0; i < warmup; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    for (int j = 0; j < output_names.size(); j++) {
      auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
      std::vector<int> output_shape = output_tensor->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 
                            1, std::multiplies<int>());
      std::vector<float> out_data;
      out_data.resize(out_num);
      output_tensor->CopyToCpu(out_data.data());
      
    }
  }

  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    for (int j = 0; j < output_names.size(); j++) {
      auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
      std::vector<int> output_shape = output_tensor->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 
                            1, std::multiplies<int>());
      std::vector<float> out_data;
      out_data.resize(out_num);
      output_tensor->CopyToCpu(out_data.data());
      cv::Mat res = cv::Mat::zeros(output_shape[3], output_shape[4], CV_32FC3);
      int pix_num = output_shape[3] * output_shape[4];
      int frame_pix_num = pix_num * 3;
      for (int frame = 0; frame < output_shape[1]; frame++) {
        int index = 0;
        for (int h = 0; h < output_shape[3]; ++h) {
            for (int w = 0; w < output_shape[4]; ++w) {
              res.at<cv::Vec3f>(h, w) = {out_data[2*pix_num+index+frame_pix_num*frame], out_data[pix_num+index+frame_pix_num*frame], out_data[index+frame_pix_num*frame]}; // R,G,B
              index+=1;
          }
        }
        result->push_back(res);
        }
    }
  }
  auto inference_end = std::chrono::steady_clock::now();
  
  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() / repeats * 1000));

}

}  // namespace PaddleGAN
