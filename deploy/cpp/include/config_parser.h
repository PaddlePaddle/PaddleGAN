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

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "yaml-cpp/yaml.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace PaddleGAN {

// Inference model configuration parser
class ConfigPaser {
 public:
  ConfigPaser() {}

  ~ConfigPaser() {}

  bool load_config(const std::string& config_dir) {
    // Load as a YAML::Node
    YAML::Node config;
    config = YAML::LoadFile(config_dir);

    // Get runtime mode : fluid, trt_fp16, trt_fp32
    if (config["mode"].IsDefined()) {
      mode_ = config["mode"].as<std::string>();
    } else {
      std::cerr << "Please set mode, "
                << "support value : fluid/trt_fp16/trt_fp32."
                << std::endl;
      return false;
    }


    // Get min_subgraph_size for tensorrt
    if (config["min_subgraph_size"].IsDefined()) {
      min_subgraph_size_ = config["min_subgraph_size"].as<int>();
    } else {
      std::cerr << "Please set min_subgraph_size." << std::endl;
      return false;
    }
    
    // Get Preprocess for preprocessing
    if (config["Preprocess"].IsDefined()) {
      preprocess_info_ = config["Preprocess"];
    } else {
      std::cerr << "Please set Preprocess." << std::endl;
      return false;
    }
    

    // Get use_dynamic_shape for TensorRT
    if (config["use_dynamic_shape"].IsDefined()) {
      use_dynamic_shape_ = config["use_dynamic_shape"].as<bool>();
    } else {
      std::cerr << "Please set use_dynamic_shape." << std::endl;
      return false;
    }
  }

  std::string mode_;
  int min_subgraph_size_;
  YAML::Node preprocess_info_;
  bool use_dynamic_shape_;
};

}  // namespace PaddleGAN

