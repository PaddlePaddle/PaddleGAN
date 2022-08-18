#include "include/vsr.h"
#include <iostream>

namespace PaddleGAN {

// VSR load model and initialize predictor
void VSR::LoadModel(const std::string& model_path, 
                    const std::string& param_path) {
    paddle_infer::Config config;
    config.SetModel(model_path, param_path);
    if (this->device_ == "GPU") {
        config.EnableUseGpu(200, this->gpu_id_);
    }
    else {
        config.DisableGpu();
        if (this->use_mkldnn_) {
            config.EnableMKLDNN();
            // cache 10 for mkldnn to avoid memory leak; copy from paddleseg
            config.SetMkldnnCacheCapacity(10);
        }
        config.SetCpuMathLibraryNumThreads(this->cpu_threads_);
    }
    
    config.SwitchUseFeedFetchOps(false);
    config.SwitchIrOptim(true);
    config.EnableMemoryOptim();
    config.DisableGlogInfo();
    this->predictor_ = paddle_infer::CreatePredictor(config);
}

std::vector<float> VSR::Preprocess(cv::Mat& img) {
    cv::Mat new_img;
    img.copyTo(new_img);    
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // transform 1 image
    this->normalize_op_.Run(&new_img, this->mean_, this->scale_, true);
    std::vector<float> unroll(1 * 3 * new_img.rows * new_img.cols, 0.0f);
    this->permute_op_.Run(&new_img, unroll.data());
    return unroll;
}

void VSR::Run(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat>* result) {
    int frame_num = imgs.size();
    int rows = imgs[0].rows;
    int cols = imgs[0].cols;
    
    // Preprocess
    // initialize a fixed size unroll vector to store processed img
    std::vector<float> in_data_all;

    for (int i = 0; i < frame_num; i++) {
        cv::Mat im = imgs[i];
        std::vector<float> unroll = this->Preprocess(im);
        in_data_all.insert(in_data_all.end(), unroll.begin(), unroll.end());
    }

    // Set input
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({1, frame_num, 3, rows, cols});
    input_t->CopyFromCpu(in_data_all.data());

    // Run
    this->predictor_->Run();

    // Get output
    auto output_names = this->predictor_->GetOutputNames();
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<float> out_data;
    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());

    // group to image
    cv::Mat res = cv::Mat::zeros(output_shape[3], output_shape[4], CV_32FC3); // RGB image
    int pix_num = output_shape[3] * output_shape[4];
    int frame_pix_num = pix_num * 3;
    for (int frame = 0; frame < output_shape[1]; frame++) {
    int index = 0;
    for (int h = 0; h < output_shape[3]; ++h) {
        for (int w = 0; w < output_shape[4]; ++w) {
            res.at<cv::Vec3f>(h, w) = {out_data[2*pix_num+index+frame_pix_num*frame], out_data[pix_num+index+frame_pix_num*frame], out_data[index+frame_pix_num*frame]};
            index+=1;
        }
    }
    result->push_back(res);
    }
}

}