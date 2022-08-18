#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <ctime>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "include/process_op.h"
#include "paddle_inference_api.h"

namespace PaddleGAN {

class VSR {
public:
    explicit VSR(const std::string& model_path,
                 const std::string& param_path,
                 const std::string& device,
                 const int& gpu_id,
                 const bool& use_mkldnn,
                 const int& cpu_threads) {

        this->device_ = device;
        this->gpu_id_ = gpu_id;
        this->use_mkldnn_ = use_mkldnn_;
        this->cpu_threads_ = cpu_threads;

        LoadModel(model_path, param_path);
    }

    // Load paddle inference model
    void LoadModel(const std::string& model_path, const std::string& param_path);

    // Run predictor
    void Run(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat>* result = nullptr);

private:
    std::shared_ptr<paddle_infer::Predictor> predictor_;

    std::string device_ = "GPU";
    int gpu_id_ = 0;
    bool use_mkldnn_ = false;
    int cpu_threads_ = 1;

    std::vector<float> mean_ = {0., 0., 0.};
    std::vector<float> scale_ = {1., 1., 1.};

    // pre/post-process
    Permute permute_op_;
    Normalize normalize_op_;
    std::vector<float> Preprocess(cv::Mat& img);
};

}
