#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <math.h>
#include <numeric>

#include <include/process_op.h>

// RGB -> CHW
void Permute::Run(const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int ch = im->channels();
  for (int i = 0; i < ch; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void Normalize::Run(cv::Mat *im, const std::vector<float> &mean,
                    const std::vector<float> &scale, const bool is_scale) {
  double e = 1.0;
  if (is_scale) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);

  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) * scale[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) * scale[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) * scale[2];
    }
  }
}

