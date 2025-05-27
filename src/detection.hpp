#pragma once
#include "config_parser.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <algorithm> 

namespace detection_inference{

void preprocess(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &input_batch, 
    std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
    const nvinfer1::Dims3 &inputDim
);

void postprocess_yolo(
    std::vector<float> &featureVector, std::vector<cv::Rect> &bboxes_out, 
    std::vector<float> &scores_out, std::vector<int> &labels_out, 
    const float &probabilityThreshold,const float &nmsThreshold,
    const size_t &numClasses,const nvinfer1::Dims &outputDim,
    const float &m_ratio_x, const float &m_ratio_y,
    const uint16_t &input_width,const uint16_t &input_height
);

} // namespace detection_inference