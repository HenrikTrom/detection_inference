#pragma once

#include<vector>
#include<time.h>
#include <utility>
#include <opencv2/core/cuda.hpp>

namespace detection_inference{

struct output_postprocess {
    std::array<std::vector<cv::Rect>, BATCH_SIZE> bboxes;
    std::array<std::vector<float>, BATCH_SIZE> scores; 
    std::array<std::vector<int>, BATCH_SIZE> labels; 
};

} // namespace detection_inference