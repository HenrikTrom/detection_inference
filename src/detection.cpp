#include "detection.hpp"

namespace detection_inference{

void preprocess(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &input_batch, std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
    const nvinfer1::Dims3 &inputDim
){
    inputs.clear();

    std::vector<cv::cuda::GpuMat> input;
    for (size_t j = 0; j < BATCH_SIZE; ++j) { // For each element we want to add to the batch...
        // You can choose to resize by scaling, adding padding, or a combination
        // of the two in order to maintain the aspect ratio You can use the
        // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
        // maintain the aspect ratio (adds padding where necessary to achieve
        // this).
        cv::cuda::GpuMat resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(
            input_batch.at(j), inputDim.d[1], inputDim.d[2]
        );
        // You could also perform a resize operation without maintaining aspect
        // ratio with the use of padding by using the following instead:
        //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
        //            inputDim.d[1])); // TRT dims are (height, width) whereas
        //            OpenCV is (width, height)
        input.emplace_back(std::move(resized));
    }
    inputs.emplace_back(std::move(input));
}


// TODO: chunk feature vector into pieces
void postprocess_yolo(
    std::vector<float> &featureVector, std::vector<cv::Rect> &bboxes_out, 
    std::vector<float> &scores_out, std::vector<int> &labels_out, 
    const float &probabilityThreshold, const float &nmsThreshold,
    const size_t &numClasses, const nvinfer1::Dims &outputDim,
    const float &m_ratio_x,const float &m_ratio_y, 
    const uint16_t &input_width, const uint16_t &input_height
){
    const auto &numChannels = outputDim.d[1];
    const auto &numAnchors = outputDim.d[2];
    // std::cout<<"numAnchors: "<<numAnchors<<std::endl;
    // std::cout<<"numChannels: "<<numChannels<<std::endl;

    std::vector<cv::Rect> _bboxes;
    std::vector<float>    _scores;
    std::vector<int>      _labels;
    std::vector<int>      _indices;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    float x, y, w, h;
    float x0, y0, x1, y1;
    float *rowPtr;
    float *bboxesPtr;
    float *scoresPtr;
    float *maxSPtr;
    float score;
    // std::cout<<numAnchors<<std::endl;
    for (int i = 0; i < numAnchors; i++)
    {
        rowPtr = output.row(i).ptr<float>();
        bboxesPtr = rowPtr;
        scoresPtr = rowPtr + 4;
        maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        score = *maxSPtr;
        if (score > probabilityThreshold)
        {
            x = *bboxesPtr++;
            y = *bboxesPtr++;
            w = *bboxesPtr++;
            h = *bboxesPtr;

            x0 = std::clamp((x - 0.5f * w) * m_ratio_x, 0.f, (float) input_width);
            x1 = std::clamp((x + 0.5f * w) * m_ratio_x, 0.f, (float) input_width);
            y0 = std::clamp((y - 0.5f * h) * m_ratio_y, 0.f, (float) input_height);
            y1 = std::clamp((y + 0.5f * h) * m_ratio_y, 0.f, (float) input_height);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            _bboxes.push_back(bbox);
            _labels.push_back(label);
            _scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(_bboxes, _scores, _labels, probabilityThreshold, nmsThreshold, _indices);

    // Choose the top k detections
    bboxes_out.clear();
    scores_out.clear();
    labels_out.clear();
    for (auto &chosenIdx : _indices)
    {
        bboxes_out.push_back(_bboxes[chosenIdx]);
        scores_out.push_back(_scores[chosenIdx]);
        labels_out.push_back(_labels[chosenIdx]);
    }
}

} // namespace detection_inference