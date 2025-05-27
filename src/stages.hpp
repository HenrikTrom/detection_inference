#pragma once
#include "cpp_utils/StageBase.h"
#include "detection.hpp"
#include "config_parser.hpp"
#include "data.hpp"

namespace detection_inference{

// NN
class NNStage : public cpp_utils::StageBase<
    std::vector<std::vector<cv::cuda::GpuMat>>, std::vector<std::vector<std::vector<float>>>
>
{
private:
    // inputDims;
    const config_detection &cfg;
    std::unique_ptr<Engine<float>> engine;
    bool ProcessFunction(
        std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
        std::vector<std::vector<std::vector<float>>> &EncodedOutputStream
    );
    double dt;
    std::size_t n_samples;

public:
    NNStage(config_detection &cfg, std::unique_ptr<Engine<float>> engine);
    ~NNStage();
    void Terminate(void);
};

// PREPROCESS
class PreProcessStage : public cpp_utils::StageBase<
    std::array<cv::cuda::GpuMat, BATCH_SIZE>, std::vector<std::vector<cv::cuda::GpuMat>>
>
{
private:
    const config_detection &cfg;
    bool ProcessFunction(
        std::array<cv::cuda::GpuMat, BATCH_SIZE> &inputs, 
        std::vector<std::vector<cv::cuda::GpuMat>> &outputs
    );

public:
    PreProcessStage(const config_detection &cfg);
    ~PreProcessStage();
    void Terminate(void);
};
// POSTPROCESS
class PostProcessStage : public cpp_utils::StageBase<
    std::vector<std::vector<std::vector<float>>>, output_postprocess
>
{
private:
    const config_detection &cfg;
    bool ProcessFunction(
        std::vector<std::vector<std::vector<float>>> &inputs, 
        output_postprocess &outputs
    );


public:
    PostProcessStage(const config_detection &cfg);
    ~PostProcessStage();
    void Terminate(void);
};

// Detection Stage 
class DetectionModule
{
private:
    // stages
    std::unique_ptr<PreProcessStage> preprocess_stage;
    std::unique_ptr<NNStage> nn_stage;
    std::unique_ptr<PostProcessStage> postprocess_stage;
    // Threads
    void ThreadPreprocessNN();
    void ThreadNNPostProcess();

    std::unique_ptr<std::thread> ThreadHandlePreprocessNN;
    std::unique_ptr<std::thread> ThreadHandleNNProstprocess;

    bool ShouldClose = false;
    bool IsReady_flag = false;

public:
    DetectionModule(config_detection &cfg, std::unique_ptr<Engine<float>> engine);
    ~DetectionModule();
    bool Get(output_postprocess &DataOut);
    uint16_t GetInFIFOSize(void);
    uint16_t GetOutFIFOSize(void);
    void InPost(std::array<cv::cuda::GpuMat, BATCH_SIZE> &PreProcessIn);
    bool IsReady(void);
    void Terminate(void);
};

} // namespace detection_inference