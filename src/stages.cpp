#include "stages.hpp"

namespace detection_inference{

// DETECTION STAGE
NNStage::NNStage(config_detection &cfg, std::unique_ptr<Engine<float>> engine) : 
    cfg(cfg), engine(std::move(engine)){
    
    this->ThreadHandle.reset(new std::thread(&NNStage::ThreadFunction, this));

}

NNStage::~NNStage(){}

bool NNStage::ProcessFunction(
    std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
    std::vector<std::vector<std::vector<float>>> &outputs
)
{
    #ifdef USE_DEBUG_TIME_LOGGING
        this->t1 = std::chrono::steady_clock::now();
    #endif
    this->engine->runInference(inputs, outputs);

    #ifdef USE_DEBUG_TIME_LOGGING
        this->t2 = std::chrono::steady_clock::now();
        this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
        this->n_iterations++;
        this->total_dt += this->duration;
    #endif
    return true;
    
}   

void NNStage::Terminate(void)
{
    this->ShouldClose = true;
    this->ThreadHandle->join();
    #ifdef USE_DEBUG_TIME_LOGGING
        spdlog::info(
            "Average Detection NN Inference Time: {} milliseconds over {} samples", 
            this->total_dt.count()/this->n_iterations, 
            this->n_iterations
        );
    #endif
}


// PREPROCESS STAGE
PreProcessStage::PreProcessStage(const config_detection &cfg) : cfg(cfg){    
    this->ThreadHandle.reset(new std::thread(&PreProcessStage::ThreadFunction, this));
}

PreProcessStage::~PreProcessStage(){}

bool PreProcessStage::ProcessFunction(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &inputs, 
    std::vector<std::vector<cv::cuda::GpuMat>> &outputs
)
{
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t1 = std::chrono::steady_clock::now();
    #endif
    preprocess(inputs, outputs, this->cfg.inputDims[0]);
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t2 = std::chrono::steady_clock::now();
        this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
        this->n_iterations++;
        this->total_dt += this->duration;
    #endif
    return true;
}


void PreProcessStage::Terminate(void)
{
    this->ShouldClose = true;
    this->ThreadHandle->join();
    #ifdef USE_DEBUG_TIME_LOGGING
        spdlog::info(
            "Average Detection PreProcess Time: {} milliseconds over {} samples", 
            this->total_dt.count()/this->n_iterations, 
            this->n_iterations
        );
    #endif

}

// POSTPROCESS STAGE
PostProcessStage::PostProcessStage(const config_detection &cfg) : cfg(cfg){
    this->ThreadHandle.reset(new std::thread(&PostProcessStage::ThreadFunction, this));
}

PostProcessStage::~PostProcessStage(){}

bool PostProcessStage::ProcessFunction(
    std::vector<std::vector<std::vector<float>>> &inputs,
    output_postprocess &outputs
)
{
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t1 = std::chrono::steady_clock::now();
    #endif
    for (size_t i = 0; i < BATCH_SIZE; i++)
    {
        postprocess_yolo(
            inputs.at(i).at(0), 
            outputs.bboxes.at(i), outputs.scores.at(i), outputs.labels.at(i),
            this->cfg.probabilityThreshold, this->cfg.nmsThreshold, 
            this->cfg.numClasses, this->cfg.outputDims[0], 
            this->cfg.m_ratio_x, this->cfg.m_ratio_x,  
            this->cfg.input_width, this->cfg.input_height
            
        );
    }
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t2 = std::chrono::steady_clock::now();
        this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
        this->n_iterations++;
        this->total_dt += this->duration;
    #endif
    return true;
}



void PostProcessStage::Terminate(void)
{
    this->ShouldClose = true;
    this->ThreadHandle->join();
    #ifdef USE_DEBUG_TIME_LOGGING
        spdlog::info(
            "Average Detection PostProcess Time: {} milliseconds over {} samples", 
            this->total_dt.count()/this->n_iterations, 
            this->n_iterations
        );
    #endif
}

// DETECTION STAGE
DetectionModule::DetectionModule(config_detection &cfg, std::unique_ptr<Engine<float>> engine){
    this->nn_stage.reset(new NNStage(cfg, std::move(engine)));
    while (!nn_stage->IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    this->preprocess_stage.reset(new PreProcessStage(cfg));
    while (!preprocess_stage->IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    this->postprocess_stage.reset(new PostProcessStage(cfg));
    while (!postprocess_stage->IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    this->ThreadHandlePreprocessNN.reset(new std::thread(&DetectionModule::ThreadPreprocessNN, this)); 
    this->ThreadHandleNNProstprocess.reset(new std::thread(&DetectionModule::ThreadNNPostProcess, this)); 
    this->IsReady_flag=true;
}

DetectionModule::~DetectionModule(){}

bool DetectionModule::Get(output_postprocess &DataOut)
{
    if (this->postprocess_stage->GetOutFIFOSize()!=0)
    {
        return this->postprocess_stage->Get(DataOut);;
    }
    return false;
}

void DetectionModule::ThreadPreprocessNN(){
    while (!this->ShouldClose) {
        std::vector<std::vector<cv::cuda::GpuMat>> NNIn;
        if (this->preprocess_stage->Get(NNIn)){
            this->nn_stage->Post(NNIn);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void DetectionModule::ThreadNNPostProcess(){
    while (!this->ShouldClose) {
        std::vector<std::vector<std::vector<float>>> PostIn;
        if (this->nn_stage->Get(PostIn)){
            this->postprocess_stage->Post(PostIn);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}


uint16_t DetectionModule::GetInFIFOSize(void)
{
    return this->preprocess_stage->GetInFIFOSize();
}

uint16_t DetectionModule::GetOutFIFOSize(void)
{
    return this->postprocess_stage->GetOutFIFOSize();
}

void DetectionModule::InPost(std::array<cv::cuda::GpuMat, BATCH_SIZE> &PreProcessIn){
    this->preprocess_stage->Post(PreProcessIn);       
}

bool DetectionModule::IsReady(void)
{
    return this->IsReady_flag;
}

void DetectionModule::Terminate(void){
    this->ShouldClose=true;
    this->ThreadHandlePreprocessNN->join();
    this->ThreadHandleNNProstprocess->join();

    this->preprocess_stage->Terminate();
    this->nn_stage->Terminate();
    this->postprocess_stage->Terminate();
}

} // namespace detection_inference