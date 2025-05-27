#pragma once
#include "tensorrt-cpp-api/engine.h"
#include "cpp_utils/jsontools.h"
#include "config.h"


namespace detection_inference{

constexpr int DOC_BUFFER = 65536;

struct config_detection{

    std::string onnxModelPath = "";
    std::string trtModelPath = "";
    std::string engineFileDir = "";
    uint32_t modelInputWidth = 0;

    uint16_t input_height = 0;
    uint16_t input_width = 0;
    uint16_t input_depth = 0;
    std::string precision = "";
    Precision precision_e;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.;
    // Max number of detected objects to return
    int topK = 0;
    uint16_t calibrationBatchSize = 0;
    std::size_t numClasses = 4; 
    std::vector<std::string> classNames;
    std::vector<nvinfer1::Dims3> inputDims;
    std::vector<nvinfer1::Dims> outputDims;
    float m_ratio_x = 0.;
    float m_ratio_y = 0.;

};

Precision stringToPrecision(const std::string &precisionStr);

/** 
 * @brief Load the configuration from a JSON file
 * @param [in]  cfg_path Path to the JSON configuration file
 * @param [out] cfg Reference to the config_detection structure to be filled
 * @return true if the configuration was loaded successfully, false otherwise
**/
bool load_config(const std::string &cfg_path, config_detection &cfg);

/**
 * @brief Load the TensorRT engine from a file
 * @param [in]  cfg_path Path to the JSON configuration file
 * @param [out] cfg Reference to the config_detection structure to be filled
 * @param [out] engine Pointer to the TensorRT engine
 * @return true if the engine was loaded successfully, false otherwise
**/
bool load_cfg_engine(
    const std::string cfg_path, config_detection &cfg, 
    std::unique_ptr<Engine<float>> &engine
);
   
}