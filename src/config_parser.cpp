#include "config_parser.hpp"

namespace detection_inference{

bool load_config(const std::string &cfg_path, config_detection &cfg){
    std::string schemepath = std::string(CONFIG_DIR)+"/detection_config.scheme.json"; // compare with scheme
    rapidjson::Document doc;
    if(!cpp_utils::load_json_with_schema(cfg_path, schemepath, DOC_BUFFER, doc)){
        const std::string msg = "Unable to load pipeline config: " + cfg_path;
        spdlog::error(msg);
        throw std::runtime_error(msg);
        return false;
    }
    // detection
    cfg.onnxModelPath = doc["onnxModelPath"].GetString();
    cfg.trtModelPath = doc["trtModelPath"].GetString();
    cfg.engineFileDir = doc["engineFileDir"].GetString();
    cfg.modelInputWidth = (uint32_t) doc["modelInputWidth"].GetUint();
    cfg.input_width = (uint16_t) doc["input_width"].GetUint();
    cfg.input_height = (uint16_t) doc["input_height"].GetUint();
    cfg.precision = doc["precision"].GetString();
    cfg.probabilityThreshold = doc["probabilityThreshold"].GetFloat();
    cfg.nmsThreshold = doc["nmsThreshold"].GetFloat();
    cfg.topK = doc["topK"].GetInt();
    cfg.calibrationBatchSize = (uint16_t) doc["calibrationBatchSize"].GetUint();

    cfg.precision_e = stringToPrecision(cfg.precision);
    const rapidjson::Value& classNames = doc["class_names"];
    for (rapidjson::SizeType i = 0; i < classNames.Size(); i++) {
        cfg.classNames.push_back(classNames[i].GetString());
    }
    cfg.numClasses = (uint16_t) cfg.classNames.size();
    return true;
}

Precision stringToPrecision(const std::string &precisionStr)
{
    if (precisionStr == "FP32")
    {
        return Precision::FP32;
    }
    else if (precisionStr == "FP16")
    {
        return Precision::FP16;
    }
    else if (precisionStr == "INT8")
    {
        return Precision::INT8;
    }
    else
    {
        const std::string msg = "Unknown precision: " + precisionStr;
        spdlog::error(msg);
        throw std::invalid_argument(msg);
    }
}

bool load_cfg_engine(const std::string cfg_path, config_detection &cfg, std::unique_ptr<Engine<float>> &engine){
    
    load_config(cfg_path, cfg);
    Options options;
    
    options.precision = stringToPrecision(cfg.precision);

    options.calibrationDataDirectoryPath = "";
    options.optBatchSize = BATCH_SIZE;
    options.maxBatchSize = BATCH_SIZE;
    options.calibrationBatchSize = cfg.calibrationBatchSize;

    options.engineFileDir = cfg.engineFileDir;
    options.optInputWidth = cfg.modelInputWidth;
    options.minInputWidth = cfg.modelInputWidth;
    options.maxInputWidth = cfg.modelInputWidth;

    engine.reset(new Engine<float>{options});

    const std::array<float, 3> subVals{0.f, 0.f, 0.f};
    const std::array<float, 3> divVals{1.f, 1.f, 1.f};
    const bool normalize = true;

    if (!cfg.onnxModelPath.empty()) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = engine->buildLoadNetwork(cfg.onnxModelPath, subVals, divVals, normalize);
        if (!succ) {
            const std::string msg = "Unable to build or load TensorRT engine.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    } else {
        // Load the TensorRT engine file directly
        bool succ = engine->loadNetwork(cfg.trtModelPath, subVals, divVals, normalize);
        if (!succ) {
            const std::string msg = "Unable to load TensorRT engine.";
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
    }

    cfg.inputDims = engine->getInputDims();
    cfg.outputDims = engine->getOutputDims();

    cfg.m_ratio_x = 1.f / (
        cfg.inputDims[0].d[1] / static_cast<float>(cfg.input_width)
    );
    cfg.m_ratio_y = 1.f / (
        cfg.inputDims[0].d[2] / static_cast<float>(cfg.input_height)
    );
    return true;
}

} // namespace detection_inference