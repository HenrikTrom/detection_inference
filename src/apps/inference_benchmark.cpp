#include "../utils.hpp"

using namespace detection_inference;

int main(int argc, char *argv[]) {
    // Load inference data
    std::array<cv::cuda::GpuMat, BATCH_SIZE> PreProcessIn;
    std::array<cv::Mat, BATCH_SIZE> cpuImgs;
    std::string resources = std::string(CONFIG_DIR)+"/../inputs/";

    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".jpg"
    );

    load_image_data(PreProcessIn, cpuImgs, fnames, resources);

    // Load config, engine and module
    config_detection cfg;
    std::unique_ptr<Engine<float>> engine;
    load_cfg_engine(
        std::string(CONFIG_DIR)+"/yolo_default_config.json", 
        cfg,
        engine
    );

    DetectionModule detection_module(cfg, std::move(engine));

    while (!detection_module.IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::thread feeder{feeder_thread_benchmark, &detection_module, PreProcessIn};
    spdlog::info("Feeder thread started");
    spdlog::info("Inferencing samples...");

    // Run inference
    std::size_t count{0};
    output_postprocess PostProcessOut;
    while (count != MAX_ITER){
        if(detection_module.Get(PostProcessOut)){
            count++;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    if (feeder.joinable()){
        feeder.join();
    }
    spdlog::info("Joined feeder thread");
    detection_module.Terminate();

    for (std::size_t i = 0; i < PostProcessOut.bboxes.size(); i++) { // for each img in batch
        for (std::size_t j = 0; j < PostProcessOut.bboxes.at(i).size(); j++) {
            cv::Rect &bbox = PostProcessOut.bboxes.at(i).at(j);
            cv::rectangle(cpuImgs.at(i), bbox, cv::Scalar(0, 255, 0), 2); // Green color with thickness 1
            cv::putText(
                cpuImgs.at(i), std::to_string(PostProcessOut.labels.at(i).at(j)), cv::Point(bbox.x, bbox.y), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2
            );
        }
        std::string filename = std::string(CONFIG_DIR)+"/../outputs/"+fnames.at(i)+"_out.jpg";
        spdlog::info("Saving {}", filename);
        cv::imwrite(filename, cpuImgs.at(i));
    }

    return 0;
}
