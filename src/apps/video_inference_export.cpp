#include "../utils.hpp"

using namespace detection_inference;
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    if (argc != 2)
    {
        std::string error_msg = "Expected missing argument for video dir";
        spdlog::error(error_msg);
        return 1;
    }
    std::string folder = std::string(argv[1]);

    std::string resources = std::string(CONFIG_DIR)+"/../"+folder+"/";

    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".mp4"
    );
    cpp_utils::SyncVideoIterator<BATCH_SIZE> video_iter(resources, fnames);
    DetectionLogger logger(resources, fnames);

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
    std::thread feeder{feeder_thread_video, &detection_module, &video_iter};
    spdlog::info("Feeder thread started");
    spdlog::info("Inferencing samples...");
    
    std::size_t counter{0};
    std::size_t max_elements = video_iter.get_framecount();
    while (counter != (max_elements)){
        output_postprocess PostProcessOut;
        if(detection_module.Get(PostProcessOut)){
            logger.log(PostProcessOut);
            counter++;
        };
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    if (feeder.joinable()){
        feeder.join();
    }
    spdlog::info("Joined feeder thread");
    detection_module.Terminate();
    logger.write();

    return 0;
}
