#include "utils.hpp"

namespace detection_inference{

void load_image_data(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &PreProcessIn, std::array<cv::Mat, BATCH_SIZE> &cpuImgs, 
    std::array<std::string, BATCH_SIZE> fnames, std::string resources
) {
    for (uint8_t i =0; i<BATCH_SIZE; i++){
        std::string inputImage = resources+fnames.at(i)+".jpg";
        cpuImgs.at(i) = cv::imread(inputImage);
        if (cpuImgs.at(i).empty()){
            const std::string msg = "Unable to read image at path: " + inputImage;
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
        PreProcessIn.at(i).upload(cpuImgs.at(i));
        cv::cuda::cvtColor(PreProcessIn.at(i), PreProcessIn.at(i), cv::COLOR_BGR2RGB);
    }
}

void feeder_thread_benchmark(DetectionModule *stage, std::array<cv::cuda::GpuMat, BATCH_SIZE> PreProcessIn) {
    cpp_utils::ProgressBar progressBar(MAX_ITER);
    progressBar.update(0);
    for (std::size_t i = 1; i <= MAX_ITER; i++)
    {
        if (stage->GetInFIFOSize() < cpp_utils::MAXINFIFOSIZE)
        {
            std::array<cv::cuda::GpuMat, BATCH_SIZE> tmp = PreProcessIn;
            stage->InPost(tmp);
            progressBar.update(i);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(MAX_INFERENCE_SLEEP_MS));
    }
    progressBar.finish();
}

void feeder_thread_video(
    DetectionModule *stage, 
    cpp_utils::SyncVideoIterator<BATCH_SIZE> *video_iter
) {
    const std::size_t n_frames = video_iter->get_framecount();
    cpp_utils::ProgressBar progressBar(n_frames);
    progressBar.update(0);
    for (std::size_t i = 1; i <= n_frames; i++)
    {
        if (stage->GetInFIFOSize() < cpp_utils::MAXINFIFOSIZE)
        {
            std::array<cv::cuda::GpuMat, BATCH_SIZE> PreProcessIn;
            std::array<cv::Mat, BATCH_SIZE> tmp;
            video_iter->get_next(tmp);
            for(uint16_t cidx = 0; cidx<BATCH_SIZE; cidx++) {
                PreProcessIn.at(cidx).upload(tmp.at(cidx));
                cv::cuda::cvtColor(PreProcessIn.at(cidx), PreProcessIn.at(cidx), cv::COLOR_BGR2RGB);
            }
            stage->InPost(PreProcessIn);
            progressBar.update(i);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(MAX_INFERENCE_SLEEP_MS));
    }
    progressBar.finish();

}

}// namespace detection_inference