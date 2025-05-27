#pragma once
#include "stages.hpp"
#include "cpp_utils/clitools.h"
#include "cpp_utils/opencvtools.h"

namespace detection_inference{

constexpr size_t MAX_ITER = 1000; // benchmark samples
constexpr size_t MAX_INFERENCE_SLEEP_MS = 10; // Fastest benchmark inference time in milliseconds

void load_image_data(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &PreProcessIn, std::array<cv::Mat, BATCH_SIZE> &cpuImgs, 
    std::array<std::string, BATCH_SIZE> fnames, std::string resources
);

void feeder_thread_benchmark(DetectionModule *stage, std::array<cv::cuda::GpuMat, BATCH_SIZE> PreProcessIn);

void feeder_thread_video(
    DetectionModule *stage, 
    cpp_utils::SyncVideoIterator<BATCH_SIZE> *video_iter
);

class DetectionLogger
{
public:
DetectionLogger(const std::string &resources, std::array<std::string, BATCH_SIZE> fnames){
    this->resources = resources;
    this->fnames = fnames;
};
~DetectionLogger(){};
void log(output_postprocess PostProcessOut){
    this->log_q.push(PostProcessOut);
};
bool write(){
    std::array<rapidjson::Document, BATCH_SIZE> docs;
    for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        docs.at(i).SetObject();
    }
    int count{0};
    while (!this->log_q.empty()){
        count++;
        output_postprocess PostProcessOut = this->log_q.front();
        this->log_q.pop();
        for (std::size_t i = 0; i < BATCH_SIZE; i++) {
            rapidjson::Value img_obj(rapidjson::kObjectType);

            rapidjson::Document::AllocatorType& allocator = docs.at(i).GetAllocator();
            rapidjson::Value scores_arr(rapidjson::kArrayType);
            rapidjson::Value labels_arr(rapidjson::kArrayType);
            rapidjson::Value x_arr(rapidjson::kArrayType);
            rapidjson::Value y_arr(rapidjson::kArrayType);
            rapidjson::Value w_arr(rapidjson::kArrayType);
            rapidjson::Value h_arr(rapidjson::kArrayType);
            for (std::size_t j = 0; j < PostProcessOut.bboxes.at(i).size(); j++) {
                x_arr.PushBack(PostProcessOut.bboxes.at(i).at(j).x, allocator);
                y_arr.PushBack(PostProcessOut.bboxes.at(i).at(j).y, allocator);
                w_arr.PushBack(PostProcessOut.bboxes.at(i).at(j).width, allocator);
                h_arr.PushBack(PostProcessOut.bboxes.at(i).at(j).height, allocator);
                scores_arr.PushBack(PostProcessOut.scores.at(i).at(j), allocator);
                labels_arr.PushBack(PostProcessOut.labels.at(i).at(j), allocator);
            
            }
            img_obj.AddMember("x", x_arr, allocator);
            img_obj.AddMember("y", y_arr, allocator);
            img_obj.AddMember("w", w_arr, allocator);
            img_obj.AddMember("h", h_arr, allocator);
            img_obj.AddMember("scores", scores_arr, allocator);
            img_obj.AddMember("labels", labels_arr, allocator);
            rapidjson::Value key;
            key.SetString(std::to_string(count).c_str(), allocator);
            docs.at(i).AddMember(key, img_obj, allocator);
        }
    }
    //write docs to file
    for (std::size_t i = 0; i < BATCH_SIZE; i++){
        std::string filename = this->resources+this->fnames.at(i)+".json";
        spdlog::info("Saving {}", filename);
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::string error_msg = "Unable to open file: "+filename;
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
            return false;
        }
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        writer.SetIndent(' ', 4);
        docs.at(i).Accept(writer);
        ofs << buffer.GetString();
        ofs.close();
    }
    return true;
}

private:
    std::string resources;
    std::array<std::string, BATCH_SIZE> fnames;
    std::queue<output_postprocess> log_q;

};


class SyncDetectionIterator{
public:
SyncDetectionIterator(const std::string &resources, const std::array<std::string, BATCH_SIZE> &fnames){
    for (std::size_t i = 0; i<BATCH_SIZE; i++){
        rapidjson::Document doc;
        cpp_utils::read_json_document(resources+fnames.at(i)+".json", 65536, doc);
        for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) // for s in samples
        {
            std::vector<uint16_t> tmp_x, tmp_y, tmp_w, tmp_h, tmp_l;
            std::vector<float> tmp_c;
            const std::string key = it->name.GetString();  // e.g. "19"
            const auto& obj = it->value;

            for (const auto& v : obj["x"].GetArray()){
                tmp_x.push_back(static_cast<uint16_t>(v.GetInt()));
            }
            for (const auto& v : obj["y"].GetArray()){
                tmp_y.push_back(static_cast<uint16_t>(v.GetInt()));
            }
            for (const auto& v : obj["w"].GetArray()){
                tmp_w.push_back(static_cast<uint16_t>(v.GetInt()));
            }
            for (const auto& v : obj["h"].GetArray()){
                tmp_h.push_back(static_cast<uint16_t>(v.GetInt()));
            }
            for (const auto& v : obj["scores"].GetArray()){
                tmp_c.push_back(v.GetFloat());
            }
            for (const auto& v : obj["labels"].GetArray()){
                tmp_l.push_back(static_cast<uint16_t>(v.GetInt()));
            }

            this->x.at(i).push_back(tmp_x);
            this->y.at(i).push_back(tmp_y);
            this->w.at(i).push_back(tmp_w);
            this->h.at(i).push_back(tmp_h);
            this->c.at(i).push_back(tmp_c);
            this->l.at(i).push_back(tmp_l);
        }
    }
}
~SyncDetectionIterator(){};

void get(
    std::array<std::vector<uint16_t>, BATCH_SIZE> &batch_x,
    std::array<std::vector<uint16_t>, BATCH_SIZE> &batch_y,
    std::array<std::vector<uint16_t>, BATCH_SIZE> &batch_w,
    std::array<std::vector<uint16_t>, BATCH_SIZE> &batch_h,
    std::array<std::vector<float>, BATCH_SIZE> &batch_c,
    std::array<std::vector<uint16_t>, BATCH_SIZE> &batch_l
){
    for (std::size_t i = 0; i<BATCH_SIZE; i++){
        batch_x.at(i) = this->x.at(i).at(this->frame_idx);
        batch_y.at(i) = this->y.at(i).at(this->frame_idx);
        batch_w.at(i) = this->w.at(i).at(this->frame_idx);
        batch_h.at(i) = this->h.at(i).at(this->frame_idx);
        batch_c.at(i) = this->c.at(i).at(this->frame_idx);
        batch_l.at(i) = this->l.at(i).at(this->frame_idx);
    }
    this->frame_idx++;
}

private:
// batch, samples, detections
std::array<std::vector<std::vector<uint16_t>>, BATCH_SIZE> x;
std::array<std::vector<std::vector<uint16_t>>, BATCH_SIZE> y;
std::array<std::vector<std::vector<uint16_t>>, BATCH_SIZE> w;
std::array<std::vector<std::vector<uint16_t>>, BATCH_SIZE> h;
std::array<std::vector<std::vector<float>>, BATCH_SIZE> c;
std::array<std::vector<std::vector<uint16_t>>, BATCH_SIZE> l;
std::size_t frame_idx{0};

void reset(){
    spdlog::info("Resetting SyncDetectionIterator");
    this->frame_idx = 0;
}

};

} // namespace detection_inference