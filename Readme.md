# 🚀 Detection-Inference

A high-performance, multi-threaded C++ pipeline for real-time **multi-camera object detection** using [YOLOv8](https://docs.ultralytics.com/).  

Developed as part of [my PhD thesis](todo-thesis-link) to enable **3D object detection** and generate proposals for my [keypoint inference pipeline](https://github.com/HenrikTrom/pose-inference).

This module supports deployment in robotic systems for real-time tracking and perception and is part of my  **ROS/ROS2** [real-time 3D tracker](https://github.com/HenrikTrom/Docker-Flir-Multi-Camera).


![System Setup](content/4cams.gif)

## 🧪 Test results

* Intel(R) Xeon(R) W-2145 CPU @ 3.70GHz, Nvidia 2080 super, Ubuntu 20.04, CUDA 11.8, TensorRT 8.6.1.6, OpenCV 4.10.0 with Yolov8 and BATCH_SIZE of 5 -> **Preprocess: ~2ms, NN inference ~7ms, Postprocess: ~5ms (1000 samples)**
<!-- * Ubuntu 20.04, CUDA 12.3, TensorRT 10.6.1.6, OpenCV 4.10.0 -->


## 📑 Citation

If you use this software, please use the GitHub **“Cite this repository”** button at the top(-right) of this page.

## Environment

This repository is designed to run inside the Docker 🐳 container provided here:  
[OpenCV-TRT-DEV](https://github.com/HenrikTrom/Docker-OpenCV-TensorRT-Dev)

It includes all necessary dependencies (CUDA, cuDNN, OpenCV, TensorRT, CMake).

### Prerequisites

In addition to the libraries installed in the container, this project relies on:

- 📦 [tensorrt-cpp-api (fork)](https://github.com/HenrikTrom/tensorrt-cpp-api)  
  *(Originally by [cyrusbehr](https://github.com/cyrusbehr/tensorrt-cpp-api))*
- 🧵 [cpp-utils](https://github.com/HenrikTrom/cpp_utils)  
  *(Handles multithreading, JSON config parsing, and utility tools)*


#### Environment Variables

Set the required variables (usually done via `.env` or your shell):

```bash
OPENCV_VERSION=4.10.0     # Your installed OpenCV version
N_CAMERAS=5               # Optional: sets system-wide batch size
```

> If `N_CAMERAS` is not set, CMake will default to a batch size of **5**.

Use the `trt.sh` script in `./scripts` to convert your .onnx model to a fixed batch size.

#### Notes

* The batch size is treated as a **hardware constraint**, defined by the number of connected cameras.
* You can change the default batch size in `CMakeLists.txt` to fit your system.
* Although this repo is optimized for YOLOv8 models, you can modify the post-processing stage to support **any ONNX-compatible detection model**.

###  Installation

Run the provided installation script:

```bash
sudo ./build_install.sh
```

This will configure the build system, compile the inference pipeline, and generate the binaries.


---

### 🧠 Model Requirements

This repo is designed for *trained* [YOLOv8](https://docs.ultralytics.com/) `.onnx` models.
The model must be **exported with a fixed batch size** to match the number of cameras used in your setup.

Adapt the configuration files in the `cfg/` folder to reflect your system and model setup.

---

## Executables

### Benchmark

After configuring your setup:

```bash
./build/inference_benchmark
```

This runs the inference pipeline, processes multi-camera input, and saves images with overlayed bounding boxes and labels to the `inputs/` folder.

### Video Inference Export

This executable iterates over a directory of synchronized .mp4 videos and saves the result for each video in a .json file. 

This example usage assumes <BATCH_SIZE> .mp4 videos in an arbitrary `./test` directory

```bash
./build/video_inference_export test
```

### BBox Overlay

This executable iterates over a directory of synchronized .mp4 videos and exported inference results (from `./build/video_inference_export`). It generates new .mp4 videos with detections and a tiled video similar to the .gif in this readme.

This example usage assumes <BATCH_SIZE> .mp4 videos and .json files in an arbitrary `./test` directory

```bash
./build/bbox_overlay test
```

---

## 📷 Applications

This inference module is optimized for:

* Real-time multi-camera tracking
* Robotics & embedded systems
* Preprocessing for downstream pipelines (e.g. keypoint tracking)
