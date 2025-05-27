# Command to convert YOLOv8 to TensorRT engine 

BATCH_SIZE=5
/opt/vision_dependencies/tensorrt/TensorRT-${TENSORRT_VERSION}/targets/x86_64-linux-gnu/bin/trtexec \
    --onnx=/home/docker/modules/detection-inference/models/yolov8n_coco_human${BATCH_SIZE}.onnx \
    --shapes=images:${BATCH_SIZE}x3x640x640 \
    --saveEngine=/home/docker/modules/detection-inference/models/yolov8n_coco_human${BATCH_SIZE}.engine

# Command to convert dynamic onnx model to fixed shape
python3 -m onnxruntime.tools.make_dynamic_shape_fixed   --input_name images   --input_shape 5,3,640,640   /home/docker/modules/detection-inference/models/yolov8n_coco_human5.onnx   /home/docker/modules/detection-inference/models/yolov8n_coco_human5_fixed.onnx
