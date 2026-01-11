#include "dnn_models.h"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>

namespace cyxwiz {

// ============================================================================
// Enum String Conversions
// ============================================================================

std::string DNNModelTypeToString(DNNModelType type) {
    switch (type) {
        case DNNModelType::Custom:        return "custom";
        case DNNModelType::FaceDetector:  return "face_detector";
        case DNNModelType::YOLOv4:        return "yolov4";
        case DNNModelType::YOLOv4Tiny:    return "yolov4_tiny";
        case DNNModelType::MobileNetSSD:  return "mobilenet_ssd";
        case DNNModelType::OpenPose:      return "openpose";
        case DNNModelType::FaceLandmark:  return "face_landmark";
        case DNNModelType::AgeGender:     return "age_gender";
        case DNNModelType::TextDetection: return "text_detection";
        default:                          return "custom";
    }
}

DNNModelType StringToDNNModelType(const std::string& str) {
    if (str == "custom")         return DNNModelType::Custom;
    if (str == "face_detector")  return DNNModelType::FaceDetector;
    if (str == "yolov4")         return DNNModelType::YOLOv4;
    if (str == "yolov4_tiny")    return DNNModelType::YOLOv4Tiny;
    if (str == "mobilenet_ssd")  return DNNModelType::MobileNetSSD;
    if (str == "openpose")       return DNNModelType::OpenPose;
    if (str == "face_landmark")  return DNNModelType::FaceLandmark;
    if (str == "age_gender")     return DNNModelType::AgeGender;
    if (str == "text_detection") return DNNModelType::TextDetection;
    return DNNModelType::Custom;
}

std::string DNNBackendToString(DNNBackend backend) {
    switch (backend) {
        case DNNBackend::Default:          return "default";
        case DNNBackend::OpenCV:           return "opencv";
        case DNNBackend::CUDA:             return "cuda";
        case DNNBackend::OpenCL:           return "opencl";
        case DNNBackend::Vulkan:           return "vulkan";
        case DNNBackend::InferenceEngine:  return "openvino";
        default:                           return "default";
    }
}

DNNBackend StringToDNNBackend(const std::string& str) {
    if (str == "default")   return DNNBackend::Default;
    if (str == "opencv")    return DNNBackend::OpenCV;
    if (str == "cuda")      return DNNBackend::CUDA;
    if (str == "opencl")    return DNNBackend::OpenCL;
    if (str == "vulkan")    return DNNBackend::Vulkan;
    if (str == "openvino")  return DNNBackend::InferenceEngine;
    return DNNBackend::Default;
}

std::string DNNTargetToString(DNNTarget target) {
    switch (target) {
        case DNNTarget::CPU:         return "cpu";
        case DNNTarget::OpenCL:      return "opencl";
        case DNNTarget::OpenCL_FP16: return "opencl_fp16";
        case DNNTarget::CUDA:        return "cuda";
        case DNNTarget::CUDA_FP16:   return "cuda_fp16";
        case DNNTarget::Vulkan:      return "vulkan";
        case DNNTarget::FPGA:        return "fpga";
        default:                     return "cpu";
    }
}

DNNTarget StringToDNNTarget(const std::string& str) {
    if (str == "cpu")         return DNNTarget::CPU;
    if (str == "opencl")      return DNNTarget::OpenCL;
    if (str == "opencl_fp16") return DNNTarget::OpenCL_FP16;
    if (str == "cuda")        return DNNTarget::CUDA;
    if (str == "cuda_fp16")   return DNNTarget::CUDA_FP16;
    if (str == "vulkan")      return DNNTarget::Vulkan;
    if (str == "fpga")        return DNNTarget::FPGA;
    return DNNTarget::CPU;
}

// ============================================================================
// DNNModel Implementation
// ============================================================================

struct DNNModel::Impl {
    cv::dnn::Net net;
    bool loaded = false;
    int cv_backend = cv::dnn::DNN_BACKEND_DEFAULT;
    int cv_target = cv::dnn::DNN_TARGET_CPU;
};

DNNModel::DNNModel() : impl_(std::make_unique<Impl>()) {}

DNNModel::~DNNModel() = default;

DNNModel::DNNModel(DNNModel&&) noexcept = default;
DNNModel& DNNModel::operator=(DNNModel&&) noexcept = default;

bool DNNModel::Load(const std::string& model_path,
                    const std::string& config_path,
                    DNNModelType type) {
    DNNModelInfo info;
    info.type = type;
    info.model_path = model_path;
    info.config_path = config_path;
    return Load(info);
}

bool DNNModel::Load(const DNNModelInfo& info) {
    Unload();
    info_ = info;

    // Configure defaults based on type
    ConfigureForType(info_.type);

    try {
        // Determine file format
        std::filesystem::path model_ext(info_.model_path);
        std::string ext = model_ext.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".onnx") {
            // ONNX format
            impl_->net = cv::dnn::readNetFromONNX(info_.model_path);
        } else if (ext == ".pb") {
            // TensorFlow format
            impl_->net = cv::dnn::readNetFromTensorflow(info_.model_path, info_.config_path);
        } else if (ext == ".caffemodel") {
            // Caffe format
            impl_->net = cv::dnn::readNetFromCaffe(info_.config_path, info_.model_path);
        } else if (ext == ".weights") {
            // Darknet/YOLO format
            impl_->net = cv::dnn::readNetFromDarknet(info_.config_path, info_.model_path);
        } else if (ext == ".t7" || ext == ".net") {
            // Torch format
            impl_->net = cv::dnn::readNetFromTorch(info_.model_path);
        } else {
            // Try to auto-detect
            impl_->net = cv::dnn::readNet(info_.model_path, info_.config_path);
        }

        if (impl_->net.empty()) {
            last_error_ = "Failed to load network: empty network";
            return false;
        }

        // Set backend and target
        impl_->net.setPreferableBackend(impl_->cv_backend);
        impl_->net.setPreferableTarget(impl_->cv_target);

        // Get output layer names if not specified
        if (info_.output_layers.empty()) {
            info_.output_layers = impl_->net.getUnconnectedOutLayersNames();
        }

        impl_->loaded = true;
        spdlog::info("DNNModel: Loaded model from {}", info_.model_path);
        return true;

    } catch (const cv::Exception& e) {
        last_error_ = std::string("OpenCV DNN error: ") + e.what();
        spdlog::error("DNNModel: {}", last_error_);
        return false;
    } catch (const std::exception& e) {
        last_error_ = std::string("Error loading model: ") + e.what();
        spdlog::error("DNNModel: {}", last_error_);
        return false;
    }
}

void DNNModel::Unload() {
    impl_->net = cv::dnn::Net();
    impl_->loaded = false;
    info_ = DNNModelInfo();
}

bool DNNModel::IsLoaded() const {
    return impl_->loaded;
}

void DNNModel::SetBackendAndTarget(DNNBackend backend, DNNTarget target) {
    // Map to OpenCV constants
    switch (backend) {
        case DNNBackend::Default:
            impl_->cv_backend = cv::dnn::DNN_BACKEND_DEFAULT;
            break;
        case DNNBackend::OpenCV:
            impl_->cv_backend = cv::dnn::DNN_BACKEND_OPENCV;
            break;
        case DNNBackend::CUDA:
            impl_->cv_backend = cv::dnn::DNN_BACKEND_CUDA;
            break;
        case DNNBackend::InferenceEngine:
            impl_->cv_backend = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
            break;
        default:
            impl_->cv_backend = cv::dnn::DNN_BACKEND_DEFAULT;
    }

    switch (target) {
        case DNNTarget::CPU:
            impl_->cv_target = cv::dnn::DNN_TARGET_CPU;
            break;
        case DNNTarget::OpenCL:
            impl_->cv_target = cv::dnn::DNN_TARGET_OPENCL;
            break;
        case DNNTarget::OpenCL_FP16:
            impl_->cv_target = cv::dnn::DNN_TARGET_OPENCL_FP16;
            break;
        case DNNTarget::CUDA:
            impl_->cv_target = cv::dnn::DNN_TARGET_CUDA;
            break;
        case DNNTarget::CUDA_FP16:
            impl_->cv_target = cv::dnn::DNN_TARGET_CUDA_FP16;
            break;
        case DNNTarget::Vulkan:
            impl_->cv_target = cv::dnn::DNN_TARGET_VULKAN;
            break;
        default:
            impl_->cv_target = cv::dnn::DNN_TARGET_CPU;
    }

    if (impl_->loaded) {
        impl_->net.setPreferableBackend(impl_->cv_backend);
        impl_->net.setPreferableTarget(impl_->cv_target);
    }
}

void DNNModel::ConfigureForType(DNNModelType type) {
    switch (type) {
        case DNNModelType::FaceDetector:
            if (info_.input_width == 0) info_.input_width = 300;
            if (info_.input_height == 0) info_.input_height = 300;
            info_.scale_factor = 1.0f;
            info_.mean = {104.0f, 177.0f, 123.0f};
            info_.swap_rb = false;
            break;

        case DNNModelType::YOLOv4:
        case DNNModelType::YOLOv4Tiny:
            if (info_.input_width == 0) info_.input_width = 416;
            if (info_.input_height == 0) info_.input_height = 416;
            info_.scale_factor = 1.0f / 255.0f;
            info_.mean = {};
            info_.swap_rb = true;
            break;

        case DNNModelType::MobileNetSSD:
            if (info_.input_width == 0) info_.input_width = 300;
            if (info_.input_height == 0) info_.input_height = 300;
            info_.scale_factor = 0.007843f;
            info_.mean = {127.5f, 127.5f, 127.5f};
            info_.swap_rb = false;
            break;

        case DNNModelType::OpenPose:
            if (info_.input_width == 0) info_.input_width = 368;
            if (info_.input_height == 0) info_.input_height = 368;
            info_.scale_factor = 1.0f / 255.0f;
            info_.mean = {};
            info_.swap_rb = false;
            break;

        default:
            // Use defaults
            if (info_.input_width == 0) info_.input_width = 224;
            if (info_.input_height == 0) info_.input_height = 224;
            info_.scale_factor = 1.0f / 255.0f;
            break;
    }
}

std::vector<std::vector<float>> DNNModel::Forward(
    const std::vector<float>& image_data,
    int width, int height, int channels
) {
    std::vector<std::vector<float>> outputs;

    if (!impl_->loaded) {
        last_error_ = "Model not loaded";
        return outputs;
    }

    try {
        // Convert float image to cv::Mat
        cv::Mat img;
        if (channels == 1) {
            img = cv::Mat(height, width, CV_32FC1, const_cast<float*>(image_data.data()));
        } else if (channels == 3) {
            img = cv::Mat(height, width, CV_32FC3, const_cast<float*>(image_data.data()));
        } else {
            last_error_ = "Unsupported channel count";
            return outputs;
        }

        // Convert to 8-bit if needed and scale
        cv::Mat img8u;
        img.convertTo(img8u, CV_8U, 255.0);

        // Create blob
        cv::Scalar mean_scalar;
        if (info_.mean.size() >= 3) {
            mean_scalar = cv::Scalar(info_.mean[0], info_.mean[1], info_.mean[2]);
        } else if (info_.mean.size() == 1) {
            mean_scalar = cv::Scalar(info_.mean[0], info_.mean[0], info_.mean[0]);
        }

        cv::Mat blob = cv::dnn::blobFromImage(
            img8u,
            info_.scale_factor,
            cv::Size(info_.input_width, info_.input_height),
            mean_scalar,
            info_.swap_rb,
            false  // crop
        );

        // Set input and run forward pass
        impl_->net.setInput(blob);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<cv::Mat> output_blobs;
        impl_->net.forward(output_blobs, info_.output_layers);
        auto end = std::chrono::high_resolution_clock::now();

        inference_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

        // Convert outputs to float vectors
        for (const auto& output_blob : output_blobs) {
            std::vector<float> data;
            if (output_blob.isContinuous()) {
                data.assign(output_blob.ptr<float>(), output_blob.ptr<float>() + output_blob.total());
            } else {
                cv::Mat flat = output_blob.reshape(1, 1);
                data.assign(flat.ptr<float>(), flat.ptr<float>() + flat.total());
            }
            outputs.push_back(data);
        }

        return outputs;

    } catch (const cv::Exception& e) {
        last_error_ = std::string("Forward pass error: ") + e.what();
        spdlog::error("DNNModel: {}", last_error_);
        return outputs;
    }
}

std::vector<Detection> DNNModel::Detect(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    float confidence_threshold,
    float nms_threshold
) {
    std::vector<Detection> detections;

    auto outputs = Forward(image_data, width, height, channels);
    if (outputs.empty()) {
        return detections;
    }

    // Parse based on model type
    switch (info_.type) {
        case DNNModelType::YOLOv4:
        case DNNModelType::YOLOv4Tiny:
            return ParseYOLOOutput(outputs, width, height, confidence_threshold, nms_threshold);

        case DNNModelType::MobileNetSSD:
        case DNNModelType::FaceDetector:
            return ParseSSDOutput(outputs, width, height, confidence_threshold);

        default:
            // Try SSD format as default
            return ParseSSDOutput(outputs, width, height, confidence_threshold);
    }
}

std::vector<FaceDetection> DNNModel::DetectFaces(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    float confidence_threshold
) {
    std::vector<FaceDetection> faces;

    auto detections = Detect(image_data, width, height, channels, confidence_threshold, 0.4f);

    for (const auto& det : detections) {
        FaceDetection face;
        face.x = det.x;
        face.y = det.y;
        face.width = det.width;
        face.height = det.height;
        face.confidence = det.confidence;
        faces.push_back(face);
    }

    return faces;
}

std::vector<PoseResult> DNNModel::EstimatePose(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    float threshold
) {
    std::vector<PoseResult> poses;

    auto outputs = Forward(image_data, width, height, channels);
    if (outputs.empty() || outputs[0].empty()) {
        return poses;
    }

    // OpenPose output parsing
    // Output shape: [1, num_points, h, w]
    // This is a simplified implementation
    const int num_points = 18;  // COCO body model
    const std::vector<std::string> point_names = {
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
        "LEye", "REar", "LEar"
    };

    const std::vector<std::pair<int, int>> skeleton = {
        {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7},
        {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
        {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}
    };

    PoseResult pose;
    pose.skeleton = skeleton;

    // Parse heatmaps (simplified - would need proper peak detection)
    int out_h = info_.input_height / 8;
    int out_w = info_.input_width / 8;

    for (int i = 0; i < num_points && i < static_cast<int>(outputs[0].size()) / (out_h * out_w); ++i) {
        float max_val = 0.0f;
        int max_x = 0, max_y = 0;

        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                int idx = i * out_h * out_w + y * out_w + x;
                if (idx < static_cast<int>(outputs[0].size())) {
                    float val = outputs[0][idx];
                    if (val > max_val) {
                        max_val = val;
                        max_x = x;
                        max_y = y;
                    }
                }
            }
        }

        PoseKeypoint kp;
        kp.id = i;
        kp.name = i < static_cast<int>(point_names.size()) ? point_names[i] : "";
        kp.x = static_cast<float>(max_x) / out_w;
        kp.y = static_cast<float>(max_y) / out_h;
        kp.confidence = max_val;

        if (max_val >= threshold) {
            pose.keypoints.push_back(kp);
        }
    }

    if (!pose.keypoints.empty()) {
        poses.push_back(pose);
    }

    return poses;
}

std::vector<ClassificationResult> DNNModel::Classify(
    const std::vector<float>& image_data,
    int width, int height, int channels,
    int top_k
) {
    std::vector<ClassificationResult> results;

    auto outputs = Forward(image_data, width, height, channels);
    if (outputs.empty() || outputs[0].empty()) {
        return results;
    }

    // Get softmax output
    const auto& probs = outputs[0];

    // Create index-probability pairs
    std::vector<std::pair<int, float>> indexed_probs;
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed_probs.emplace_back(static_cast<int>(i), probs[i]);
    }

    // Sort by probability
    std::partial_sort(indexed_probs.begin(),
                     indexed_probs.begin() + std::min(top_k, static_cast<int>(indexed_probs.size())),
                     indexed_probs.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

    // Get top K results
    for (int i = 0; i < top_k && i < static_cast<int>(indexed_probs.size()); ++i) {
        ClassificationResult res;
        res.class_id = indexed_probs[i].first;
        res.confidence = indexed_probs[i].second;
        if (res.class_id < static_cast<int>(info_.class_labels.size())) {
            res.class_name = info_.class_labels[res.class_id];
        }
        results.push_back(res);
    }

    return results;
}

std::vector<Detection> DNNModel::ParseYOLOOutput(
    const std::vector<std::vector<float>>& outputs,
    int img_width, int img_height,
    float conf_threshold, float nms_threshold
) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outputs) {
        // YOLO output format: [batch, num_detections, 5 + num_classes]
        // Each detection: [center_x, center_y, width, height, obj_conf, class_probs...]
        int num_classes = static_cast<int>(info_.class_labels.size());
        if (num_classes == 0) num_classes = 80;  // COCO default

        int stride = 5 + num_classes;
        int num_detections = static_cast<int>(output.size()) / stride;

        for (int i = 0; i < num_detections; ++i) {
            int base = i * stride;
            float obj_conf = output[base + 4];

            if (obj_conf < conf_threshold) continue;

            // Find best class
            int best_class = 0;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                float score = output[base + 5 + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            float confidence = obj_conf * best_score;
            if (confidence < conf_threshold) continue;

            // Get bounding box
            float cx = output[base + 0];
            float cy = output[base + 1];
            float w = output[base + 2];
            float h = output[base + 3];

            int x = static_cast<int>((cx - w / 2) * img_width);
            int y = static_cast<int>((cy - h / 2) * img_height);
            int bw = static_cast<int>(w * img_width);
            int bh = static_cast<int>(h * img_height);

            class_ids.push_back(best_class);
            confidences.push_back(confidence);
            boxes.emplace_back(x, y, bw, bh);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.x = static_cast<float>(boxes[idx].x) / img_width;
        det.y = static_cast<float>(boxes[idx].y) / img_height;
        det.width = static_cast<float>(boxes[idx].width) / img_width;
        det.height = static_cast<float>(boxes[idx].height) / img_height;

        if (det.class_id < static_cast<int>(info_.class_labels.size())) {
            det.class_name = info_.class_labels[det.class_id];
        }

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> DNNModel::ParseSSDOutput(
    const std::vector<std::vector<float>>& outputs,
    int img_width, int img_height,
    float conf_threshold
) {
    // SSD outputs are already normalized, dimensions not needed
    (void)img_width;
    (void)img_height;

    std::vector<Detection> detections;

    if (outputs.empty() || outputs[0].empty()) {
        return detections;
    }

    // SSD output format: [1, 1, num_detections, 7]
    // Each detection: [batch_id, class_id, confidence, x1, y1, x2, y2]
    const auto& output = outputs[0];
    int num_detections = static_cast<int>(output.size()) / 7;

    for (int i = 0; i < num_detections; ++i) {
        int base = i * 7;
        float confidence = output[base + 2];

        if (confidence < conf_threshold) continue;

        Detection det;
        det.class_id = static_cast<int>(output[base + 1]);
        det.confidence = confidence;
        det.x = output[base + 3];
        det.y = output[base + 4];
        det.width = output[base + 5] - det.x;
        det.height = output[base + 6] - det.y;

        // Clamp to [0, 1]
        det.x = std::max(0.0f, std::min(1.0f, det.x));
        det.y = std::max(0.0f, std::min(1.0f, det.y));
        det.width = std::max(0.0f, std::min(1.0f - det.x, det.width));
        det.height = std::max(0.0f, std::min(1.0f - det.y, det.height));

        if (det.class_id < static_cast<int>(info_.class_labels.size())) {
            det.class_name = info_.class_labels[det.class_id];
        }

        detections.push_back(det);
    }

    return detections;
}

// ============================================================================
// DNNModelRegistry Implementation
// ============================================================================

DNNModelRegistry& DNNModelRegistry::Instance() {
    static DNNModelRegistry instance;
    return instance;
}

DNNModelRegistry::DNNModelRegistry() {
    RegisterBuiltins();
}

void DNNModelRegistry::Register(const std::string& name, const DNNModelInfo& info) {
    models_[name] = info;
}

const DNNModelInfo* DNNModelRegistry::Get(const std::string& name) const {
    auto it = models_.find(name);
    if (it != models_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> DNNModelRegistry::GetModelNames() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : models_) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

void DNNModelRegistry::SetModelsDirectory(const std::string& path) {
    models_dir_ = path;
}

bool DNNModelRegistry::IsAvailable(const std::string& name) const {
    auto* info = Get(name);
    if (!info) return false;

    std::filesystem::path model_path = info->model_path;
    if (!models_dir_.empty() && model_path.is_relative()) {
        model_path = std::filesystem::path(models_dir_) / model_path;
    }

    if (!std::filesystem::exists(model_path)) {
        return false;
    }

    if (!info->config_path.empty()) {
        std::filesystem::path config_path = info->config_path;
        if (!models_dir_.empty() && config_path.is_relative()) {
            config_path = std::filesystem::path(models_dir_) / config_path;
        }
        if (!std::filesystem::exists(config_path)) {
            return false;
        }
    }

    return true;
}

void DNNModelRegistry::RegisterBuiltins() {
    // Face Detector (OpenCV DNN Face)
    {
        DNNModelInfo info;
        info.type = DNNModelType::FaceDetector;
        info.name = "OpenCV Face Detector";
        info.description = "SSD-based face detector (res10_300x300)";
        info.model_path = "res10_300x300_ssd_iter_140000.caffemodel";
        info.config_path = "deploy.prototxt";
        info.input_width = 300;
        info.input_height = 300;
        info.scale_factor = 1.0f;
        info.mean = {104.0f, 177.0f, 123.0f};
        info.swap_rb = false;
        models_["face_detector"] = info;
    }

    // YOLO v4
    {
        DNNModelInfo info;
        info.type = DNNModelType::YOLOv4;
        info.name = "YOLOv4";
        info.description = "YOLO v4 object detector (COCO 80 classes)";
        info.model_path = "yolov4.weights";
        info.config_path = "yolov4.cfg";
        info.labels_path = "coco.names";
        info.input_width = 416;
        info.input_height = 416;
        info.scale_factor = 1.0f / 255.0f;
        info.swap_rb = true;
        models_["yolov4"] = info;
    }

    // YOLO v4 Tiny
    {
        DNNModelInfo info;
        info.type = DNNModelType::YOLOv4Tiny;
        info.name = "YOLOv4 Tiny";
        info.description = "YOLO v4 Tiny - faster, less accurate";
        info.model_path = "yolov4-tiny.weights";
        info.config_path = "yolov4-tiny.cfg";
        info.labels_path = "coco.names";
        info.input_width = 416;
        info.input_height = 416;
        info.scale_factor = 1.0f / 255.0f;
        info.swap_rb = true;
        models_["yolov4_tiny"] = info;
    }

    // MobileNet SSD
    {
        DNNModelInfo info;
        info.type = DNNModelType::MobileNetSSD;
        info.name = "MobileNet SSD";
        info.description = "Lightweight object detector";
        info.model_path = "mobilenet_ssd_deploy.caffemodel";
        info.config_path = "mobilenet_ssd_deploy.prototxt";
        info.input_width = 300;
        info.input_height = 300;
        info.scale_factor = 0.007843f;
        info.mean = {127.5f, 127.5f, 127.5f};
        info.swap_rb = false;
        info.class_labels = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        };
        models_["mobilenet_ssd"] = info;
    }

    // OpenPose
    {
        DNNModelInfo info;
        info.type = DNNModelType::OpenPose;
        info.name = "OpenPose (COCO)";
        info.description = "Body pose estimation (18 keypoints)";
        info.model_path = "pose_iter_440000.caffemodel";
        info.config_path = "pose_deploy_linevec.prototxt";
        info.input_width = 368;
        info.input_height = 368;
        info.scale_factor = 1.0f / 255.0f;
        info.swap_rb = false;
        models_["openpose"] = info;
    }
}

} // namespace cyxwiz
