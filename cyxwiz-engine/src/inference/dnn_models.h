#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace cyxwiz {

/**
 * Supported DNN model types
 */
enum class DNNModelType {
    Custom,           // User-provided model
    FaceDetector,     // SSD face detector (res10_300x300)
    YOLOv4,           // YOLO v4 object detector
    YOLOv4Tiny,       // YOLO v4 tiny (faster, less accurate)
    MobileNetSSD,     // MobileNet SSD object detector
    OpenPose,         // OpenPose body pose estimator
    FaceLandmark,     // Facial landmark detector
    AgeGender,        // Age and gender classification
    TextDetection     // EAST text detector
};

/**
 * DNN backend options
 */
enum class DNNBackend {
    Default,          // OpenCV default
    OpenCV,           // Pure OpenCV
    CUDA,             // NVIDIA CUDA
    OpenCL,           // OpenCL (GPU)
    Vulkan,           // Vulkan compute
    InferenceEngine   // Intel OpenVINO
};

/**
 * DNN target device
 */
enum class DNNTarget {
    CPU,              // CPU execution
    OpenCL,           // OpenCL GPU
    OpenCL_FP16,      // OpenCL GPU with FP16
    CUDA,             // NVIDIA CUDA
    CUDA_FP16,        // NVIDIA CUDA with FP16
    Vulkan,           // Vulkan compute
    FPGA              // FPGA (Intel)
};

/**
 * Detection result from object detection models
 */
struct Detection {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    float x = 0.0f;          // Bounding box (normalized 0-1)
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

/**
 * Face detection result
 */
struct FaceDetection {
    float x = 0.0f;          // Bounding box (normalized 0-1)
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float confidence = 0.0f;
    std::vector<float> landmarks;  // Optional facial landmarks
};

/**
 * Pose keypoint
 */
struct PoseKeypoint {
    float x = 0.0f;          // Position (normalized 0-1)
    float y = 0.0f;
    float confidence = 0.0f;
    int id = -1;             // Keypoint ID (e.g., 0=nose, 1=neck, etc.)
    std::string name;        // Keypoint name
};

/**
 * Pose estimation result
 */
struct PoseResult {
    std::vector<PoseKeypoint> keypoints;
    std::vector<std::pair<int, int>> skeleton;  // Connections between keypoints
    float confidence = 0.0f;
};

/**
 * Classification result
 */
struct ClassificationResult {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
};

/**
 * Model metadata
 */
struct DNNModelInfo {
    DNNModelType type = DNNModelType::Custom;
    std::string name;
    std::string description;
    std::string model_path;       // Path to model file (.caffemodel, .weights, .pb, .onnx)
    std::string config_path;      // Path to config file (.prototxt, .cfg, .pbtxt)
    std::string labels_path;      // Path to class labels file
    int input_width = 0;          // Expected input width
    int input_height = 0;         // Expected input height
    int input_channels = 3;       // Number of input channels
    float scale_factor = 1.0f;    // Input scale factor (e.g., 1/255)
    std::vector<float> mean;      // Mean values for normalization
    bool swap_rb = true;          // Swap R and B channels (BGR to RGB)
    std::vector<std::string> output_layers;  // Output layer names
    std::vector<std::string> class_labels;   // Class names for detection/classification
};

/**
 * OpenCV DNN Model wrapper
 *
 * Provides unified interface for various pre-trained models:
 * - Face detection (SSD)
 * - Object detection (YOLO, MobileNet SSD)
 * - Pose estimation (OpenPose)
 * - Classification
 */
class DNNModel {
public:
    DNNModel();
    ~DNNModel();

    // Non-copyable, movable
    DNNModel(const DNNModel&) = delete;
    DNNModel& operator=(const DNNModel&) = delete;
    DNNModel(DNNModel&&) noexcept;
    DNNModel& operator=(DNNModel&&) noexcept;

    /**
     * Load a model from files
     * @param model_path Path to model weights
     * @param config_path Path to model config (optional for ONNX)
     * @param type Model type for automatic configuration
     * @return true if loaded successfully
     */
    bool Load(const std::string& model_path,
              const std::string& config_path = "",
              DNNModelType type = DNNModelType::Custom);

    /**
     * Load a model with full info
     * @param info Model information and configuration
     * @return true if loaded successfully
     */
    bool Load(const DNNModelInfo& info);

    /**
     * Unload the model
     */
    void Unload();

    /**
     * Check if model is loaded
     */
    bool IsLoaded() const;

    /**
     * Set backend and target
     * @param backend Computation backend
     * @param target Target device
     */
    void SetBackendAndTarget(DNNBackend backend, DNNTarget target);

    /**
     * Run object detection
     * @param image_data Float image (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels (3 for RGB)
     * @param confidence_threshold Minimum confidence
     * @param nms_threshold NMS IoU threshold
     * @return Vector of detections
     */
    std::vector<Detection> Detect(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        float confidence_threshold = 0.5f,
        float nms_threshold = 0.4f
    );

    /**
     * Run face detection
     * @param image_data Float image (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param confidence_threshold Minimum confidence
     * @return Vector of face detections
     */
    std::vector<FaceDetection> DetectFaces(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        float confidence_threshold = 0.5f
    );

    /**
     * Run pose estimation
     * @param image_data Float image (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param threshold Keypoint confidence threshold
     * @return Vector of pose results
     */
    std::vector<PoseResult> EstimatePose(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        float threshold = 0.1f
    );

    /**
     * Run classification
     * @param image_data Float image (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @param top_k Return top K results
     * @return Vector of classification results
     */
    std::vector<ClassificationResult> Classify(
        const std::vector<float>& image_data,
        int width, int height, int channels,
        int top_k = 5
    );

    /**
     * Get raw network output
     * @param image_data Float image (0-1 range) [HWC format]
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Raw output blobs (flattened)
     */
    std::vector<std::vector<float>> Forward(
        const std::vector<float>& image_data,
        int width, int height, int channels
    );

    /**
     * Get model info
     */
    const DNNModelInfo& GetInfo() const { return info_; }

    /**
     * Get last error
     */
    const std::string& GetLastError() const { return last_error_; }

    /**
     * Get inference time (last run, in ms)
     */
    double GetInferenceTimeMs() const { return inference_time_ms_; }

private:
    // Implementation details hidden
    struct Impl;
    std::unique_ptr<Impl> impl_;

    DNNModelInfo info_;
    std::string last_error_;
    double inference_time_ms_ = 0.0;

    // Configure model based on type
    void ConfigureForType(DNNModelType type);

    // Parse YOLO output
    std::vector<Detection> ParseYOLOOutput(
        const std::vector<std::vector<float>>& outputs,
        int img_width, int img_height,
        float conf_threshold, float nms_threshold
    );

    // Parse SSD output
    std::vector<Detection> ParseSSDOutput(
        const std::vector<std::vector<float>>& outputs,
        int img_width, int img_height,
        float conf_threshold
    );
};

/**
 * DNN Model Registry - Singleton for managing pre-configured models
 */
class DNNModelRegistry {
public:
    static DNNModelRegistry& Instance();

    /**
     * Register a model configuration
     * @param name Unique model name
     * @param info Model configuration
     */
    void Register(const std::string& name, const DNNModelInfo& info);

    /**
     * Get model info by name
     * @param name Model name
     * @return Model info or nullptr if not found
     */
    const DNNModelInfo* Get(const std::string& name) const;

    /**
     * Get all registered model names
     */
    std::vector<std::string> GetModelNames() const;

    /**
     * Register built-in model configurations
     */
    void RegisterBuiltins();

    /**
     * Check if a model is available (files exist)
     * @param name Model name
     * @return true if model files exist
     */
    bool IsAvailable(const std::string& name) const;

    /**
     * Set the models directory
     * @param path Path to directory containing model files
     */
    void SetModelsDirectory(const std::string& path);

    /**
     * Get the models directory
     */
    const std::string& GetModelsDirectory() const { return models_dir_; }

private:
    DNNModelRegistry();
    ~DNNModelRegistry() = default;

    std::unordered_map<std::string, DNNModelInfo> models_;
    std::string models_dir_;
};

// Enum string conversions
std::string DNNModelTypeToString(DNNModelType type);
DNNModelType StringToDNNModelType(const std::string& str);
std::string DNNBackendToString(DNNBackend backend);
DNNBackend StringToDNNBackend(const std::string& str);
std::string DNNTargetToString(DNNTarget target);
DNNTarget StringToDNNTarget(const std::string& str);

} // namespace cyxwiz
