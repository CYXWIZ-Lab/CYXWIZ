#pragma once

#include "../panel.h"
#include "../../inference/dnn_models.h"
#include <imgui.h>
#include <vector>
#include <string>
#include <memory>

namespace cyxwiz {

// Forward declarations
class AnnotationManager;
class DataRegistry;

// Model/Backend/Target option strings (defined in cpp)
extern const char* const MODEL_TYPES[];
extern const char* const BACKENDS[];
extern const char* const TARGETS[];
constexpr int NUM_MODEL_TYPES = 9;
constexpr int NUM_BACKENDS = 6;
constexpr int NUM_TARGETS = 7;

/**
 * DNN Inference Panel - GUI for running inference with pre-trained models
 *
 * Features:
 * - Load DNN models (YOLO, OpenPose, face detector, classifier)
 * - Run inference on single images or dataset samples
 * - Visualize results (bounding boxes, keypoints, classifications)
 * - Save detections to AnnotationManager
 * - Auto-annotate entire datasets
 */
class DNNInferencePanel : public Panel {
public:
    DNNInferencePanel();
    ~DNNInferencePanel() override;

    void Render() override;
    const char* GetIcon() const override { return "\xef\x97\x9c"; }  // FA_BRAIN

private:
    // Render sub-sections
    void RenderModelConfig();
    void RenderInputSource();
    void RenderInferenceSettings();
    void RenderResults();
    void RenderResultsImage();
    void RenderResultsList();
    void RenderActions();

    // Actions
    void LoadModel();
    void RunInference();
    void AutoAnnotateBatch();
    void SaveDetectionToAnnotation(size_t detection_idx);
    void SaveAllToAnnotations();
    void ExportResultsJSON();
    void LoadImageFromDataset();
    void LoadImageFromFile();

    // Drawing overlays on image
    void DrawDetectionBoxes(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max);
    void DrawPoseKeypoints(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max);
    void DrawFaceBoxes(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max);

    // Helper to get model type name
    static const char* GetModelTypeName(DNNModelType type);
    static const char* GetBackendName(DNNBackend backend);
    static const char* GetTargetName(DNNTarget target);

private:
    // Model state
    std::unique_ptr<DNNModel> model_;
    int model_type_idx_ = 2;       // Default: YOLOv4
    int backend_idx_ = 2;          // Default: CUDA
    int target_idx_ = 3;           // Default: CUDA
    char model_path_[512] = "";
    char config_path_[512] = "";
    char labels_path_[512] = "";
    bool model_loaded_ = false;
    std::string model_status_ = "No model loaded";

    // Input source
    enum class InputSource { File, Dataset };
    InputSource input_source_ = InputSource::Dataset;
    char file_path_[512] = "";
    std::vector<std::string> dataset_names_;
    int selected_dataset_idx_ = -1;
    size_t current_image_idx_ = 0;
    size_t total_images_ = 0;

    // Inference settings
    float confidence_threshold_ = 0.5f;
    float nms_threshold_ = 0.4f;
    int top_k_ = 5;  // For classification

    // Current image
    std::vector<float> current_image_;
    int image_width_ = 0;
    int image_height_ = 0;
    int image_channels_ = 0;
    uint32_t image_texture_ = 0;
    bool has_image_ = false;

    // Inference results
    std::vector<Detection> detections_;
    std::vector<FaceDetection> face_detections_;
    std::vector<PoseResult> poses_;
    std::vector<ClassificationResult> classifications_;
    float inference_time_ms_ = 0.0f;
    bool has_results_ = false;

    // Display options
    bool show_boxes_ = true;
    bool show_labels_ = true;
    bool show_confidence_ = true;
    bool show_keypoints_ = true;
    float box_thickness_ = 2.0f;

    // Colors for different classes (cycled)
    static constexpr int NUM_COLORS = 10;
    static const ImU32 CLASS_COLORS[NUM_COLORS];
};

} // namespace cyxwiz
