// dnn_inference_panel.cpp - DNN Inference Panel implementation
#include "dnn_inference_panel.h"
#include "../../core/data_registry.h"
#include "../../core/texture_manager.h"
#include "../../core/file_dialogs.h"
#include "../../core/annotation_manager.h"
#include "../icons.h"

#include <imgui.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>

namespace cyxwiz {

// Model type options
const char* const MODEL_TYPES[] = {
    "Custom", "Face Detector", "YOLOv4", "YOLOv4-tiny",
    "MobileNet SSD", "OpenPose", "Face Landmark", "Age/Gender", "Text Detection"
};

// Backend options
const char* const BACKENDS[] = {
    "Default", "OpenCV", "CUDA", "OpenCL", "Vulkan", "OpenVINO"
};

// Target options
const char* const TARGETS[] = {
    "CPU", "OpenCL", "OpenCL FP16", "CUDA", "CUDA FP16", "Vulkan", "FPGA"
};

// Class colors for detections (10 distinct colors)
const ImU32 DNNInferencePanel::CLASS_COLORS[NUM_COLORS] = {
    IM_COL32(255, 0, 0, 255),      // Red
    IM_COL32(0, 255, 0, 255),      // Green
    IM_COL32(0, 0, 255, 255),      // Blue
    IM_COL32(255, 255, 0, 255),    // Yellow
    IM_COL32(255, 0, 255, 255),    // Magenta
    IM_COL32(0, 255, 255, 255),    // Cyan
    IM_COL32(255, 128, 0, 255),    // Orange
    IM_COL32(128, 0, 255, 255),    // Purple
    IM_COL32(0, 255, 128, 255),    // Spring Green
    IM_COL32(255, 128, 128, 255),  // Coral
};

DNNInferencePanel::DNNInferencePanel()
    : Panel("DNN Inference", false) {
    model_ = std::make_unique<DNNModel>();
}

DNNInferencePanel::~DNNInferencePanel() {
    if (image_texture_ != 0) {
        TextureManager::Instance().DeleteTexture(image_texture_);
    }
}

void DNNInferencePanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("DNN Inference###DNNInference", &visible_)) {
        RenderModelConfig();
        ImGui::Separator();
        RenderInputSource();
        ImGui::Separator();
        RenderInferenceSettings();
        ImGui::Separator();
        RenderResults();
        ImGui::Separator();
        RenderActions();
    }
    ImGui::End();
}

void DNNInferencePanel::RenderModelConfig() {
    ImGui::Text("%s Model Configuration", ICON_FA_BRAIN);
    ImGui::Spacing();

    // Model type, backend, target dropdowns
    ImGui::PushItemWidth(120);
    if (ImGui::Combo("Type", &model_type_idx_, MODEL_TYPES, NUM_MODEL_TYPES)) {
        model_loaded_ = false;
        model_status_ = "Model type changed - reload required";
    }
    ImGui::SameLine();
    if (ImGui::Combo("Backend", &backend_idx_, BACKENDS, NUM_BACKENDS)) {
        if (model_loaded_ && model_) {
            model_->SetBackendAndTarget(
                static_cast<DNNBackend>(backend_idx_),
                static_cast<DNNTarget>(target_idx_)
            );
        }
    }
    ImGui::SameLine();
    if (ImGui::Combo("Target", &target_idx_, TARGETS, NUM_TARGETS)) {
        if (model_loaded_ && model_) {
            model_->SetBackendAndTarget(
                static_cast<DNNBackend>(backend_idx_),
                static_cast<DNNTarget>(target_idx_)
            );
        }
    }
    ImGui::PopItemWidth();

    // Model file path
    ImGui::InputText("Model", model_path_, sizeof(model_path_));
    ImGui::SameLine();
    if (ImGui::Button("Browse##Model")) {
        auto result = FileDialogs::OpenFile(
            "Select Model File",
            {{"DNN Models", "weights,caffemodel,pb,onnx,t7,net"},
             {"YOLO Weights", "weights"},
             {"Caffe Model", "caffemodel"},
             {"TensorFlow", "pb"},
             {"ONNX", "onnx"},
             {"All Files", "*"}}
        );
        if (result) {
            strncpy(model_path_, result->c_str(), sizeof(model_path_) - 1);
            model_loaded_ = false;
        }
    }

    // Config file path
    ImGui::InputText("Config", config_path_, sizeof(config_path_));
    ImGui::SameLine();
    if (ImGui::Button("Browse##Config")) {
        auto result = FileDialogs::OpenFile(
            "Select Config File",
            {{"Config Files", "cfg,prototxt,pbtxt,json"},
             {"YOLO Config", "cfg"},
             {"Caffe Proto", "prototxt"},
             {"TensorFlow Proto", "pbtxt"},
             {"All Files", "*"}}
        );
        if (result) {
            strncpy(config_path_, result->c_str(), sizeof(config_path_) - 1);
            model_loaded_ = false;
        }
    }

    // Labels file path
    ImGui::InputText("Labels", labels_path_, sizeof(labels_path_));
    ImGui::SameLine();
    if (ImGui::Button("Browse##Labels")) {
        auto result = FileDialogs::OpenFile(
            "Select Labels File",
            {{"Labels", "names,txt,labels"},
             {"All Files", "*"}}
        );
        if (result) {
            strncpy(labels_path_, result->c_str(), sizeof(labels_path_) - 1);
        }
    }

    // Load button and status
    if (ImGui::Button(ICON_FA_DOWNLOAD " Load Model")) {
        LoadModel();
    }
    ImGui::SameLine();
    if (model_loaded_) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%s %s", ICON_FA_CHECK, model_status_.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "%s %s", ICON_FA_TRIANGLE_EXCLAMATION, model_status_.c_str());
    }
}

void DNNInferencePanel::RenderInputSource() {
    ImGui::Text("%s Input Source", ICON_FA_IMAGE);
    ImGui::Spacing();

    // Radio buttons for input source
    if (ImGui::RadioButton("Single File", input_source_ == InputSource::File)) {
        input_source_ = InputSource::File;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Dataset", input_source_ == InputSource::Dataset)) {
        input_source_ = InputSource::Dataset;
        // Refresh dataset list
        auto& registry = DataRegistry::Instance();
        dataset_names_ = registry.GetDatasetNames();
    }

    if (input_source_ == InputSource::File) {
        // File input
        ImGui::InputText("Image File", file_path_, sizeof(file_path_));
        ImGui::SameLine();
        if (ImGui::Button("Browse##Image")) {
            auto result = FileDialogs::OpenImage();
            if (result) {
                strncpy(file_path_, result->c_str(), sizeof(file_path_) - 1);
                LoadImageFromFile();
            }
        }
    } else {
        // Dataset input
        auto& registry = DataRegistry::Instance();
        dataset_names_ = registry.GetDatasetNames();

        if (dataset_names_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                "No datasets loaded. Load a dataset first.");
        } else {
            // Dataset dropdown
            std::vector<const char*> names_cstr;
            for (const auto& name : dataset_names_) {
                names_cstr.push_back(name.c_str());
            }

            if (ImGui::Combo("Dataset", &selected_dataset_idx_,
                            names_cstr.data(), static_cast<int>(names_cstr.size()))) {
                // Dataset changed - update total images and load first image
                auto handle = registry.GetDataset(dataset_names_[selected_dataset_idx_]);
                if (handle.IsValid()) {
                    total_images_ = handle.Size();
                    current_image_idx_ = 0;
                    LoadImageFromDataset();
                }
            }

            // Image navigation
            if (selected_dataset_idx_ >= 0 && total_images_ > 0) {
                ImGui::Text("Image:");
                ImGui::SameLine();
                if (ImGui::Button(ICON_FA_CHEVRON_LEFT)) {
                    if (current_image_idx_ > 0) {
                        current_image_idx_--;
                        LoadImageFromDataset();
                    }
                }
                ImGui::SameLine();
                ImGui::Text("%zu / %zu", current_image_idx_ + 1, total_images_);
                ImGui::SameLine();
                if (ImGui::Button(ICON_FA_CHEVRON_RIGHT)) {
                    if (current_image_idx_ < total_images_ - 1) {
                        current_image_idx_++;
                        LoadImageFromDataset();
                    }
                }

                // Jump to specific image
                ImGui::SameLine();
                static int goto_idx = 0;
                ImGui::PushItemWidth(80);
                ImGui::InputInt("##goto", &goto_idx, 0, 0);
                ImGui::PopItemWidth();
                ImGui::SameLine();
                if (ImGui::Button("Go")) {
                    if (goto_idx >= 1 && goto_idx <= static_cast<int>(total_images_)) {
                        current_image_idx_ = goto_idx - 1;
                        LoadImageFromDataset();
                    }
                }
            }
        }
    }
}

void DNNInferencePanel::RenderInferenceSettings() {
    ImGui::Text("%s Inference Settings", ICON_FA_SLIDERS);
    ImGui::Spacing();

    ImGui::PushItemWidth(200);
    ImGui::SliderFloat("Confidence", &confidence_threshold_, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SliderFloat("NMS", &nms_threshold_, 0.0f, 1.0f, "%.2f");
    ImGui::SameLine();
    ImGui::SliderInt("Top-K", &top_k_, 1, 20);
    ImGui::PopItemWidth();

    // Run inference button
    ImGui::Spacing();
    bool can_run = model_loaded_ && has_image_;
    if (!can_run) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_FA_PLAY " Run Inference", ImVec2(150, 0))) {
        RunInference();
    }
    if (!can_run) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (has_results_) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
            "Inference time: %.2f ms", inference_time_ms_);
    }
}

void DNNInferencePanel::RenderResults() {
    // Two-column layout: image on left, results list on right
    float panel_width = ImGui::GetContentRegionAvail().x;
    float image_panel_width = panel_width * 0.6f;
    float results_panel_width = panel_width * 0.4f - 10;

    // Left panel: Image with overlays
    ImGui::BeginChild("ImagePanel", ImVec2(image_panel_width, 350), true);
    RenderResultsImage();
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel: Results list
    ImGui::BeginChild("ResultsPanel", ImVec2(results_panel_width, 350), true);
    RenderResultsList();
    ImGui::EndChild();
}

void DNNInferencePanel::RenderResultsImage() {
    if (!has_image_) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "No image loaded. Select an image from file or dataset.");
        return;
    }

    // Calculate display size maintaining aspect ratio
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;
    float scale = std::min(avail_w / image_width_, avail_h / image_height_);
    float display_w = image_width_ * scale;
    float display_h = image_height_ * scale;

    // Center the image
    float offset_x = (avail_w - display_w) * 0.5f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);

    // Update texture if needed
    if (image_texture_ == 0 && !current_image_.empty()) {
        image_texture_ = TextureManager::Instance().CreateTextureFromFloatData(
            current_image_.data(), image_width_, image_height_, image_channels_
        );
    }

    // Get image position for overlay drawing
    ImVec2 img_pos = ImGui::GetCursorScreenPos();

    // Draw image
    if (image_texture_ != 0) {
        ImGui::Image((ImTextureID)(intptr_t)image_texture_,
                    ImVec2(display_w, display_h));
    }

    // Draw overlays on top of image
    if (has_results_) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 img_min = img_pos;
        ImVec2 img_max = ImVec2(img_pos.x + display_w, img_pos.y + display_h);

        // Draw detection boxes
        if (show_boxes_ && !detections_.empty()) {
            DrawDetectionBoxes(draw_list, img_min, img_max);
        }

        // Draw face boxes
        if (show_boxes_ && !face_detections_.empty()) {
            DrawFaceBoxes(draw_list, img_min, img_max);
        }

        // Draw pose keypoints
        if (show_keypoints_ && !poses_.empty()) {
            DrawPoseKeypoints(draw_list, img_min, img_max);
        }
    }

    // Display options
    ImGui::Checkbox("Boxes", &show_boxes_);
    ImGui::SameLine();
    ImGui::Checkbox("Labels", &show_labels_);
    ImGui::SameLine();
    ImGui::Checkbox("Confidence", &show_confidence_);
    ImGui::SameLine();
    ImGui::Checkbox("Keypoints", &show_keypoints_);
}

void DNNInferencePanel::RenderResultsList() {
    ImGui::Text("%s Results", ICON_FA_LIST);
    ImGui::Separator();

    if (!has_results_) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No results yet.");
        return;
    }

    // Object detections
    if (!detections_.empty()) {
        if (ImGui::CollapsingHeader("Object Detections", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < detections_.size(); i++) {
                const auto& det = detections_[i];
                ImU32 color = CLASS_COLORS[det.class_id % NUM_COLORS];
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::BulletText("#%zu: %s (%.1f%%)",
                    i + 1,
                    det.class_name.empty() ? std::to_string(det.class_id).c_str() : det.class_name.c_str(),
                    det.confidence * 100.0f);
                ImGui::PopStyleColor();

                ImGui::SameLine();
                char btn_id[32];
                snprintf(btn_id, sizeof(btn_id), "Save##det%zu", i);
                if (ImGui::SmallButton(btn_id)) {
                    SaveDetectionToAnnotation(i);
                }
            }
        }
    }

    // Face detections
    if (!face_detections_.empty()) {
        if (ImGui::CollapsingHeader("Face Detections", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < face_detections_.size(); i++) {
                const auto& face = face_detections_[i];
                ImGui::BulletText("#%zu: Face (%.1f%%)", i + 1, face.confidence * 100.0f);
            }
        }
    }

    // Pose results
    if (!poses_.empty()) {
        if (ImGui::CollapsingHeader("Pose Estimation", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < poses_.size(); i++) {
                const auto& pose = poses_[i];
                ImGui::BulletText("#%zu: %zu keypoints (%.1f%%)",
                    i + 1, pose.keypoints.size(), pose.confidence * 100.0f);
            }
        }
    }

    // Classification results
    if (!classifications_.empty()) {
        if (ImGui::CollapsingHeader("Classification", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < classifications_.size(); i++) {
                const auto& cls = classifications_[i];
                // Progress bar showing confidence
                ImGui::Text("%zu. %s", i + 1,
                    cls.class_name.empty() ? std::to_string(cls.class_id).c_str() : cls.class_name.c_str());
                ImGui::SameLine(150);
                ImGui::ProgressBar(cls.confidence, ImVec2(100, 0), "");
                ImGui::SameLine();
                ImGui::Text("%.1f%%", cls.confidence * 100.0f);
            }
        }
    }
}

void DNNInferencePanel::RenderActions() {
    ImGui::Text("%s Actions", ICON_FA_WAND_MAGIC_SPARKLES);
    ImGui::Spacing();

    bool has_detections = has_results_ && (!detections_.empty() || !face_detections_.empty());

    if (!has_detections) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save All to Annotations")) {
        SaveAllToAnnotations();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export Results JSON")) {
        ExportResultsJSON();
    }

    if (!has_detections) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (!model_loaded_ || input_source_ != InputSource::Dataset || selected_dataset_idx_ < 0) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Auto-Annotate Batch")) {
        AutoAnnotateBatch();
    }
    if (!model_loaded_ || input_source_ != InputSource::Dataset || selected_dataset_idx_ < 0) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
            "(Requires model + dataset)");
    }
}

void DNNInferencePanel::LoadModel() {
    if (strlen(model_path_) == 0) {
        model_status_ = "Model path is empty";
        model_loaded_ = false;
        return;
    }

    spdlog::info("DNNInferencePanel: Loading model from {}", model_path_);

    // Prepare model info
    DNNModelInfo info;
    info.type = static_cast<DNNModelType>(model_type_idx_);
    info.model_path = model_path_;
    info.config_path = config_path_;
    info.labels_path = labels_path_;

    // Load labels if provided
    if (strlen(labels_path_) > 0) {
        std::ifstream labels_file(labels_path_);
        if (labels_file.is_open()) {
            std::string line;
            while (std::getline(labels_file, line)) {
                if (!line.empty()) {
                    info.class_labels.push_back(line);
                }
            }
            spdlog::info("DNNInferencePanel: Loaded {} class labels", info.class_labels.size());
        }
    }

    // Load model
    model_ = std::make_unique<DNNModel>();
    bool success = model_->Load(info);

    if (success) {
        // Set backend and target
        model_->SetBackendAndTarget(
            static_cast<DNNBackend>(backend_idx_),
            static_cast<DNNTarget>(target_idx_)
        );

        model_loaded_ = true;
        model_status_ = std::string("Ready (") + MODEL_TYPES[model_type_idx_] +
                       ", " + BACKENDS[backend_idx_] + ")";
        spdlog::info("DNNInferencePanel: Model loaded successfully");
    } else {
        model_loaded_ = false;
        model_status_ = model_->GetLastError();
        spdlog::error("DNNInferencePanel: Failed to load model: {}", model_status_);
    }
}

void DNNInferencePanel::LoadImageFromDataset() {
    if (selected_dataset_idx_ < 0 || dataset_names_.empty()) return;

    auto& registry = DataRegistry::Instance();
    auto handle = registry.GetDataset(dataset_names_[selected_dataset_idx_]);

    if (!handle.IsValid()) {
        spdlog::error("DNNInferencePanel: Invalid dataset handle");
        return;
    }

    auto info = handle.GetInfo();
    auto [image_data, label] = handle.GetSample(current_image_idx_);

    if (image_data.empty()) {
        spdlog::error("DNNInferencePanel: Empty image data");
        return;
    }

    current_image_ = image_data;
    image_width_ = static_cast<int>(info.shape[1]);
    image_height_ = static_cast<int>(info.shape[0]);
    image_channels_ = static_cast<int>(info.shape.size() > 2 ? info.shape[2] : 1);
    has_image_ = true;

    // Delete old texture and create new one
    if (image_texture_ != 0) {
        TextureManager::Instance().DeleteTexture(image_texture_);
    }
    image_texture_ = TextureManager::Instance().CreateTextureFromFloatData(
        current_image_.data(), image_width_, image_height_, image_channels_
    );

    // Clear previous results
    has_results_ = false;
    detections_.clear();
    face_detections_.clear();
    poses_.clear();
    classifications_.clear();

    spdlog::debug("DNNInferencePanel: Loaded image {} ({}x{}x{})",
        current_image_idx_, image_width_, image_height_, image_channels_);
}

void DNNInferencePanel::LoadImageFromFile() {
    if (strlen(file_path_) == 0) return;

    int w, h;
    auto texture = TextureManager::Instance().LoadTextureFromFile(file_path_, &w, &h);

    if (texture == 0) {
        spdlog::error("DNNInferencePanel: Failed to load image from {}", file_path_);
        return;
    }

    // For file-based loading, we need to read the raw data for inference
    // This is a simplified approach - in production, use OpenCV directly
    // For now, just use the texture for display and mark that we need proper loading
    image_width_ = w;
    image_height_ = h;
    image_channels_ = 3;  // Assume RGB

    if (image_texture_ != 0) {
        TextureManager::Instance().DeleteTexture(image_texture_);
    }
    image_texture_ = texture;
    has_image_ = true;

    // Clear previous results
    has_results_ = false;
    detections_.clear();
    face_detections_.clear();
    poses_.clear();
    classifications_.clear();

    spdlog::info("DNNInferencePanel: Loaded image from file ({}x{})", w, h);
}

void DNNInferencePanel::RunInference() {
    if (!model_loaded_ || !model_ || !has_image_ || current_image_.empty()) {
        spdlog::warn("DNNInferencePanel: Cannot run inference - model or image not ready");
        return;
    }

    spdlog::info("DNNInferencePanel: Running inference...");

    DNNModelType type = static_cast<DNNModelType>(model_type_idx_);

    switch (type) {
        case DNNModelType::YOLOv4:
        case DNNModelType::YOLOv4Tiny:
        case DNNModelType::MobileNetSSD:
        case DNNModelType::Custom:
            detections_ = model_->Detect(
                current_image_, image_width_, image_height_, image_channels_,
                confidence_threshold_, nms_threshold_
            );
            spdlog::info("DNNInferencePanel: {} detections", detections_.size());
            break;

        case DNNModelType::FaceDetector:
        case DNNModelType::FaceLandmark:
            face_detections_ = model_->DetectFaces(
                current_image_, image_width_, image_height_, image_channels_,
                confidence_threshold_
            );
            spdlog::info("DNNInferencePanel: {} faces detected", face_detections_.size());
            break;

        case DNNModelType::OpenPose:
            poses_ = model_->EstimatePose(
                current_image_, image_width_, image_height_, image_channels_,
                confidence_threshold_
            );
            spdlog::info("DNNInferencePanel: {} poses detected", poses_.size());
            break;

        case DNNModelType::AgeGender:
        case DNNModelType::TextDetection:
            classifications_ = model_->Classify(
                current_image_, image_width_, image_height_, image_channels_,
                top_k_
            );
            spdlog::info("DNNInferencePanel: {} classifications", classifications_.size());
            break;
    }

    inference_time_ms_ = static_cast<float>(model_->GetInferenceTimeMs());
    has_results_ = true;
}

void DNNInferencePanel::DrawDetectionBoxes(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max) {
    float img_w = img_max.x - img_min.x;
    float img_h = img_max.y - img_min.y;

    for (const auto& det : detections_) {
        // Convert normalized coords to screen coords
        float x1 = img_min.x + det.x * img_w;
        float y1 = img_min.y + det.y * img_h;
        float x2 = x1 + det.width * img_w;
        float y2 = y1 + det.height * img_h;

        ImU32 color = CLASS_COLORS[det.class_id % NUM_COLORS];

        // Draw box
        draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), color, 0.0f, 0, box_thickness_);

        // Draw label
        if (show_labels_) {
            std::string label = det.class_name.empty() ?
                std::to_string(det.class_id) : det.class_name;
            if (show_confidence_) {
                char conf_str[16];
                snprintf(conf_str, sizeof(conf_str), " %.0f%%", det.confidence * 100);
                label += conf_str;
            }

            // Background for label
            ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
            draw_list->AddRectFilled(
                ImVec2(x1, y1 - text_size.y - 2),
                ImVec2(x1 + text_size.x + 4, y1),
                IM_COL32(0, 0, 0, 180)
            );
            draw_list->AddText(ImVec2(x1 + 2, y1 - text_size.y - 1),
                IM_COL32(255, 255, 255, 255), label.c_str());
        }
    }
}

void DNNInferencePanel::DrawFaceBoxes(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max) {
    float img_w = img_max.x - img_min.x;
    float img_h = img_max.y - img_min.y;

    for (size_t i = 0; i < face_detections_.size(); i++) {
        const auto& face = face_detections_[i];

        float x1 = img_min.x + face.x * img_w;
        float y1 = img_min.y + face.y * img_h;
        float x2 = x1 + face.width * img_w;
        float y2 = y1 + face.height * img_h;

        ImU32 color = CLASS_COLORS[i % NUM_COLORS];

        draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), color, 0.0f, 0, box_thickness_);

        if (show_labels_) {
            char label[32];
            snprintf(label, sizeof(label), "Face %.0f%%", face.confidence * 100);

            ImVec2 text_size = ImGui::CalcTextSize(label);
            draw_list->AddRectFilled(
                ImVec2(x1, y1 - text_size.y - 2),
                ImVec2(x1 + text_size.x + 4, y1),
                IM_COL32(0, 0, 0, 180)
            );
            draw_list->AddText(ImVec2(x1 + 2, y1 - text_size.y - 1),
                IM_COL32(255, 255, 255, 255), label);
        }

        // Draw landmarks if available
        if (!face.landmarks.empty() && face.landmarks.size() >= 10) {
            // Assuming 5 landmarks with x,y pairs
            for (size_t j = 0; j < face.landmarks.size(); j += 2) {
                float lx = img_min.x + face.landmarks[j] * img_w;
                float ly = img_min.y + face.landmarks[j + 1] * img_h;
                draw_list->AddCircleFilled(ImVec2(lx, ly), 3.0f, IM_COL32(0, 255, 0, 255));
            }
        }
    }
}

void DNNInferencePanel::DrawPoseKeypoints(ImDrawList* draw_list, ImVec2 img_min, ImVec2 img_max) {
    float img_w = img_max.x - img_min.x;
    float img_h = img_max.y - img_min.y;

    for (size_t p = 0; p < poses_.size(); p++) {
        const auto& pose = poses_[p];
        ImU32 color = CLASS_COLORS[p % NUM_COLORS];

        // Draw keypoints
        for (const auto& kp : pose.keypoints) {
            if (kp.confidence > confidence_threshold_) {
                float x = img_min.x + kp.x * img_w;
                float y = img_min.y + kp.y * img_h;
                draw_list->AddCircleFilled(ImVec2(x, y), 4.0f, color);
            }
        }

        // Draw skeleton connections
        for (const auto& [from, to] : pose.skeleton) {
            if (from >= 0 && to >= 0) {
                const size_t from_index = static_cast<size_t>(from);
                const size_t to_index = static_cast<size_t>(to);
                if (from_index >= pose.keypoints.size() ||
                    to_index >= pose.keypoints.size()) {
                    continue;
                }
                const auto& kp1 = pose.keypoints[from_index];
                const auto& kp2 = pose.keypoints[to_index];

                if (kp1.confidence > confidence_threshold_ && kp2.confidence > confidence_threshold_) {
                    float x1 = img_min.x + kp1.x * img_w;
                    float y1 = img_min.y + kp1.y * img_h;
                    float x2 = img_min.x + kp2.x * img_w;
                    float y2 = img_min.y + kp2.y * img_h;
                    draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), color, 2.0f);
                }
            }
        }
    }
}

void DNNInferencePanel::SaveDetectionToAnnotation(size_t detection_idx) {
    if (detection_idx >= detections_.size()) return;
    if (input_source_ != InputSource::Dataset || selected_dataset_idx_ < 0) return;

    auto& registry = DataRegistry::Instance();
    auto& ann_mgr = registry.GetAnnotationManager();

    const std::string& dataset_id = dataset_names_[selected_dataset_idx_];
    const auto& det = detections_[detection_idx];

    // Convert normalized bbox to polygon points
    std::vector<Point2D> polygon = {
        {det.x * image_width_, det.y * image_height_},
        {(det.x + det.width) * image_width_, det.y * image_height_},
        {(det.x + det.width) * image_width_, (det.y + det.height) * image_height_},
        {det.x * image_width_, (det.y + det.height) * image_height_}
    };

    // Ensure annotation set exists
    if (!ann_mgr.GetAnnotationSet(dataset_id)) {
        ann_mgr.CreateAnnotationSet(dataset_id);
    }

    // Add class if not exists
    int class_id = ann_mgr.AddClass(dataset_id,
        det.class_name.empty() ? std::to_string(det.class_id) : det.class_name);

    // Add annotation
    ann_mgr.AddAnnotation(dataset_id, current_image_idx_, class_id,
                          AnnotationType::BBox, polygon);

    spdlog::info("DNNInferencePanel: Saved detection {} to annotations", detection_idx);
}

void DNNInferencePanel::SaveAllToAnnotations() {
    if (input_source_ != InputSource::Dataset || selected_dataset_idx_ < 0) return;

    for (size_t i = 0; i < detections_.size(); i++) {
        SaveDetectionToAnnotation(i);
    }

    spdlog::info("DNNInferencePanel: Saved {} detections to annotations", detections_.size());
}

void DNNInferencePanel::ExportResultsJSON() {
    auto result = FileDialogs::SaveFile(
        "Export Results",
        {{"JSON Files", "json"}, {"All Files", "*"}},
        nullptr,
        "inference_results.json"
    );

    if (!result) return;

    nlohmann::json j;
    j["inference_time_ms"] = inference_time_ms_;
    j["image_width"] = image_width_;
    j["image_height"] = image_height_;

    // Export detections
    j["detections"] = nlohmann::json::array();
    for (const auto& det : detections_) {
        j["detections"].push_back({
            {"class_id", det.class_id},
            {"class_name", det.class_name},
            {"confidence", det.confidence},
            {"x", det.x},
            {"y", det.y},
            {"width", det.width},
            {"height", det.height}
        });
    }

    // Export face detections
    j["faces"] = nlohmann::json::array();
    for (const auto& face : face_detections_) {
        j["faces"].push_back({
            {"confidence", face.confidence},
            {"x", face.x},
            {"y", face.y},
            {"width", face.width},
            {"height", face.height}
        });
    }

    // Export classifications
    j["classifications"] = nlohmann::json::array();
    for (const auto& cls : classifications_) {
        j["classifications"].push_back({
            {"class_id", cls.class_id},
            {"class_name", cls.class_name},
            {"confidence", cls.confidence}
        });
    }

    std::ofstream file(*result);
    file << j.dump(2);
    file.close();

    spdlog::info("DNNInferencePanel: Exported results to {}", *result);
}

void DNNInferencePanel::AutoAnnotateBatch() {
    if (!model_loaded_ || input_source_ != InputSource::Dataset || selected_dataset_idx_ < 0) {
        return;
    }

    auto& registry = DataRegistry::Instance();
    auto handle = registry.GetDataset(dataset_names_[selected_dataset_idx_]);

    if (!handle.IsValid()) return;

    size_t total = handle.Size();
    auto& ann_mgr = registry.GetAnnotationManager();
    const std::string& dataset_id = dataset_names_[selected_dataset_idx_];

    // Ensure annotation set exists
    if (!ann_mgr.GetAnnotationSet(dataset_id)) {
        ann_mgr.CreateAnnotationSet(dataset_id);
    }

    spdlog::info("DNNInferencePanel: Auto-annotating {} images...", total);

    int total_detections = 0;

    for (size_t i = 0; i < total; i++) {
        auto [image_data, label] = handle.GetSample(i);
        if (image_data.empty()) continue;

        auto info = handle.GetInfo();
        int w = static_cast<int>(info.shape[1]);
        int h = static_cast<int>(info.shape[0]);
        int c = static_cast<int>(info.shape.size() > 2 ? info.shape[2] : 1);

        // Run detection
        auto dets = model_->Detect(image_data, w, h, c, confidence_threshold_, nms_threshold_);

        // Save detections
        for (const auto& det : dets) {
            std::vector<Point2D> polygon = {
                {det.x * w, det.y * h},
                {(det.x + det.width) * w, det.y * h},
                {(det.x + det.width) * w, (det.y + det.height) * h},
                {det.x * w, (det.y + det.height) * h}
            };

            int class_id = ann_mgr.AddClass(dataset_id,
                det.class_name.empty() ? std::to_string(det.class_id) : det.class_name);

            ann_mgr.AddAnnotation(dataset_id, i, class_id, AnnotationType::BBox, polygon);
            total_detections++;
        }

        // Log progress every 100 images
        if ((i + 1) % 100 == 0) {
            spdlog::info("DNNInferencePanel: Processed {}/{} images", i + 1, total);
        }
    }

    spdlog::info("DNNInferencePanel: Auto-annotation complete. {} detections saved.", total_detections);
}

const char* DNNInferencePanel::GetModelTypeName(DNNModelType type) {
    int idx = static_cast<int>(type);
    if (idx >= 0 && idx < NUM_MODEL_TYPES) {
        return MODEL_TYPES[idx];
    }
    return "Unknown";
}

const char* DNNInferencePanel::GetBackendName(DNNBackend backend) {
    int idx = static_cast<int>(backend);
    if (idx >= 0 && idx < NUM_BACKENDS) {
        return BACKENDS[idx];
    }
    return "Unknown";
}

const char* DNNInferencePanel::GetTargetName(DNNTarget target) {
    int idx = static_cast<int>(target);
    if (idx >= 0 && idx < NUM_TARGETS) {
        return TARGETS[idx];
    }
    return "Unknown";
}

} // namespace cyxwiz
