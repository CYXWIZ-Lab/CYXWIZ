#include "annotation_editor_panel.h"
#include "../../core/data_registry.h"
#include "../../core/file_dialogs.h"
#include <spdlog/spdlog.h>
#include <glad/glad.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

// Static member initialization
constexpr ImVec4 AnnotationEditorPanel::CLASS_COLORS[];

AnnotationEditorPanel::AnnotationEditorPanel()
    : Panel("Annotation Editor", false) {
    // Initialize annotation tools
    rect_tool_ = std::make_unique<RectangleSelectTool>();
    brush_tool_ = std::make_unique<BrushTool>();
    polygon_tool_ = std::make_unique<PolygonTool>();
    point_tool_ = std::make_unique<PointMarkerTool>();

    spdlog::info("AnnotationEditorPanel initialized");
}

AnnotationEditorPanel::~AnnotationEditorPanel() {
    ReleaseTexture();
}

void AnnotationEditorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200, 800), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(name_.c_str(), &visible_)) {
        // Main toolbar
        RenderToolbar();
        ImGui::Separator();

        // Three-column layout
        float available_width = ImGui::GetContentRegionAvail().x;
        float center_width = available_width - left_panel_width_ - right_panel_width_ - 20.0f;
        if (center_width < 400.0f) center_width = 400.0f;

        // Left panel: Tools + Classes
        ImGui::BeginChild("LeftPanel", ImVec2(left_panel_width_, -ImGui::GetFrameHeightWithSpacing()), true);
        RenderToolPanel();
        ImGui::Separator();
        RenderClassPanel();
        ImGui::EndChild();

        ImGui::SameLine();

        // Center panel: Image canvas
        ImGui::BeginChild("CenterPanel", ImVec2(center_width, -ImGui::GetFrameHeightWithSpacing()), true);
        RenderImageCanvas();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel: Annotations + Export
        ImGui::BeginChild("RightPanel", ImVec2(right_panel_width_, -ImGui::GetFrameHeightWithSpacing()), true);
        RenderAnnotationListPanel();
        ImGui::Separator();
        RenderExportPanel();
        ImGui::EndChild();

        // Status bar
        RenderStatusBar();
    }
    ImGui::End();
}

void AnnotationEditorPanel::RenderToolbar() {
    // Dataset selection
    ImGui::Text("Dataset:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);

    // Get available datasets from DataRegistry
    auto& registry = DataRegistry::Instance();
    auto dataset_names = registry.GetDatasetNames();

    if (ImGui::BeginCombo("##DatasetSelect", dataset_id_.empty() ? "Select dataset..." : dataset_id_.c_str())) {
        for (const auto& name : dataset_names) {
            bool is_selected = (dataset_id_ == name);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                SetDataset(name);
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Navigation controls
    ImGui::BeginDisabled(dataset_id_.empty() || current_image_idx_ == 0);
    if (ImGui::Button(ICON_FA_CHEVRON_LEFT " Prev")) {
        NavigatePrev();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    // Image index input
    ImGui::SetNextItemWidth(80);
    int display_idx = static_cast<int>(current_image_idx_ + 1);
    if (ImGui::InputInt("##ImageIdx", &display_idx, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (display_idx >= 1 && display_idx <= static_cast<int>(total_images_)) {
            NavigateToIndex(static_cast<size_t>(display_idx - 1));
        }
    }
    ImGui::SameLine();
    ImGui::Text("/ %zu", total_images_);

    ImGui::SameLine();

    ImGui::BeginDisabled(dataset_id_.empty() || current_image_idx_ >= total_images_ - 1);
    if (ImGui::Button("Next " ICON_FA_CHEVRON_RIGHT)) {
        NavigateNext();
    }
    ImGui::EndDisabled();

    // Progress bar
    if (total_images_ > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("|");
        ImGui::SameLine();

        float progress = static_cast<float>(current_image_idx_ + 1) / static_cast<float>(total_images_);
        ImGui::SetNextItemWidth(150);
        ImGui::ProgressBar(progress, ImVec2(0, 0), "");
    }

    // Zoom controls
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_MINUS)) {
        canvas_zoom_ = std::max(0.1f, canvas_zoom_ - 0.1f);
    }
    ImGui::SameLine();
    ImGui::Text("%.0f%%", canvas_zoom_ * 100.0f);
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_PLUS)) {
        canvas_zoom_ = std::min(5.0f, canvas_zoom_ + 0.1f);
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit")) {
        canvas_zoom_ = 1.0f;
        canvas_offset_ = {0.0f, 0.0f};
    }
}

void AnnotationEditorPanel::RenderToolPanel() {
    ImGui::Text(ICON_FA_SCREWDRIVER_WRENCH " Tools");
    ImGui::Separator();

    // Tool selection buttons
    RenderToolButton(ICON_FA_SQUARE, "Rectangle Selection", Tool::Rectangle);
    RenderToolButton(ICON_FA_BRUSH, "Brush Paint/Erase", Tool::Brush);
    RenderToolButton(ICON_FA_DRAW_POLYGON, "Polygon Draw", Tool::Polygon);
    RenderToolButton(ICON_FA_CROSSHAIRS, "Point Marker", Tool::PointMarker);

    ImGui::Spacing();

    // Tool-specific controls
    if (current_tool_) {
        ImGui::Separator();
        ImGui::Text("Tool Settings");
        current_tool_->RenderControls();
    }

    // Undo/Redo buttons
    ImGui::Spacing();
    ImGui::Separator();

    ImGui::BeginDisabled(!current_tool_ || !current_tool_->CanUndo());
    if (ImGui::Button(ICON_FA_ROTATE_LEFT " Undo", ImVec2(-1, 0))) {
        if (current_tool_) current_tool_->Undo();
    }
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!current_tool_ || !current_tool_->CanRedo());
    if (ImGui::Button(ICON_FA_ROTATE_RIGHT " Redo", ImVec2(-1, 0))) {
        if (current_tool_) current_tool_->Redo();
    }
    ImGui::EndDisabled();

    // Save annotation button
    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Annotation", ImVec2(-1, 0))) {
        SaveCurrentToolAnnotation();
    }
}

void AnnotationEditorPanel::RenderToolButton(const char* icon, const char* tooltip, Tool tool) {
    bool is_active = (active_tool_ == tool);
    if (is_active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }

    if (ImGui::Button(icon, ImVec2(40, 40))) {
        SetActiveTool(tool);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", tooltip);
    }

    if (is_active) {
        ImGui::PopStyleColor();
    }
    ImGui::SameLine();
}

void AnnotationEditorPanel::RenderImageCanvas() {
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

    // Draw background
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_pos,
                             ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                             IM_COL32(30, 30, 30, 255));

    // Handle input
    ImGui::InvisibleButton("canvas", canvas_size,
                           ImGuiButtonFlags_MouseButtonLeft |
                           ImGuiButtonFlags_MouseButtonRight |
                           ImGuiButtonFlags_MouseButtonMiddle);

    HandleCanvasInput();

    // Draw image if loaded
    if (image_texture_ != 0 && image_width_ > 0 && image_height_ > 0) {
        float scaled_width = static_cast<float>(image_width_) * canvas_zoom_;
        float scaled_height = static_cast<float>(image_height_) * canvas_zoom_;

        // Center image in canvas
        ImVec2 img_pos;
        img_pos.x = canvas_pos.x + (canvas_size.x - scaled_width) * 0.5f + canvas_offset_.x;
        img_pos.y = canvas_pos.y + (canvas_size.y - scaled_height) * 0.5f + canvas_offset_.y;

        ImVec2 img_end;
        img_end.x = img_pos.x + scaled_width;
        img_end.y = img_pos.y + scaled_height;

        draw_list->AddImage(
            (ImTextureID)(intptr_t)image_texture_,
            img_pos, img_end);

        // Draw tool overlay
        if (current_tool_) {
            current_tool_->RenderOverlay(draw_list, img_pos.x, img_pos.y, canvas_zoom_);
        }

        // TODO: Draw existing annotations (temporarily disabled)
    } else {
        // No image loaded message
        const char* msg = dataset_id_.empty()
            ? "Select a dataset to begin annotation"
            : "Loading image...";
        ImVec2 text_size = ImGui::CalcTextSize(msg);
        ImVec2 text_pos;
        text_pos.x = canvas_pos.x + (canvas_size.x - text_size.x) * 0.5f;
        text_pos.y = canvas_pos.y + (canvas_size.y - text_size.y) * 0.5f;
        draw_list->AddText(text_pos, IM_COL32(128, 128, 128, 255), msg);
    }
}

void AnnotationEditorPanel::HandleCanvasInput() {
    ImGuiIO& io = ImGui::GetIO();

    // Middle mouse button or right-click for panning
    if (ImGui::IsItemActive() && (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
                                   (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && active_tool_ == Tool::None))) {
        canvas_offset_.x += io.MouseDelta.x;
        canvas_offset_.y += io.MouseDelta.y;
    }

    // Scroll wheel for zoom
    if (ImGui::IsItemHovered()) {
        float wheel = io.MouseWheel;
        if (wheel != 0.0f) {
            canvas_zoom_ = std::clamp(canvas_zoom_ + wheel * 0.1f, 0.1f, 5.0f);
        }
    }

    // Tool input - simplified for now
    if (current_tool_ && image_texture_ != 0 && ImGui::IsItemHovered()) {
        ImVec2 mouse_pos = io.MousePos;
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();

        float scaled_width = static_cast<float>(image_width_) * canvas_zoom_;
        float scaled_height = static_cast<float>(image_height_) * canvas_zoom_;
        float img_x = canvas_pos.x + (canvas_size.x - scaled_width) * 0.5f + canvas_offset_.x;
        float img_y = canvas_pos.y + (canvas_size.y - scaled_height) * 0.5f + canvas_offset_.y;

        float image_x = (mouse_pos.x - img_x) / canvas_zoom_;
        float image_y = (mouse_pos.y - img_y) / canvas_zoom_;

        // Clamp to image bounds
        image_x = std::clamp(image_x, 0.0f, static_cast<float>(image_width_));
        image_y = std::clamp(image_y, 0.0f, static_cast<float>(image_height_));

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            current_tool_->OnMouseDown(image_x, image_y, MouseButton::Left);
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            current_tool_->OnMouseUp(image_x, image_y, MouseButton::Left);
        }
        current_tool_->OnMouseMove(image_x, image_y, ImGui::IsMouseDown(ImGuiMouseButton_Left));
    }
}

ImVec2 AnnotationEditorPanel::ScreenToImage(ImVec2 screen_pos) const {
    // Not used in simplified version
    return screen_pos;
}

ImVec2 AnnotationEditorPanel::ImageToScreen(ImVec2 image_pos) const {
    // Not used in simplified version
    return image_pos;
}

ImVec2 AnnotationEditorPanel::ImageToScreen(ImVec2 image_pos, ImVec2 img_origin) const {
    return ImVec2(
        img_origin.x + image_pos.x * canvas_zoom_,
        img_origin.y + image_pos.y * canvas_zoom_
    );
}

void AnnotationEditorPanel::RenderAnnotationListPanel() {
    ImGui::Text(ICON_FA_LIST " Annotations");
    ImGui::Separator();

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    const std::vector<Annotation>* anns = ann_mgr.GetAnnotations(dataset_id_, current_image_idx_);

    if (!anns || anns->empty()) {
        ImGui::TextDisabled("No annotations for this image");
        return;
    }

    ImGui::BeginChild("AnnotationList", ImVec2(0, -60), true);
    for (size_t i = 0; i < anns->size(); ++i) {
        RenderAnnotationItem((*anns)[i], static_cast<int>(i));
    }
    ImGui::EndChild();

    // Delete button
    ImGui::BeginDisabled(selected_annotation_id_ < 0);
    if (ImGui::Button(ICON_FA_TRASH " Delete Selected", ImVec2(-1, 0))) {
        DeleteSelectedAnnotation();
    }
    ImGui::EndDisabled();
}

void AnnotationEditorPanel::RenderAnnotationItem(const Annotation& ann, int index) {
    ImGui::PushID(ann.id);

    bool is_selected = (selected_annotation_id_ == ann.id);
    ImVec4 color = GetClassColor(ann.class_id);

    // Color indicator
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    draw_list->AddRectFilled(pos, ImVec2(pos.x + 4, pos.y + ImGui::GetTextLineHeight()), ImGui::ColorConvertFloat4ToU32(color));
    ImGui::Dummy(ImVec2(8, 0));
    ImGui::SameLine();

    // Selectable item
    char label[128];
    const char* type_str = (ann.type == AnnotationType::BBox) ? "Box" :
                           (ann.type == AnnotationType::Polygon) ? "Poly" : "Mask";
    snprintf(label, sizeof(label), "%s: %s (%.0f px)", type_str, ann.class_name.c_str(), ann.area);

    if (ImGui::Selectable(label, is_selected)) {
        selected_annotation_id_ = ann.id;
    }

    ImGui::PopID();
}

void AnnotationEditorPanel::RenderClassPanel() {
    ImGui::Text(ICON_FA_TAGS " Classes");
    ImGui::Separator();

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    const std::vector<std::string>& classes = ann_mgr.GetClasses(dataset_id_);

    // Class list
    ImGui::BeginChild("ClassList", ImVec2(0, 100), true);
    for (size_t i = 0; i < classes.size(); ++i) {
        ImVec4 color = GetClassColor(static_cast<int>(i));
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        bool is_selected = (current_class_id_ == static_cast<int>(i));
        if (ImGui::Selectable(classes[i].c_str(), is_selected)) {
            current_class_id_ = static_cast<int>(i);
        }

        ImGui::PopStyleColor();
    }
    ImGui::EndChild();

    // Add new class
    ImGui::SetNextItemWidth(-50);
    ImGui::InputText("##NewClass", new_class_name_, sizeof(new_class_name_));
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PLUS)) {
        AddNewClass();
    }
}

void AnnotationEditorPanel::RenderExportPanel() {
    ImGui::Text(ICON_FA_FILE_EXPORT " Export");
    ImGui::Separator();

    ImGui::InputText("##ExportPath", export_path_, sizeof(export_path_));
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FOLDER_OPEN)) {
        if (auto selected = FileDialogs::SelectOutputFolder(
                export_path_[0] == '\0' ? nullptr : export_path_)) {
            strncpy(export_path_, selected->c_str(), sizeof(export_path_) - 1);
            export_path_[sizeof(export_path_) - 1] = '\0';
        }
    }

    if (ImGui::Button("COCO JSON", ImVec2(-1, 0))) {
        ExportCOCO();
    }
    if (ImGui::Button("YOLO txt", ImVec2(-1, 0))) {
        ExportYOLO();
    }
    if (ImGui::Button("Pascal VOC", ImVec2(-1, 0))) {
        ExportVOC();
    }
}

void AnnotationEditorPanel::RenderStatusBar() {
    // Get annotation stats
    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    size_t ann_count = 0;
    size_t annotated_images = 0;

    if (ann_mgr.HasAnnotationSet(dataset_id_)) {
        const AnnotationSet* ann_set = ann_mgr.GetAnnotationSet(dataset_id_);
        if (ann_set) {
            ann_count = ann_set->GetTotalAnnotationCount();
            annotated_images = ann_set->GetAnnotatedImageCount();
        }
    }

    ImGui::Text("%s | Image: %zu/%zu | Annotations: %zu | Labeled: %zu/%zu images",
                dataset_id_.empty() ? "No dataset" : dataset_id_.c_str(),
                current_image_idx_ + 1, total_images_,
                ann_count, annotated_images, total_images_);
}

void AnnotationEditorPanel::HandleKeyboardShortcuts() {
    if (!visible_ || !focused_) return;

    ImGuiIO& io = ImGui::GetIO();

    // Navigation: Left/Right arrows
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && !io.KeyCtrl) {
        NavigatePrev();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) && !io.KeyCtrl) {
        NavigateNext();
    }

    // Tool shortcuts: 1-4
    if (ImGui::IsKeyPressed(ImGuiKey_1)) SetActiveTool(Tool::Rectangle);
    if (ImGui::IsKeyPressed(ImGuiKey_2)) SetActiveTool(Tool::Brush);
    if (ImGui::IsKeyPressed(ImGuiKey_3)) SetActiveTool(Tool::Polygon);
    if (ImGui::IsKeyPressed(ImGuiKey_4)) SetActiveTool(Tool::PointMarker);

    // Undo/Redo: Ctrl+Z, Ctrl+Y
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z) && current_tool_) {
        current_tool_->Undo();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y) && current_tool_) {
        current_tool_->Redo();
    }

    // Delete: Del key
    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        DeleteSelectedAnnotation();
    }

    // Save annotation: Enter
    if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        SaveCurrentToolAnnotation();
    }
}

void AnnotationEditorPanel::SetDataset(const std::string& dataset_id) {
    if (dataset_id_ == dataset_id) return;

    dataset_id_ = dataset_id;
    current_image_idx_ = 0;

    // Get dataset info from DataRegistry
    auto& registry = DataRegistry::Instance();
    auto handle = registry.GetDataset(dataset_id_);
    if (handle.IsValid()) {
        auto info = handle.GetInfo();
        total_images_ = info.num_samples;
    } else {
        total_images_ = 0;
    }

    // Ensure annotation set exists
    auto& ann_mgr = registry.GetAnnotationManager();
    ann_mgr.CreateAnnotationSet(dataset_id_);

    // Load first image
    LoadCurrentImage();

    spdlog::info("AnnotationEditorPanel: Set dataset to '{}' ({} images)", dataset_id_, total_images_);
}

void AnnotationEditorPanel::NavigatePrev() {
    if (current_image_idx_ > 0) {
        current_image_idx_--;
        LoadCurrentImage();
    }
}

void AnnotationEditorPanel::NavigateNext() {
    if (current_image_idx_ < total_images_ - 1) {
        current_image_idx_++;
        LoadCurrentImage();
    }
}

void AnnotationEditorPanel::NavigateToIndex(size_t index) {
    if (index < total_images_) {
        current_image_idx_ = index;
        LoadCurrentImage();
    }
}

void AnnotationEditorPanel::LoadCurrentImage() {
    if (dataset_id_.empty() || current_image_idx_ >= total_images_) {
        ReleaseTexture();
        return;
    }

    // Get image from DataRegistry
    auto& registry = DataRegistry::Instance();
    auto handle = registry.GetDataset(dataset_id_);

    if (!handle.IsValid()) {
        spdlog::warn("AnnotationEditorPanel: Dataset '{}' not found", dataset_id_);
        ReleaseTexture();
        return;
    }

    // Get sample data
    auto [image_data, label] = handle.GetSample(current_image_idx_);

    if (image_data.empty()) {
        spdlog::warn("AnnotationEditorPanel: Failed to load image {} from dataset '{}'",
                     current_image_idx_, dataset_id_);
        return;
    }

    // Get dimensions from dataset info
    auto info = handle.GetInfo();
    if (info.shape.size() >= 2) {
        // Assume shape is [H, W, C] or [H, W]
        image_height_ = static_cast<int>(info.shape[0]);
        image_width_ = static_cast<int>(info.shape[1]);
        image_channels_ = info.shape.size() >= 3 ? static_cast<int>(info.shape[2]) : 1;
    } else {
        // Fallback: try to infer from data size
        int total = static_cast<int>(image_data.size());
        image_channels_ = 3;
        image_height_ = static_cast<int>(std::sqrt(total / image_channels_));
        image_width_ = image_height_;
    }

    current_image_data_ = std::move(image_data);

    // Update tool with new image
    if (current_tool_) {
        current_tool_->SetImage(current_image_data_, image_width_, image_height_, image_channels_);
    }

    UpdateTexture();
}

void AnnotationEditorPanel::UpdateTexture() {
    if (current_image_data_.empty() || image_width_ <= 0 || image_height_ <= 0) {
        ReleaseTexture();
        return;
    }

    // Convert float data to unsigned char for OpenGL
    std::vector<unsigned char> pixels(static_cast<size_t>(image_width_) * static_cast<size_t>(image_height_) * 4);
    int src_channels = image_channels_;

    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            size_t dst_idx = (static_cast<size_t>(y) * static_cast<size_t>(image_width_) + static_cast<size_t>(x)) * 4;
            size_t src_idx = (static_cast<size_t>(y) * static_cast<size_t>(image_width_) + static_cast<size_t>(x)) * static_cast<size_t>(src_channels);

            if (src_channels >= 3) {
                pixels[dst_idx + 0] = static_cast<unsigned char>(std::clamp(current_image_data_[src_idx + 0] * 255.0f, 0.0f, 255.0f));
                pixels[dst_idx + 1] = static_cast<unsigned char>(std::clamp(current_image_data_[src_idx + 1] * 255.0f, 0.0f, 255.0f));
                pixels[dst_idx + 2] = static_cast<unsigned char>(std::clamp(current_image_data_[src_idx + 2] * 255.0f, 0.0f, 255.0f));
            } else {
                unsigned char gray = static_cast<unsigned char>(std::clamp(current_image_data_[src_idx] * 255.0f, 0.0f, 255.0f));
                pixels[dst_idx + 0] = gray;
                pixels[dst_idx + 1] = gray;
                pixels[dst_idx + 2] = gray;
            }
            pixels[dst_idx + 3] = 255;
        }
    }

    // Create or update OpenGL texture
    if (image_texture_ == 0) {
        glGenTextures(1, &image_texture_);
    }

    glBindTexture(GL_TEXTURE_2D, image_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width_, image_height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void AnnotationEditorPanel::ReleaseTexture() {
    if (image_texture_ != 0) {
        glDeleteTextures(1, &image_texture_);
        image_texture_ = 0;
    }
    image_width_ = 0;
    image_height_ = 0;
    current_image_data_.clear();
}

void AnnotationEditorPanel::SetActiveTool(Tool tool) {
    active_tool_ = tool;

    switch (tool) {
        case Tool::Rectangle:
            current_tool_ = rect_tool_.get();
            break;
        case Tool::Brush:
            current_tool_ = brush_tool_.get();
            break;
        case Tool::Polygon:
            current_tool_ = polygon_tool_.get();
            break;
        case Tool::PointMarker:
            current_tool_ = point_tool_.get();
            break;
        default:
            current_tool_ = nullptr;
            break;
    }

    // Set current image to tool
    if (current_tool_ && !current_image_data_.empty()) {
        current_tool_->SetImage(current_image_data_, image_width_, image_height_, image_channels_);
    }
}

void AnnotationEditorPanel::SaveCurrentToolAnnotation() {
    if (!current_tool_ || dataset_id_.empty()) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();

    // Get or create annotation set
    ann_mgr.CreateAnnotationSet(dataset_id_);

    // Convert tool result to annotation based on tool type
    std::vector<Point2D> polygon;
    BoundingBox bbox;

    if (active_tool_ == Tool::Rectangle) {
        RectangleSelectTool* rect_tool = static_cast<RectangleSelectTool*>(current_tool_);
        if (rect_tool->HasSelection()) {
            const AnnotationRect& sel = rect_tool->GetSelection();
            bbox.x = sel.x;
            bbox.y = sel.y;
            bbox.width = sel.width;
            bbox.height = sel.height;

            ann_mgr.AddAnnotation(dataset_id_, current_image_idx_,
                                  current_class_id_, AnnotationType::BBox,
                                  polygon, bbox, image_width_, image_height_);

            rect_tool->ClearSelection();
            spdlog::info("Saved bounding box annotation");
        }
    } else if (active_tool_ == Tool::Polygon) {
        PolygonTool* poly_tool = static_cast<PolygonTool*>(current_tool_);
        const std::vector<PolygonAnnotation>& polygons = poly_tool->GetPolygons();
        for (size_t i = 0; i < polygons.size(); ++i) {
            const PolygonAnnotation& poly = polygons[i];
            if (poly.closed && poly.vertices.size() >= 3) {
                polygon.clear();
                for (size_t j = 0; j < poly.vertices.size(); ++j) {
                    const AnnotationPoint& v = poly.vertices[j];
                    polygon.push_back({v.x, v.y});
                }
                ann_mgr.AddAnnotation(dataset_id_, current_image_idx_,
                                      current_class_id_, AnnotationType::Polygon,
                                      polygon, bbox, image_width_, image_height_);
            }
        }
        poly_tool->Clear();
        spdlog::info("Saved polygon annotation");
    }

    // Notify callback
    if (on_annotation_changed_) {
        on_annotation_changed_(dataset_id_, current_image_idx_);
    }
}

void AnnotationEditorPanel::DeleteSelectedAnnotation() {
    if (selected_annotation_id_ < 0 || dataset_id_.empty()) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    if (ann_mgr.RemoveAnnotation(dataset_id_, current_image_idx_, selected_annotation_id_)) {
        spdlog::info("Deleted annotation {}", selected_annotation_id_);
        selected_annotation_id_ = -1;

        if (on_annotation_changed_) {
            on_annotation_changed_(dataset_id_, current_image_idx_);
        }
    }
}

void AnnotationEditorPanel::AddNewClass() {
    if (strlen(new_class_name_) == 0 || dataset_id_.empty()) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    int class_id = ann_mgr.AddClass(dataset_id_, new_class_name_);
    current_class_id_ = class_id;

    spdlog::info("Added class '{}' with id {}", new_class_name_, class_id);
    new_class_name_[0] = '\0';
}

ImVec4 AnnotationEditorPanel::GetClassColor(int class_id) const {
    return CLASS_COLORS[class_id % NUM_CLASS_COLORS];
}

void AnnotationEditorPanel::ExportCOCO() {
    if (dataset_id_.empty() || strlen(export_path_) == 0) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    std::string path = std::string(export_path_) + "/annotations.json";
    if (ann_mgr.ExportCOCO(dataset_id_, path)) {
        spdlog::info("Exported COCO annotations to {}", path);
    } else {
        spdlog::error("Failed to export COCO annotations");
    }
}

void AnnotationEditorPanel::ExportYOLO() {
    if (dataset_id_.empty() || strlen(export_path_) == 0) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    std::string classes_file = std::string(export_path_) + "/classes.txt";
    if (ann_mgr.ExportYOLO(dataset_id_, export_path_, classes_file)) {
        spdlog::info("Exported YOLO annotations to {}", export_path_);
    } else {
        spdlog::error("Failed to export YOLO annotations");
    }
}

void AnnotationEditorPanel::ExportVOC() {
    if (dataset_id_.empty() || strlen(export_path_) == 0) return;

    auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    if (ann_mgr.ExportVOC(dataset_id_, export_path_)) {
        spdlog::info("Exported VOC annotations to {}", export_path_);
    } else {
        spdlog::error("Failed to export VOC annotations");
    }
}

} // namespace cyxwiz
