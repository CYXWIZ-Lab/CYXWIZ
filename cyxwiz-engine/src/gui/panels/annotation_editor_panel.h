#pragma once

#include "../panel.h"
#include "../icons.h"
#include "../../core/annotation_manager.h"
#include "../annotation/annotation_tools.h"
#include <imgui.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace cyxwiz {

/**
 * AnnotationEditorPanel - Standalone annotation workspace for image labeling
 *
 * Extracted from Dataset Manager (Phase 2 - UI Consolidation)
 *
 * Features:
 * - Tool Selection (Rectangle, Brush, Polygon, Point Marker)
 * - Image Navigator (Prev/Next/GoTo with progress indicator)
 * - Image Canvas (zoom/pan with tool overlays)
 * - Annotation List (view/select/delete annotations per image)
 * - Class Management (add/select class labels)
 * - Export Section (COCO JSON, YOLO txt, Pascal VOC XML)
 * - Status Bar (current image info, annotation count)
 */
class AnnotationEditorPanel : public Panel {
public:
    AnnotationEditorPanel();
    ~AnnotationEditorPanel() override;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_DRAW_POLYGON; }
    void HandleKeyboardShortcuts() override;

    // Set active dataset for annotation
    void SetDataset(const std::string& dataset_id);
    const std::string& GetDatasetId() const { return dataset_id_; }

    // Callback when annotations change
    using AnnotationChangedCallback = std::function<void(const std::string& dataset_id, size_t image_idx)>;
    void SetAnnotationChangedCallback(AnnotationChangedCallback callback) { on_annotation_changed_ = callback; }

private:
    // ===== Main Render Sections =====
    void RenderToolbar();
    void RenderToolPanel();
    void RenderImageCanvas();
    void RenderAnnotationListPanel();
    void RenderClassPanel();
    void RenderExportPanel();
    void RenderStatusBar();

    // ===== Image Navigation =====
    void NavigatePrev();
    void NavigateNext();
    void NavigateToIndex(size_t index);
    void LoadCurrentImage();

    size_t current_image_idx_ = 0;
    size_t total_images_ = 0;
    std::string dataset_id_;

    // Current image data
    std::vector<float> current_image_data_;
    int image_width_ = 0;
    int image_height_ = 0;
    int image_channels_ = 0;
    unsigned int image_texture_ = 0;  // OpenGL texture handle

    // ===== Tool Management =====
    enum class Tool {
        None,
        Rectangle,
        Brush,
        Polygon,
        PointMarker
    };

    void SetActiveTool(Tool tool);
    void RenderToolButton(const char* icon, const char* tooltip, Tool tool);

    Tool active_tool_ = Tool::None;
    std::unique_ptr<RectangleSelectTool> rect_tool_;
    std::unique_ptr<BrushTool> brush_tool_;
    std::unique_ptr<PolygonTool> polygon_tool_;
    std::unique_ptr<PointMarkerTool> point_tool_;
    AnnotationTool* current_tool_ = nullptr;

    // ===== Canvas State =====
    float canvas_zoom_ = 1.0f;
    ImVec2 canvas_offset_ = {0.0f, 0.0f};
    bool is_panning_ = false;
    ImVec2 pan_start_;

    void HandleCanvasInput();
    ImVec2 ScreenToImage(ImVec2 screen_pos) const;
    ImVec2 ImageToScreen(ImVec2 image_pos) const;
    ImVec2 ImageToScreen(ImVec2 image_pos, ImVec2 img_origin) const;

    // ===== Annotation List =====
    int selected_annotation_id_ = -1;
    void RenderAnnotationItem(const Annotation& ann, int index);
    void DeleteSelectedAnnotation();
    void SaveCurrentToolAnnotation();

    // ===== Class Management =====
    int current_class_id_ = 0;
    char new_class_name_[64] = "";
    void AddNewClass();
    ImVec4 GetClassColor(int class_id) const;

    // ===== Export =====
    char export_path_[512] = "";
    void ExportCOCO();
    void ExportYOLO();
    void ExportVOC();

    // ===== Layout =====
    float left_panel_width_ = 200.0f;
    float right_panel_width_ = 250.0f;

    // ===== Callbacks =====
    AnnotationChangedCallback on_annotation_changed_;

    // ===== Helpers =====
    void UpdateTexture();
    void ReleaseTexture();
    static constexpr ImVec4 CLASS_COLORS[] = {
        {0.2f, 0.8f, 0.2f, 1.0f},  // Green
        {0.8f, 0.2f, 0.2f, 1.0f},  // Red
        {0.2f, 0.2f, 0.8f, 1.0f},  // Blue
        {0.8f, 0.8f, 0.2f, 1.0f},  // Yellow
        {0.8f, 0.2f, 0.8f, 1.0f},  // Magenta
        {0.2f, 0.8f, 0.8f, 1.0f},  // Cyan
        {0.8f, 0.5f, 0.2f, 1.0f},  // Orange
        {0.5f, 0.2f, 0.8f, 1.0f},  // Purple
    };
    static constexpr int NUM_CLASS_COLORS = 8;
};

} // namespace cyxwiz
