/**
 * Dataset Panel - Interactive Tools Tab
 *
 * Provides interactive annotation and segmentation tools:
 * - GrabCut: Rectangle selection + brush refinement for foreground extraction
 * - Watershed: Marker-based region segmentation
 * - Brush: Manual mask painting
 * - Polygon: Vertex-based region annotation
 */

#include "dataset_panel.h"
#include "../annotation/annotation_tools.h"
#include "../annotation/grabcut_editor.h"
#include "../annotation/watershed_editor.h"
#include "../icons.h"
#include "../../core/image_utils.h"
#include "../../core/data_registry.h"
#include "../../core/annotation_manager.h"
#include "../../utils/labeling_utils.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace gui {

// Static state for interactive tools (shared across renders)
static struct InteractiveToolsState {
    // Tool selection
    enum class ActiveTool { None, GrabCut, Watershed, Brush, Polygon };
    ActiveTool active_tool = ActiveTool::None;

    // Tool instances
    std::unique_ptr<cyxwiz::GrabCutEditor> grabcut_editor;
    std::unique_ptr<cyxwiz::WatershedEditor> watershed_editor;
    std::unique_ptr<cyxwiz::BrushTool> brush_tool;
    std::unique_ptr<cyxwiz::PolygonTool> polygon_tool;

    // Image state
    std::vector<float> current_image;
    int image_width = 0;
    int image_height = 0;
    int image_channels = 0;
    int current_sample_idx = 0;
    bool image_loaded = false;

    // Display state
    float zoom = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    bool show_mask_overlay = true;
    float mask_opacity = 0.5f;

    // Export state
    char export_path[512] = "mask.png";
    bool export_as_binary = true;

    // Annotation state
    std::string current_dataset_id;
    int current_class_id = 0;
    char new_class_name[128] = "";
    char export_dir[512] = "annotations";
    int selected_annotation_id = -1;
    bool show_all_annotations = true;
    int goto_image_idx = 0;

    // Initialize tools
    void Initialize() {
        if (!grabcut_editor) {
            grabcut_editor = std::make_unique<cyxwiz::GrabCutEditor>();
        }
        if (!watershed_editor) {
            watershed_editor = std::make_unique<cyxwiz::WatershedEditor>();
        }
        if (!brush_tool) {
            brush_tool = std::make_unique<cyxwiz::BrushTool>();
        }
        if (!polygon_tool) {
            polygon_tool = std::make_unique<cyxwiz::PolygonTool>();
        }
    }

    // Load image into tools
    void LoadImage(const std::vector<float>& image, int w, int h, int c) {
        current_image = image;
        image_width = w;
        image_height = h;
        image_channels = c;
        image_loaded = true;

        // Set image for all tools
        if (grabcut_editor) grabcut_editor->SetImage(image, w, h, c);
        if (watershed_editor) watershed_editor->SetImage(image, w, h, c);
        if (brush_tool) brush_tool->SetImage(image, w, h, c);
        if (polygon_tool) polygon_tool->SetImage(image, w, h, c);
    }

    // Get current mask from active tool
    std::vector<float> GetCurrentMask() const {
        switch (active_tool) {
            case ActiveTool::GrabCut:
                if (grabcut_editor && grabcut_editor->HasSegmentation())
                    return grabcut_editor->GetFloatMask();
                break;
            case ActiveTool::Watershed:
                if (watershed_editor && watershed_editor->HasSegmentation())
                    return watershed_editor->GetFloatMask();
                break;
            case ActiveTool::Brush:
                if (brush_tool)
                    return brush_tool->GetMask().ToFloatMask();
                break;
            case ActiveTool::Polygon:
                if (polygon_tool)
                    return polygon_tool->GetMask().ToFloatMask();
                break;
            default:
                break;
        }
        return {};
    }

    // Check if there's a valid mask
    bool HasMask() const {
        return !GetCurrentMask().empty();
    }

} g_interactive_state;

// Helper to convert screen coords to image coords
static bool ScreenToImageCoords(float screen_x, float screen_y,
                                 float img_offset_x, float img_offset_y,
                                 float scale, int img_w, int img_h,
                                 float& out_x, float& out_y) {
    out_x = (screen_x - img_offset_x) / scale;
    out_y = (screen_y - img_offset_y) / scale;
    return out_x >= 0 && out_x < img_w && out_y >= 0 && out_y < img_h;
}

void DatasetPanel::RenderInteractiveToolsTab() {
    auto& state = g_interactive_state;

    state.Initialize();

    // Get annotation manager
    auto& ann_mgr = cyxwiz::DataRegistry::Instance().GetAnnotationManager();

    // Update dataset ID if we have a valid dataset
    if (current_dataset_.IsValid()) {
        state.current_dataset_id = current_dataset_.GetName();
        // Ensure annotation set exists
        if (!ann_mgr.HasAnnotationSet(state.current_dataset_id)) {
            ann_mgr.CreateAnnotationSet(state.current_dataset_id);
        }
    }

    // Tool selection toolbar
    ImGui::Text("Interactive Annotation Tools");

    // =========================================================================
    // Batch Navigation Section
    // =========================================================================
    if (current_dataset_.IsValid()) {
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 400);
        ImGui::Text("Dataset: %s (%zu images)", state.current_dataset_id.c_str(),
                    current_dataset_.Size());
    }
    ImGui::Separator();

    // Navigation controls (only show if dataset is loaded)
    if (current_dataset_.IsValid()) {
        size_t dataset_size = current_dataset_.Size();

        // Prev button
        if (ImGui::Button(ICON_FA_CHEVRON_LEFT " Prev")) {
            if (state.current_sample_idx > 0) {
                state.current_sample_idx--;
                // Load new image
                auto [sample, label] = current_dataset_.GetSample(state.current_sample_idx);
                int width = static_cast<int>(cached_info_.shape[0]);
                int height = static_cast<int>(cached_info_.shape[1]);
                int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;
                state.LoadImage(sample, width, height, channels);
            }
        }
        ImGui::SameLine();

        // Image counter
        ImGui::Text("%d / %zu", state.current_sample_idx + 1, dataset_size);
        ImGui::SameLine();

        // Next button
        if (ImGui::Button("Next " ICON_FA_CHEVRON_RIGHT)) {
            if (state.current_sample_idx < static_cast<int>(dataset_size) - 1) {
                state.current_sample_idx++;
                // Load new image
                auto [sample, label] = current_dataset_.GetSample(state.current_sample_idx);
                int width = static_cast<int>(cached_info_.shape[0]);
                int height = static_cast<int>(cached_info_.shape[1]);
                int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;
                state.LoadImage(sample, width, height, channels);
            }
        }
        ImGui::SameLine();

        // Go to specific image
        ImGui::SetNextItemWidth(80);
        if (ImGui::InputInt("##goto", &state.goto_image_idx, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
            state.goto_image_idx = std::clamp(state.goto_image_idx, 0, static_cast<int>(dataset_size) - 1);
            state.current_sample_idx = state.goto_image_idx;
            auto [sample, label] = current_dataset_.GetSample(state.current_sample_idx);
            int width = static_cast<int>(cached_info_.shape[0]);
            int height = static_cast<int>(cached_info_.shape[1]);
            int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;
            state.LoadImage(sample, width, height, channels);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(Enter to jump)");

        // Show annotation count for current image
        auto* anns = ann_mgr.GetAnnotations(state.current_dataset_id, state.current_sample_idx);
        if (anns && !anns->empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "[%zu annotations]", anns->size());
        }

        ImGui::Separator();
    }

    // =========================================================================

    // Tool buttons
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

    const char* tool_names[] = { "Select", "GrabCut", "Watershed", "Brush", "Polygon" };
    const char* tool_tips[] = {
        "No tool selected - pan and zoom only",
        "GrabCut: Rectangle selection for foreground extraction",
        "Watershed: Marker-based region segmentation",
        "Brush: Manual mask painting",
        "Polygon: Click to add vertices, close to fill"
    };

    for (int i = 0; i < 5; ++i) {
        bool is_selected = (static_cast<int>(state.active_tool) == i);

        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }

        if (ImGui::Button(tool_names[i], ImVec2(70, 30))) {
            state.active_tool = static_cast<InteractiveToolsState::ActiveTool>(i);
        }

        if (is_selected) {
            ImGui::PopStyleColor();
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", tool_tips[i]);
        }

        if (i < 4) ImGui::SameLine();
    }

    ImGui::PopStyleVar();

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Zoom controls
    if (ImGui::Button("-##zoom")) {
        state.zoom = std::max(0.1f, state.zoom * 0.8f);
    }
    ImGui::SameLine();
    ImGui::Text("%.0f%%", state.zoom * 100);
    ImGui::SameLine();
    if (ImGui::Button("+##zoom")) {
        state.zoom = std::min(10.0f, state.zoom * 1.25f);
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit")) {
        state.zoom = 1.0f;
        state.pan_x = 0;
        state.pan_y = 0;
    }

    ImGui::Separator();

    // Main area: image view + controls
    float controls_width = 280.0f;
    ImVec2 avail = ImGui::GetContentRegionAvail();

    // Image viewport
    ImGui::BeginChild("ImageViewport", ImVec2(avail.x - controls_width - 10, avail.y), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        size_t expected_sz = static_cast<size_t>(state.image_width) * state.image_height * state.image_channels;
        bool valid_img = state.image_loaded && state.image_width > 0 && state.image_height > 0 && state.image_channels > 0 && state.current_image.size() == expected_sz;
        if (!valid_img) {
            if (state.image_loaded) state.image_loaded = false;
            ImVec2 win_size = ImGui::GetWindowSize();
            ImVec2 text_size = ImGui::CalcTextSize("No image loaded");
            ImGui::SetCursorPos(ImVec2((win_size.x - text_size.x) / 2, win_size.y / 2 - 30));
            ImGui::TextDisabled("No image loaded");

            ImGui::SetCursorPos(ImVec2((win_size.x - 160) / 2, win_size.y / 2));
            ImGui::TextWrapped("Load an image dataset and select a sample from the Preview tab to begin annotating.");
        } else {
            // Render image with annotations
            ImVec2 win_pos = ImGui::GetCursorScreenPos();
            ImVec2 win_size = ImGui::GetContentRegionAvail();

            float scale = state.zoom;
            float img_w = state.image_width * scale;
            float img_h = state.image_height * scale;

            // Center image
            float offset_x = win_pos.x + (win_size.x - img_w) / 2 + state.pan_x;
            float offset_y = win_pos.y + (win_size.y - img_h) / 2 + state.pan_y;

            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Draw image background (checkerboard for transparency)
            draw_list->AddRectFilled(
                ImVec2(offset_x, offset_y),
                ImVec2(offset_x + img_w, offset_y + img_h),
                IM_COL32(60, 60, 60, 255)
            );

            // Draw simple visualization of image (grayscale intensity blocks)
            int step = std::max(1, static_cast<int>(4 / scale));
            for (int y = 0; y < state.image_height; y += step) {
                for (int x = 0; x < state.image_width; x += step) {
                    int idx = (y * state.image_width + x) * state.image_channels;
                    float gray = state.current_image[idx];
                    if (state.image_channels >= 3) {
                        gray = 0.299f * state.current_image[idx] +
                               0.587f * state.current_image[idx + 1] +
                               0.114f * state.current_image[idx + 2];
                    }
                    uint8_t v = static_cast<uint8_t>(gray * 255);

                    ImVec2 p1(offset_x + x * scale, offset_y + y * scale);
                    ImVec2 p2(offset_x + (x + step) * scale, offset_y + (y + step) * scale);
                    draw_list->AddRectFilled(p1, p2, IM_COL32(v, v, v, 255));
                }
            }

            // Draw mask overlay
            if (state.show_mask_overlay) {
                std::vector<float> mask;
                if (state.active_tool == InteractiveToolsState::ActiveTool::GrabCut &&
                    state.grabcut_editor && state.grabcut_editor->HasSegmentation()) {
                    mask = state.grabcut_editor->GetFloatMask();
                } else if (state.active_tool == InteractiveToolsState::ActiveTool::Watershed &&
                           state.watershed_editor && state.watershed_editor->HasSegmentation()) {
                    mask = state.watershed_editor->GetFloatMask();
                } else if (state.active_tool == InteractiveToolsState::ActiveTool::Brush &&
                           state.brush_tool) {
                    mask = state.brush_tool->GetMask().ToFloatMask();
                } else if (state.active_tool == InteractiveToolsState::ActiveTool::Polygon &&
                           state.polygon_tool) {
                    mask = state.polygon_tool->GetMask().ToFloatMask();
                }

                if (!mask.empty()) {
                    uint8_t alpha = static_cast<uint8_t>(state.mask_opacity * 200);
                    for (int y = 0; y < state.image_height; y += step) {
                        for (int x = 0; x < state.image_width; x += step) {
                            if (mask[y * state.image_width + x] > 0.5f) {
                                ImVec2 p1(offset_x + x * scale, offset_y + y * scale);
                                ImVec2 p2(offset_x + (x + step) * scale, offset_y + (y + step) * scale);
                                draw_list->AddRectFilled(p1, p2, IM_COL32(0, 255, 0, alpha));
                            }
                        }
                    }
                }
            }

            // Draw tool overlay
            switch (state.active_tool) {
                case InteractiveToolsState::ActiveTool::GrabCut:
                    if (state.grabcut_editor) {
                        state.grabcut_editor->RenderOverlay(draw_list, offset_x, offset_y, scale);
                    }
                    break;
                case InteractiveToolsState::ActiveTool::Watershed:
                    if (state.watershed_editor) {
                        state.watershed_editor->RenderOverlay(draw_list, offset_x, offset_y, scale);
                    }
                    break;
                case InteractiveToolsState::ActiveTool::Brush:
                    if (state.brush_tool) {
                        state.brush_tool->RenderOverlay(draw_list, offset_x, offset_y, scale);
                    }
                    break;
                case InteractiveToolsState::ActiveTool::Polygon:
                    if (state.polygon_tool) {
                        state.polygon_tool->RenderOverlay(draw_list, offset_x, offset_y, scale);
                    }
                    break;
                default:
                    break;
            }

            // Handle mouse input
            if (ImGui::IsWindowHovered()) {
                ImGuiIO& io = ImGui::GetIO();
                ImVec2 mouse = io.MousePos;

                float img_x, img_y;
                bool in_image = ScreenToImageCoords(mouse.x, mouse.y, offset_x, offset_y, scale,
                                                     state.image_width, state.image_height,
                                                     img_x, img_y);

                // Zoom with scroll
                if (io.MouseWheel != 0 && !io.KeyCtrl) {
                    float old_zoom = state.zoom;
                    state.zoom *= (io.MouseWheel > 0) ? 1.1f : 0.9f;
                    state.zoom = std::clamp(state.zoom, 0.1f, 10.0f);
                }

                // Pan with middle mouse or when no tool is selected
                if (io.MouseDown[2] || (state.active_tool == InteractiveToolsState::ActiveTool::None && io.MouseDown[0])) {
                    state.pan_x += io.MouseDelta.x;
                    state.pan_y += io.MouseDelta.y;
                }

                // Tool input
                if (in_image && state.active_tool != InteractiveToolsState::ActiveTool::None) {
                    cyxwiz::MouseButton button = cyxwiz::MouseButton::None;
                    if (io.MouseDown[0]) button = cyxwiz::MouseButton::Left;
                    else if (io.MouseDown[1]) button = cyxwiz::MouseButton::Right;

                    bool clicked = ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1);
                    bool released = ImGui::IsMouseReleased(0) || ImGui::IsMouseReleased(1);
                    bool dragging = io.MouseDown[0] || io.MouseDown[1];

                    switch (state.active_tool) {
                        case InteractiveToolsState::ActiveTool::GrabCut:
                            if (state.grabcut_editor) {
                                if (clicked) state.grabcut_editor->OnMouseDown(img_x, img_y, button);
                                if (released) state.grabcut_editor->OnMouseUp(img_x, img_y, button);
                                state.grabcut_editor->OnMouseMove(img_x, img_y, dragging);
                                if (io.MouseWheel != 0 && io.KeyCtrl) {
                                    state.grabcut_editor->OnMouseScroll(io.MouseWheel);
                                }
                            }
                            break;
                        case InteractiveToolsState::ActiveTool::Watershed:
                            if (state.watershed_editor) {
                                if (clicked) state.watershed_editor->OnMouseDown(img_x, img_y, button);
                                if (released) state.watershed_editor->OnMouseUp(img_x, img_y, button);
                                state.watershed_editor->OnMouseMove(img_x, img_y, dragging);
                                if (io.MouseWheel != 0 && io.KeyCtrl) {
                                    state.watershed_editor->OnMouseScroll(io.MouseWheel);
                                }
                            }
                            break;
                        case InteractiveToolsState::ActiveTool::Brush:
                            if (state.brush_tool) {
                                if (clicked) state.brush_tool->OnMouseDown(img_x, img_y, button);
                                if (released) state.brush_tool->OnMouseUp(img_x, img_y, button);
                                state.brush_tool->OnMouseMove(img_x, img_y, dragging);
                                if (io.MouseWheel != 0 && io.KeyCtrl) {
                                    state.brush_tool->OnMouseScroll(io.MouseWheel);
                                }
                            }
                            break;
                        case InteractiveToolsState::ActiveTool::Polygon:
                            if (state.polygon_tool) {
                                if (clicked) state.polygon_tool->OnMouseDown(img_x, img_y, button);
                                if (released) state.polygon_tool->OnMouseUp(img_x, img_y, button);
                                state.polygon_tool->OnMouseMove(img_x, img_y, dragging);
                            }
                            break;
                        default:
                            break;
                    }
                }

                // Keyboard shortcuts
                if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                    switch (state.active_tool) {
                        case InteractiveToolsState::ActiveTool::GrabCut:
                            if (state.grabcut_editor) state.grabcut_editor->OnKeyDown(ImGuiKey_Enter);
                            break;
                        case InteractiveToolsState::ActiveTool::Watershed:
                            if (state.watershed_editor) state.watershed_editor->OnKeyDown(ImGuiKey_Enter);
                            break;
                        case InteractiveToolsState::ActiveTool::Polygon:
                            if (state.polygon_tool) state.polygon_tool->OnKeyDown(ImGuiKey_Enter);
                            break;
                        default:
                            break;
                    }
                }

                if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                    switch (state.active_tool) {
                        case InteractiveToolsState::ActiveTool::GrabCut:
                            if (state.grabcut_editor) state.grabcut_editor->OnKeyDown(ImGuiKey_Escape);
                            break;
                        case InteractiveToolsState::ActiveTool::Watershed:
                            if (state.watershed_editor) state.watershed_editor->OnKeyDown(ImGuiKey_Escape);
                            break;
                        case InteractiveToolsState::ActiveTool::Polygon:
                            if (state.polygon_tool) state.polygon_tool->OnKeyDown(ImGuiKey_Escape);
                            break;
                        default:
                            break;
                    }
                }

                // Undo/Redo
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z)) {
                    if (state.brush_tool && state.active_tool == InteractiveToolsState::ActiveTool::Brush) {
                        state.brush_tool->Undo();
                    }
                    if (state.polygon_tool && state.active_tool == InteractiveToolsState::ActiveTool::Polygon) {
                        state.polygon_tool->Undo();
                    }
                }
                if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y)) {
                    if (state.brush_tool && state.active_tool == InteractiveToolsState::ActiveTool::Brush) {
                        state.brush_tool->Redo();
                    }
                    if (state.polygon_tool && state.active_tool == InteractiveToolsState::ActiveTool::Polygon) {
                        state.polygon_tool->Redo();
                    }
                }
            }
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Tool controls panel
    ImGui::BeginChild("ToolControls", ImVec2(controls_width, avail.y), true);
    {
        // Image info
        if (state.image_loaded) {
            ImGui::Text("Image: %dx%d, %d ch", state.image_width, state.image_height, state.image_channels);
            ImGui::Separator();
        }

        // Mask display options
        ImGui::Checkbox("Show Mask Overlay", &state.show_mask_overlay);
        if (state.show_mask_overlay) {
            ImGui::SliderFloat("Opacity", &state.mask_opacity, 0.0f, 1.0f);
        }

        ImGui::Separator();

        // Tool-specific controls
        switch (state.active_tool) {
            case InteractiveToolsState::ActiveTool::GrabCut:
                if (state.grabcut_editor) {
                    state.grabcut_editor->RenderControls();
                }
                break;
            case InteractiveToolsState::ActiveTool::Watershed:
                if (state.watershed_editor) {
                    state.watershed_editor->RenderControls();
                }
                break;
            case InteractiveToolsState::ActiveTool::Brush:
                if (state.brush_tool) {
                    state.brush_tool->RenderControls();
                }
                break;
            case InteractiveToolsState::ActiveTool::Polygon:
                if (state.polygon_tool) {
                    state.polygon_tool->RenderControls();
                }
                break;
            default:
                ImGui::TextDisabled("Select a tool to begin annotating");
                ImGui::Spacing();
                ImGui::TextWrapped("Available tools:");
                ImGui::BulletText("GrabCut - Draw rectangle, then refine with brush");
                ImGui::BulletText("Watershed - Place markers for region segmentation");
                ImGui::BulletText("Brush - Paint foreground/background directly");
                ImGui::BulletText("Polygon - Draw polygon regions");
                break;
        }

        // Export section
        if (state.image_loaded) {
            ImGui::Separator();
            ImGui::Text("Export Mask");

            ImGui::InputText("Path", state.export_path, sizeof(state.export_path));
            ImGui::Checkbox("Binary Mask", &state.export_as_binary);

            if (ImGui::Button("Export Mask", ImVec2(-1, 0))) {
                std::vector<float> mask;
                switch (state.active_tool) {
                    case InteractiveToolsState::ActiveTool::GrabCut:
                        if (state.grabcut_editor) mask = state.grabcut_editor->GetFloatMask();
                        break;
                    case InteractiveToolsState::ActiveTool::Watershed:
                        if (state.watershed_editor) mask = state.watershed_editor->GetFloatMask();
                        break;
                    case InteractiveToolsState::ActiveTool::Brush:
                        if (state.brush_tool) mask = state.brush_tool->GetMask().ToFloatMask();
                        break;
                    case InteractiveToolsState::ActiveTool::Polygon:
                        if (state.polygon_tool) mask = state.polygon_tool->GetMask().ToFloatMask();
                        break;
                    default:
                        break;
                }

                if (!mask.empty()) {
                    if (cyxwiz::ImageUtils::SaveMask(state.export_path, mask,
                                                      state.image_width, state.image_height,
                                                      state.export_as_binary)) {
                        spdlog::info("Exported mask to: {}", state.export_path);
                    }
                } else {
                    spdlog::warn("No mask to export");
                }
            }
        }

        // =====================================================================
        // Annotation Management Section
        // =====================================================================
        if (!state.current_dataset_id.empty()) {
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), ICON_FA_TAGS " Annotations");

            // Current image annotations list
            auto* anns = ann_mgr.GetAnnotations(state.current_dataset_id, state.current_sample_idx);
            if (anns && !anns->empty()) {
                ImGui::Text("Image %d: %zu annotations", state.current_sample_idx, anns->size());

                ImGui::BeginChild("AnnotationList", ImVec2(-1, 120), true);
                for (const auto& ann : *anns) {
                    bool selected = (state.selected_annotation_id == ann.id);
                    std::string label = std::to_string(ann.id) + ": " + ann.class_name +
                                        " (" + cyxwiz::AnnotationTypeToString(ann.type) + ")";

                    if (ImGui::Selectable(label.c_str(), selected)) {
                        state.selected_annotation_id = selected ? -1 : ann.id;
                    }

                    // Delete button on same line
                    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 50);
                    ImGui::PushID(ann.id);
                    if (ImGui::SmallButton(ICON_FA_TRASH)) {
                        ann_mgr.RemoveAnnotation(state.current_dataset_id, state.current_sample_idx, ann.id);
                    }
                    ImGui::PopID();
                }
                ImGui::EndChild();
            } else {
                ImGui::TextDisabled("No annotations for this image");
            }

            // Class selector
            ImGui::Separator();
            ImGui::Text("Class:");
            const auto& classes = ann_mgr.GetClasses(state.current_dataset_id);

            if (!classes.empty()) {
                if (ImGui::BeginCombo("##class", classes[state.current_class_id].c_str())) {
                    for (int i = 0; i < static_cast<int>(classes.size()); ++i) {
                        bool is_selected = (state.current_class_id == i);
                        if (ImGui::Selectable(classes[i].c_str(), is_selected)) {
                            state.current_class_id = i;
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                ImGui::TextDisabled("No classes defined");
            }

            // Add new class
            ImGui::SetNextItemWidth(100);
            ImGui::InputText("##newclass", state.new_class_name, sizeof(state.new_class_name));
            ImGui::SameLine();
            if (ImGui::Button("+ Add Class")) {
                if (strlen(state.new_class_name) > 0) {
                    state.current_class_id = ann_mgr.AddClass(state.current_dataset_id, state.new_class_name);
                    state.new_class_name[0] = '\0';
                }
            }

            // Add annotation from current mask
            ImGui::Separator();
            bool has_mask = state.HasMask();
            if (!has_mask) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button(ICON_FA_PLUS " Add Annotation from Mask", ImVec2(-1, 0))) {
                std::vector<float> mask = state.GetCurrentMask();
                if (!mask.empty() && state.image_width > 0 && state.image_height > 0) {
                    // Extract contours from mask
                    auto contours = cyxwiz::LabelingUtils::ExtractContours(
                        mask, state.image_width, state.image_height,
                        cyxwiz::ContourMode::External, cyxwiz::ContourApprox::Simple);

                    if (!contours.empty()) {
                        // Use largest contour
                        auto& largest = contours[0];
                        for (const auto& c : contours) {
                            if (c.points.size() > largest.points.size()) {
                                largest = c;
                            }
                        }

                        // Add annotation
                        int ann_id = ann_mgr.AddAnnotation(
                            state.current_dataset_id,
                            state.current_sample_idx,
                            state.current_class_id,
                            cyxwiz::AnnotationType::Polygon,
                            largest.points,
                            cyxwiz::BoundingBox{},
                            state.image_width,
                            state.image_height);

                        spdlog::info("Added annotation {} to image {}", ann_id, state.current_sample_idx);
                    }
                }
            }
            if (ImGui::IsItemHovered() && !has_mask) {
                ImGui::SetTooltip("Use a tool to create a mask first");
            }

            if (!has_mask) {
                ImGui::EndDisabled();
            }

            // Export annotations
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.8f, 1.0f), ICON_FA_DOWNLOAD " Export Annotations");
            ImGui::InputText("Dir##export", state.export_dir, sizeof(state.export_dir));

            if (ImGui::Button("COCO JSON", ImVec2(-1, 0))) {
                std::string coco_path = std::string(state.export_dir) + "/annotations.json";
                if (ann_mgr.ExportCOCO(state.current_dataset_id, coco_path)) {
                    spdlog::info("Exported COCO annotations to: {}", coco_path);
                }
            }
            if (ImGui::Button("YOLO txt", ImVec2(-1, 0))) {
                if (ann_mgr.ExportYOLO(state.current_dataset_id, state.export_dir,
                                        std::string(state.export_dir) + "/classes.txt")) {
                    spdlog::info("Exported YOLO annotations to: {}", state.export_dir);
                }
            }
            if (ImGui::Button("Pascal VOC", ImVec2(-1, 0))) {
                if (ann_mgr.ExportVOC(state.current_dataset_id, state.export_dir)) {
                    spdlog::info("Exported VOC annotations to: {}", state.export_dir);
                }
            }

            // Show stats
            auto* ann_set = ann_mgr.GetAnnotationSet(state.current_dataset_id);
            if (ann_set) {
                ImGui::Separator();
                ImGui::TextDisabled("Total: %zu annotations, %zu images",
                                    ann_set->GetTotalAnnotationCount(),
                                    ann_set->GetAnnotatedImageCount());
            }
        }

        // Load from dataset button
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Load Image");

        // Check if dataset is loaded with image data
        bool can_load_from_dataset = current_dataset_.IsValid() &&
                                      !cached_info_.shape.empty() &&
                                      cached_info_.shape.size() >= 2;

        if (!can_load_from_dataset) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Load from Dataset (Preview)", ImVec2(-1, 0))) {
            // Get current preview sample from dataset
            auto [sample, label] = current_dataset_.GetSample(preview_sample_idx_);

            int width = static_cast<int>(cached_info_.shape[0]);
            int height = static_cast<int>(cached_info_.shape[1]);
            int channels = cached_info_.shape.size() > 2 ? static_cast<int>(cached_info_.shape[2]) : 1;

            if (sample.size() == static_cast<size_t>(width * height * channels)) {
                state.LoadImage(sample, width, height, channels);
                state.current_sample_idx = preview_sample_idx_;
                spdlog::info("Loaded dataset sample {} ({}x{}x{})", preview_sample_idx_, width, height, channels);
            } else {
                spdlog::warn("Sample size mismatch: expected {}, got {}",
                            width * height * channels, sample.size());
            }
        }
        if (ImGui::IsItemHovered() && !can_load_from_dataset) {
            ImGui::SetTooltip("Load a dataset first in the Preview tab");
        }

        if (!can_load_from_dataset) {
            ImGui::EndDisabled();
        }

        // Load test image button (for development)
        if (ImGui::Button("Load Test Image (64x64)", ImVec2(-1, 0))) {
            // Create a simple test image
            std::vector<float> test_image(64 * 64 * 3);
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    int idx = (y * 64 + x) * 3;
                    // Gradient pattern
                    test_image[idx + 0] = x / 64.0f;  // R
                    test_image[idx + 1] = y / 64.0f;  // G
                    test_image[idx + 2] = 0.5f;       // B
                }
            }
            state.LoadImage(test_image, 64, 64, 3);
            spdlog::info("Loaded test image 64x64");
        }
    }
    ImGui::EndChild();
}

} // namespace gui
