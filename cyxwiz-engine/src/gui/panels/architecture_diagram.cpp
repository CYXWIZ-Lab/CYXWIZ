#include "architecture_diagram.h"
#include "../node_editor.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

ArchitectureDiagram::ArchitectureDiagram() {
}

void ArchitectureDiagram::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_DIAGRAM_PROJECT " Architecture Diagram", &visible_, ImGuiWindowFlags_MenuBar)) {
        RenderToolbar();

        if (!analysis_.is_valid) {
            if (analysis_.error_message.empty()) {
                ImGui::TextDisabled("No model to visualize. Create a node graph first.");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", analysis_.error_message.c_str());
            }

            if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
                RefreshDiagram();
            }
        } else {
            RenderDiagram();
        }
    }
    ImGui::End();
}

void ArchitectureDiagram::RenderToolbar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::Text("Layout Mode:");
            if (ImGui::RadioButton("Vertical", layout_mode_ == LayoutMode::Vertical)) {
                layout_mode_ = LayoutMode::Vertical;
                ComputeLayout();
            }
            if (ImGui::RadioButton("Horizontal", layout_mode_ == LayoutMode::Horizontal)) {
                layout_mode_ = LayoutMode::Horizontal;
                ComputeLayout();
            }
            if (ImGui::RadioButton("Compact", layout_mode_ == LayoutMode::Compact)) {
                layout_mode_ = LayoutMode::Compact;
                ComputeLayout();
            }
            ImGui::Separator();
            ImGui::Checkbox("Show Shapes", &show_shapes_);
            ImGui::Checkbox("Show Parameters", &show_params_);
            ImGui::Checkbox("Show Legend", &show_legend_);
            ImGui::Checkbox("Show Connections", &show_connections_);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Zoom")) {
            if (ImGui::MenuItem("Zoom In", "+")) zoom_ = std::min(zoom_ * 1.2f, 3.0f);
            if (ImGui::MenuItem("Zoom Out", "-")) zoom_ = std::max(zoom_ / 1.2f, 0.3f);
            if (ImGui::MenuItem("Reset Zoom", "0")) {
                zoom_ = 1.0f;
                pan_offset_ = ImVec2(0, 0);
            }
            ImGui::Separator();
            ImGui::Text("Current: %.0f%%", zoom_ * 100.0f);
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Toolbar buttons
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshDiagram();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_PLUS)) {
        zoom_ = std::min(zoom_ * 1.2f, 3.0f);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_MINUS)) {
        zoom_ = std::max(zoom_ / 1.2f, 0.3f);
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_EXPAND)) {
        zoom_ = 1.0f;
        pan_offset_ = ImVec2(0, 0);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("| Drag to pan, Scroll to zoom");

    ImGui::Separator();
}

void ArchitectureDiagram::RenderDiagram() {
    if (analysis_.layers.empty()) {
        ImGui::TextDisabled("No layers to display");
        return;
    }

    // Get available region
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();

    // Reserve space for legend
    float legend_height = show_legend_ ? 80.0f : 0.0f;
    canvas_size.y -= legend_height;

    // Create invisible button for interaction
    ImGui::InvisibleButton("canvas", canvas_size,
                          ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);

    bool is_hovered = ImGui::IsItemHovered();
    bool is_active = ImGui::IsItemActive();

    // Handle panning
    if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 delta = ImGui::GetIO().MouseDelta;
        pan_offset_.x += delta.x;
        pan_offset_.y += delta.y;
    }

    // Handle zooming with scroll
    if (is_hovered) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            float old_zoom = zoom_;
            zoom_ = std::clamp(zoom_ + wheel * 0.1f, 0.3f, 3.0f);

            // Zoom toward mouse position
            ImVec2 mouse_pos = ImGui::GetIO().MousePos;
            ImVec2 mouse_rel = ImVec2(mouse_pos.x - canvas_pos.x - pan_offset_.x,
                                       mouse_pos.y - canvas_pos.y - pan_offset_.y);
            float zoom_factor = zoom_ / old_zoom;
            pan_offset_.x -= mouse_rel.x * (zoom_factor - 1.0f);
            pan_offset_.y -= mouse_rel.y * (zoom_factor - 1.0f);
        }
    }

    // Draw background
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                             IM_COL32(25, 25, 30, 255));

    // Draw grid
    ImU32 grid_color = IM_COL32(50, 50, 60, 255);
    float grid_step = 50.0f * zoom_;
    for (float x = fmod(pan_offset_.x, grid_step); x < canvas_size.x; x += grid_step) {
        draw_list->AddLine(ImVec2(canvas_pos.x + x, canvas_pos.y),
                          ImVec2(canvas_pos.x + x, canvas_pos.y + canvas_size.y), grid_color);
    }
    for (float y = fmod(pan_offset_.y, grid_step); y < canvas_size.y; y += grid_step) {
        draw_list->AddLine(ImVec2(canvas_pos.x, canvas_pos.y + y),
                          ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + y), grid_color);
    }

    // Push clipping rect
    draw_list->PushClipRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), true);

    // Detect hover
    hovered_layer_ = -1;
    ImVec2 mouse_pos = ImGui::GetIO().MousePos;

    // Draw connections first (behind blocks)
    if (show_connections_) {
        for (size_t i = 0; i < analysis_.layers.size() - 1; ++i) {
            ImVec2 from_pos = GetBlockPosition(static_cast<int>(i));
            ImVec2 from_size = GetBlockSize(analysis_.layers[i]);
            ImVec2 to_pos = GetBlockPosition(static_cast<int>(i + 1));
            ImVec2 to_size = GetBlockSize(analysis_.layers[i + 1]);

            // Transform positions
            ImVec2 from_screen = ImVec2(canvas_pos.x + pan_offset_.x + from_pos.x * zoom_,
                                         canvas_pos.y + pan_offset_.y + from_pos.y * zoom_);
            ImVec2 to_screen = ImVec2(canvas_pos.x + pan_offset_.x + to_pos.x * zoom_,
                                       canvas_pos.y + pan_offset_.y + to_pos.y * zoom_);

            // Connection points
            ImVec2 from_center, to_center;
            if (layout_mode_ == LayoutMode::Vertical) {
                from_center = ImVec2(from_screen.x + from_size.x * zoom_ / 2,
                                     from_screen.y + from_size.y * zoom_);
                to_center = ImVec2(to_screen.x + to_size.x * zoom_ / 2, to_screen.y);
            } else {
                from_center = ImVec2(from_screen.x + from_size.x * zoom_,
                                     from_screen.y + from_size.y * zoom_ / 2);
                to_center = ImVec2(to_screen.x, to_screen.y + to_size.y * zoom_ / 2);
            }

            DrawConnection(draw_list, from_center, to_center, IM_COL32(150, 150, 180, 200));
        }
    }

    // Draw layer blocks
    for (size_t i = 0; i < analysis_.layers.size(); ++i) {
        const auto& layer = analysis_.layers[i];
        ImVec2 pos = GetBlockPosition(static_cast<int>(i));
        ImVec2 size = GetBlockSize(layer);

        // Transform to screen space
        ImVec2 screen_pos = ImVec2(canvas_pos.x + pan_offset_.x + pos.x * zoom_,
                                    canvas_pos.y + pan_offset_.y + pos.y * zoom_);
        ImVec2 screen_size = ImVec2(size.x * zoom_, size.y * zoom_);

        // Check hover
        if (mouse_pos.x >= screen_pos.x && mouse_pos.x <= screen_pos.x + screen_size.x &&
            mouse_pos.y >= screen_pos.y && mouse_pos.y <= screen_pos.y + screen_size.y) {
            hovered_layer_ = static_cast<int>(i);
        }

        DrawLayerBlock(draw_list, layer, static_cast<int>(i), screen_pos, screen_size);
    }

    // Pop clipping rect
    draw_list->PopClipRect();

    // Hover tooltip
    if (hovered_layer_ >= 0 && hovered_layer_ < static_cast<int>(analysis_.layers.size())) {
        const auto& layer = analysis_.layers[hovered_layer_];
        ImGui::BeginTooltip();
        ImGui::Text("%s", layer.name.empty() ? GetNodeTypeName(layer.type).c_str() : layer.name.c_str());
        ImGui::Separator();
        ImGui::Text("Type: %s", GetNodeTypeName(layer.type).c_str());
        ImGui::Text("Input: %s", FormatShape(layer.input_shape).c_str());
        ImGui::Text("Output: %s", FormatShape(layer.output_shape).c_str());
        if (layer.parameters > 0) {
            ImGui::Text("Parameters: %s", FormatParameterCount(layer.parameters).c_str());
        }
        if (layer.flops > 0) {
            ImGui::Text("FLOPs: %s", FormatFLOPs(layer.flops).c_str());
        }
        ImGui::EndTooltip();
    }

    // Render legend
    if (show_legend_) {
        ImGui::SetCursorScreenPos(ImVec2(canvas_pos.x, canvas_pos.y + canvas_size.y + 5));
        RenderLegend();
    }
}

void ArchitectureDiagram::DrawLayerBlock(ImDrawList* draw_list, const LayerAnalysis& layer,
                                         int index, ImVec2 pos, ImVec2 size) {
    bool is_hovered = (index == hovered_layer_);
    bool is_selected = (index == selected_layer_);

    ImU32 bg_color = GetLayerColor(layer.type);
    ImU32 border_color = GetLayerBorderColor(layer.type);

    if (is_hovered) {
        // Brighten on hover
        bg_color = IM_COL32(
            std::min(255, (int)((bg_color & 0xFF) * 1.3f)),
            std::min(255, (int)(((bg_color >> 8) & 0xFF) * 1.3f)),
            std::min(255, (int)(((bg_color >> 16) & 0xFF) * 1.3f)),
            255
        );
    }

    // Draw rounded rectangle
    float rounding = 8.0f * zoom_;
    draw_list->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), bg_color, rounding);
    draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), border_color, rounding, 0, 2.0f);

    if (is_selected) {
        draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                          IM_COL32(255, 200, 100, 255), rounding, 0, 3.0f);
    }

    // Draw text
    std::string type_name = GetNodeTypeName(layer.type);
    std::string display_name = layer.name.empty() ? type_name : layer.name;

    // Truncate if necessary
    float max_text_width = size.x - 10.0f;
    while (display_name.length() > 3 && ImGui::CalcTextSize(display_name.c_str()).x * zoom_ > max_text_width) {
        display_name = display_name.substr(0, display_name.length() - 4) + "...";
    }

    ImVec2 text_size = ImGui::CalcTextSize(display_name.c_str());
    float text_x = pos.x + (size.x - text_size.x * zoom_) / 2;
    float text_y = pos.y + 5 * zoom_;

    // Scale font (not ideal but works)
    draw_list->AddText(ImVec2(text_x, text_y), IM_COL32(255, 255, 255, 255), display_name.c_str());

    // Draw shape info
    if (show_shapes_ && !layer.output_shape.empty()) {
        std::string shape_text = FormatShape(layer.output_shape);
        if (shape_text.length() > 15) {
            shape_text = shape_text.substr(0, 12) + "...";
        }
        ImVec2 shape_size = ImGui::CalcTextSize(shape_text.c_str());
        float shape_x = pos.x + (size.x - shape_size.x * zoom_) / 2;
        float shape_y = pos.y + size.y / 2;
        draw_list->AddText(ImVec2(shape_x, shape_y), IM_COL32(200, 200, 200, 255), shape_text.c_str());
    }

    // Draw param count
    if (show_params_ && layer.parameters > 0) {
        std::string param_text = FormatParameterCount(layer.parameters);
        ImVec2 param_size = ImGui::CalcTextSize(param_text.c_str());
        float param_x = pos.x + (size.x - param_size.x * zoom_) / 2;
        float param_y = pos.y + size.y - 20 * zoom_;
        draw_list->AddText(ImVec2(param_x, param_y), IM_COL32(180, 180, 180, 200), param_text.c_str());
    }
}

void ArchitectureDiagram::DrawConnection(ImDrawList* draw_list, ImVec2 from, ImVec2 to, ImU32 color) {
    // Draw bezier curve or straight line
    if (layout_mode_ == LayoutMode::Vertical) {
        // Vertical bezier
        float mid_y = (from.y + to.y) / 2;
        draw_list->AddBezierCubic(from,
                                   ImVec2(from.x, mid_y),
                                   ImVec2(to.x, mid_y),
                                   to, color, 2.0f);
    } else {
        // Horizontal bezier
        float mid_x = (from.x + to.x) / 2;
        draw_list->AddBezierCubic(from,
                                   ImVec2(mid_x, from.y),
                                   ImVec2(mid_x, to.y),
                                   to, color, 2.0f);
    }

    // Draw arrow head
    DrawArrowHead(draw_list, to, ImVec2(to.x - from.x, to.y - from.y), 8.0f * zoom_, color);
}

void ArchitectureDiagram::DrawArrowHead(ImDrawList* draw_list, ImVec2 tip, ImVec2 direction, float size, ImU32 color) {
    // Normalize direction
    float len = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    if (len < 0.001f) return;
    direction.x /= len;
    direction.y /= len;

    // Perpendicular
    ImVec2 perp(-direction.y, direction.x);

    ImVec2 p1(tip.x - direction.x * size + perp.x * size * 0.5f,
              tip.y - direction.y * size + perp.y * size * 0.5f);
    ImVec2 p2(tip.x - direction.x * size - perp.x * size * 0.5f,
              tip.y - direction.y * size - perp.y * size * 0.5f);

    draw_list->AddTriangleFilled(tip, p1, p2, color);
}

void ArchitectureDiagram::RenderLegend() {
    ImGui::Text("Legend:");
    ImGui::SameLine();

    struct LegendItem {
        const char* name;
        ImU32 color;
    };

    std::vector<LegendItem> items = {
        {"Dense", IM_COL32(60, 150, 80, 255)},
        {"Conv", IM_COL32(80, 120, 180, 255)},
        {"Pool", IM_COL32(180, 120, 80, 255)},
        {"Norm", IM_COL32(150, 80, 150, 255)},
        {"Activation", IM_COL32(200, 150, 50, 255)},
        {"Recurrent", IM_COL32(80, 150, 180, 255)},
        {"Attention", IM_COL32(180, 80, 120, 255)},
    };

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float x = ImGui::GetCursorScreenPos().x + 60;
    float y = ImGui::GetCursorScreenPos().y;

    for (const auto& item : items) {
        draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + 15, y + 15), item.color, 3.0f);
        draw_list->AddText(ImVec2(x + 20, y), IM_COL32(200, 200, 200, 255), item.name);
        x += ImGui::CalcTextSize(item.name).x + 40;
    }

    ImGui::Dummy(ImVec2(0, 20));
}

void ArchitectureDiagram::ComputeLayout() {
    if (analysis_.layers.empty()) return;

    block_positions_.clear();
    block_sizes_.clear();

    float x = block_padding_;
    float y = block_padding_;
    int cols = 1;
    int current_col = 0;

    if (layout_mode_ == LayoutMode::Compact) {
        // Calculate columns for compact mode
        cols = std::max(1, static_cast<int>(std::sqrt(analysis_.layers.size())));
    }

    for (size_t i = 0; i < analysis_.layers.size(); ++i) {
        block_positions_.push_back(ImVec2(x, y));
        block_sizes_.push_back(GetBlockSize(analysis_.layers[i]));

        switch (layout_mode_) {
            case LayoutMode::Vertical:
                y += block_height_ + block_spacing_;
                break;
            case LayoutMode::Horizontal:
                x += block_width_ + block_spacing_;
                break;
            case LayoutMode::Compact:
                current_col++;
                if (current_col >= cols) {
                    current_col = 0;
                    x = block_padding_;
                    y += block_height_ + block_spacing_;
                } else {
                    x += block_width_ + block_spacing_;
                }
                break;
        }
    }
}

ImVec2 ArchitectureDiagram::GetBlockPosition(int layer_index) const {
    if (layer_index < 0 || layer_index >= static_cast<int>(block_positions_.size())) {
        return ImVec2(0, 0);
    }
    return block_positions_[layer_index];
}

ImVec2 ArchitectureDiagram::GetBlockSize(const LayerAnalysis& layer) const {
    return ImVec2(block_width_, block_height_);
}

ImU32 ArchitectureDiagram::GetLayerColor(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
            return IM_COL32(60, 150, 80, 255);

        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
        case gui::NodeType::DepthwiseConv2D:
            return IM_COL32(80, 120, 180, 255);

        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
        case gui::NodeType::AdaptiveAvgPool:
            return IM_COL32(180, 120, 80, 255);

        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
        case gui::NodeType::GroupNorm:
        case gui::NodeType::InstanceNorm:
            return IM_COL32(150, 80, 150, 255);

        case gui::NodeType::ReLU:
        case gui::NodeType::LeakyReLU:
        case gui::NodeType::PReLU:
        case gui::NodeType::ELU:
        case gui::NodeType::SELU:
        case gui::NodeType::GELU:
        case gui::NodeType::Swish:
        case gui::NodeType::Mish:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return IM_COL32(200, 150, 50, 255);

        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
        case gui::NodeType::Bidirectional:
            return IM_COL32(80, 150, 180, 255);

        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention:
        case gui::NodeType::CrossAttention:
        case gui::NodeType::LinearAttention:
        case gui::NodeType::TransformerEncoder:
        case gui::NodeType::TransformerDecoder:
            return IM_COL32(180, 80, 120, 255);

        case gui::NodeType::Dropout:
            return IM_COL32(100, 100, 100, 255);

        case gui::NodeType::Flatten:
        case gui::NodeType::Reshape:
        case gui::NodeType::Permute:
            return IM_COL32(120, 120, 150, 255);

        default:
            return IM_COL32(100, 100, 100, 255);
    }
}

ImU32 ArchitectureDiagram::GetLayerBorderColor(gui::NodeType type) const {
    ImU32 base = GetLayerColor(type);
    // Brighter border
    return IM_COL32(
        std::min(255, (int)((base & 0xFF) * 1.5f)),
        std::min(255, (int)(((base >> 8) & 0xFF) * 1.5f)),
        std::min(255, (int)(((base >> 16) & 0xFF) * 1.5f)),
        255
    );
}

std::string ArchitectureDiagram::GetLayerCategory(gui::NodeType type) const {
    switch (type) {
        case gui::NodeType::Dense:
            return "Dense";
        case gui::NodeType::Conv1D:
        case gui::NodeType::Conv2D:
        case gui::NodeType::Conv3D:
            return "Convolution";
        case gui::NodeType::MaxPool2D:
        case gui::NodeType::AvgPool2D:
        case gui::NodeType::GlobalMaxPool:
        case gui::NodeType::GlobalAvgPool:
            return "Pooling";
        case gui::NodeType::BatchNorm:
        case gui::NodeType::LayerNorm:
            return "Normalization";
        case gui::NodeType::ReLU:
        case gui::NodeType::Sigmoid:
        case gui::NodeType::Tanh:
        case gui::NodeType::Softmax:
            return "Activation";
        case gui::NodeType::LSTM:
        case gui::NodeType::GRU:
        case gui::NodeType::RNN:
            return "Recurrent";
        case gui::NodeType::MultiHeadAttention:
        case gui::NodeType::SelfAttention:
            return "Attention";
        default:
            return "Other";
    }
}

void ArchitectureDiagram::RefreshDiagram() {
    if (!node_editor_) {
        analysis_.is_valid = false;
        analysis_.error_message = "No node editor connected";
        return;
    }

    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();

    if (nodes.empty()) {
        analysis_.is_valid = false;
        analysis_.error_message = "";
        return;
    }

    analysis_ = analyzer_.AnalyzeGraph(nodes, links, 1);
    ComputeLayout();

    if (analysis_.is_valid) {
        spdlog::info("Architecture diagram refreshed: {} layers", analysis_.layers.size());
    }
}

} // namespace cyxwiz
