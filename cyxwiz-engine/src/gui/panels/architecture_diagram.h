#pragma once

#include "../../core/model_analyzer.h"
#include <imgui.h>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * ArchitectureDiagram - Visual network architecture viewer
 *
 * Displays a clean block diagram of the neural network architecture:
 * - Color-coded layer blocks by type
 * - Connections showing data flow
 * - Zoom and pan controls
 * - Layer details on hover
 */
class ArchitectureDiagram {
public:
    ArchitectureDiagram();
    ~ArchitectureDiagram() = default;

    void Render();

    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }
    void RefreshDiagram();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }

private:
    void RenderToolbar();
    void RenderDiagram();
    void RenderLegend();

    // Drawing helpers
    void DrawLayerBlock(ImDrawList* draw_list, const LayerAnalysis& layer,
                        int index, ImVec2 pos, ImVec2 size);
    void DrawConnection(ImDrawList* draw_list, ImVec2 from, ImVec2 to, ImU32 color);
    void DrawArrowHead(ImDrawList* draw_list, ImVec2 tip, ImVec2 direction, float size, ImU32 color);

    // Layout computation
    void ComputeLayout();
    ImVec2 GetBlockPosition(int layer_index) const;
    ImVec2 GetBlockSize(const LayerAnalysis& layer) const;

    // Color helpers
    ImU32 GetLayerColor(gui::NodeType type) const;
    ImU32 GetLayerBorderColor(gui::NodeType type) const;
    std::string GetLayerCategory(gui::NodeType type) const;

    gui::NodeEditor* node_editor_ = nullptr;
    ModelAnalyzer analyzer_;
    ModelAnalysis analysis_;

    bool visible_ = false;

    // Layout settings
    enum class LayoutMode { Vertical, Horizontal, Compact };
    LayoutMode layout_mode_ = LayoutMode::Vertical;

    // View controls
    float zoom_ = 1.0f;
    ImVec2 pan_offset_ = {0, 0};
    bool is_panning_ = false;
    ImVec2 pan_start_;

    // Block dimensions
    float block_width_ = 140.0f;
    float block_height_ = 60.0f;
    float block_spacing_ = 40.0f;
    float block_padding_ = 20.0f;

    // Computed layout positions
    std::vector<ImVec2> block_positions_;
    std::vector<ImVec2> block_sizes_;

    // Interaction state
    int hovered_layer_ = -1;
    int selected_layer_ = -1;

    // Display options
    bool show_shapes_ = true;
    bool show_params_ = true;
    bool show_legend_ = true;
    bool show_connections_ = true;
};

} // namespace cyxwiz
