#pragma once

#include "../../core/model_analyzer.h"
#include <string>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * ModelSummaryPanel - Displays model architecture details
 *
 * Shows:
 * - Layer table with name, type, output shape, parameters, FLOPs
 * - Summary statistics (total params, FLOPs, memory)
 * - Shape flow visualization
 * - Export options (text, JSON)
 */
class ModelSummaryPanel {
public:
    ModelSummaryPanel();
    ~ModelSummaryPanel() = default;

    void Render();

    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }
    void RefreshAnalysis();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }

private:
    void RenderToolbar();
    void RenderLayerTable();
    void RenderSummaryStats();
    void RenderShapeFlow();
    void RenderExportOptions();
    void RenderDetailPopup();

    // Export functions
    void ExportAsText();
    void ExportAsJson();
    void CopyToClipboard();

    gui::NodeEditor* node_editor_ = nullptr;
    ModelAnalyzer analyzer_;
    ModelAnalysis analysis_;

    bool visible_ = false;
    bool auto_refresh_ = true;
    int selected_layer_ = -1;
    bool show_flops_ = true;
    bool show_memory_ = false;
    bool show_non_trainable_ = false;
    int batch_size_ = 1;

    // Export dialog state
    bool show_export_dialog_ = false;
    std::string export_path_;

    // Detail popup state
    bool show_detail_popup_ = false;
    int detail_layer_index_ = -1;
};

} // namespace cyxwiz
