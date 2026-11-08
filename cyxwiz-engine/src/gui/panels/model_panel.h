#pragma once

// Tools > Model (TOFIX129 D11): one window for the model the graph on the
// canvas builds. Summary: every layer as compiled - output shape, learnable
// parameters, multiply-adds and activation memory - from the same background
// compile as the Properties panel (core/model_summary_presentation), with
// the selected layer's numbers worked out, advice, Show on canvas, Copy as
// text and Export CSV.

#include "../../core/model_summary_presentation.h"

#include <cstdint>
#include <string>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class ModelPanel {
public:
    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on a tab ("summary").
    void Open(const std::string& tab);
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

private:
    void RenderSummary();
    void RenderTiles();
    void RenderLayers(float height);
    void RenderSide();
    void Refresh();

    bool visible_ = false;
    std::string open_tab_;      // the tab to select on the next frame
    gui::NodeEditor* node_editor_ = nullptr;

    ModelSummary summary_;
    uint64_t summary_key_ = 0;  // the compile the summary was built from
    int batch_ = 0;             // 0: the graph's batch size
    int selected_node_ = -1;
    size_t selected_row_ = 0;
    std::string message_;
    bool message_error_ = false;
};

}  // namespace cyxwiz
