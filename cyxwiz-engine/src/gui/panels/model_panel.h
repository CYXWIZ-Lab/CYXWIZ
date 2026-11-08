#pragma once

// Tools > Model (TOFIX129 D11): one window for the model the graph on the
// canvas builds. Summary: every layer as compiled - output shape, learnable
// parameters, multiply-adds and activation memory - from the same background
// compile as the Properties panel (core/model_summary_presentation), with
// the selected layer's numbers worked out, advice, Show on canvas, Copy as
// text and Export CSV. Learning rate: a range test (Smith) on the graph's
// Train rows through Train's own path - the rate rises every step, the
// trained model in use is kept - its curve, the suggested rate and Use,
// which sets it on the optimizer node.

#include "../../core/learning_rate_range_test.h"
#include "../../core/model_summary_presentation.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class ModelPanel {
public:
    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on a tab ("summary", "learning_rate").
    void Open(const std::string& tab);
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }
    // Starts a range test through Train's launcher: "" when it starts (or
    // waits for the data), else why not.
    using StartRangeTest = std::function<std::string(const LearningRateRangeSettings&)>;
    void SetStartRangeTest(StartRangeTest start) { start_range_test_ = std::move(start); }
    // The launch stopped before training (Train's checks): why.
    void RangeLaunchEnded(const std::string& why);

private:
    void RenderSummary();
    void RenderTiles();
    void RenderLayers(float height);
    void RenderSide();
    void Refresh();
    void RenderLearningRate();
    void RenderRangeSettings();
    void RenderRangeResult();

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

    // Learning rate.
    StartRangeTest start_range_test_;
    LearningRateRangeSettings range_ = [] {
        LearningRateRangeSettings s;
        s.enabled = true;
        return s;
    }();
    bool range_running_ = false;          // started; TrainingManager has the run
    uint64_t range_seen_serial_ = 0;      // the last finished test taken in
    std::vector<LearningRateRangePoint> range_points_;
    std::string range_stop_reason_;
    std::string range_message_;
    bool range_message_error_ = false;
    bool steepest_rule_ = true;
};

}  // namespace cyxwiz
