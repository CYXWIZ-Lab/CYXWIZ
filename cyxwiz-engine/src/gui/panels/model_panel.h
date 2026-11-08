#pragma once

// Tools > Model (TOFIX129 D11): one window for the model the graph on the
// canvas builds. Summary: every layer as compiled - output shape, learnable
// parameters, multiply-adds and activation memory - from the same background
// compile as the Properties panel (core/model_summary_presentation), with
// the selected layer's numbers worked out, advice, Show on canvas, Copy as
// text and Export CSV. Learning rate: a range test (Smith) on the graph's
// Train rows through Train's own path - the rate rises every step, the
// trained model in use is kept - its curve, the suggested rate and Use,
// which sets it on the optimizer node. Explain: Grad-CAM, saliency and
// Integrated Gradients of the trained model for a row the last test kept
// (core/model_explain), worked out in the background on the model's device.

#include "../../core/learning_rate_range_test.h"
#include "../../core/model_explain.h"
#include "../../core/model_projector.h"
#include "../../core/model_summary_presentation.h"
#include "../../core/training_health_presentation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

class ModelPanel {
public:
    ~ModelPanel();
    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on a tab ("summary", "learning_rate", "explain", "health", "projector").
    void Open(const std::string& tab);
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }
    // Starts a range test through Train's launcher: "" when it starts (or
    // waits for the data), else why not.
    using StartRangeTest = std::function<std::string(const LearningRateRangeSettings&)>;
    void SetStartRangeTest(StartRangeTest start) { start_range_test_ = std::move(start); }
    // The launch stopped before training (Train's checks): why.
    void RangeLaunchEnded(const std::string& why);
    // The health of the training running now, else of the last one, with
    // the graph's layer names (also the Training Dashboard's notice).
    HealthView CurrentHealth() const;
    // Projector's Send to Projection tab: a table of the layer's features, by name.
    using SendTable = std::function<void(std::shared_ptr<arrow::Table>, const std::string&)>;
    void SetSendToProjection(SendTable send) { send_to_projection_ = std::move(send); }

private:
    void RenderSummary();
    void RenderTiles();
    void RenderLayers(float height);
    void RenderSide();
    void Refresh();
    void RenderLearningRate();
    void RenderRangeSettings();
    void RenderRangeResult();
    void RenderExplain();
    void RenderExplainControls(const std::vector<ExplainRowChoice>& choices, const std::vector<std::string>& class_names);
    void RenderExplainResult(const std::vector<std::string>& class_names);
    void StartExplain(const std::vector<ExplainRowChoice>& choices);
    void UploadExplainImages();
    void ReleaseExplainImages();
    void RenderHealth();
    void RenderProjector();
    void RenderProjectorPlot(const std::vector<std::string>& class_names);

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

    // Explain.
    struct ExplainJob {
        std::mutex mutex;
        bool done = false;
        ExplainOutcome outcome;
    };
    std::shared_ptr<ExplainJob> explain_job_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
    bool explain_wrong_only_ = false;
    int explain_sample_ = -1;        // the chosen test sample
    int explain_target_mode_ = 0;    // 0 predicted, 1 true class, 2 a class typed
    int explain_class_ = 0;
    int ig_steps_ = 64;
    std::string explain_asked_;      // the request in flight or shown (sample, target, steps)
    std::string explain_shown_;
    double test_seen_ = -1.0;        // identity of the test whose rows are listed
    ExplainOutcome explain_;
    ExplainRowChoice explain_row_;
    unsigned int explain_images_[4] = {0, 0, 0, 0};  // input, Grad-CAM, saliency, Integrated Gradients
    int explain_image_size_[2] = {0, 0};
    std::string explain_message_;

    // Health.
    size_t health_selected_ = 0;  // the selected row's layer

    // Projector.
    struct ProjectorJob {
        std::mutex mutex;
        bool done = false;
        ProjectorOutcome outcome;
    };
    std::shared_ptr<ProjectorJob> projector_job_;
    size_t projector_layer_ = static_cast<size_t>(-1);
    int projector_method_ = 0;        // ProjectorMethod
    int projector_colour_ = 0;        // 0 true class, 1 predicted, 2 right / wrong
    int projector_only_ = -1;         // a class picked in the legend
    std::string projector_asked_, projector_shown_;
    ProjectorOutcome projector_;
    std::string projector_message_;
    SendTable send_to_projection_;
};

}  // namespace cyxwiz
