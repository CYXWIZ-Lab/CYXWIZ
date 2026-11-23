#pragma once

#include "../../core/engine_config.h"
#include "../../core/nas_evaluator.h"
#include "../../core/nas_presentation.h"
#include "../../core/trial_runner.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

// The architecture search's settings (the panel and Preferences > Search):
// versions per round, rounds, kept, changed and made-from-two shares; with
// `train_best`, also how many of the best to train. True when one changed.
bool ArchitectureSearchFields(NASSearchConfig& config, int* train_best);

/**
 * Neural Architecture Search (Tools > Model; TOFIX178 T4 H2, approved mockup)
 *
 * Reads the canvas graph and its compile (input and output come from the
 * graph). The quick score card (no training); changes to the model, each a
 * step with its score, applied to the canvas (Ctrl+Z undoes) or dropped;
 * an evolutionary search on the quick score whose best versions can be
 * trained the way Train does (Train's checks and data preparation, then
 * core/trial_runner) and applied.
 */
class NASPanel {
public:
    NASPanel();
    ~NASPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }
    // Train the best N: Train's checks and preparation for `nodes` / `links`
    // (the graph as read), ending in BeginTraining or NotStarted.
    // `launch_pending` is true while that launch waits (data loads, the
    // memory question).
    void SetTrainCallback(std::function<void(std::vector<gui::MLNode>, std::vector<gui::NodeLink>)> start,
                          std::function<bool()> launch_pending) {
        start_training_ = std::move(start);
        launch_pending_ = std::move(launch_pending);
    }
    // What Train would train, prepared: the versions train (main thread).
    void BeginTraining(TrialSetup setup);
    // The launch stopped before the training ("Not started: ...").
    void NotStarted(const std::string& why);

    // Preparing or training versions.
    bool IsTraining() const { return train_stage_ != TrainStage::Idle; }

private:
    enum class TrainStage { Idle, Preparing, Training };

    // A trained version's row.
    struct Trained {
        bool done = false;
        bool ok = false;
        TrialScore score;
        std::string error;
        int epoch = 0;
        int epochs = 0;
    };
    struct Step {
        std::string text;
        double score = 0.0;
    };

    void ReadGraph();
    void Rescore();
    void Mutate(MutationType type);
    void ReplaceCanvas(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links);
    void StartSearch();
    void StopSearch();
    void PollWork();
    void StartTraining();

    void RenderHeader();
    void RenderScoreCard();
    void RenderChange();
    void RenderSearch();
    void RenderVersions();

    bool visible_ = false;
    gui::NodeEditor* node_editor_ = nullptr;
    std::function<void(std::vector<gui::MLNode>, std::vector<gui::NodeLink>)> start_training_;
    std::function<bool()> launch_pending_;

    // The graph as read, and the version being changed.
    bool read_ = false;
    std::string read_problem_;  // why nothing can be scored
    uint64_t read_serial_ = 0;  // the compile read (LiveGraphCompile::ResultSerial)
    uint64_t read_revision_ = 0;  // the canvas read (LiveGraphCompile::GraphRevision)
    int rebase_frames_ = 0;       // after Apply: the replaced canvas becomes the graph read
    bool canvas_changed_ = false;
    std::string graph_name_;
    nas_presentation::GraphShape shape_;
    std::vector<gui::MLNode> graph_nodes_;
    std::vector<gui::NodeLink> graph_links_;
    std::vector<gui::MLNode> nodes_;
    std::vector<gui::NodeLink> links_;
    ArchitectureScore score_;
    std::vector<Step> steps_;
    std::string change_note_;

    // Search automatically (the saved defaults are followed while idle).
    NASSearchConfig search_config_;
    int train_best_ = 3;
    std::optional<core::ArchitectureSearchDefaults> followed_defaults_;
    std::unique_ptr<std::thread> search_thread_;
    std::atomic<bool> searching_{false};
    std::atomic<bool> stop_search_{false};
    std::atomic<int> search_round_{0};
    std::atomic<double> search_best_{0.0};
    std::mutex mutex_;  // result_, trained_
    NASSearchResult result_;
    bool has_result_ = false;

    // Train the best N.
    TrainStage train_stage_ = TrainStage::Idle;
    std::vector<NASCandidate> to_train_;
    std::vector<Trained> trained_;
    std::unique_ptr<std::thread> train_thread_;
    std::atomic<bool> stop_training_{false};
    std::atomic<bool> training_finished_{false};
    std::string not_started_;
    std::string versions_note_;

};

} // namespace cyxwiz
