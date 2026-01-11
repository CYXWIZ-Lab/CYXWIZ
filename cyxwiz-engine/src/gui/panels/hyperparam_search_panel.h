#pragma once

#include "gui/panel.h"
#include "core/hyperparam_search.h"
#include <memory>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * Hyperparameter Search Panel
 *
 * Provides UI for configuring and running hyperparameter optimization:
 * - Define search space (parameters with ranges)
 * - Select search strategy (Grid, Random, Bayesian)
 * - View real-time progress and results
 * - Compare trial results in sortable table
 */
class HyperparamSearchPanel : public Panel {
public:
    HyperparamSearchPanel();
    ~HyperparamSearchPanel() override = default;

    void Render() override;

    // Set node editor for getting model configuration
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

private:
    void RenderSearchConfig();
    void RenderParameterSpace();
    void RenderProgress();
    void RenderResults();
    void RenderAddParameterDialog();

    // Actions
    void StartSearch();
    void StopSearch();
    void AddParameter();
    void RemoveParameter(size_t index);

    gui::NodeEditor* node_editor_ = nullptr;

    // Search engine
    std::unique_ptr<HyperparamSearch> search_engine_;
    SearchConfig config_;

    // UI state
    bool show_add_param_dialog_ = false;
    bool is_running_ = false;

    // Add parameter dialog state
    char param_name_[128] = "";
    int param_type_ = 0;  // 0=continuous, 1=discrete, 2=logscale, 3=categorical
    float param_min_ = 0.0f;
    float param_max_ = 1.0f;
    int param_step_ = 1;
    char param_options_[512] = "";  // Comma-separated for categorical

    // Results state
    std::vector<TrialResult> cached_results_;
    int sort_column_ = 0;
    bool sort_ascending_ = true;

    // Progress state
    int current_trial_ = 0;
    int total_trials_ = 0;
    float current_progress_ = 0.0f;
    float elapsed_seconds_ = 0.0f;

    // Best result tracking
    int best_trial_id_ = -1;
    float best_val_loss_ = std::numeric_limits<float>::max();
    float best_val_accuracy_ = 0.0f;
};

} // namespace cyxwiz
