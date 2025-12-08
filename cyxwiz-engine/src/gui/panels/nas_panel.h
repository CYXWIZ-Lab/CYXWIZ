#pragma once

#include "../../core/nas_evaluator.h"
#include "../node_editor.h"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <functional>

namespace cyxwiz {

/**
 * NASPanel - Neural Architecture Search Tool
 *
 * Features:
 * - Architecture scoring (complexity, efficiency, depth)
 * - Architecture mutation controls
 * - Architecture suggestions by task type
 * - Evolutionary search with visualization
 * - Generation progress tracking
 * - Best architecture preview
 * - Apply architecture to node editor
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

    // Set callback to get current architecture from node editor
    void SetGetArchitectureCallback(
        std::function<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>()> cb);

    // Set callback to apply architecture to node editor
    void SetApplyArchitectureCallback(
        std::function<void(const std::vector<gui::MLNode>&, const std::vector<gui::NodeLink>&)> cb);

private:
    void RenderToolbar();
    void RenderInputConfig();
    void RenderCurrentScore();
    void RenderMutationControls();
    void RenderSuggestions();
    void RenderEvolutionarySearch();
    void RenderSearchProgress();
    void RenderSearchResults();
    void RenderExportOptions();

    void ScoreCurrentArchitecture();
    void MutateArchitecture(MutationType type);
    void GenerateSuggestions();
    void RunEvolutionarySearch();

    bool visible_ = false;

    // Callbacks
    std::function<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>()>
        get_architecture_callback_;
    std::function<void(const std::vector<gui::MLNode>&, const std::vector<gui::NodeLink>&)>
        apply_architecture_callback_;

    // Input configuration
    int input_type_ = 0;  // 0=Tabular, 1=Image
    int input_features_ = 784;
    int input_channels_ = 1;
    int input_height_ = 28;
    int input_width_ = 28;
    int output_size_ = 10;
    int task_type_ = 0;  // 0=Classification, 1=Regression, 2=Image Classification

    // Current architecture
    std::vector<gui::MLNode> current_nodes_;
    std::vector<gui::NodeLink> current_links_;
    ArchitectureScore current_score_;
    bool has_score_ = false;

    // Suggestions
    std::vector<std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>> suggestions_;
    std::vector<ArchitectureScore> suggestion_scores_;
    int selected_suggestion_ = -1;

    // Evolutionary search config
    NASSearchConfig search_config_;
    bool show_search_config_ = false;

    // Search results
    NASSearchResult search_result_;
    bool has_search_result_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::atomic<bool> is_searching_{false};
    std::atomic<int> search_generation_{0};
    std::atomic<double> search_best_score_{0.0};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
