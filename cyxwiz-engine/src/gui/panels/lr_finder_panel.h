#pragma once

#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <mutex>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * LRFinderPanel - Learning Rate Finder Tool
 *
 * Implements the Learning Rate Range Test to help find optimal learning rate:
 * - Gradually increases learning rate over mini-batches
 * - Records loss at each step
 * - Plots loss vs learning rate
 * - Suggests optimal LR based on steepest gradient
 *
 * Based on the technique from "Cyclical Learning Rates for Training Neural Networks"
 * by Leslie N. Smith.
 */
class LRFinderPanel {
public:
    LRFinderPanel();
    ~LRFinderPanel();

    void Render();

    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }

private:
    void RenderParameters();
    void RenderProgress();
    void RenderResults();
    void RenderSuggestions();

    // LR finder execution
    void StartLRFinder();
    void StopLRFinder();
    void SimulateLRFinder();  // Simulated for demo (actual would integrate with TrainingExecutor)

    // Analysis helpers
    float FindSteepestGradient();  // Non-const: updates steepest_idx_
    float FindMinLoss() const;
    void SmoothLossCurve();

    gui::NodeEditor* node_editor_ = nullptr;

    bool visible_ = false;

    // Parameters
    float start_lr_ = 1e-7f;
    float end_lr_ = 10.0f;
    int num_iterations_ = 100;
    int batch_size_ = 32;
    int schedule_type_ = 1;  // 0=linear, 1=exponential
    float smooth_factor_ = 0.05f;  // Exponential moving average factor

    // Results
    std::vector<float> learning_rates_;
    std::vector<float> losses_;
    std::vector<float> smoothed_losses_;
    float suggested_lr_ = 0.0f;
    float min_loss_ = 0.0f;
    float min_loss_lr_ = 0.0f;
    int steepest_idx_ = -1;

    // State
    std::atomic<bool> is_running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<float> progress_{0.0f};
    std::atomic<int> current_iteration_{0};
    std::string status_message_;
    std::unique_ptr<std::thread> finder_thread_;
    std::mutex results_mutex_;

    // UI state
    bool show_log_scale_ = true;
    bool show_smoothed_ = true;
    bool show_suggestions_ = true;
    bool show_gradient_ = false;
    bool results_available_ = false;
};

} // namespace cyxwiz
