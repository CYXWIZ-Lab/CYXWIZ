#include "lr_finder_panel.h"
#include "../node_editor.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <random>

namespace cyxwiz {

LRFinderPanel::LRFinderPanel() {
}

LRFinderPanel::~LRFinderPanel() {
    StopLRFinder();
    if (finder_thread_ && finder_thread_->joinable()) {
        finder_thread_->join();
    }
}

void LRFinderPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS_CHART " Learning Rate Finder", &visible_)) {

        // Tab bar for different sections
        if (ImGui::BeginTabBar("LRFinderTabs")) {
            if (ImGui::BeginTabItem(ICON_FA_SLIDERS " Parameters")) {
                RenderParameters();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Results")) {
                RenderResults();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem(ICON_FA_LIGHTBULB " Suggestions")) {
                RenderSuggestions();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        // Always show progress at bottom when running
        if (is_running_.load()) {
            ImGui::Separator();
            RenderProgress();
        }
    }
    ImGui::End();
}

void LRFinderPanel::RenderParameters() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Learning Rate Range Test Parameters");
    ImGui::Separator();
    ImGui::Spacing();

    // Start/End LR
    ImGui::Text("Learning Rate Range:");
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("Start LR", &start_lr_, 0.0f, 0.0f, "%.1e");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    ImGui::InputFloat("End LR", &end_lr_, 0.0f, 0.0f, "%.1e");

    // Validate
    start_lr_ = std::max(1e-10f, start_lr_);
    end_lr_ = std::max(start_lr_ * 10, end_lr_);

    ImGui::Spacing();

    // Iterations and batch size
    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("Iterations", &num_iterations_);
    num_iterations_ = std::clamp(num_iterations_, 10, 1000);

    ImGui::SetNextItemWidth(150);
    ImGui::InputInt("Batch Size", &batch_size_);
    batch_size_ = std::clamp(batch_size_, 1, 512);

    ImGui::Spacing();

    // Schedule type
    ImGui::Text("LR Schedule:");
    ImGui::RadioButton("Exponential", &schedule_type_, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Linear", &schedule_type_, 0);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Exponential: LR increases geometrically (recommended)");
        ImGui::Text("Linear: LR increases linearly");
        ImGui::EndTooltip();
    }

    ImGui::Spacing();

    // Smoothing factor
    ImGui::SetNextItemWidth(150);
    ImGui::SliderFloat("Smoothing", &smooth_factor_, 0.0f, 0.5f, "%.2f");
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Exponential moving average smoothing for loss curve");
        ImGui::Text("Higher = smoother curve, lower = more responsive");
        ImGui::EndTooltip();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Action buttons
    bool can_run = !is_running_.load() && node_editor_ != nullptr;

    if (!can_run && !is_running_.load()) {
        ImGui::TextDisabled("No node editor connected");
    }

    if (is_running_.load()) {
        if (ImGui::Button(ICON_FA_STOP " Stop", ImVec2(120, 30))) {
            StopLRFinder();
        }
    } else {
        if (ImGui::Button(ICON_FA_PLAY " Start", ImVec2(120, 30))) {
            StartLRFinder();
        }
    }

    ImGui::SameLine();
    if (results_available_) {
        if (ImGui::Button(ICON_FA_TRASH " Clear Results", ImVec2(120, 30))) {
            std::lock_guard<std::mutex> lock(results_mutex_);
            learning_rates_.clear();
            losses_.clear();
            smoothed_losses_.clear();
            results_available_ = false;
            suggested_lr_ = 0.0f;
        }
    }

    ImGui::Spacing();

    // Info box
    ImGui::TextWrapped(
        "The Learning Rate Range Test gradually increases the learning rate "
        "from a very small value to a large value while recording the loss. "
        "The optimal learning rate is typically found where the loss decreases "
        "most steeply, before it starts increasing again."
    );
}

void LRFinderPanel::RenderProgress() {
    float progress = progress_.load();
    int current = current_iteration_.load();

    ImGui::Text(ICON_FA_SPINNER " Running LR Finder...");
    ImGui::ProgressBar(progress, ImVec2(-1, 0), "");
    ImGui::Text("Iteration: %d / %d", current, num_iterations_);

    if (!status_message_.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", status_message_.c_str());
    }
}

void LRFinderPanel::RenderResults() {
    // Display options
    ImGui::Checkbox("Log Scale X", &show_log_scale_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Smoothed", &show_smoothed_);
    ImGui::SameLine();
    ImGui::Checkbox("Show Gradient", &show_gradient_);

    ImGui::Separator();

    if (!results_available_ && !is_running_.load()) {
        ImGui::TextDisabled("No results yet. Run the LR finder first.");
        return;
    }

    std::lock_guard<std::mutex> lock(results_mutex_);

    if (learning_rates_.empty()) {
        ImGui::TextDisabled("Waiting for results...");
        return;
    }

    // Plot
    ImPlotFlags plot_flags = ImPlotFlags_NoTitle;
    if (show_log_scale_) {
        plot_flags |= ImPlotFlags_None;  // Log scale set via axis
    }

    if (ImPlot::BeginPlot("##LRFinderPlot", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Learning Rate", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        if (show_log_scale_) {
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        }

        // Raw loss
        if (!show_smoothed_ || smoothed_losses_.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(0.4f, 0.6f, 1.0f, 0.7f), 1.5f);
            ImPlot::PlotLine("Loss", learning_rates_.data(), losses_.data(),
                            static_cast<int>(losses_.size()));
        }

        // Smoothed loss
        if (show_smoothed_ && !smoothed_losses_.empty()) {
            ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), 2.0f);
            ImPlot::PlotLine("Smoothed Loss", learning_rates_.data(), smoothed_losses_.data(),
                            static_cast<int>(smoothed_losses_.size()));
        }

        // Mark suggested LR
        if (suggested_lr_ > 0.0f) {
            double suggested_arr[] = {static_cast<double>(suggested_lr_), static_cast<double>(suggested_lr_)};
            double y_range[] = {0.0, static_cast<double>(min_loss_ * 3)};
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotLine("Suggested LR", suggested_arr, y_range, 2);

            // Annotation
            ImPlot::Annotation(suggested_lr_, min_loss_ * 1.5,
                              ImVec4(1.0f, 0.5f, 0.0f, 1.0f), ImVec2(10, -10), true,
                              "Suggested: %.2e", suggested_lr_);
        }

        // Mark min loss
        if (min_loss_lr_ > 0.0f) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8.0f, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
            ImPlot::PlotScatter("Min Loss", &min_loss_lr_, &min_loss_, 1);
        }

        ImPlot::EndPlot();
    }

    // Statistics
    ImGui::Separator();
    ImGui::Text("Statistics:");
    ImGui::BulletText("Total iterations: %zu", losses_.size());
    ImGui::BulletText("Min loss: %.4f at LR = %.2e", min_loss_, min_loss_lr_);
    if (suggested_lr_ > 0.0f) {
        ImGui::BulletText("Suggested LR: %.2e", suggested_lr_);
    }
}

void LRFinderPanel::RenderSuggestions() {
    if (!results_available_) {
        ImGui::TextDisabled("Run the LR finder first to get suggestions.");
        return;
    }

    std::lock_guard<std::mutex> lock(results_mutex_);

    ImGui::Text(ICON_FA_LIGHTBULB " Learning Rate Recommendations");
    ImGui::Separator();
    ImGui::Spacing();

    if (suggested_lr_ > 0.0f) {
        // Main suggestion
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.9f, 0.2f, 1.0f));
        ImGui::Text("Recommended Learning Rate: %.2e", suggested_lr_);
        ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::TextWrapped(
            "This is the learning rate where the loss was decreasing most rapidly. "
            "It's typically a good starting point for training."
        );

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Alternative suggestions
        ImGui::Text("Alternative Suggestions:");

        float conservative_lr = suggested_lr_ * 0.1f;
        float aggressive_lr = suggested_lr_ * 3.0f;

        ImGui::BulletText("Conservative: %.2e (slower but more stable)", conservative_lr);
        ImGui::BulletText("Aggressive: %.2e (faster but may diverge)", aggressive_lr);

        ImGui::Spacing();

        // Learning rate schedules
        ImGui::Text("Recommended Schedules:");
        ImGui::BulletText("One Cycle: max_lr = %.2e", suggested_lr_);
        ImGui::BulletText("Cosine Annealing: initial_lr = %.2e", suggested_lr_ * 0.5f);
        ImGui::BulletText("Step Decay: initial_lr = %.2e, decay every 10 epochs", suggested_lr_);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Copy buttons
        if (ImGui::Button("Copy Suggested LR")) {
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "%.2e", suggested_lr_);
            ImGui::SetClipboardText(buffer);
            spdlog::info("Copied LR to clipboard: {}", buffer);
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f),
                          "Could not determine optimal learning rate.");
        ImGui::TextWrapped(
            "The loss curve may not have a clear minimum. Try:\n"
            "- Increasing the number of iterations\n"
            "- Using a wider LR range\n"
            "- Checking that the model and data are valid"
        );
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Tips
    ImGui::Text(ICON_FA_CIRCLE_INFO " Tips:");
    ImGui::TextWrapped(
        "1. The suggested LR is where the loss decreases fastest\n"
        "2. For fine-tuning, use a smaller LR (1/10 to 1/100)\n"
        "3. Use learning rate warmup for transformer models\n"
        "4. If training diverges, reduce LR by half"
    );
}

void LRFinderPanel::StartLRFinder() {
    if (is_running_.load()) return;

    spdlog::info("Starting LR finder: {} iterations, LR from {} to {}",
                num_iterations_, start_lr_, end_lr_);

    // Clear previous results
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        learning_rates_.clear();
        losses_.clear();
        smoothed_losses_.clear();
        results_available_ = false;
        suggested_lr_ = 0.0f;
    }

    is_running_.store(true);
    stop_requested_.store(false);
    progress_.store(0.0f);
    current_iteration_.store(0);
    status_message_ = "Initializing...";

    // Start finder thread
    finder_thread_ = std::make_unique<std::thread>([this]() {
        SimulateLRFinder();
    });
}

void LRFinderPanel::StopLRFinder() {
    if (!is_running_.load()) return;

    spdlog::info("Stopping LR finder");
    stop_requested_.store(true);

    if (finder_thread_ && finder_thread_->joinable()) {
        finder_thread_->join();
    }
    finder_thread_.reset();

    is_running_.store(false);
    status_message_ = "Stopped by user";
}

void LRFinderPanel::SimulateLRFinder() {
    // This is a simulation for demo purposes
    // Real implementation would integrate with TrainingExecutor

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.1f);

    float lr_mult = (schedule_type_ == 1) ?
        std::pow(end_lr_ / start_lr_, 1.0f / num_iterations_) :
        (end_lr_ - start_lr_) / num_iterations_;

    float current_lr = start_lr_;
    float loss = 2.0f;  // Starting loss
    float best_loss = loss;
    float smoothed = loss;

    for (int i = 0; i < num_iterations_ && !stop_requested_.load(); ++i) {
        // Simulate loss behavior:
        // - Initially decreases as LR increases
        // - Then increases rapidly when LR is too high

        // Optimal LR around 0.01
        float optimal_lr = 0.01f;
        float lr_ratio = current_lr / optimal_lr;

        // Loss model: decreases until optimal, then explodes
        float base_loss;
        if (lr_ratio < 1.0f) {
            // Before optimal: gradual decrease
            base_loss = 2.0f - 1.5f * lr_ratio + 0.5f * lr_ratio * lr_ratio;
        } else {
            // After optimal: exponential increase
            base_loss = 0.5f + std::pow(lr_ratio - 1.0f, 2.0f) * 2.0f;
        }

        loss = base_loss + noise(gen) * 0.15f;
        loss = std::max(0.01f, loss);

        // Exponential smoothing
        smoothed = smooth_factor_ * loss + (1.0f - smooth_factor_) * smoothed;

        // Store results
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            learning_rates_.push_back(current_lr);
            losses_.push_back(loss);
            smoothed_losses_.push_back(smoothed);

            if (smoothed < best_loss) {
                best_loss = smoothed;
                min_loss_ = smoothed;
                min_loss_lr_ = current_lr;
            }
        }

        // Update LR
        if ((schedule_type_ == 1)) {
            current_lr *= lr_mult;
        } else {
            current_lr += lr_mult;
        }

        // Update progress
        current_iteration_.store(i + 1);
        progress_.store(static_cast<float>(i + 1) / num_iterations_);
        status_message_ = fmt::format("LR: {:.2e}, Loss: {:.4f}", current_lr, loss);

        // Small delay for visualization
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Analyze results
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        SmoothLossCurve();
        suggested_lr_ = FindSteepestGradient();
        results_available_ = true;
    }

    is_running_.store(false);
    status_message_ = stop_requested_.load() ? "Stopped" : "Complete";

    spdlog::info("LR finder complete. Suggested LR: {}", suggested_lr_);
}

float LRFinderPanel::FindSteepestGradient() {
    if (smoothed_losses_.size() < 10) return 0.0f;

    // Find point with steepest negative gradient (loss decreasing fastest)
    float max_gradient = 0.0f;
    int best_idx = -1;

    for (size_t i = 5; i < smoothed_losses_.size() - 5; ++i) {
        // Use central difference for gradient
        float gradient = (smoothed_losses_[i - 5] - smoothed_losses_[i + 5]) /
                        (std::log(learning_rates_[i + 5]) - std::log(learning_rates_[i - 5]));

        if (gradient > max_gradient) {
            max_gradient = gradient;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx > 0) {
        // Return LR slightly before the steepest point
        int suggested_idx = std::max(0, best_idx - 5);
        steepest_idx_ = suggested_idx;
        return learning_rates_[suggested_idx];
    }

    return 0.0f;
}

float LRFinderPanel::FindMinLoss() const {
    if (smoothed_losses_.empty()) return 0.0f;

    auto min_it = std::min_element(smoothed_losses_.begin(), smoothed_losses_.end());
    return *min_it;
}

void LRFinderPanel::SmoothLossCurve() {
    // Already smoothed during collection, but can apply additional smoothing here
}

} // namespace cyxwiz
