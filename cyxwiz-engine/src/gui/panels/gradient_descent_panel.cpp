#include "gradient_descent_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

GradientDescentPanel::GradientDescentPanel() {
    current_x_ = { start_x_, start_y_ };
    velocity_ = { 0.0, 0.0 };
    m_ = { 0.0, 0.0 };
    v_ = { 0.0, 0.0 };
}

GradientDescentPanel::~GradientDescentPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void GradientDescentPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_CHART_LINE " Gradient Descent Visualizer###GradientDescentPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float settings_width = 280.0f;

        // Left column - Settings
        ImGui::BeginChild("GDSettings", ImVec2(settings_width, 0), true);
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Visualization
        ImGui::BeginChild("GDVisualization", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderVisualization();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void GradientDescentPanel::RenderToolbar() {
    bool can_run = !is_computing_;

    if (!can_run) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_PLAY " Run")) {
        RunOptimization();
    }
    if (!can_run) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!can_run || !has_result_) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_FORWARD_STEP " Step")) {
        RunStep();
    }
    if (!can_run || !has_result_) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ROTATE_LEFT " Reset")) {
        Reset();
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (has_result_) {
        if (result_.converged) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                ICON_FA_CHECK " Converged in %d iterations", result_.iterations);
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
                ICON_FA_HOURGLASS " Stopped at %d iterations", result_.iterations);
        }
    }
}

void GradientDescentPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Separator();

    RenderFunctionSelector();
    ImGui::Spacing();
    RenderAlgorithmSelector();
    ImGui::Spacing();
    RenderParameters();
    ImGui::Spacing();
    RenderStartPoint();

    if (has_result_) {
        ImGui::Spacing();
        ImGui::Separator();
        RenderStats();
    }
}

void GradientDescentPanel::RenderFunctionSelector() {
    ImGui::Text("Test Function:");
    int func_idx = static_cast<int>(selected_function_);
    if (ImGui::Combo("##Function", &func_idx, function_names_, 6)) {
        selected_function_ = static_cast<TestFunction>(func_idx);
        contour_generated_ = false;
        has_result_ = false;

        // Adjust view range based on function
        switch (selected_function_) {
            case TestFunction::Rosenbrock:
                x_min_ = -2.5; x_max_ = 2.5;
                y_min_ = -1.5; y_max_ = 3.5;
                start_x_ = -2.0; start_y_ = 2.0;
                break;
            case TestFunction::Sphere:
                x_min_ = -3.0; x_max_ = 3.0;
                y_min_ = -3.0; y_max_ = 3.0;
                start_x_ = 2.0; start_y_ = 2.0;
                break;
            case TestFunction::Rastrigin:
                x_min_ = -5.12; x_max_ = 5.12;
                y_min_ = -5.12; y_max_ = 5.12;
                start_x_ = 4.0; start_y_ = 4.0;
                break;
            case TestFunction::Beale:
                x_min_ = -4.5; x_max_ = 4.5;
                y_min_ = -4.5; y_max_ = 4.5;
                start_x_ = 3.0; start_y_ = 3.0;
                break;
            case TestFunction::Booth:
                x_min_ = -10.0; x_max_ = 10.0;
                y_min_ = -10.0; y_max_ = 10.0;
                start_x_ = 5.0; start_y_ = 5.0;
                break;
            case TestFunction::Himmelblau:
                x_min_ = -5.0; x_max_ = 5.0;
                y_min_ = -5.0; y_max_ = 5.0;
                start_x_ = 0.0; start_y_ = 0.0;
                break;
        }
    }

    // Show function formula
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    switch (selected_function_) {
        case TestFunction::Rosenbrock:
            ImGui::TextWrapped("f(x,y) = (1-x)^2 + 100(y-x^2)^2");
            ImGui::TextWrapped("Min at (1, 1)");
            break;
        case TestFunction::Sphere:
            ImGui::TextWrapped("f(x,y) = x^2 + y^2");
            ImGui::TextWrapped("Min at (0, 0)");
            break;
        case TestFunction::Rastrigin:
            ImGui::TextWrapped("f(x,y) = 20 + x^2 + y^2 - 10cos(2pi*x) - 10cos(2pi*y)");
            ImGui::TextWrapped("Min at (0, 0)");
            break;
        case TestFunction::Beale:
            ImGui::TextWrapped("f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2");
            ImGui::TextWrapped("Min at (3, 0.5)");
            break;
        case TestFunction::Booth:
            ImGui::TextWrapped("f(x,y) = (x+2y-7)^2 + (2x+y-5)^2");
            ImGui::TextWrapped("Min at (1, 3)");
            break;
        case TestFunction::Himmelblau:
            ImGui::TextWrapped("f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2");
            ImGui::TextWrapped("4 minima");
            break;
    }
    ImGui::PopStyleColor();
}

void GradientDescentPanel::RenderAlgorithmSelector() {
    ImGui::Text("Algorithm:");
    int alg_idx = static_cast<int>(selected_algorithm_);
    if (ImGui::Combo("##Algorithm", &alg_idx, algorithm_names_, 4)) {
        selected_algorithm_ = static_cast<Algorithm>(alg_idx);
    }
}

void GradientDescentPanel::RenderParameters() {
    ImGui::Text("Parameters:");

    float lr = static_cast<float>(learning_rate_);
    if (ImGui::SliderFloat("Learning Rate", &lr, 0.0001f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic)) {
        learning_rate_ = lr;
    }

    if (selected_algorithm_ == Algorithm::Momentum) {
        float mom = static_cast<float>(momentum_);
        if (ImGui::SliderFloat("Momentum", &mom, 0.0f, 0.99f, "%.2f")) {
            momentum_ = mom;
        }
    }

    if (selected_algorithm_ == Algorithm::Adam) {
        float b1 = static_cast<float>(beta1_);
        float b2 = static_cast<float>(beta2_);
        if (ImGui::SliderFloat("Beta1", &b1, 0.0f, 0.999f, "%.3f")) {
            beta1_ = b1;
        }
        if (ImGui::SliderFloat("Beta2", &b2, 0.0f, 0.9999f, "%.4f")) {
            beta2_ = b2;
        }
    }

    if (selected_algorithm_ == Algorithm::RMSprop) {
        float dr = static_cast<float>(decay_rate_);
        if (ImGui::SliderFloat("Decay Rate", &dr, 0.0f, 0.999f, "%.3f")) {
            decay_rate_ = dr;
        }
    }

    ImGui::SliderInt("Max Iterations", &max_iterations_, 10, 10000);

    float tol = static_cast<float>(tolerance_);
    if (ImGui::SliderFloat("Tolerance", &tol, 1e-10f, 1e-2f, "%.2e", ImGuiSliderFlags_Logarithmic)) {
        tolerance_ = tol;
    }
}

void GradientDescentPanel::RenderStartPoint() {
    ImGui::Text("Starting Point:");

    float sx = static_cast<float>(start_x_);
    float sy = static_cast<float>(start_y_);

    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("X0", &sx, 0.1f, 1.0f, "%.2f")) {
        start_x_ = sx;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("Y0", &sy, 0.1f, 1.0f, "%.2f")) {
        start_y_ = sy;
    }
}

void GradientDescentPanel::RenderLoadingIndicator() {
    ImGui::Text("Computing optimization...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void GradientDescentPanel::RenderVisualization() {
    // Tab bar for different visualizations
    if (ImGui::BeginTabBar("GDVisualizationTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_MOUNTAIN " Contour")) {
            viz_tab_ = 0;
            RenderContourPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Convergence")) {
            viz_tab_ = 1;
            RenderConvergencePlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_ARROW_TREND_DOWN " Gradient")) {
            viz_tab_ = 2;
            RenderGradientPlot();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void GradientDescentPanel::RenderContourPlot() {
    // Generate contour data if needed
    if (!contour_generated_) {
        GenerateContourData();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##ContourPlot", avail)) {
        ImPlot::SetupAxes("x", "y");
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min_, x_max_);
        ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_, y_max_);

        // Draw contour as heatmap
        if (!contour_data_.empty()) {
            int nx = static_cast<int>(contour_x_.size());
            int ny = static_cast<int>(contour_y_.size());

            // Flatten contour data for heatmap
            std::vector<double> flat_data;
            flat_data.reserve(nx * ny);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    flat_data.push_back(std::log10(contour_data_[i][j] + 1.0));
                }
            }

            ImPlot::PlotHeatmap("##Contour", flat_data.data(), ny, nx,
                0, 0, nullptr, ImPlotPoint(x_min_, y_min_), ImPlotPoint(x_max_, y_max_));
        }

        // Draw optimization path
        if (has_result_ && !result_.path.empty()) {
            std::vector<double> path_x, path_y;
            for (const auto& pt : result_.path) {
                if (pt.size() >= 2) {
                    path_x.push_back(pt[0]);
                    path_y.push_back(pt[1]);
                }
            }

            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), 2.0f);
            ImPlot::PlotLine("Path", path_x.data(), path_y.data(), static_cast<int>(path_x.size()));

            // Mark start point
            double sx = path_x.front(), sy = path_y.front();
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(0.2f, 0.8f, 0.2f, 1.0f), 2.0f);
            ImPlot::PlotScatter("Start", &sx, &sy, 1);

            // Mark end point
            double ex = path_x.back(), ey = path_y.back();
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 10, ImVec4(1.0f, 0.8f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotScatter("End", &ex, &ey, 1);
        }

        ImPlot::EndPlot();
    }
}

void GradientDescentPanel::RenderConvergencePlot() {
    if (!has_result_ || result_.objective_history.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Run optimization to see convergence plot");
        return;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##ConvergencePlot", avail)) {
        ImPlot::SetupAxes("Iteration", "Objective f(x)");

        // Create iteration indices
        std::vector<double> iterations;
        for (size_t i = 0; i < result_.objective_history.size(); ++i) {
            iterations.push_back(static_cast<double>(i));
        }

        // Use log scale for y-axis if values span large range
        double min_val = *std::min_element(result_.objective_history.begin(), result_.objective_history.end());
        double max_val = *std::max_element(result_.objective_history.begin(), result_.objective_history.end());

        if (max_val / (min_val + 1e-10) > 100) {
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        }

        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("f(x)", iterations.data(), result_.objective_history.data(),
            static_cast<int>(result_.objective_history.size()));

        ImPlot::EndPlot();
    }
}

void GradientDescentPanel::RenderGradientPlot() {
    if (!has_result_ || gradient_norms_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Run optimization to see gradient plot");
        return;
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##GradientPlot", avail)) {
        ImPlot::SetupAxes("Iteration", "||grad f(x)||");
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);

        std::vector<double> iterations;
        for (size_t i = 0; i < gradient_norms_.size(); ++i) {
            iterations.push_back(static_cast<double>(i));
        }

        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.4f, 0.2f, 1.0f), 2.0f);
        ImPlot::PlotLine("||grad||", iterations.data(), gradient_norms_.data(),
            static_cast<int>(gradient_norms_.size()));

        // Draw tolerance line
        double tol_x[2] = { 0, static_cast<double>(gradient_norms_.size()) };
        double tol_y[2] = { tolerance_, tolerance_ };
        ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), 1.0f);
        ImPlot::PlotLine("Tolerance", tol_x, tol_y, 2);

        ImPlot::EndPlot();
    }
}

void GradientDescentPanel::RenderStats() {
    ImGui::Text(ICON_FA_CHART_PIE " Results:");

    if (!result_.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", result_.error_message.c_str());
        return;
    }

    ImGui::Text("Iterations: %d", result_.iterations);
    ImGui::Text("Final f(x): %.6e", result_.final_objective);
    ImGui::Text("Gradient norm: %.6e", result_.gradient_norm);

    if (result_.solution.size() >= 2) {
        ImGui::Text("Solution: (%.4f, %.4f)", result_.solution[0], result_.solution[1]);
    }

    if (result_.converged) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_CHECK " Converged");
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), ICON_FA_CLOCK " Max iterations reached");
    }
}

void GradientDescentPanel::RunOptimization() {
    if (is_computing_) return;

    // Get objective and gradient functions based on selection
    std::function<double(const std::vector<double>&)> objective;
    std::function<std::vector<double>(const std::vector<double>&)> gradient;

    switch (selected_function_) {
        case TestFunction::Rosenbrock:
            objective = Optimization::Rosenbrock;
            gradient = Optimization::RosenbrockGradient;
            break;
        case TestFunction::Sphere:
            objective = Optimization::Sphere;
            gradient = Optimization::SphereGradient;
            break;
        case TestFunction::Rastrigin:
            objective = Optimization::Rastrigin;
            gradient = Optimization::RastriginGradient;
            break;
        case TestFunction::Beale:
            objective = Optimization::Beale;
            gradient = Optimization::BealeGradient;
            break;
        case TestFunction::Booth:
            objective = Optimization::Booth;
            gradient = Optimization::BoothGradient;
            break;
        case TestFunction::Himmelblau:
            objective = Optimization::Himmelblau;
            gradient = Optimization::HimmelblauGradient;
            break;
    }

    std::vector<double> x0 = { start_x_, start_y_ };
    Algorithm alg = selected_algorithm_;
    double lr = learning_rate_;
    double mom = momentum_;
    double b1 = beta1_, b2 = beta2_;
    double dr = decay_rate_;
    double eps = epsilon_;
    int max_iter = max_iterations_;
    double tol = tolerance_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, objective, gradient, x0, alg, lr, mom, b1, b2, dr, eps, max_iter, tol]() {
        GradientDescentResult res;

        switch (alg) {
            case Algorithm::VanillaGD:
                res = Optimization::GradientDescent(objective, gradient, x0, lr, max_iter, tol);
                break;
            case Algorithm::Momentum:
                res = Optimization::MomentumGD(objective, gradient, x0, lr, mom, max_iter, tol);
                break;
            case Algorithm::Adam:
                res = Optimization::Adam(objective, gradient, x0, lr, b1, b2, eps, max_iter, tol);
                break;
            case Algorithm::RMSprop:
                res = Optimization::RMSprop(objective, gradient, x0, lr, dr, eps, max_iter, tol);
                break;
        }

        // Compute gradient norms from path
        std::vector<double> grad_norms;
        for (const auto& pt : res.path) {
            auto g = gradient(pt);
            double norm = 0.0;
            for (double gi : g) norm += gi * gi;
            grad_norms.push_back(std::sqrt(norm));
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            gradient_norms_ = std::move(grad_norms);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void GradientDescentPanel::RunStep() {
    // Single step optimization (for visualization purposes)
    // This is a simplified version - in a real implementation you'd want
    // to maintain state across steps properly
}

void GradientDescentPanel::Reset() {
    has_result_ = false;
    result_ = GradientDescentResult();
    gradient_norms_.clear();
    current_x_ = { start_x_, start_y_ };
    velocity_ = { 0.0, 0.0 };
    m_ = { 0.0, 0.0 };
    v_ = { 0.0, 0.0 };
    current_step_ = 0;
    error_message_.clear();
}

void GradientDescentPanel::GenerateContourData() {
    std::function<double(const std::vector<double>&)> func;

    switch (selected_function_) {
        case TestFunction::Rosenbrock:
            func = Optimization::Rosenbrock;
            break;
        case TestFunction::Sphere:
            func = Optimization::Sphere;
            break;
        case TestFunction::Rastrigin:
            func = Optimization::Rastrigin;
            break;
        case TestFunction::Beale:
            func = Optimization::Beale;
            break;
        case TestFunction::Booth:
            func = Optimization::Booth;
            break;
        case TestFunction::Himmelblau:
            func = Optimization::Himmelblau;
            break;
    }

    contour_data_ = Optimization::GenerateContourData(func, x_min_, x_max_, y_min_, y_max_, contour_resolution_);

    // Generate x and y axes
    contour_x_.clear();
    contour_y_.clear();
    double dx = (x_max_ - x_min_) / (contour_resolution_ - 1);
    double dy = (y_max_ - y_min_) / (contour_resolution_ - 1);

    for (int i = 0; i < contour_resolution_; ++i) {
        contour_x_.push_back(x_min_ + i * dx);
        contour_y_.push_back(y_min_ + i * dy);
    }

    contour_generated_ = true;
}

} // namespace cyxwiz
