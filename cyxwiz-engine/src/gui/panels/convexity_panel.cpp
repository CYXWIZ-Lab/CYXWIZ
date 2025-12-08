#include "convexity_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

ConvexityPanel::ConvexityPanel() {
}

ConvexityPanel::~ConvexityPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void ConvexityPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(850, 650), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_SQUARE_ROOT_VARIABLE " Convexity Analyzer###ConvexityPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float settings_width = 260.0f;

        // Left column - Settings
        ImGui::BeginChild("ConvSettings", ImVec2(settings_width, 0), true);
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Results
        ImGui::BeginChild("ConvResults", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderResults();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void ConvexityPanel::RenderToolbar() {
    bool can_analyze = !is_computing_;

    if (!can_analyze) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS_CHART " Analyze")) {
        AnalyzeConvexity();
    }
    if (!can_analyze) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_DICE " Sample Points")) {
        show_sample_analysis_ = true;
        // Generate sample points and analyze each
        sample_points_.clear();
        sample_results_.clear();
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                double x = x_min_ + (x_max_ - x_min_) * i / 4.0;
                double y = y_min_ + (y_max_ - y_min_) * j / 4.0;
                sample_points_.push_back({x, y});
            }
        }
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        has_result_ = false;
        result_ = ConvexityResult();
        sample_points_.clear();
        sample_results_.clear();
        show_sample_analysis_ = false;
        surface_generated_ = false;
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (has_result_) {
        if (result_.is_strictly_convex) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_CHECK " Strictly Convex");
        } else if (result_.is_convex) {
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ICON_FA_CHECK " Convex");
        } else if (result_.is_strictly_concave) {
            ImGui::TextColored(ImVec4(0.2f, 0.2f, 0.8f, 1.0f), ICON_FA_ARROW_DOWN " Strictly Concave");
        } else if (result_.is_concave) {
            ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.8f, 1.0f), ICON_FA_ARROW_DOWN " Concave");
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), ICON_FA_XMARK " Non-Convex");
        }
    }
}

void ConvexityPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Separator();

    RenderFunctionSelector();
    ImGui::Spacing();
    RenderPointInput();

    // Numerical delta
    ImGui::Spacing();
    ImGui::Text("Numerical Settings:");
    float d = static_cast<float>(delta_);
    if (ImGui::SliderFloat("Delta (h)", &d, 1e-8f, 1e-2f, "%.2e", ImGuiSliderFlags_Logarithmic)) {
        delta_ = d;
    }

    // Plot range
    ImGui::Spacing();
    ImGui::Text("Plot Range:");
    float xmin = static_cast<float>(x_min_);
    float xmax = static_cast<float>(x_max_);
    float ymin = static_cast<float>(y_min_);
    float ymax = static_cast<float>(y_max_);

    if (ImGui::DragFloatRange2("X Range", &xmin, &xmax, 0.1f, -20.0f, 20.0f, "%.1f", "%.1f")) {
        x_min_ = xmin; x_max_ = xmax;
        surface_generated_ = false;
    }
    if (ImGui::DragFloatRange2("Y Range", &ymin, &ymax, 0.1f, -20.0f, 20.0f, "%.1f", "%.1f")) {
        y_min_ = ymin; y_max_ = ymax;
        surface_generated_ = false;
    }

    if (has_result_) {
        ImGui::Spacing();
        ImGui::Separator();
        RenderAnalysisText();
    }
}

void ConvexityPanel::RenderFunctionSelector() {
    ImGui::Text("Test Function:");
    int func_idx = static_cast<int>(selected_function_);
    if (ImGui::Combo("##ConvFunction", &func_idx, function_names_, 7)) {
        selected_function_ = static_cast<TestFunction>(func_idx);
        surface_generated_ = false;
        has_result_ = false;

        // Set appropriate ranges
        switch (selected_function_) {
            case TestFunction::Rosenbrock:
                x_min_ = -2.0; x_max_ = 2.0;
                y_min_ = -1.0; y_max_ = 3.0;
                break;
            case TestFunction::Sphere:
                x_min_ = -3.0; x_max_ = 3.0;
                y_min_ = -3.0; y_max_ = 3.0;
                break;
            case TestFunction::Rastrigin:
                x_min_ = -5.0; x_max_ = 5.0;
                y_min_ = -5.0; y_max_ = 5.0;
                break;
            default:
                x_min_ = -5.0; x_max_ = 5.0;
                y_min_ = -5.0; y_max_ = 5.0;
                break;
        }
    }
}

void ConvexityPanel::RenderPointInput() {
    ImGui::Text("Analysis Point:");

    float px = static_cast<float>(point_x_);
    float py = static_cast<float>(point_y_);

    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("x", &px, 0.1f, 1.0f, "%.3f")) {
        point_x_ = px;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("y", &py, 0.1f, 1.0f, "%.3f")) {
        point_y_ = py;
    }
}

void ConvexityPanel::RenderLoadingIndicator() {
    ImGui::Text("Analyzing convexity...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void ConvexityPanel::RenderResults() {
    if (ImGui::BeginTabBar("ConvexityTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_MOUNTAIN " Surface")) {
            viz_tab_ = 0;
            RenderSurfacePlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE_CELLS " Hessian")) {
            viz_tab_ = 1;
            RenderHessianMatrix();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_WAVE_SQUARE " Eigenvalues")) {
            viz_tab_ = 2;
            RenderEigenvalues();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void ConvexityPanel::RenderHessianMatrix() {
    if (!has_result_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Analyze' to compute Hessian matrix");
        return;
    }

    ImGui::Text("Hessian Matrix H at (%.3f, %.3f):", point_x_, point_y_);
    ImGui::Spacing();

    if (result_.hessian.size() >= 2 && result_.hessian[0].size() >= 2) {
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(10, 5));
        if (ImGui::BeginTable("HessianTable", 3, ImGuiTableFlags_Borders)) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableSetupColumn("dx", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("dy", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableHeadersRow();

            // Row 1
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("dx");
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.hessian[0][0]);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.hessian[0][1]);

            // Row 2
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("dy");
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.hessian[1][0]);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.hessian[1][1]);

            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }

    ImGui::Spacing();
    ImGui::Text("Interpretation:");
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    ImGui::TextWrapped("The Hessian matrix contains second-order partial derivatives. "
                       "Its eigenvalues determine convexity: all positive = convex, "
                       "all negative = concave, mixed = saddle point.");
    ImGui::PopStyleColor();
}

void ConvexityPanel::RenderEigenvalues() {
    if (!has_result_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Analyze' to compute eigenvalues");
        return;
    }

    ImGui::Text("Hessian Eigenvalues:");
    ImGui::Spacing();

    for (size_t i = 0; i < result_.eigenvalues.size(); ++i) {
        double ev = result_.eigenvalues[i];
        ImVec4 color;
        if (ev > 1e-10) {
            color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f); // Positive - green
        } else if (ev < -1e-10) {
            color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f); // Negative - red
        } else {
            color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f); // Zero - yellow
        }

        ImGui::TextColored(color, "lambda_%zu = %.6f", i + 1, ev);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Summary:");
    ImGui::Text("Min eigenvalue: %.6f", result_.min_eigenvalue);
    ImGui::Text("Max eigenvalue: %.6f", result_.max_eigenvalue);

    // Visualization - eigenvalue bar
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.y > 100 && !result_.eigenvalues.empty()) {
        if (ImPlot::BeginPlot("##EigenvalueBar", ImVec2(-1, 150))) {
            ImPlot::SetupAxes("Eigenvalue Index", "Value");

            std::vector<double> indices;
            for (size_t i = 0; i < result_.eigenvalues.size(); ++i) {
                indices.push_back(static_cast<double>(i));
            }

            ImPlot::PlotBars("Eigenvalues", indices.data(), result_.eigenvalues.data(),
                static_cast<int>(result_.eigenvalues.size()), 0.5);

            // Draw zero line
            double zero_x[2] = { -0.5, static_cast<double>(result_.eigenvalues.size()) - 0.5 };
            double zero_y[2] = { 0.0, 0.0 };
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), 1.0f);
            ImPlot::PlotLine("Zero", zero_x, zero_y, 2);

            ImPlot::EndPlot();
        }
    }
}

void ConvexityPanel::RenderSurfacePlot() {
    if (!surface_generated_) {
        GenerateSurfaceData();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##SurfacePlot", avail)) {
        ImPlot::SetupAxes("x", "y");
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min_, x_max_);
        ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_, y_max_);

        // Draw surface as heatmap
        if (!surface_data_.empty()) {
            int n = static_cast<int>(surface_data_.size());

            std::vector<double> flat_data;
            flat_data.reserve(n * n);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    double val = surface_data_[i][j];
                    flat_data.push_back(std::log10(std::abs(val) + 1.0));
                }
            }

            ImPlot::PlotHeatmap("##Surface", flat_data.data(), n, n,
                0, 0, nullptr, ImPlotPoint(x_min_, y_min_), ImPlotPoint(x_max_, y_max_));
        }

        // Mark analysis point
        if (has_result_) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 12, ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotScatter("Analysis Point", &point_x_, &point_y_, 1);
        }

        // Mark sample points if shown
        if (show_sample_analysis_ && !sample_points_.empty()) {
            std::vector<double> sx, sy;
            for (const auto& pt : sample_points_) {
                sx.push_back(pt.first);
                sy.push_back(pt.second);
            }
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6, ImVec4(1.0f, 1.0f, 0.0f, 0.8f), 1.0f);
            ImPlot::PlotScatter("Sample Points", sx.data(), sy.data(), static_cast<int>(sx.size()));
        }

        ImPlot::EndPlot();
    }
}

void ConvexityPanel::RenderAnalysisText() {
    ImGui::Text(ICON_FA_CHART_PIE " Analysis:");

    if (!result_.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", result_.error_message.c_str());
        return;
    }

    ImGui::TextWrapped("%s", result_.analysis.c_str());
}

void ConvexityPanel::AnalyzeConvexity() {
    if (is_computing_) return;

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
        default:
            func = Optimization::Sphere;
            break;
    }

    std::vector<double> point = { point_x_, point_y_ };
    double delta = delta_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, func, point, delta]() {
        auto res = Optimization::AnalyzeConvexity(func, point, delta);

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void ConvexityPanel::GenerateSurfaceData() {
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
        default:
            func = Optimization::Sphere;
            break;
    }

    surface_data_ = Optimization::GenerateContourData(func, x_min_, x_max_, y_min_, y_max_, resolution_);
    surface_generated_ = true;
}

} // namespace cyxwiz
