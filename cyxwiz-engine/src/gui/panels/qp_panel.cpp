#include "qp_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

QPPanel::QPPanel() {
    // Initialize with simple 2x2 problem: min 0.5*x'Qx + c'x
    Q_matrix_ = {
        { 2.0, 0.0 },
        { 0.0, 2.0 }
    };
    c_vector_ = { -2.0, -5.0 };

    // Default constraint: x + y <= 3
    constraint_matrix_ = {
        { 1.0, 1.0 }
    };
    constraint_rhs_ = { 3.0 };
}

QPPanel::~QPPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void QPPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_SQUARE_POLL_VERTICAL " Quadratic Programming###QPPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float input_width = 320.0f;

        // Left column - Problem Input
        ImGui::BeginChild("QPInput", ImVec2(input_width, 0), true);
        RenderProblemInput();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Results
        ImGui::BeginChild("QPResults", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderResults();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void QPPanel::RenderToolbar() {
    bool can_solve = !is_computing_;

    if (!can_solve) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_PLAY " Solve")) {
        Solve();
    }
    if (!can_solve) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!can_solve) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_BOLT " Unconstrained")) {
        SolveUnconstrained();
    }
    if (!can_solve) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_PLUS " Add Constraint")) {
        AddConstraint();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        has_result_ = false;
        result_ = QPResult();
        contour_generated_ = false;
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    // Presets dropdown
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("Preset", &selected_preset_, preset_names_, 4)) {
        if (selected_preset_ > 0) {
            LoadPreset(selected_preset_);
        }
    }

    if (has_result_) {
        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();

        if (result_.status == "Optimal") {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                ICON_FA_CHECK " Optimal: %.4f", result_.objective_value);
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
                ICON_FA_EXCLAMATION " %s", result_.status.c_str());
        }
    }
}

void QPPanel::RenderProblemInput() {
    ImGui::Text(ICON_FA_COG " Problem: min 0.5*x'Qx + c'x");
    ImGui::Separator();

    RenderQMatrix();
    ImGui::Spacing();
    RenderCVector();
    ImGui::Spacing();
    RenderConstraints();
}

void QPPanel::RenderQMatrix() {
    ImGui::Text("Q Matrix (symmetric):");

    int n = static_cast<int>(Q_matrix_.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ImGui::PushID(i * 10 + j);
            ImGui::SetNextItemWidth(50);
            float val = static_cast<float>(Q_matrix_[i][j]);
            if (ImGui::InputFloat("##q", &val, 0, 0, "%.1f")) {
                Q_matrix_[i][j] = val;
                // Keep symmetric
                if (i != j) Q_matrix_[j][i] = val;
                contour_generated_ = false;
            }
            if (j < n - 1) ImGui::SameLine();
            ImGui::PopID();
        }
    }
}

void QPPanel::RenderCVector() {
    ImGui::Text("c Vector (linear term):");

    for (size_t i = 0; i < c_vector_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 50));
        ImGui::SetNextItemWidth(60);
        float val = static_cast<float>(c_vector_[i]);
        if (ImGui::InputFloat("##c", &val, 0, 0, "%.2f")) {
            c_vector_[i] = val;
            contour_generated_ = false;
        }
        ImGui::SameLine();
        ImGui::Text("x%zu", i + 1);
        if (i < c_vector_.size() - 1) {
            ImGui::SameLine();
            ImGui::Text(",");
            ImGui::SameLine();
        }
        ImGui::PopID();
    }
}

void QPPanel::RenderConstraints() {
    ImGui::Text("Constraints (Ax <= b):");

    if (constraint_matrix_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(no constraints)");
        return;
    }

    for (size_t i = 0; i < constraint_matrix_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i + 100));

        // Coefficients
        for (size_t j = 0; j < constraint_matrix_[i].size(); ++j) {
            ImGui::PushID(static_cast<int>(j));
            ImGui::SetNextItemWidth(40);
            float val = static_cast<float>(constraint_matrix_[i][j]);
            if (ImGui::InputFloat("##a", &val, 0, 0, "%.1f")) {
                constraint_matrix_[i][j] = val;
            }
            ImGui::SameLine();
            ImGui::Text("x%zu", j + 1);
            if (j < constraint_matrix_[i].size() - 1) {
                ImGui::SameLine();
                ImGui::Text("+");
                ImGui::SameLine();
            }
            ImGui::PopID();
        }

        ImGui::SameLine();
        ImGui::Text("<=");
        ImGui::SameLine();

        // RHS
        ImGui::SetNextItemWidth(50);
        float rhs = static_cast<float>(constraint_rhs_[i]);
        if (ImGui::InputFloat("##b", &rhs, 0, 0, "%.1f")) {
            constraint_rhs_[i] = rhs;
        }

        ImGui::SameLine();

        // Delete button
        if (ImGui::Button(ICON_FA_TRASH "##del")) {
            RemoveConstraint(static_cast<int>(i));
            ImGui::PopID();
            break;
        }

        ImGui::PopID();
    }
}

void QPPanel::RenderLoadingIndicator() {
    ImGui::Text("Solving QP problem...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void QPPanel::RenderResults() {
    if (ImGui::BeginTabBar("QPResultsTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_MOUNTAIN " Contour")) {
            viz_tab_ = 0;
            RenderContourPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_LIST " Solution")) {
            viz_tab_ = 1;
            RenderSolutionDetails();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void QPPanel::RenderContourPlot() {
    if (!contour_generated_) {
        GenerateContourData();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##QPContour", avail)) {
        ImPlot::SetupAxes("x1", "x2");
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min_, x_max_);
        ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_, y_max_);

        // Draw contour heatmap
        if (!contour_data_.empty()) {
            int n = static_cast<int>(contour_data_.size());

            std::vector<double> flat_data;
            flat_data.reserve(n * n);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    flat_data.push_back(contour_data_[i][j]);
                }
            }

            ImPlot::PlotHeatmap("##Contour", flat_data.data(), n, n,
                0, 0, nullptr, ImPlotPoint(x_min_, y_min_), ImPlotPoint(x_max_, y_max_));
        }

        // Draw constraint lines
        for (size_t i = 0; i < constraint_matrix_.size(); ++i) {
            if (constraint_matrix_[i].size() >= 2) {
                double a1 = constraint_matrix_[i][0];
                double a2 = constraint_matrix_[i][1];
                double b = constraint_rhs_[i];

                std::vector<double> line_x, line_y;

                if (std::abs(a2) > 1e-10) {
                    for (double x = x_min_; x <= x_max_; x += 0.2) {
                        double y = (b - a1 * x) / a2;
                        if (y >= y_min_ && y <= y_max_) {
                            line_x.push_back(x);
                            line_y.push_back(y);
                        }
                    }
                }

                if (!line_x.empty()) {
                    std::string label = "C" + std::to_string(i + 1);
                    ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 0.8f), 2.0f);
                    ImPlot::PlotLine(label.c_str(), line_x.data(), line_y.data(),
                        static_cast<int>(line_x.size()));
                }
            }
        }

        // Draw optimal point
        if (has_result_ && result_.success && result_.solution.size() >= 2) {
            double opt_x = result_.solution[0];
            double opt_y = result_.solution[1];
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 12, ImVec4(1.0f, 0.8f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotScatter("Optimal", &opt_x, &opt_y, 1);
        }

        ImPlot::EndPlot();
    }
}

void QPPanel::RenderSolutionDetails() {
    if (!has_result_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Solve' to find optimal solution");
        return;
    }

    ImGui::Text("Status: %s", result_.status.c_str());
    ImGui::Separator();

    if (!result_.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
            "Error: %s", result_.error_message.c_str());
        return;
    }

    ImGui::Text("Optimal Objective Value: %.6f", result_.objective_value);
    ImGui::Spacing();

    ImGui::Text("Decision Variables:");
    for (size_t i = 0; i < result_.solution.size(); ++i) {
        ImGui::BulletText("x%zu = %.6f", i + 1, result_.solution[i]);
    }

    ImGui::Spacing();
    ImGui::Text("Iterations: %d", result_.iterations);

    if (!result_.lagrange_multipliers.empty()) {
        ImGui::Spacing();
        ImGui::Text("Lagrange Multipliers:");
        for (size_t i = 0; i < result_.lagrange_multipliers.size(); ++i) {
            ImGui::BulletText("mu%zu = %.6f", i + 1, result_.lagrange_multipliers[i]);
        }
    }
}

void QPPanel::Solve() {
    if (is_computing_) return;

    auto Q = Q_matrix_;
    auto c = c_vector_;
    auto A = constraint_matrix_;
    auto b = constraint_rhs_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, Q, c, A, b]() {
        auto res = Optimization::SolveQP(Q, c, A, b);

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void QPPanel::SolveUnconstrained() {
    if (is_computing_) return;

    auto Q = Q_matrix_;
    auto c = c_vector_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, Q, c]() {
        auto res = Optimization::SolveUnconstrainedQP(Q, c);

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void QPPanel::AddConstraint() {
    std::vector<double> new_row(c_vector_.size(), 0.0);
    constraint_matrix_.push_back(new_row);
    constraint_rhs_.push_back(0.0);
}

void QPPanel::RemoveConstraint(int index) {
    if (index >= 0 && index < static_cast<int>(constraint_matrix_.size())) {
        constraint_matrix_.erase(constraint_matrix_.begin() + index);
        constraint_rhs_.erase(constraint_rhs_.begin() + index);
    }
}

void QPPanel::LoadPreset(int preset) {
    switch (preset) {
        case 1: // Portfolio Optimization
            Q_matrix_ = {
                { 0.04, 0.02 },
                { 0.02, 0.09 }
            };
            c_vector_ = { -0.10, -0.15 };  // Expected returns (negative for min)
            constraint_matrix_ = {
                { 1.0, 1.0 },   // Budget constraint
                { -1.0, 0.0 },  // Non-negative x1
                { 0.0, -1.0 }   // Non-negative x2
            };
            constraint_rhs_ = { 1.0, 0.0, 0.0 };
            x_min_ = -0.5; x_max_ = 1.5;
            y_min_ = -0.5; y_max_ = 1.5;
            break;

        case 2: // Regularized Regression
            Q_matrix_ = {
                { 2.0, 1.0 },
                { 1.0, 2.0 }
            };
            c_vector_ = { -4.0, -6.0 };
            constraint_matrix_.clear();
            constraint_rhs_.clear();
            x_min_ = -2.0; x_max_ = 4.0;
            y_min_ = -2.0; y_max_ = 4.0;
            break;

        case 3: // Simple Quadratic
            Q_matrix_ = {
                { 1.0, 0.0 },
                { 0.0, 1.0 }
            };
            c_vector_ = { 0.0, 0.0 };
            constraint_matrix_ = {
                { 1.0, 0.0 },
                { 0.0, 1.0 },
                { -1.0, -1.0 }
            };
            constraint_rhs_ = { 2.0, 2.0, -1.0 };
            x_min_ = -1.0; x_max_ = 3.0;
            y_min_ = -1.0; y_max_ = 3.0;
            break;
    }

    contour_generated_ = false;
    has_result_ = false;
}

void QPPanel::GenerateContourData() {
    // Generate contour data for the QP objective function
    int n = resolution_;
    double dx = (x_max_ - x_min_) / (n - 1);
    double dy = (y_max_ - y_min_) / (n - 1);

    contour_data_.resize(n);
    for (int i = 0; i < n; ++i) {
        contour_data_[i].resize(n);
        double x1 = x_min_ + i * dx;
        for (int j = 0; j < n; ++j) {
            double x2 = y_min_ + j * dy;
            std::vector<double> x = { x1, x2 };

            // Compute 0.5 * x'Qx + c'x
            double val = 0.0;

            // Quadratic term
            for (size_t k = 0; k < Q_matrix_.size(); ++k) {
                for (size_t l = 0; l < Q_matrix_[k].size(); ++l) {
                    val += 0.5 * Q_matrix_[k][l] * x[k] * x[l];
                }
            }

            // Linear term
            for (size_t k = 0; k < c_vector_.size(); ++k) {
                val += c_vector_[k] * x[k];
            }

            contour_data_[i][j] = val;
        }
    }

    contour_generated_ = true;
}

} // namespace cyxwiz
