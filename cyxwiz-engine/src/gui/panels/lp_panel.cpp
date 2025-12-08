#include "lp_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

LPPanel::LPPanel() {
    // Initialize with a simple 2-variable problem
    objective_coeffs_ = { 3.0, 2.0 };
    lower_bounds_ = { 0.0, 0.0 };
    upper_bounds_ = { 100.0, 100.0 };

    // Default constraints: x + y <= 4, x + 2y <= 6
    constraint_matrix_ = {
        { 1.0, 1.0 },
        { 1.0, 2.0 }
    };
    constraint_rhs_ = { 4.0, 6.0 };
    constraint_types_ = { 0, 0 }; // <=
}

LPPanel::~LPPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void LPPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_CHART_PIE " Linear Programming###LPPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float input_width = 320.0f;

        // Left column - Problem Input
        ImGui::BeginChild("LPInput", ImVec2(input_width, 0), true);
        RenderProblemInput();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Results
        ImGui::BeginChild("LPResults", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderResults();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void LPPanel::RenderToolbar() {
    bool can_solve = !is_computing_ && !objective_coeffs_.empty();

    if (!can_solve) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_PLAY " Solve")) {
        Solve();
    }
    if (!can_solve) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_PLUS " Add Constraint")) {
        AddConstraint();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        has_result_ = false;
        result_ = LPResult();
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

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (has_result_) {
        if (result_.status == "Optimal") {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                ICON_FA_CHECK " Optimal: %.2f", result_.objective_value);
        } else if (result_.status == "Unbounded") {
            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
                ICON_FA_INFINITY " Unbounded");
        } else if (result_.status == "Infeasible") {
            ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f),
                ICON_FA_XMARK " Infeasible");
        }
    }
}

void LPPanel::RenderProblemInput() {
    ImGui::Text(ICON_FA_COG " Problem Definition");
    ImGui::Separator();

    RenderObjective();
    ImGui::Spacing();
    RenderConstraints();
}

void LPPanel::RenderObjective() {
    ImGui::Text("Objective:");

    // Maximize/Minimize toggle
    if (ImGui::RadioButton("Maximize", maximize_)) maximize_ = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("Minimize", !maximize_)) maximize_ = false;

    // Objective coefficients
    ImGui::Text("Z =");
    ImGui::SameLine();

    for (size_t i = 0; i < objective_coeffs_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::SetNextItemWidth(50);
        float val = static_cast<float>(objective_coeffs_[i]);
        if (ImGui::InputFloat("##c", &val, 0, 0, "%.1f")) {
            objective_coeffs_[i] = val;
        }
        ImGui::SameLine();
        ImGui::Text("x%zu", i + 1);
        if (i < objective_coeffs_.size() - 1) {
            ImGui::SameLine();
            ImGui::Text("+");
            ImGui::SameLine();
        }
        ImGui::PopID();
    }
}

void LPPanel::RenderConstraints() {
    ImGui::Text("Constraints:");

    // Non-negativity
    ImGui::Checkbox("x >= 0 (non-negative)", &use_bounds_);

    ImGui::Spacing();

    // Constraint table
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

        // Constraint type
        ImGui::SetNextItemWidth(50);
        ImGui::Combo("##type", &constraint_types_[i], constraint_type_names_, 3);

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

void LPPanel::RenderLoadingIndicator() {
    ImGui::Text("Solving LP problem...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void LPPanel::RenderResults() {
    if (ImGui::BeginTabBar("LPResultsTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_DRAW_POLYGON " Feasible Region")) {
            viz_tab_ = 0;
            RenderFeasibleRegion();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_LIST " Solution")) {
            viz_tab_ = 1;
            RenderSolutionDetails();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Sensitivity")) {
            viz_tab_ = 2;
            RenderSensitivity();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void LPPanel::RenderFeasibleRegion() {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##FeasibleRegion", avail)) {
        ImPlot::SetupAxes("x1", "x2");
        ImPlot::SetupAxisLimits(ImAxis_X1, x_min_, x_max_);
        ImPlot::SetupAxisLimits(ImAxis_Y1, y_min_, y_max_);

        // Draw constraint lines
        for (size_t i = 0; i < constraint_matrix_.size(); ++i) {
            if (constraint_matrix_[i].size() >= 2) {
                double a1 = constraint_matrix_[i][0];
                double a2 = constraint_matrix_[i][1];
                double b = constraint_rhs_[i];

                // Line: a1*x + a2*y = b
                std::vector<double> line_x, line_y;

                if (std::abs(a2) > 1e-10) {
                    // y = (b - a1*x) / a2
                    for (double x = x_min_; x <= x_max_; x += 0.5) {
                        double y = (b - a1 * x) / a2;
                        if (y >= y_min_ && y <= y_max_) {
                            line_x.push_back(x);
                            line_y.push_back(y);
                        }
                    }
                } else if (std::abs(a1) > 1e-10) {
                    // x = b / a1
                    double x = b / a1;
                    if (x >= x_min_ && x <= x_max_) {
                        line_x = { x, x };
                        line_y = { y_min_, y_max_ };
                    }
                }

                if (!line_x.empty()) {
                    std::string label = "C" + std::to_string(i + 1);
                    ImPlot::PlotLine(label.c_str(), line_x.data(), line_y.data(),
                        static_cast<int>(line_x.size()));
                }
            }
        }

        // Draw non-negativity constraints
        if (use_bounds_) {
            double ax_x[2] = { 0, 0 };
            double ax_y[2] = { y_min_, y_max_ };
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), 1.0f);
            ImPlot::PlotLine("x1=0", ax_x, ax_y, 2);

            double ay_x[2] = { x_min_, x_max_ };
            double ay_y[2] = { 0, 0 };
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), 1.0f);
            ImPlot::PlotLine("x2=0", ay_x, ay_y, 2);
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

void LPPanel::RenderSolutionDetails() {
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

    if (result_.status == "Optimal") {
        ImGui::Text("Optimal Objective Value: %.4f", result_.objective_value);
        ImGui::Spacing();

        ImGui::Text("Decision Variables:");
        for (size_t i = 0; i < result_.solution.size(); ++i) {
            ImGui::BulletText("x%zu = %.4f", i + 1, result_.solution[i]);
        }

        ImGui::Spacing();
        ImGui::Text("Iterations: %d", result_.iterations);

        // Active constraints
        ImGui::Spacing();
        ImGui::Text("Active Constraints:");
        for (size_t i = 0; i < result_.active_constraints.size(); ++i) {
            if (result_.active_constraints[i]) {
                ImGui::BulletText("Constraint %zu is binding", i + 1);
            }
        }
    } else if (result_.status == "Unbounded") {
        ImGui::TextWrapped("The problem is unbounded. The objective function can be "
                          "improved indefinitely.");
    } else if (result_.status == "Infeasible") {
        ImGui::TextWrapped("The problem has no feasible solution. The constraints "
                          "are contradictory.");
    }
}

void LPPanel::RenderSensitivity() {
    if (!has_result_ || result_.status != "Optimal") {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Solve an optimal LP to see sensitivity analysis");
        return;
    }

    ImGui::Text("Dual Variables (Shadow Prices):");
    ImGui::Separator();

    if (result_.dual_variables.empty()) {
        ImGui::Text("No dual variables available");
        return;
    }

    for (size_t i = 0; i < result_.dual_variables.size(); ++i) {
        double dv = result_.dual_variables[i];
        ImGui::Text("y%zu = %.4f", i + 1, dv);

        // Interpretation
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
        if (std::abs(dv) > 1e-6) {
            ImGui::SameLine();
            ImGui::Text("(+1 RHS -> %+.4f obj)", dv);
        } else {
            ImGui::SameLine();
            ImGui::Text("(non-binding)");
        }
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    ImGui::TextWrapped("Shadow prices indicate how much the objective function "
                      "would improve if the constraint RHS increased by 1 unit.");
}

void LPPanel::Solve() {
    if (is_computing_) return;

    // Build constraint types as strings
    std::vector<std::string> types;
    for (int t : constraint_types_) {
        if (t == 0) types.push_back("<=");
        else if (t == 1) types.push_back(">=");
        else types.push_back("=");
    }

    auto c = objective_coeffs_;
    auto A = constraint_matrix_;
    auto b = constraint_rhs_;
    bool max = maximize_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, c, A, b, types, max]() {
        auto res = Optimization::SolveLP(c, A, b, types, max);

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void LPPanel::AddConstraint() {
    std::vector<double> new_row(objective_coeffs_.size(), 0.0);
    constraint_matrix_.push_back(new_row);
    constraint_rhs_.push_back(0.0);
    constraint_types_.push_back(0); // <=
}

void LPPanel::RemoveConstraint(int index) {
    if (index >= 0 && index < static_cast<int>(constraint_matrix_.size())) {
        constraint_matrix_.erase(constraint_matrix_.begin() + index);
        constraint_rhs_.erase(constraint_rhs_.begin() + index);
        constraint_types_.erase(constraint_types_.begin() + index);
    }
}

void LPPanel::LoadPreset(int preset) {
    switch (preset) {
        case 1: // Production Planning
            objective_coeffs_ = { 5.0, 4.0 };  // Profit per unit
            constraint_matrix_ = {
                { 2.0, 1.0 },  // Machine A hours
                { 1.0, 2.0 },  // Machine B hours
                { 1.0, 1.0 }   // Raw material
            };
            constraint_rhs_ = { 100.0, 80.0, 60.0 };
            constraint_types_ = { 0, 0, 0 }; // All <=
            maximize_ = true;
            x_max_ = 60.0;
            y_max_ = 60.0;
            break;

        case 2: // Diet Problem
            objective_coeffs_ = { 0.5, 0.8 };  // Cost per unit
            constraint_matrix_ = {
                { 10.0, 5.0 },   // Calories
                { 2.0, 8.0 },    // Protein
                { 5.0, 3.0 }     // Vitamins
            };
            constraint_rhs_ = { 200.0, 50.0, 100.0 };
            constraint_types_ = { 1, 1, 1 }; // All >=
            maximize_ = false;
            x_max_ = 30.0;
            y_max_ = 30.0;
            break;

        case 3: // Transportation
            objective_coeffs_ = { 8.0, 6.0 };
            constraint_matrix_ = {
                { 1.0, 1.0 },
                { 1.0, 0.0 },
                { 0.0, 1.0 }
            };
            constraint_rhs_ = { 100.0, 70.0, 50.0 };
            constraint_types_ = { 0, 0, 0 };
            maximize_ = false;
            x_max_ = 100.0;
            y_max_ = 100.0;
            break;
    }

    has_result_ = false;
}

} // namespace cyxwiz
