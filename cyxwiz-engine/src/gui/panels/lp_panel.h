#pragma once

#include <cyxwiz/optimization.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

/**
 * LPPanel - Linear Programming Solver
 *
 * Features:
 * - Objective function input (c vector)
 * - Constraint matrix editor (A matrix, b vector, types)
 * - Add/remove constraint rows
 * - Graphical 2D visualization for 2-variable problems
 * - Feasible region shading
 * - Optimal point marker
 * - Sensitivity analysis (shadow prices)
 */
class LPPanel {
public:
    LPPanel();
    ~LPPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderProblemInput();
    void RenderObjective();
    void RenderConstraints();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderFeasibleRegion();
    void RenderSolutionDetails();
    void RenderSensitivity();

    void Solve();
    void AddConstraint();
    void RemoveConstraint(int index);
    void LoadPreset(int preset);

    bool visible_ = false;

    // Problem type
    bool maximize_ = true;

    // Number of variables (for 2D visualization, typically 2)
    int num_variables_ = 2;

    // Objective function coefficients c
    std::vector<double> objective_coeffs_;

    // Constraint matrix A, RHS b, and types
    std::vector<std::vector<double>> constraint_matrix_;
    std::vector<double> constraint_rhs_;
    std::vector<int> constraint_types_; // 0: <=, 1: >=, 2: =
    const char* constraint_type_names_[3] = { "<=", ">=", "=" };

    // Variable bounds
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    bool use_bounds_ = false;

    // Results
    LPResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Plot range for 2D visualization
    double x_min_ = 0.0, x_max_ = 20.0;
    double y_min_ = 0.0, y_max_ = 20.0;

    // Presets
    int selected_preset_ = -1;
    const char* preset_names_[4] = { "None", "Production Planning", "Diet Problem", "Transportation" };

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Feasible Region, 1: Solution, 2: Sensitivity
};

} // namespace cyxwiz
