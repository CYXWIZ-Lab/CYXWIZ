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
 * QPPanel - Quadratic Programming Solver
 *
 * Features:
 * - Q matrix editor (quadratic term)
 * - c vector input (linear term)
 * - Constraint editor (Ax <= b)
 * - 2D contour plot for 2-variable problems
 * - Elliptical level sets visualization
 * - Active constraint highlighting
 */
class QPPanel {
public:
    QPPanel();
    ~QPPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderProblemInput();
    void RenderQMatrix();
    void RenderCVector();
    void RenderConstraints();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderContourPlot();
    void RenderSolutionDetails();

    void Solve();
    void SolveUnconstrained();
    void AddConstraint();
    void RemoveConstraint(int index);
    void LoadPreset(int preset);
    void GenerateContourData();

    bool visible_ = false;

    // Number of variables
    int num_variables_ = 2;

    // Quadratic term Q (must be symmetric positive semi-definite)
    std::vector<std::vector<double>> Q_matrix_;

    // Linear term c
    std::vector<double> c_vector_;

    // Constraint matrix A, RHS b
    std::vector<std::vector<double>> constraint_matrix_;
    std::vector<double> constraint_rhs_;

    // Results
    QPResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Contour data for visualization
    std::vector<std::vector<double>> contour_data_;
    bool contour_generated_ = false;

    // Plot range
    double x_min_ = -5.0, x_max_ = 5.0;
    double y_min_ = -5.0, y_max_ = 5.0;
    int resolution_ = 50;

    // Presets
    int selected_preset_ = -1;
    const char* preset_names_[4] = { "None", "Portfolio", "Regularized Regression", "Simple Quadratic" };

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Contour, 1: Solution
};

} // namespace cyxwiz
