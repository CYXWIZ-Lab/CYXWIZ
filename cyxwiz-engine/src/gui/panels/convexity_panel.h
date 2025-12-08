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
 * ConvexityPanel - Convexity Analysis Tool
 *
 * Features:
 * - Function input (2D test functions)
 * - Hessian computation at user-specified points
 * - Eigenvalue display
 * - Convexity classification (convex, strictly convex, concave, non-convex)
 * - 3D surface plot with convexity visualization
 * - Level set visualization
 */
class ConvexityPanel {
public:
    ConvexityPanel();
    ~ConvexityPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSettings();
    void RenderFunctionSelector();
    void RenderPointInput();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderHessianMatrix();
    void RenderEigenvalues();
    void RenderSurfacePlot();
    void RenderAnalysisText();

    void AnalyzeConvexity();
    void GenerateSurfaceData();

    bool visible_ = false;

    // Function selection
    enum class TestFunction { Rosenbrock, Sphere, Rastrigin, Beale, Booth, Himmelblau, Custom };
    TestFunction selected_function_ = TestFunction::Sphere;
    const char* function_names_[7] = { "Rosenbrock", "Sphere", "Rastrigin", "Beale", "Booth", "Himmelblau", "Custom" };

    // Point to analyze
    double point_x_ = 0.0;
    double point_y_ = 0.0;
    double delta_ = 1e-5;  // For numerical Hessian

    // Plot range
    double x_min_ = -3.0, x_max_ = 3.0;
    double y_min_ = -3.0, y_max_ = 3.0;
    int resolution_ = 40;

    // Results
    ConvexityResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Surface data for visualization
    std::vector<std::vector<double>> surface_data_;
    bool surface_generated_ = false;

    // Multi-point analysis
    std::vector<std::pair<double, double>> sample_points_;
    std::vector<ConvexityResult> sample_results_;
    bool show_sample_analysis_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Surface, 1: Hessian, 2: Eigenvalues
};

} // namespace cyxwiz
