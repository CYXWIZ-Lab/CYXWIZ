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
 * IntegrationPanel - Numerical Integration Tool
 *
 * Features:
 * - Function input (presets)
 * - Integration bounds [a, b]
 * - Method selector: Trapezoid, Simpson, Romberg, Adaptive, Gaussian
 * - Subdivision count (n)
 * - Visual: Area under curve shading
 * - Error estimate display
 * - Comparison of methods
 */
class IntegrationPanel {
public:
    IntegrationPanel();
    ~IntegrationPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSettings();
    void RenderFunctionSelector();
    void RenderBoundsInput();
    void RenderMethodSelector();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderIntegralValue();
    void RenderComparison();
    void RenderAreaPlot();

    void Compute();
    void ComputeComparison();
    void GeneratePlotData();

    bool visible_ = false;

    // Test functions
    enum class TestFunction { Sin, Cos, Polynomial, Exp, Gaussian, Rational };
    TestFunction selected_function_ = TestFunction::Sin;
    const char* function_names_[6] = { "sin(x)", "cos(x)", "x^2", "exp(-x)", "exp(-x^2)", "1/(1+x^2)" };

    // Integration method
    enum class Method { Trapezoid, Simpson, Midpoint, Romberg, Adaptive, Gaussian };
    Method method_ = Method::Simpson;
    const char* method_names_[6] = { "Trapezoid", "Simpson", "Midpoint", "Romberg", "Adaptive", "Gaussian" };

    // Integration bounds
    double lower_bound_ = 0.0;
    double upper_bound_ = 3.14159265358979;  // pi

    // Parameters
    int num_subdivisions_ = 100;
    double adaptive_tolerance_ = 1e-6;
    int gaussian_points_ = 5;
    int romberg_iterations_ = 10;

    // Results
    IntegralResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Comparison results
    std::vector<IntegralResult> comparison_results_;
    double analytical_integral_ = 0.0;  // If known

    // Plot data
    std::vector<double> plot_x_, plot_y_;
    std::vector<double> fill_x_, fill_y_;
    bool plot_generated_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Value, 1: Plot, 2: Comparison
};

} // namespace cyxwiz
