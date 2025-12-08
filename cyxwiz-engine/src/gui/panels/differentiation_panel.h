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
 * DifferentiationPanel - Numerical Differentiation Tool
 *
 * Features:
 * - Function input (presets or expressions)
 * - Point selector for evaluation
 * - Method selector: Forward, Backward, Central differences
 * - Step size (h) control
 * - Gradient visualization (quiver plot for 2D)
 * - Higher-order derivatives option
 * - Comparison table of methods
 */
class DifferentiationPanel {
public:
    DifferentiationPanel();
    ~DifferentiationPanel();

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
    void RenderMethodSelector();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderDerivativeValue();
    void RenderComparison();
    void RenderFunctionPlot();
    void RenderGradientPlot();

    void Compute();
    void ComputeComparison();
    void GeneratePlotData();

    bool visible_ = false;

    // Function type
    enum class FunctionType { SingleVar, MultiVar };
    FunctionType function_type_ = FunctionType::SingleVar;

    // Single variable test functions
    enum class SingleVarFunction { Sin, Cos, Exp, Polynomial, Log };
    SingleVarFunction single_func_ = SingleVarFunction::Sin;
    const char* single_func_names_[5] = { "sin(x)", "cos(x)", "exp(x)", "x^2 + 2x", "ln(x)" };

    // Multi variable test functions (2D)
    enum class MultiVarFunction { Sphere, Rosenbrock, Himmelblau };
    MultiVarFunction multi_func_ = MultiVarFunction::Sphere;
    const char* multi_func_names_[3] = { "x^2 + y^2", "Rosenbrock", "Himmelblau" };

    // Differentiation method
    enum class Method { Forward, Backward, Central };
    Method method_ = Method::Central;
    const char* method_names_[3] = { "Forward", "Backward", "Central" };

    // Evaluation point
    double point_x_ = 1.0;
    double point_y_ = 1.0;  // For 2D functions

    // Step size
    double step_h_ = 1e-5;

    // Options
    bool compute_second_derivative_ = false;
    bool show_comparison_ = true;

    // Results
    DerivativeResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Comparison results
    DerivativeResult forward_result_;
    DerivativeResult backward_result_;
    DerivativeResult central_result_;
    double analytical_derivative_ = 0.0;  // If known

    // Plot data
    std::vector<double> plot_x_, plot_y_;
    std::vector<double> derivative_y_;
    double plot_x_min_ = -5.0, plot_x_max_ = 5.0;
    int plot_points_ = 200;
    bool plot_generated_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Value, 1: Comparison, 2: Plot
};

} // namespace cyxwiz
