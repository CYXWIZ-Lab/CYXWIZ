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
 * GradientDescentPanel - Optimization Visualizer
 *
 * Features:
 * - Objective function selector (presets: Rosenbrock, Sphere, Rastrigin, etc.)
 * - Algorithm selector: Vanilla GD, Momentum, Adam, RMSprop
 * - Parameters: learning rate, momentum, iterations, tolerance
 * - 2D contour plot with optimization path overlay
 * - Convergence plot (objective vs iteration)
 * - Gradient magnitude history
 * - Step-by-step visualization
 */
class GradientDescentPanel {
public:
    GradientDescentPanel();
    ~GradientDescentPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderSettings();
    void RenderFunctionSelector();
    void RenderAlgorithmSelector();
    void RenderParameters();
    void RenderStartPoint();
    void RenderLoadingIndicator();
    void RenderVisualization();
    void RenderContourPlot();
    void RenderConvergencePlot();
    void RenderGradientPlot();
    void RenderStats();

    void RunOptimization();
    void RunStep();
    void Reset();
    void GenerateContourData();

    bool visible_ = false;

    // Function selection
    enum class TestFunction { Rosenbrock, Sphere, Rastrigin, Beale, Booth, Himmelblau };
    TestFunction selected_function_ = TestFunction::Rosenbrock;
    const char* function_names_[6] = { "Rosenbrock", "Sphere", "Rastrigin", "Beale", "Booth", "Himmelblau" };

    // Algorithm selection
    enum class Algorithm { VanillaGD, Momentum, Adam, RMSprop };
    Algorithm selected_algorithm_ = Algorithm::Adam;
    const char* algorithm_names_[4] = { "Vanilla GD", "Momentum", "Adam", "RMSprop" };

    // Parameters
    double learning_rate_ = 0.01;
    double momentum_ = 0.9;
    double beta1_ = 0.9;   // Adam
    double beta2_ = 0.999; // Adam
    double decay_rate_ = 0.99; // RMSprop
    double epsilon_ = 1e-8;
    int max_iterations_ = 1000;
    double tolerance_ = 1e-6;

    // Starting point
    double start_x_ = -2.0;
    double start_y_ = 2.0;

    // Contour plot range
    double x_min_ = -3.0, x_max_ = 3.0;
    double y_min_ = -3.0, y_max_ = 3.0;
    int contour_resolution_ = 50;

    // Results
    GradientDescentResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Contour data for visualization
    std::vector<std::vector<double>> contour_data_;
    std::vector<double> contour_x_, contour_y_;
    bool contour_generated_ = false;

    // Step-by-step mode
    bool step_mode_ = false;
    int current_step_ = 0;
    std::vector<double> current_x_;
    std::vector<double> velocity_;  // For momentum
    std::vector<double> m_, v_;     // For Adam

    // Gradient history for plotting
    std::vector<double> gradient_norms_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Visualization tab
    int viz_tab_ = 0; // 0: Contour, 1: Convergence, 2: Gradient
};

} // namespace cyxwiz
