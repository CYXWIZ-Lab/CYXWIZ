#include "differentiation_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

DifferentiationPanel::DifferentiationPanel() {
}

DifferentiationPanel::~DifferentiationPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void DifferentiationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(850, 650), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_SUPERSCRIPT " Numerical Differentiation###DifferentiationPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float settings_width = 260.0f;

        // Left column - Settings
        ImGui::BeginChild("DiffSettings", ImVec2(settings_width, 0), true);
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Results
        ImGui::BeginChild("DiffResults", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderResults();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void DifferentiationPanel::RenderToolbar() {
    bool can_compute = !is_computing_;

    if (!can_compute) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_CALCULATOR " Compute")) {
        Compute();
    }
    if (!can_compute) ImGui::EndDisabled();

    ImGui::SameLine();

    if (!can_compute) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_TABLE " Compare All")) {
        ComputeComparison();
    }
    if (!can_compute) ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_TRASH " Clear")) {
        has_result_ = false;
        result_ = DerivativeResult();
        plot_generated_ = false;
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            ICON_FA_CHECK " f'(x) = %.6e", result_.value);
    }
}

void DifferentiationPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Separator();

    RenderFunctionSelector();
    ImGui::Spacing();
    RenderPointInput();
    ImGui::Spacing();
    RenderMethodSelector();

    // Step size
    ImGui::Spacing();
    ImGui::Text("Step Size:");
    float h = static_cast<float>(step_h_);
    if (ImGui::SliderFloat("h", &h, 1e-10f, 1e-1f, "%.2e", ImGuiSliderFlags_Logarithmic)) {
        step_h_ = h;
        plot_generated_ = false;
    }

    // Options
    ImGui::Spacing();
    ImGui::Checkbox("Second derivative", &compute_second_derivative_);
    ImGui::Checkbox("Show comparison", &show_comparison_);
}

void DifferentiationPanel::RenderFunctionSelector() {
    ImGui::Text("Function Type:");
    if (ImGui::RadioButton("Single Variable", function_type_ == FunctionType::SingleVar)) {
        function_type_ = FunctionType::SingleVar;
        plot_generated_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Multi Variable", function_type_ == FunctionType::MultiVar)) {
        function_type_ = FunctionType::MultiVar;
        plot_generated_ = false;
    }

    ImGui::Spacing();

    if (function_type_ == FunctionType::SingleVar) {
        ImGui::Text("f(x):");
        int idx = static_cast<int>(single_func_);
        if (ImGui::Combo("##SingleFunc", &idx, single_func_names_, 5)) {
            single_func_ = static_cast<SingleVarFunction>(idx);
            plot_generated_ = false;
            has_result_ = false;
        }
    } else {
        ImGui::Text("f(x, y):");
        int idx = static_cast<int>(multi_func_);
        if (ImGui::Combo("##MultiFunc", &idx, multi_func_names_, 3)) {
            multi_func_ = static_cast<MultiVarFunction>(idx);
            has_result_ = false;
        }
    }
}

void DifferentiationPanel::RenderPointInput() {
    ImGui::Text("Evaluation Point:");

    float px = static_cast<float>(point_x_);
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("x", &px, 0.1f, 1.0f, "%.4f")) {
        point_x_ = px;
    }

    if (function_type_ == FunctionType::MultiVar) {
        float py = static_cast<float>(point_y_);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputFloat("y", &py, 0.1f, 1.0f, "%.4f")) {
            point_y_ = py;
        }
    }
}

void DifferentiationPanel::RenderMethodSelector() {
    ImGui::Text("Method:");
    int idx = static_cast<int>(method_);
    if (ImGui::Combo("##Method", &idx, method_names_, 3)) {
        method_ = static_cast<Method>(idx);
    }

    // Show formula
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    switch (method_) {
        case Method::Forward:
            ImGui::TextWrapped("f'(x) = [f(x+h) - f(x)] / h");
            ImGui::TextWrapped("O(h) error");
            break;
        case Method::Backward:
            ImGui::TextWrapped("f'(x) = [f(x) - f(x-h)] / h");
            ImGui::TextWrapped("O(h) error");
            break;
        case Method::Central:
            ImGui::TextWrapped("f'(x) = [f(x+h) - f(x-h)] / 2h");
            ImGui::TextWrapped("O(h^2) error");
            break;
    }
    ImGui::PopStyleColor();
}

void DifferentiationPanel::RenderLoadingIndicator() {
    ImGui::Text("Computing derivatives...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void DifferentiationPanel::RenderResults() {
    if (ImGui::BeginTabBar("DiffResultsTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CALCULATOR " Value")) {
            viz_tab_ = 0;
            RenderDerivativeValue();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Comparison")) {
            viz_tab_ = 1;
            RenderComparison();
            ImGui::EndTabItem();
        }
        if (function_type_ == FunctionType::SingleVar) {
            if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Plot")) {
                viz_tab_ = 2;
                RenderFunctionPlot();
                ImGui::EndTabItem();
            }
        } else {
            if (ImGui::BeginTabItem(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Gradient")) {
                viz_tab_ = 2;
                RenderGradientPlot();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
}

void DifferentiationPanel::RenderDerivativeValue() {
    if (!has_result_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Compute' to calculate derivative");
        return;
    }

    if (!result_.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
            "Error: %s", result_.error_message.c_str());
        return;
    }

    if (function_type_ == FunctionType::SingleVar) {
        ImGui::Text("First Derivative:");
        ImGui::BulletText("f'(%.4f) = %.10f", point_x_, result_.value);

        // Show all three estimates
        ImGui::Spacing();
        ImGui::Text("Method Estimates:");
        ImGui::BulletText("Forward:  %.10f", result_.forward_estimate);
        ImGui::BulletText("Backward: %.10f", result_.backward_estimate);
        ImGui::BulletText("Central:  %.10f", result_.central_estimate);

        // Analytical derivative if known
        ImGui::Spacing();
        switch (single_func_) {
            case SingleVarFunction::Sin:
                analytical_derivative_ = std::cos(point_x_);
                ImGui::Text("Analytical: cos(%.4f) = %.10f", point_x_, analytical_derivative_);
                ImGui::Text("Error: %.2e", std::abs(result_.value - analytical_derivative_));
                break;
            case SingleVarFunction::Cos:
                analytical_derivative_ = -std::sin(point_x_);
                ImGui::Text("Analytical: -sin(%.4f) = %.10f", point_x_, analytical_derivative_);
                ImGui::Text("Error: %.2e", std::abs(result_.value - analytical_derivative_));
                break;
            case SingleVarFunction::Exp:
                analytical_derivative_ = std::exp(point_x_);
                ImGui::Text("Analytical: exp(%.4f) = %.10f", point_x_, analytical_derivative_);
                ImGui::Text("Error: %.2e", std::abs(result_.value - analytical_derivative_));
                break;
            case SingleVarFunction::Polynomial:
                analytical_derivative_ = 2.0 * point_x_ + 2.0;
                ImGui::Text("Analytical: 2x + 2 = %.10f", analytical_derivative_);
                ImGui::Text("Error: %.2e", std::abs(result_.value - analytical_derivative_));
                break;
            case SingleVarFunction::Log:
                if (point_x_ > 0) {
                    analytical_derivative_ = 1.0 / point_x_;
                    ImGui::Text("Analytical: 1/x = %.10f", analytical_derivative_);
                    ImGui::Text("Error: %.2e", std::abs(result_.value - analytical_derivative_));
                }
                break;
        }
    } else {
        // Multi-variable gradient
        ImGui::Text("Gradient:");
        if (result_.gradient.size() >= 2) {
            ImGui::BulletText("df/dx = %.10f", result_.gradient[0]);
            ImGui::BulletText("df/dy = %.10f", result_.gradient[1]);
        }

        if (!result_.hessian.empty()) {
            ImGui::Spacing();
            ImGui::Text("Hessian:");
            for (size_t i = 0; i < result_.hessian.size(); ++i) {
                std::string row = "";
                for (size_t j = 0; j < result_.hessian[i].size(); ++j) {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.4f ", result_.hessian[i][j]);
                    row += buf;
                }
                ImGui::BulletText("%s", row.c_str());
            }
        }
    }
}

void DifferentiationPanel::RenderComparison() {
    if (!show_comparison_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Enable 'Show comparison' and click 'Compare All'");
        return;
    }

    ImGui::Text("Method Comparison at x = %.4f", point_x_);
    ImGui::Separator();

    if (ImGui::BeginTable("ComparisonTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Method");
        ImGui::TableSetupColumn("Estimate");
        ImGui::TableSetupColumn("Error Order");
        ImGui::TableSetupColumn("Abs Error");
        ImGui::TableHeadersRow();

        // Forward
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Forward");
        ImGui::TableNextColumn(); ImGui::Text("%.10f", forward_result_.value);
        ImGui::TableNextColumn(); ImGui::Text("O(h)");
        ImGui::TableNextColumn();
        if (std::abs(analytical_derivative_) > 1e-15) {
            ImGui::Text("%.2e", std::abs(forward_result_.value - analytical_derivative_));
        } else {
            ImGui::Text("-");
        }

        // Backward
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Backward");
        ImGui::TableNextColumn(); ImGui::Text("%.10f", backward_result_.value);
        ImGui::TableNextColumn(); ImGui::Text("O(h)");
        ImGui::TableNextColumn();
        if (std::abs(analytical_derivative_) > 1e-15) {
            ImGui::Text("%.2e", std::abs(backward_result_.value - analytical_derivative_));
        } else {
            ImGui::Text("-");
        }

        // Central
        ImGui::TableNextRow();
        ImGui::TableNextColumn(); ImGui::Text("Central");
        ImGui::TableNextColumn(); ImGui::Text("%.10f", central_result_.value);
        ImGui::TableNextColumn(); ImGui::Text("O(h^2)");
        ImGui::TableNextColumn();
        if (std::abs(analytical_derivative_) > 1e-15) {
            ImGui::Text("%.2e", std::abs(central_result_.value - analytical_derivative_));
        } else {
            ImGui::Text("-");
        }

        // Analytical (if known)
        if (std::abs(analytical_derivative_) > 1e-15) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Analytical");
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.10f", analytical_derivative_);
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Exact");
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "0");
        }

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::TextWrapped("Central difference has O(h^2) error, making it more accurate "
                      "for the same step size h compared to forward/backward O(h).");
}

void DifferentiationPanel::RenderFunctionPlot() {
    if (!plot_generated_) {
        GeneratePlotData();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##FunctionPlot", avail)) {
        ImPlot::SetupAxes("x", "y");

        // Plot function
        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("f(x)", plot_x_.data(), plot_y_.data(), static_cast<int>(plot_x_.size()));

        // Plot derivative
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 2.0f);
        ImPlot::PlotLine("f'(x)", plot_x_.data(), derivative_y_.data(), static_cast<int>(plot_x_.size()));

        // Mark evaluation point
        if (has_result_) {
            double px = point_x_;
            double py = 0.0;

            // Get function value at point
            switch (single_func_) {
                case SingleVarFunction::Sin: py = std::sin(px); break;
                case SingleVarFunction::Cos: py = std::cos(px); break;
                case SingleVarFunction::Exp: py = std::exp(px); break;
                case SingleVarFunction::Polynomial: py = px * px + 2 * px; break;
                case SingleVarFunction::Log: py = (px > 0) ? std::log(px) : 0; break;
            }

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(1.0f, 0.8f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotScatter("Point", &px, &py, 1);
        }

        ImPlot::EndPlot();
    }
}

void DifferentiationPanel::RenderGradientPlot() {
    // For 2D functions, show a quiver plot of the gradient field
    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##GradientPlot", avail)) {
        ImPlot::SetupAxes("x", "y");
        ImPlot::SetupAxisLimits(ImAxis_X1, -3.0, 3.0);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -3.0, 3.0);

        // Generate gradient field
        std::function<double(const std::vector<double>&)> func;
        switch (multi_func_) {
            case MultiVarFunction::Sphere:
                func = Optimization::Sphere;
                break;
            case MultiVarFunction::Rosenbrock:
                func = Optimization::Rosenbrock;
                break;
            case MultiVarFunction::Himmelblau:
                func = Optimization::Himmelblau;
                break;
        }

        // Draw arrows at grid points
        for (double x = -2.5; x <= 2.5; x += 0.5) {
            for (double y = -2.5; y <= 2.5; y += 0.5) {
                auto grad_result = Optimization::NumericalGradient(func, {x, y}, step_h_);
                if (grad_result.success && grad_result.gradient.size() >= 2) {
                    double gx = grad_result.gradient[0];
                    double gy = grad_result.gradient[1];

                    // Normalize and scale
                    double mag = std::sqrt(gx * gx + gy * gy);
                    if (mag > 1e-10) {
                        double scale = 0.3 / std::max(1.0, mag);
                        gx *= scale;
                        gy *= scale;

                        double arrow_x[2] = { x, x + gx };
                        double arrow_y[2] = { y, y + gy };
                        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 0.8f), 1.5f);
                        ImPlot::PlotLine("##arrow", arrow_x, arrow_y, 2);
                    }
                }
            }
        }

        // Mark evaluation point
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 10, ImVec4(1.0f, 0.8f, 0.0f, 1.0f), 2.0f);
        ImPlot::PlotScatter("Point", &point_x_, &point_y_, 1);

        ImPlot::EndPlot();
    }
}

void DifferentiationPanel::Compute() {
    if (is_computing_) return;

    SingleVarFunction sf = single_func_;
    MultiVarFunction mf = multi_func_;
    FunctionType ft = function_type_;
    double x = point_x_;
    double y = point_y_;
    double h = step_h_;
    std::string meth = (method_ == Method::Forward) ? "forward" :
                       (method_ == Method::Backward) ? "backward" : "central";

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, sf, mf, ft, x, y, h, meth]() {
        DerivativeResult res;

        if (ft == FunctionType::SingleVar) {
            std::function<double(double)> func;
            switch (sf) {
                case SingleVarFunction::Sin:
                    func = [](double t) { return std::sin(t); };
                    break;
                case SingleVarFunction::Cos:
                    func = [](double t) { return std::cos(t); };
                    break;
                case SingleVarFunction::Exp:
                    func = [](double t) { return std::exp(t); };
                    break;
                case SingleVarFunction::Polynomial:
                    func = [](double t) { return t * t + 2 * t; };
                    break;
                case SingleVarFunction::Log:
                    func = [](double t) { return (t > 0) ? std::log(t) : 0; };
                    break;
            }

            res = Optimization::CompareDerivativeMethods(func, x, h);
        } else {
            std::function<double(const std::vector<double>&)> func;
            switch (mf) {
                case MultiVarFunction::Sphere:
                    func = Optimization::Sphere;
                    break;
                case MultiVarFunction::Rosenbrock:
                    func = Optimization::Rosenbrock;
                    break;
                case MultiVarFunction::Himmelblau:
                    func = Optimization::Himmelblau;
                    break;
            }

            res = Optimization::NumericalGradient(func, {x, y}, h);
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void DifferentiationPanel::ComputeComparison() {
    if (function_type_ != FunctionType::SingleVar) return;

    std::function<double(double)> func;
    switch (single_func_) {
        case SingleVarFunction::Sin:
            func = [](double t) { return std::sin(t); };
            analytical_derivative_ = std::cos(point_x_);
            break;
        case SingleVarFunction::Cos:
            func = [](double t) { return std::cos(t); };
            analytical_derivative_ = -std::sin(point_x_);
            break;
        case SingleVarFunction::Exp:
            func = [](double t) { return std::exp(t); };
            analytical_derivative_ = std::exp(point_x_);
            break;
        case SingleVarFunction::Polynomial:
            func = [](double t) { return t * t + 2 * t; };
            analytical_derivative_ = 2 * point_x_ + 2;
            break;
        case SingleVarFunction::Log:
            func = [](double t) { return (t > 0) ? std::log(t) : 0; };
            analytical_derivative_ = (point_x_ > 0) ? 1.0 / point_x_ : 0;
            break;
    }

    forward_result_ = Optimization::NumericalDerivative(func, point_x_, step_h_, "forward");
    backward_result_ = Optimization::NumericalDerivative(func, point_x_, step_h_, "backward");
    central_result_ = Optimization::NumericalDerivative(func, point_x_, step_h_, "central");
}

void DifferentiationPanel::GeneratePlotData() {
    plot_x_.clear();
    plot_y_.clear();
    derivative_y_.clear();

    std::function<double(double)> func;
    std::function<double(double)> deriv;

    switch (single_func_) {
        case SingleVarFunction::Sin:
            func = [](double t) { return std::sin(t); };
            deriv = [](double t) { return std::cos(t); };
            break;
        case SingleVarFunction::Cos:
            func = [](double t) { return std::cos(t); };
            deriv = [](double t) { return -std::sin(t); };
            break;
        case SingleVarFunction::Exp:
            func = [](double t) { return std::exp(t); };
            deriv = [](double t) { return std::exp(t); };
            plot_x_max_ = 3.0;  // Limit for exp
            break;
        case SingleVarFunction::Polynomial:
            func = [](double t) { return t * t + 2 * t; };
            deriv = [](double t) { return 2 * t + 2; };
            break;
        case SingleVarFunction::Log:
            func = [](double t) { return (t > 0.1) ? std::log(t) : std::log(0.1); };
            deriv = [](double t) { return (t > 0.1) ? 1.0 / t : 10.0; };
            plot_x_min_ = 0.1;
            break;
    }

    double dx = (plot_x_max_ - plot_x_min_) / (plot_points_ - 1);
    for (int i = 0; i < plot_points_; ++i) {
        double x = plot_x_min_ + i * dx;
        plot_x_.push_back(x);
        plot_y_.push_back(func(x));
        derivative_y_.push_back(deriv(x));
    }

    plot_generated_ = true;
}

} // namespace cyxwiz
