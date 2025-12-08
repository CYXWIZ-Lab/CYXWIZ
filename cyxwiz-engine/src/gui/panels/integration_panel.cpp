#include "integration_panel.h"
#include "../icons.h"
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace cyxwiz {

IntegrationPanel::IntegrationPanel() {
}

IntegrationPanel::~IntegrationPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void IntegrationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(850, 650), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_SQUARE_ROOT_VARIABLE " Numerical Integration###IntegrationPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        // Two-column layout
        float settings_width = 260.0f;

        // Left column - Settings
        ImGui::BeginChild("IntSettings", ImVec2(settings_width, 0), true);
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        // Right column - Results
        ImGui::BeginChild("IntResults", ImVec2(0, 0), true);
        if (is_computing_) {
            RenderLoadingIndicator();
        } else {
            RenderResults();
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void IntegrationPanel::RenderToolbar() {
    bool can_compute = !is_computing_;

    if (!can_compute) ImGui::BeginDisabled();
    if (ImGui::Button(ICON_FA_CALCULATOR " Integrate")) {
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
        result_ = IntegralResult();
        comparison_results_.clear();
        plot_generated_ = false;
    }

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    if (has_result_ && result_.success) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            ICON_FA_CHECK " Integral = %.8f", result_.value);
    }
}

void IntegrationPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Separator();

    RenderFunctionSelector();
    ImGui::Spacing();
    RenderBoundsInput();
    ImGui::Spacing();
    RenderMethodSelector();
}

void IntegrationPanel::RenderFunctionSelector() {
    ImGui::Text("Function f(x):");
    int idx = static_cast<int>(selected_function_);
    if (ImGui::Combo("##Function", &idx, function_names_, 6)) {
        selected_function_ = static_cast<TestFunction>(idx);
        plot_generated_ = false;
        has_result_ = false;

        // Set appropriate default bounds
        switch (selected_function_) {
            case TestFunction::Sin:
            case TestFunction::Cos:
                lower_bound_ = 0.0;
                upper_bound_ = 3.14159265358979;
                break;
            case TestFunction::Polynomial:
                lower_bound_ = 0.0;
                upper_bound_ = 2.0;
                break;
            case TestFunction::Exp:
                lower_bound_ = 0.0;
                upper_bound_ = 5.0;
                break;
            case TestFunction::Gaussian:
                lower_bound_ = -3.0;
                upper_bound_ = 3.0;
                break;
            case TestFunction::Rational:
                lower_bound_ = 0.0;
                upper_bound_ = 1.0;
                break;
        }
    }

    // Show analytical integral if known
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    switch (selected_function_) {
        case TestFunction::Sin:
            ImGui::TextWrapped("Integral of sin(x) = -cos(x)");
            break;
        case TestFunction::Cos:
            ImGui::TextWrapped("Integral of cos(x) = sin(x)");
            break;
        case TestFunction::Polynomial:
            ImGui::TextWrapped("Integral of x^2 = x^3/3");
            break;
        case TestFunction::Exp:
            ImGui::TextWrapped("Integral of exp(-x) = -exp(-x)");
            break;
        case TestFunction::Gaussian:
            ImGui::TextWrapped("Gaussian: sqrt(pi)*erf(x)");
            break;
        case TestFunction::Rational:
            ImGui::TextWrapped("Integral = arctan(x)");
            break;
    }
    ImGui::PopStyleColor();
}

void IntegrationPanel::RenderBoundsInput() {
    ImGui::Text("Integration Bounds:");

    float a = static_cast<float>(lower_bound_);
    float b = static_cast<float>(upper_bound_);

    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("a (lower)", &a, 0.1f, 1.0f, "%.4f")) {
        lower_bound_ = a;
        plot_generated_ = false;
    }

    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("b (upper)", &b, 0.1f, 1.0f, "%.4f")) {
        upper_bound_ = b;
        plot_generated_ = false;
    }
}

void IntegrationPanel::RenderMethodSelector() {
    ImGui::Text("Method:");
    int idx = static_cast<int>(method_);
    if (ImGui::Combo("##Method", &idx, method_names_, 6)) {
        method_ = static_cast<Method>(idx);
    }

    // Method-specific parameters
    ImGui::Spacing();

    switch (method_) {
        case Method::Trapezoid:
        case Method::Simpson:
        case Method::Midpoint:
            ImGui::SliderInt("Subdivisions", &num_subdivisions_, 2, 1000);
            break;
        case Method::Romberg:
            ImGui::SliderInt("Max Iterations", &romberg_iterations_, 2, 20);
            break;
        case Method::Adaptive:
            {
                float tol = static_cast<float>(adaptive_tolerance_);
                if (ImGui::SliderFloat("Tolerance", &tol, 1e-12f, 1e-2f, "%.2e", ImGuiSliderFlags_Logarithmic)) {
                    adaptive_tolerance_ = tol;
                }
            }
            break;
        case Method::Gaussian:
            ImGui::SliderInt("Quadrature Points", &gaussian_points_, 1, 10);
            break;
    }

    // Show error formula
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));
    switch (method_) {
        case Method::Trapezoid:
            ImGui::TextWrapped("Error: O(h^2)");
            break;
        case Method::Simpson:
            ImGui::TextWrapped("Error: O(h^4)");
            break;
        case Method::Midpoint:
            ImGui::TextWrapped("Error: O(h^2)");
            break;
        case Method::Romberg:
            ImGui::TextWrapped("Richardson extrapolation");
            break;
        case Method::Adaptive:
            ImGui::TextWrapped("Auto-adjusts step size");
            break;
        case Method::Gaussian:
            ImGui::TextWrapped("Exact for poly degree 2n-1");
            break;
    }
    ImGui::PopStyleColor();
}

void IntegrationPanel::RenderLoadingIndicator() {
    ImGui::Text("Computing integral...");
    float progress = 0.5f + 0.5f * std::sin(ImGui::GetTime() * 5.0f);
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
}

void IntegrationPanel::RenderResults() {
    if (ImGui::BeginTabBar("IntResultsTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CALCULATOR " Value")) {
            viz_tab_ = 0;
            RenderIntegralValue();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Plot")) {
            viz_tab_ = 1;
            RenderAreaPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Comparison")) {
            viz_tab_ = 2;
            RenderComparison();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void IntegrationPanel::RenderIntegralValue() {
    if (!has_result_) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Integrate' to compute the integral");
        return;
    }

    if (!result_.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
            "Error: %s", result_.error_message.c_str());
        return;
    }

    ImGui::Text("Result:");
    ImGui::BulletText("Integral value: %.12f", result_.value);
    ImGui::BulletText("Method used: %s", result_.method_used.c_str());
    ImGui::BulletText("Function evaluations: %d", result_.function_evaluations);

    if (result_.absolute_error > 0) {
        ImGui::BulletText("Estimated error: %.2e", result_.absolute_error);
    }

    // Analytical value comparison
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Comparison with Analytical:");

    switch (selected_function_) {
        case TestFunction::Sin:
            analytical_integral_ = -std::cos(upper_bound_) + std::cos(lower_bound_);
            break;
        case TestFunction::Cos:
            analytical_integral_ = std::sin(upper_bound_) - std::sin(lower_bound_);
            break;
        case TestFunction::Polynomial:
            analytical_integral_ = (std::pow(upper_bound_, 3) - std::pow(lower_bound_, 3)) / 3.0;
            break;
        case TestFunction::Exp:
            analytical_integral_ = -std::exp(-upper_bound_) + std::exp(-lower_bound_);
            break;
        case TestFunction::Gaussian:
            // sqrt(pi) * (erf(b) - erf(a)) / 2
            analytical_integral_ = std::sqrt(3.14159265358979) * (std::erf(upper_bound_) - std::erf(lower_bound_)) / 2.0;
            break;
        case TestFunction::Rational:
            analytical_integral_ = std::atan(upper_bound_) - std::atan(lower_bound_);
            break;
    }

    ImGui::BulletText("Analytical: %.12f", analytical_integral_);
    ImGui::BulletText("Numerical:  %.12f", result_.value);

    double abs_error = std::abs(result_.value - analytical_integral_);
    double rel_error = abs_error / std::max(std::abs(analytical_integral_), 1e-15);

    if (abs_error < 1e-10) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            "Absolute error: %.2e (excellent)", abs_error);
    } else if (abs_error < 1e-6) {
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.2f, 1.0f),
            "Absolute error: %.2e (good)", abs_error);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f),
            "Absolute error: %.2e", abs_error);
    }

    ImGui::Text("Relative error: %.2e", rel_error);
}

void IntegrationPanel::RenderAreaPlot() {
    if (!plot_generated_) {
        GeneratePlotData();
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    if (ImPlot::BeginPlot("##AreaPlot", avail)) {
        ImPlot::SetupAxes("x", "f(x)");

        // Plot function
        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("f(x)", plot_x_.data(), plot_y_.data(), static_cast<int>(plot_x_.size()));

        // Shade area under curve
        if (!fill_x_.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.6f, 1.0f, 0.3f));
            ImPlot::PlotShaded("Area", fill_x_.data(), fill_y_.data(), static_cast<int>(fill_x_.size()), 0.0);
        }

        // Mark integration bounds
        double bounds_y_range[2] = { -10, 10 };
        double a_x[2] = { lower_bound_, lower_bound_ };
        double b_x[2] = { upper_bound_, upper_bound_ };
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.3f, 0.8f), 1.5f);
        ImPlot::PlotLine("##a", a_x, bounds_y_range, 2);
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.3f, 0.8f), 1.5f);
        ImPlot::PlotLine("##b", b_x, bounds_y_range, 2);

        ImPlot::EndPlot();
    }
}

void IntegrationPanel::RenderComparison() {
    if (comparison_results_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "Click 'Compare All' to compare integration methods");
        return;
    }

    ImGui::Text("Method Comparison for integral from %.2f to %.2f", lower_bound_, upper_bound_);
    ImGui::Separator();

    if (ImGui::BeginTable("ComparisonTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Method");
        ImGui::TableSetupColumn("Value");
        ImGui::TableSetupColumn("Error");
        ImGui::TableSetupColumn("Evaluations");
        ImGui::TableSetupColumn("Est. Error");
        ImGui::TableHeadersRow();

        for (const auto& res : comparison_results_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", res.method_used.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.10f", res.value);
            ImGui::TableNextColumn();
            double err = std::abs(res.value - analytical_integral_);
            if (err < 1e-10) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.2e", err);
            } else if (err < 1e-6) {
                ImGui::TextColored(ImVec4(0.6f, 0.8f, 0.2f, 1.0f), "%.2e", err);
            } else {
                ImGui::Text("%.2e", err);
            }
            ImGui::TableNextColumn();
            ImGui::Text("%d", res.function_evaluations);
            ImGui::TableNextColumn();
            if (res.absolute_error > 0) {
                ImGui::Text("%.2e", res.absolute_error);
            } else {
                ImGui::Text("-");
            }
        }

        // Analytical row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Analytical");
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.10f", analytical_integral_);
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "0");
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "N/A");
        ImGui::TableNextColumn();
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "0");

        ImGui::EndTable();
    }
}

void IntegrationPanel::Compute() {
    if (is_computing_) return;

    TestFunction sf = selected_function_;
    Method m = method_;
    double a = lower_bound_;
    double b = upper_bound_;
    int n = num_subdivisions_;
    double tol = adaptive_tolerance_;
    int gp = gaussian_points_;
    int ri = romberg_iterations_;

    is_computing_ = true;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    compute_thread_ = std::make_unique<std::thread>([this, sf, m, a, b, n, tol, gp, ri]() {
        std::function<double(double)> func;
        switch (sf) {
            case TestFunction::Sin:
                func = [](double x) { return std::sin(x); };
                break;
            case TestFunction::Cos:
                func = [](double x) { return std::cos(x); };
                break;
            case TestFunction::Polynomial:
                func = [](double x) { return x * x; };
                break;
            case TestFunction::Exp:
                func = [](double x) { return std::exp(-x); };
                break;
            case TestFunction::Gaussian:
                func = [](double x) { return std::exp(-x * x); };
                break;
            case TestFunction::Rational:
                func = [](double x) { return 1.0 / (1.0 + x * x); };
                break;
        }

        IntegralResult res;
        switch (m) {
            case Method::Trapezoid:
                res = Optimization::TrapezoidalRule(func, a, b, n);
                break;
            case Method::Simpson:
                res = Optimization::SimpsonsRule(func, a, b, n);
                break;
            case Method::Midpoint:
                res = Optimization::Integrate(func, a, b, "midpoint", n);
                break;
            case Method::Romberg:
                res = Optimization::RombergIntegration(func, a, b, ri);
                break;
            case Method::Adaptive:
                res = Optimization::AdaptiveIntegrate(func, a, b, tol);
                break;
            case Method::Gaussian:
                res = Optimization::GaussianQuadrature(func, a, b, gp);
                break;
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            result_ = std::move(res);
            has_result_ = true;
        }

        is_computing_ = false;
    });
}

void IntegrationPanel::ComputeComparison() {
    std::function<double(double)> func;
    switch (selected_function_) {
        case TestFunction::Sin:
            func = [](double x) { return std::sin(x); };
            analytical_integral_ = -std::cos(upper_bound_) + std::cos(lower_bound_);
            break;
        case TestFunction::Cos:
            func = [](double x) { return std::cos(x); };
            analytical_integral_ = std::sin(upper_bound_) - std::sin(lower_bound_);
            break;
        case TestFunction::Polynomial:
            func = [](double x) { return x * x; };
            analytical_integral_ = (std::pow(upper_bound_, 3) - std::pow(lower_bound_, 3)) / 3.0;
            break;
        case TestFunction::Exp:
            func = [](double x) { return std::exp(-x); };
            analytical_integral_ = -std::exp(-upper_bound_) + std::exp(-lower_bound_);
            break;
        case TestFunction::Gaussian:
            func = [](double x) { return std::exp(-x * x); };
            analytical_integral_ = std::sqrt(3.14159265358979) * (std::erf(upper_bound_) - std::erf(lower_bound_)) / 2.0;
            break;
        case TestFunction::Rational:
            func = [](double x) { return 1.0 / (1.0 + x * x); };
            analytical_integral_ = std::atan(upper_bound_) - std::atan(lower_bound_);
            break;
    }

    comparison_results_.clear();
    comparison_results_.push_back(Optimization::TrapezoidalRule(func, lower_bound_, upper_bound_, num_subdivisions_));
    comparison_results_.push_back(Optimization::SimpsonsRule(func, lower_bound_, upper_bound_, num_subdivisions_));
    comparison_results_.push_back(Optimization::RombergIntegration(func, lower_bound_, upper_bound_, romberg_iterations_));
    comparison_results_.push_back(Optimization::AdaptiveIntegrate(func, lower_bound_, upper_bound_, adaptive_tolerance_));
    comparison_results_.push_back(Optimization::GaussianQuadrature(func, lower_bound_, upper_bound_, gaussian_points_));
}

void IntegrationPanel::GeneratePlotData() {
    plot_x_.clear();
    plot_y_.clear();
    fill_x_.clear();
    fill_y_.clear();

    std::function<double(double)> func;
    switch (selected_function_) {
        case TestFunction::Sin:
            func = [](double x) { return std::sin(x); };
            break;
        case TestFunction::Cos:
            func = [](double x) { return std::cos(x); };
            break;
        case TestFunction::Polynomial:
            func = [](double x) { return x * x; };
            break;
        case TestFunction::Exp:
            func = [](double x) { return std::exp(-x); };
            break;
        case TestFunction::Gaussian:
            func = [](double x) { return std::exp(-x * x); };
            break;
        case TestFunction::Rational:
            func = [](double x) { return 1.0 / (1.0 + x * x); };
            break;
    }

    // Generate full function plot
    double margin = (upper_bound_ - lower_bound_) * 0.2;
    double x_min = lower_bound_ - margin;
    double x_max = upper_bound_ + margin;
    int num_points = 200;
    double dx = (x_max - x_min) / (num_points - 1);

    for (int i = 0; i < num_points; ++i) {
        double x = x_min + i * dx;
        plot_x_.push_back(x);
        plot_y_.push_back(func(x));
    }

    // Generate fill data for integration region
    int fill_points = 100;
    double fill_dx = (upper_bound_ - lower_bound_) / (fill_points - 1);
    for (int i = 0; i < fill_points; ++i) {
        double x = lower_bound_ + i * fill_dx;
        fill_x_.push_back(x);
        fill_y_.push_back(func(x));
    }

    plot_generated_ = true;
}

} // namespace cyxwiz
