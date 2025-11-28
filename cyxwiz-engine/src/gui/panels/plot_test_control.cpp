#include "plot_test_control.h"
#include "../../plotting/test_data_generator.h"
#include "../../plotting/plot_manager.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

PlotTestControlPanel::PlotTestControlPanel()
    : Panel("Plot Test Control", false)  // Hidden by default - use toolbar button to show
    , selected_plot_type_(0)
    , selected_backend_(0)
    , selected_test_data_(0)
{
}

void PlotTestControlPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(name_.c_str(), &visible_);

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Test Plotting System");
    ImGui::Separator();
    ImGui::Spacing();

    // Plot Type Selection
    ImGui::Text("Plot Type:");
    const char* plot_types[] = {
        "Line Plot",
        "Scatter Plot",
        "Bar Chart",
        "Histogram",
        "Box Plot",
        "Stem Plot",
        "Stairs Plot",
        "Pie Chart",
        "Heatmap",
        "Polar Plot",
        "3D Surface",
        "3D Scatter",
        "3D Line"
    };
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##PlotType", &selected_plot_type_, plot_types, IM_ARRAYSIZE(plot_types));

    ImGui::Spacing();
    ImGui::Spacing();

    // Backend Selection
    ImGui::Text("Backend:");
    const char* backends[] = {
        "ImPlot (Real-time)",
        "Matplotlib (Offline)"
    };
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##Backend", &selected_backend_, backends, IM_ARRAYSIZE(backends));

    ImGui::Spacing();
    ImGui::Spacing();

    // Test Data Selection
    ImGui::Text("Test Data:");
    const char* test_data[] = {
        "Sine Wave",
        "Cosine Wave",
        "Normal Distribution",
        "Exponential Decay",
        "Random Scatter",
        "Linear",
        "Polynomial",
        "Damped Oscillation"
    };
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##TestData", &selected_test_data_, test_data, IM_ARRAYSIZE(test_data));

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Calculate button sizes - two buttons side by side with default height
    float available_width = ImGui::GetContentRegionAvail().x;
    float button_spacing = ImGui::GetStyle().ItemSpacing.x;
    float button_width = (available_width - button_spacing) * 0.5f;
    ImVec2 button_size = ImVec2(button_width, 0);  // Use default button height

    // Generate Plot button (default color)
    if (ImGui::Button("Generate Plot", button_size)) {
        GeneratePlot();
    }

    // Clear All Button (same line, default color)
    ImGui::SameLine();
    if (ImGui::Button("Clear All", button_size)) {
        ClearAllPlots();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Info section
    ImGui::TextWrapped("Use this panel to test different plot types with various test data patterns.");
    ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.9f, 1.0f, 1.0f));
    ImGui::BulletText("ImPlot is faster for real-time updates");
    ImGui::PopStyleColor();

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.5f, 1.0f));
    ImGui::BulletText("Matplotlib is better for exports");
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Configuration Info
    ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f), "Configuration:");
    ImGui::Indent();
    ImGui::Text("Data Points: 100");
    ImGui::Text("Auto-generated with real math");
    ImGui::Unindent();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Statistics
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Plots Created: %zu", plot_windows_.size());

    ImGui::End();

    // Render all plot windows created by this panel
    for (auto& plot_window : plot_windows_) {
        if (plot_window) {
            plot_window->Render();
        }
    }
}

void PlotTestControlPanel::GeneratePlot() {
    using namespace cyxwiz::plotting;

    // Plot type names for logging and titles
    const char* plot_type_names[] = {
        "Line Plot",
        "Scatter Plot",
        "Bar Chart",
        "Histogram",
        "Box Plot",
        "Stem Plot",
        "Stairs Plot",
        "Pie Chart",
        "Heatmap",
        "Polar Plot",
        "3D Surface",
        "3D Scatter",
        "3D Line"
    };

    const char* backend_names[] = {
        "ImPlot (Real-time)",
        "Matplotlib (Offline)"
    };

    const char* test_data_names[] = {
        "Sine Wave",
        "Cosine Wave",
        "Normal Distribution",
        "Exponential Decay",
        "Random Scatter",
        "Linear",
        "Polynomial",
        "Damped Oscillation"
    };

    // Determine backend type
    PlotManager::BackendType backend = (selected_backend_ == 0) ?
        PlotManager::BackendType::ImPlot :
        PlotManager::BackendType::Matplotlib;

    // Generate real data based on selection
    TestDataGenerator::DataSeries data;
    const size_t num_points = 100;

    switch (selected_test_data_) {
        case 0: // Sine Wave
            data = TestDataGenerator::PlotSine(1.0, 1.0, 0.0, 0.0, 2 * M_PI, num_points);
            break;
        case 1: // Cosine Wave
            data = TestDataGenerator::PlotCosine(1.0, 1.0, 0.0, 0.0, 2 * M_PI, num_points);
            break;
        case 2: // Normal Distribution
            data.x.resize(num_points);
            data.y = TestDataGenerator::GenerateNormal(num_points, 0.0, 1.0);
            for (size_t i = 0; i < num_points; ++i) {
                data.x[i] = static_cast<double>(i);
            }
            break;
        case 3: // Exponential Decay
            data = TestDataGenerator::PlotExponential(1.0, -0.5, 0.0, 10.0, num_points);
            break;
        case 4: // Random Scatter
            data.x = TestDataGenerator::GenerateUniform(num_points, 0.0, 10.0);
            data.y = TestDataGenerator::GenerateUniform(num_points, 0.0, 10.0);
            break;
        case 5: // Linear
            data = TestDataGenerator::PlotFunction(
                [](double x) { return 2.0 * x + 1.0; },
                0.0, 10.0, num_points
            );
            break;
        case 6: // Polynomial (x^2 - 2x + 1)
            data = TestDataGenerator::PlotPolynomial({1.0, -2.0, 1.0}, -5.0, 5.0, num_points);
            break;
        case 7: // Damped Oscillation
            data = TestDataGenerator::PlotFunction(
                [](double x) { return std::exp(-0.2 * x) * std::sin(2.0 * x); },
                0.0, 20.0, num_points
            );
            break;
        default:
            data = TestDataGenerator::PlotSine(1.0, 1.0, 0.0, 0.0, 2 * M_PI, num_points);
            break;
    }

    // Create plot using PlotManager
    PlotManager& plot_mgr = PlotManager::GetInstance();

    // Map plot type to PlotManager::PlotType
    PlotManager::PlotType plot_type;
    switch (selected_plot_type_) {
        case 0: plot_type = PlotManager::PlotType::Line; break;
        case 1: plot_type = PlotManager::PlotType::Scatter; break;
        case 2: plot_type = PlotManager::PlotType::Bar; break;
        case 3: plot_type = PlotManager::PlotType::Histogram; break;
        case 4: plot_type = PlotManager::PlotType::BoxPlot; break;
        case 5: plot_type = PlotManager::PlotType::StemLeaf; break;
        case 6: plot_type = PlotManager::PlotType::Line; break;  // Stairs -> Line for now
        case 7: plot_type = PlotManager::PlotType::Bar; break;   // Pie -> Bar for now
        case 8: plot_type = PlotManager::PlotType::Heatmap; break;
        case 9: plot_type = PlotManager::PlotType::Line; break;  // Polar -> Line for now
        case 10: plot_type = PlotManager::PlotType::Line; break; // 3D Surface -> Line for now
        case 11: plot_type = PlotManager::PlotType::Scatter; break; // 3D Scatter
        case 12: plot_type = PlotManager::PlotType::Line; break; // 3D Line
        default: plot_type = PlotManager::PlotType::Line; break;
    }

    // Create plot configuration
    PlotManager::PlotConfig config;
    config.title = std::string(plot_type_names[selected_plot_type_]) + " - " + test_data_names[selected_test_data_];
    config.x_label = "X";
    config.y_label = "Y";
    config.type = plot_type;
    config.backend = backend;
    config.show_legend = true;
    config.show_grid = true;
    config.width = 800;
    config.height = 600;

    // Create plot
    std::string plot_id = plot_mgr.CreatePlot(config);

    // Create dataset and add data
    PlotDataset dataset;
    dataset.AddSeries("test_data");

    auto* series = dataset.GetSeries("test_data");
    if (series) {
        for (size_t i = 0; i < data.x.size(); ++i) {
            series->AddPoint(data.x[i], data.y[i]);
        }
    }

    // Add dataset to plot
    plot_mgr.AddDataset(plot_id, "test_data", dataset);

    // Reuse existing plot window or create new one
    if (!current_plot_window_ || !current_plot_window_->IsVisible()) {
        // Create new plot window to visualize
        PlotWindow::PlotWindowType window_type;
        switch (selected_plot_type_) {
            case 0: window_type = PlotWindow::PlotWindowType::Line2D; break;
            case 1: window_type = PlotWindow::PlotWindowType::Scatter2D; break;
            case 2: window_type = PlotWindow::PlotWindowType::Bar; break;
            case 3: window_type = PlotWindow::PlotWindowType::Histogram; break;
            case 4: window_type = PlotWindow::PlotWindowType::BoxPlot; break;
            case 5: window_type = PlotWindow::PlotWindowType::Stem; break;
            case 6: window_type = PlotWindow::PlotWindowType::Stair; break;
            case 7: window_type = PlotWindow::PlotWindowType::PieChart; break;
            case 8: window_type = PlotWindow::PlotWindowType::Heatmap; break;
            case 9: window_type = PlotWindow::PlotWindowType::Polar; break;
            case 10: window_type = PlotWindow::PlotWindowType::Surface3D; break;
            case 11: window_type = PlotWindow::PlotWindowType::Scatter3D; break;
            case 12: window_type = PlotWindow::PlotWindowType::Line3D; break;
            default: window_type = PlotWindow::PlotWindowType::Line2D; break;
        }

        current_plot_window_ = std::make_shared<PlotWindow>(config.title, window_type, false);
        plot_windows_.push_back(current_plot_window_);
    }

    // Update the plot window with new data
    current_plot_window_->SetPlotId(plot_id);  // Connect the window to the plot we created
    current_plot_window_->SetName(config.title);  // Update window title

    spdlog::info("Generated test plot '{}': Type='{}', Backend='{}', Data='{}', Points={}",
                 plot_id,
                 plot_type_names[selected_plot_type_],
                 backend_names[selected_backend_],
                 test_data_names[selected_test_data_],
                 data.x.size());
}

void PlotTestControlPanel::ClearAllPlots() {
    // Close all plot windows
    for (auto& plot_window : plot_windows_) {
        if (plot_window) {
            plot_window->Hide();
        }
    }

    // Clear the vector and current window reference
    plot_windows_.clear();
    current_plot_window_.reset();

    spdlog::info("Cleared all test plots");
}

} // namespace cyxwiz
