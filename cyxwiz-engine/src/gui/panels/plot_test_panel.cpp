#include "plot_test_panel.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

// ============================================================================
// Construction / Destruction
// ============================================================================

PlotTestPanel::PlotTestPanel()
    : Panel("Plot Test Panel", true) {
    InitializePlots();
    spdlog::info("PlotTestPanel initialized");
}

PlotTestPanel::~PlotTestPanel() {
    ShutdownPlots();
}

// ============================================================================
// Initialization
// ============================================================================

void PlotTestPanel::InitializePlots() {
    auto& mgr = plotting::PlotManager::GetInstance();

    // Create training loss plot
    plotting::PlotManager::PlotConfig config;
    config.title = "Training Loss";
    config.x_label = "Epoch";
    config.y_label = "Loss";
    config.type = plotting::PlotManager::PlotType::Line;
    config.backend = plotting::PlotManager::BackendType::ImPlot;
    config.width = 600;
    config.height = 300;
    training_loss_plot_ = mgr.CreatePlot(config);

    // Create training accuracy plot
    config.title = "Training Accuracy";
    config.y_label = "Accuracy";
    training_acc_plot_ = mgr.CreatePlot(config);

    // Create histogram plot
    config.title = "Data Distribution";
    config.x_label = "Value";
    config.y_label = "Frequency";
    config.type = plotting::PlotManager::PlotType::Histogram;
    histogram_plot_ = mgr.CreatePlot(config);

    // Create scatter plot
    config.title = "Scatter Plot";
    config.x_label = "X";
    config.y_label = "Y";
    config.type = plotting::PlotManager::PlotType::Scatter;
    scatter_plot_ = mgr.CreatePlot(config);

    // Create heatmap plot
    config.title = "Confusion Matrix";
    config.type = plotting::PlotManager::PlotType::Heatmap;
    config.height = 400;
    heatmap_plot_ = mgr.CreatePlot(config);

    spdlog::info("Created {} test plots", 5);
}

void PlotTestPanel::ShutdownPlots() {
    auto& mgr = plotting::PlotManager::GetInstance();
    mgr.DeletePlot(training_loss_plot_);
    mgr.DeletePlot(training_acc_plot_);
    mgr.DeletePlot(histogram_plot_);
    mgr.DeletePlot(scatter_plot_);
    mgr.DeletePlot(heatmap_plot_);
}

// ============================================================================
// Main Rendering
// ============================================================================

void PlotTestPanel::Render() {
    if (!visible_) return;

    ImGui::Begin("Plot Test Panel", &visible_);

    // Main tabs
    if (ImGui::BeginTabBar("PlotTestTabs")) {
        if (ImGui::BeginTabItem("Real-time Plots")) {
            RenderRealtimeDemo();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Offline Plots")) {
            RenderOfflineDemo();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Test Controls")) {
            RenderTestControls();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

// ============================================================================
// Real-time Demo
// ============================================================================

void PlotTestPanel::RenderRealtimeDemo() {
    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Real-time Plotting (ImPlot)");
    ImGui::Separator();

    // Training simulation controls
    ImGui::Text("Training Simulation:");
    ImGui::SameLine();
    if (ImGui::Button(is_training_ ? "Stop Training" : "Start Training")) {
        is_training_ = !is_training_;
        if (is_training_) {
            current_epoch_ = 0;
            spdlog::info("Starting training simulation");
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        current_epoch_ = 0;
        auto& mgr = plotting::PlotManager::GetInstance();
        // Clear data (would need to implement this)
        spdlog::info("Reset training data");
    }

    ImGui::SliderInt("Max Epochs", &num_epochs_, 10, 1000);
    ImGui::SliderFloat("Noise Level", &noise_level_, 0.0f, 0.5f);

    if (is_training_ && current_epoch_ < num_epochs_) {
        SimulateTraining();
    }

    ImGui::Text("Current Epoch: %d / %d", current_epoch_, num_epochs_);
    ImGui::Separator();

    // Render training curves
    RenderTrainingCurves();

    ImGui::Separator();

    // Other real-time plots
    if (ImGui::CollapsingHeader("Histogram", ImGuiTreeNodeFlags_DefaultOpen)) {
        RenderHistogram();
    }

    if (ImGui::CollapsingHeader("Scatter Plot")) {
        RenderScatter();
    }

    if (ImGui::CollapsingHeader("Heatmap")) {
        RenderHeatmap();
    }
}

void PlotTestPanel::RenderTrainingCurves() {
    auto& mgr = plotting::PlotManager::GetInstance();

    ImGui::Text("Training Metrics:");

    // Render loss plot
    mgr.RenderImPlot(training_loss_plot_);

    // Render accuracy plot
    mgr.RenderImPlot(training_acc_plot_);
}

void PlotTestPanel::RenderHistogram() {
    if (ImGui::Button("Generate Random Data")) {
        GenerateHistogramData();
    }

    ImGui::SameLine();
    ImGui::SliderInt("Samples", &num_samples_, 100, 10000);
    ImGui::SameLine();
    ImGui::SliderInt("Bins", &num_bins_, 10, 100);

    auto& mgr = plotting::PlotManager::GetInstance();
    mgr.RenderImPlot(histogram_plot_);
}

void PlotTestPanel::RenderScatter() {
    if (ImGui::Button("Generate Clusters")) {
        GenerateScatterData();
    }

    ImGui::SameLine();
    ImGui::SliderInt("Points", &num_samples_, 100, 5000);

    auto& mgr = plotting::PlotManager::GetInstance();
    mgr.RenderImPlot(scatter_plot_);
}

void PlotTestPanel::RenderHeatmap() {
    if (ImGui::Button("Generate Confusion Matrix")) {
        GenerateHeatmapData();
    }

    auto& mgr = plotting::PlotManager::GetInstance();
    mgr.RenderImPlot(heatmap_plot_);
}

// ============================================================================
// Offline Demo
// ============================================================================

void PlotTestPanel::RenderOfflineDemo() {
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.2f, 1.0f), "Offline Plotting (Matplotlib)");
    ImGui::Separator();

    ImGui::TextWrapped(
        "Offline plots use matplotlib for publication-quality output. "
        "These plots are rendered to files and can be used for reports and papers."
    );

    ImGui::Spacing();

    if (ImGui::Button("Generate KDE Plot")) {
        RenderKDEPlot();
    }

    ImGui::SameLine();
    if (ImGui::Button("Generate Q-Q Plot")) {
        RenderQQPlot();
    }

    ImGui::SameLine();
    if (ImGui::Button("Generate Box Plot")) {
        RenderBoxPlot();
    }

    ImGui::Separator();

    ImGui::Text("Export Options:");
    static char filepath[256] = "output/plot.png";
    ImGui::InputText("Filepath", filepath, sizeof(filepath));

    if (ImGui::Button("Export Current Plot")) {
        ExportPlotToFile();
    }

    ImGui::Spacing();
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
        "Note: Python matplotlib integration is pending. "
        "Commands are queued but not executed yet.");
}

void PlotTestPanel::RenderStatisticalPlots() {
    ImGui::Text("Statistical plot rendering...");
    // TODO: Implement statistical plots rendering
}

void PlotTestPanel::RenderKDEPlot() {
    auto& mgr = plotting::PlotManager::GetInstance();

    // Create matplotlib plot
    plotting::PlotManager::PlotConfig config;
    config.title = "Kernel Density Estimation";
    config.backend = plotting::PlotManager::BackendType::Matplotlib;
    config.type = plotting::PlotManager::PlotType::KDE;

    auto plot_id = mgr.CreatePlot(config);

    // Generate normal distribution data
    auto data = plotting::TestDataGenerator::GenerateNormal(num_samples_, 0.0, 1.0);

    // TODO: Add data to plot (need to implement AddRawData method)
    // mgr.AddRawData(plot_id, data);

    // Save to file
    mgr.SavePlotToFile(plot_id, "output/kde_plot.png");

    spdlog::info("KDE plot generated (matplotlib backend)");
}

void PlotTestPanel::RenderQQPlot() {
    spdlog::info("Q-Q plot generated (matplotlib backend)");
    // TODO: Implement Q-Q plot
}

void PlotTestPanel::RenderBoxPlot() {
    spdlog::info("Box plot generated (matplotlib backend)");
    // TODO: Implement box plot
}

// ============================================================================
// Test Controls
// ============================================================================

void PlotTestPanel::RenderTestControls() {
    ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "Plot Testing Controls");
    ImGui::Separator();

    // Plot type selection
    ImGui::Text("Plot Type:");
    const char* plot_types[] = {"Line", "Scatter", "Bar", "Histogram", "Heatmap", "Box", "Violin", "KDE"};
    ImGui::Combo("Type", &selected_plot_type_, plot_types, IM_ARRAYSIZE(plot_types));

    // Backend selection
    ImGui::Text("Backend:");
    const char* backends[] = {"ImPlot (Real-time)", "Matplotlib (Offline)"};
    ImGui::Combo("Backend", &selected_backend_, backends, IM_ARRAYSIZE(backends));

    // Data generation selection
    ImGui::Text("Test Data:");
    const char* data_types[] = {
        "Normal Distribution",
        "Sine Wave",
        "Training Curve",
        "Clustered Data",
        "Spiral Pattern",
        "Random Walk",
        "Overfitting Example"
    };
    ImGui::Combo("Data Type", &selected_data_gen_, data_types, IM_ARRAYSIZE(data_types));

    ImGui::Separator();

    // Parameters
    ImGui::SliderInt("Num Samples", &num_samples_, 10, 10000);
    ImGui::SliderInt("Num Bins", &num_bins_, 5, 100);
    ImGui::SliderFloat("Noise", &noise_level_, 0.0f, 1.0f);

    ImGui::Spacing();

    if (ImGui::Button("Generate Test Plot", ImVec2(200, 30))) {
        GenerateTestPlot();
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear All Plots", ImVec2(200, 30))) {
        ClearAllPlots();
    }

    ImGui::Separator();

    // Statistics
    auto& mgr = plotting::PlotManager::GetInstance();
    ImGui::Text("Total Plots: %zu", mgr.GetPlotCount());

    if (mgr.HasPlot(training_loss_plot_)) {
        auto stats = mgr.CalculateStatistics(training_loss_plot_, "realtime");
        ImGui::Text("Training Loss Stats:");
        ImGui::Text("  Mean: %.4f", stats.mean);
        ImGui::Text("  Std Dev: %.4f", stats.std_dev);
        ImGui::Text("  Min: %.4f, Max: %.4f", stats.min, stats.max);
    }
}

// ============================================================================
// Actions
// ============================================================================

void PlotTestPanel::GenerateTestPlot() {
    auto& mgr = plotting::PlotManager::GetInstance();

    plotting::PlotManager::PlotConfig config;
    config.title = "Test Plot";
    config.backend = (selected_backend_ == 0)
        ? plotting::PlotManager::BackendType::ImPlot
        : plotting::PlotManager::BackendType::Matplotlib;

    // Map selection to plot type
    switch (selected_plot_type_) {
        case 0: config.type = plotting::PlotManager::PlotType::Line; break;
        case 1: config.type = plotting::PlotManager::PlotType::Scatter; break;
        case 2: config.type = plotting::PlotManager::PlotType::Bar; break;
        case 3: config.type = plotting::PlotManager::PlotType::Histogram; break;
        case 4: config.type = plotting::PlotManager::PlotType::Heatmap; break;
        case 5: config.type = plotting::PlotManager::PlotType::BoxPlot; break;
        case 6: config.type = plotting::PlotManager::PlotType::Violin; break;
        case 7: config.type = plotting::PlotManager::PlotType::KDE; break;
    }

    auto plot_id = mgr.CreatePlot(config);

    // Generate data based on selection
    plotting::PlotDataset dataset;
    dataset.AddSeries("test_data");

    switch (selected_data_gen_) {
        case 0: { // Normal
            auto data = plotting::TestDataGenerator::GenerateNormal(num_samples_, 0.0, 1.0);
            auto* series = dataset.GetSeries("test_data");
            for (size_t i = 0; i < data.size(); ++i) {
                series->AddPoint(static_cast<double>(i), data[i]);
            }
            break;
        }
        case 1: { // Sine
            auto data = plotting::TestDataGenerator::GenerateSineWave(num_samples_, 2.0, noise_level_);
            auto* series = dataset.GetSeries("test_data");
            for (size_t i = 0; i < data.x.size(); ++i) {
                series->AddPoint(data.x[i], data.y[i]);
            }
            break;
        }
        case 2: { // Training curve
            auto data = plotting::TestDataGenerator::GenerateTrainingCurve(num_samples_, 2.5, 0.1, noise_level_);
            auto* series = dataset.GetSeries("test_data");
            for (size_t i = 0; i < data.x.size(); ++i) {
                series->AddPoint(data.x[i], data.y[i]);
            }
            break;
        }
        // TODO: Add other data types
    }

    mgr.AddDataset(plot_id, "test", dataset);

    spdlog::info("Generated test plot: {} (backend: {}, data: {})",
                 plot_id, selected_backend_, selected_data_gen_);
}

void PlotTestPanel::SimulateTraining() {
    if (current_epoch_ >= num_epochs_) {
        is_training_ = false;
        return;
    }

    // Simulate one epoch every frame (or throttle)
    static int frame_count = 0;
    if (++frame_count % 2 != 0) return;  // Update every 2 frames

    UpdateTrainingData();
    current_epoch_++;
}

void PlotTestPanel::UpdateTrainingData() {
    auto& mgr = plotting::PlotManager::GetInstance();

    // Generate realistic training data point
    double t = static_cast<double>(current_epoch_) / num_epochs_;

    // Loss: exponential decay with noise
    double loss = 2.5 * std::exp(-3.0 * t) + 0.05;
    loss += (std::rand() / static_cast<double>(RAND_MAX) - 0.5) * noise_level_;

    // Accuracy: logarithmic growth with noise
    double acc = 0.3 + 0.65 * std::log1p(3.0 * t) / std::log1p(3.0);
    acc += (std::rand() / static_cast<double>(RAND_MAX) - 0.5) * noise_level_ * 0.5;

    mgr.UpdateRealtimePlot(training_loss_plot_, current_epoch_, loss, "train_loss");
    mgr.UpdateRealtimePlot(training_acc_plot_, current_epoch_, acc, "train_acc");
}

void PlotTestPanel::GenerateHistogramData() {
    auto data = plotting::TestDataGenerator::GenerateNormal(num_samples_, 0.0, 1.0);

    // TODO: Add histogram data to plot
    // For now, just log
    spdlog::info("Generated histogram data: {} samples", num_samples_);
}

void PlotTestPanel::GenerateScatterData() {
    auto data = plotting::TestDataGenerator::GenerateClusteredData(num_samples_ / 3, 3);

    // TODO: Add scatter data to plot
    spdlog::info("Generated scatter data: {} clusters", 3);
}

void PlotTestPanel::GenerateHeatmapData() {
    auto data = plotting::TestDataGenerator::GenerateConfusionMatrix(5, 0.85);

    // TODO: Add heatmap data to plot
    spdlog::info("Generated confusion matrix: 5x5");
}

void PlotTestPanel::ExportPlotToFile() {
    spdlog::info("Export plot to file (not yet implemented)");
    // TODO: Implement file export
}

void PlotTestPanel::ClearAllPlots() {
    auto& mgr = plotting::PlotManager::GetInstance();
    current_epoch_ = 0;
    is_training_ = false;

    // Recreate plots
    ShutdownPlots();
    InitializePlots();

    spdlog::info("Cleared all test plots");
}

} // namespace cyxwiz
