/**
 * PlotWindow - Static/Offline Plotting using Matplotlib Backend
 *
 * ARCHITECTURE NOTES:
 * - ImPlot Backend: Real-time plotting ONLY (Training Dashboard, Plot Test Panel)
 * - Matplotlib Backend: Static/offline plots (all Plots menu items)
 *
 * This window is used for static plots that are NOT updated in real-time.
 * For real-time streaming data, use the Plot Test Panel or Training Dashboard.
 */

#include "plot_window.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <filesystem>

// stb_image for loading PNG images (implementation is in application.cpp)
#include <stb_image.h>

// OpenGL for texture management - use GLAD for modern OpenGL
#include <glad/glad.h>

namespace cyxwiz {

PlotWindow::PlotWindow(const std::string& title, PlotWindowType type, bool auto_generate)
    : Panel(title, true)
    , type_(type)
    , auto_generated_(auto_generate)
    , show_controls_(true)
    , num_points_(100)
    , noise_level_(0.1f)
    , num_bins_(20)
    , matplotlib_texture_id_(0)
    , matplotlib_image_width_(0)
    , matplotlib_image_height_(0)
{
    // Only initialize plot if we're auto-generating
    // If not auto-generating, the plot_id will be set externally via SetPlotId()
    if (auto_generate) {
        InitializePlot();
        GenerateDefaultData();
    }
}

PlotWindow::~PlotWindow() {
    UnloadMatplotlibTexture();

    if (!plot_id_.empty()) {
        auto& plot_mgr = plotting::PlotManager::GetInstance();
        plot_mgr.DeletePlot(plot_id_);
    }

    // Clean up temporary file
    if (!matplotlib_temp_file_.empty() && std::filesystem::exists(matplotlib_temp_file_)) {
        std::filesystem::remove(matplotlib_temp_file_);
    }
}

void PlotWindow::Render() {
    if (!visible_) return;

    ImGui::Begin(name_.c_str(), &visible_, ImGuiWindowFlags_MenuBar);

    RenderMenuBar();
    RenderControls();
    ImGui::Separator();
    RenderPlot();

    ImGui::End();
}

void PlotWindow::RenderMenuBar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save As...", "Ctrl+S")) {
                SaveToFile("plot_export.png");
            }
            if (ImGui::MenuItem("Export to PNG")) {
                SaveToFile("plot_export.png");
            }
            if (ImGui::MenuItem("Export to SVG")) {
                SaveToFile("plot_export.svg");
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Close", "Ctrl+W")) {
                visible_ = false;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Plot Settings")) {
                show_controls_ = !show_controls_;
            }
            if (ImGui::MenuItem("Copy Data")) {
                spdlog::info("Copy data to clipboard (not implemented yet)");
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Reset Zoom")) {
                spdlog::info("Reset zoom (not implemented yet)");
            }
            if (ImGui::MenuItem("Auto-fit")) {
                spdlog::info("Auto-fit axes (not implemented yet)");
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Show Controls", nullptr, show_controls_)) {
                show_controls_ = !show_controls_;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Insert")) {
            if (ImGui::MenuItem("Add Series")) {
                spdlog::info("Add series (not implemented yet)");
            }
            if (ImGui::MenuItem("Add Annotation")) {
                spdlog::info("Add annotation (not implemented yet)");
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tools")) {
            if (ImGui::MenuItem("Data Statistics")) {
                // Show statistics
                auto& plot_mgr = plotting::PlotManager::GetInstance();
                // auto stats = plot_mgr.CalculateStatistics(plot_id_, "default");
                spdlog::info("Show statistics (not implemented yet)");
            }
            if (ImGui::MenuItem("Fit Curve")) {
                spdlog::info("Curve fitting (not implemented yet)");
            }
            if (ImGui::MenuItem("Regenerate Data")) {
                RegenerateData();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Plot Test Panel")) {
                spdlog::info("Toggle Plot Test Panel (needs main window integration)");
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Plotting Guide")) {
                spdlog::info("Open plotting guide");
            }
            if (ImGui::MenuItem("About")) {
                spdlog::info("About CyxWiz Plotting System");
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void PlotWindow::InitializePlot() {
    auto& plot_mgr = plotting::PlotManager::GetInstance();

    plotting::PlotManager::PlotConfig config;
    config.title = name_;
    config.x_label = "X";
    config.y_label = "Y";
    // Use Matplotlib for static/offline plots (NOT real-time)
    config.backend = plotting::PlotManager::BackendType::Matplotlib;
    config.auto_fit = true;
    config.show_legend = true;
    config.show_grid = true;
    config.width = 800;
    config.height = 600;

    // Set plot type based on window type
    switch (type_) {
        case PlotWindowType::Line2D:
            config.type = plotting::PlotManager::PlotType::Line;
            break;
        case PlotWindowType::Scatter2D:
            config.type = plotting::PlotManager::PlotType::Scatter;
            break;
        case PlotWindowType::Bar:
            config.type = plotting::PlotManager::PlotType::Bar;
            break;
        case PlotWindowType::Histogram:
            config.type = plotting::PlotManager::PlotType::Histogram;
            break;
        case PlotWindowType::BoxPlot:
            config.type = plotting::PlotManager::PlotType::BoxPlot;
            break;
        case PlotWindowType::Heatmap:
            config.type = plotting::PlotManager::PlotType::Heatmap;
            break;
        default:
            config.type = plotting::PlotManager::PlotType::Line;
            break;
    }

    plot_id_ = plot_mgr.CreatePlot(config);
    spdlog::info("Created plot window '{}' with ID: {}", name_, plot_id_);
}

void PlotWindow::RenderControls() {
    if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Plot Type: %s", name_.c_str());

        if (ImGui::Button("Regenerate Data")) {
            RegenerateData();
        }

        ImGui::SameLine();
        if (ImGui::Button("Save to File")) {
            SaveToFile("plot_export.png");
        }

        ImGui::SliderInt("Points", &num_points_, 10, 1000);
        ImGui::SliderFloat("Noise", &noise_level_, 0.0f, 1.0f);

        if (type_ == PlotWindowType::Histogram) {
            ImGui::SliderInt("Bins", &num_bins_, 5, 100);
        }
    }
}

void PlotWindow::RenderPlot() {
    auto& plot_mgr = plotting::PlotManager::GetInstance();

    // Check which backend this plot is using
    auto config = plot_mgr.GetPlotConfig(plot_id_);

    if (config.backend == plotting::PlotManager::BackendType::ImPlot) {
        // Real-time plotting using ImPlot - render directly
        plot_mgr.RenderImPlot(plot_id_);
    } else if (config.backend == plotting::PlotManager::BackendType::Matplotlib) {
        // Offline plotting using Matplotlib - render image if available

        // If we don't have a texture yet, generate the plot and load it
        if (matplotlib_texture_id_ == 0) {
            // Create temporary filename
            matplotlib_temp_file_ = "temp_matplotlib_plot_" + plot_id_ + ".png";

            // Save plot to temporary file
            if (plot_mgr.SavePlotToFile(plot_id_, matplotlib_temp_file_)) {
                // Load the image as an OpenGL texture
                LoadMatplotlibImage(matplotlib_temp_file_);
            }
        }

        // Display the plot image if loaded
        if (matplotlib_texture_id_ != 0) {
            // Calculate display size to fit window while maintaining aspect ratio
            ImVec2 available = ImGui::GetContentRegionAvail();
            float aspect_ratio = static_cast<float>(matplotlib_image_width_) / matplotlib_image_height_;

            ImVec2 display_size;
            if (available.x / aspect_ratio <= available.y) {
                // Width-constrained
                display_size.x = available.x;
                display_size.y = available.x / aspect_ratio;
            } else {
                // Height-constrained
                display_size.y = available.y - 80;  // Leave room for buttons
                display_size.x = display_size.y * aspect_ratio;
            }

            // Center the image
            float offset_x = (available.x - display_size.x) * 0.5f;
            if (offset_x > 0) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
            }

            // Render the texture with proper UV coordinates
            ImGui::Image(
                (ImTextureID)(intptr_t)matplotlib_texture_id_,
                display_size,
                ImVec2(0, 0),  // UV0 - top-left
                ImVec2(1, 1)   // UV1 - bottom-right
            );

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Export buttons
            if (ImGui::Button("Export to PNG", ImVec2(150, 30))) {
                SaveToFile("matplotlib_plot.png");
            }
            ImGui::SameLine();
            if (ImGui::Button("Export to SVG", ImVec2(150, 30))) {
                SaveToFile("matplotlib_plot.svg");
            }
            ImGui::SameLine();
            if (ImGui::Button("Regenerate", ImVec2(150, 30))) {
                // Unload current texture and regenerate
                UnloadMatplotlibTexture();
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Failed to load matplotlib plot image");
            ImGui::Text("Plot ID: %s", plot_id_.c_str());
            ImGui::Text("Temp file: %s", matplotlib_temp_file_.c_str());
        }
    }
}

void PlotWindow::SetDataGenerator(std::function<void()> generator) {
    data_generator_ = generator;
}

void PlotWindow::RegenerateData() {
    if (data_generator_) {
        data_generator_();
    } else {
        GenerateDefaultData();
    }
}

void PlotWindow::SaveToFile(const std::string& filepath) {
    auto& plot_mgr = plotting::PlotManager::GetInstance();
    if (plot_mgr.SavePlotToFile(plot_id_, filepath)) {
        spdlog::info("Plot saved to: {}", filepath);
    } else {
        spdlog::warn("Failed to save plot to: {}", filepath);
    }
}

void PlotWindow::GenerateDefaultData() {
    switch (type_) {
        case PlotWindowType::Line2D:
            GenerateLineData();
            break;
        case PlotWindowType::Scatter2D:
            GenerateScatterData();
            break;
        case PlotWindowType::Bar:
            GenerateBarData();
            break;
        case PlotWindowType::Stem:
            GenerateStemData();
            break;
        case PlotWindowType::Stair:
            GenerateStairData();
            break;
        case PlotWindowType::Histogram:
            GenerateHistogramData();
            break;
        case PlotWindowType::PieChart:
            GeneratePieData();
            break;
        case PlotWindowType::BoxPlot:
            GenerateBoxPlotData();
            break;
        case PlotWindowType::Polar:
            GeneratePolarData();
            break;
        case PlotWindowType::Heatmap:
            GenerateHeatmapData();
            break;
        case PlotWindowType::Surface3D:
            Generate3DSurfaceData();
            break;
        case PlotWindowType::Scatter3D:
            Generate3DScatterData();
            break;
        case PlotWindowType::Line3D:
            Generate3DLineData();
            break;
        case PlotWindowType::Parametric:
            GenerateParametricData();
            break;
    }
}

// ============================================================================
// Type-specific Data Generators
// ============================================================================

void PlotWindow::GenerateLineData() {
    auto data = plotting::TestDataGenerator::PlotSine(1.0, 1.0, 0.0, 0.0, 2 * 3.14159, num_points_);
    plotting::TestDataGenerator::AddNoise(data.y, noise_level_);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("sine_wave");
    auto* series = dataset.GetSeries("sine_wave");
    if (series) {
        series->x_data = data.x;
        series->y_data = data.y;
    }
    plot_mgr.AddDataset(plot_id_, "sine_wave", dataset);
}

void PlotWindow::GenerateScatterData() {
    auto scatter_data = plotting::TestDataGenerator::GenerateClusteredData(num_points_ / 3, 3);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("clusters");
    auto* series = dataset.GetSeries("clusters");
    if (series) {
        series->x_data = scatter_data.x;
        series->y_data = scatter_data.y;
    }
    plot_mgr.AddDataset(plot_id_, "clusters", dataset);
}

void PlotWindow::GenerateBarData() {
    auto cat_data = plotting::TestDataGenerator::GenerateCategoricalData(10, 0.0, 100.0);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    std::vector<double> x_vals;
    for (size_t i = 0; i < cat_data.values.size(); ++i) {
        x_vals.push_back(static_cast<double>(i));
    }
    plotting::PlotDataset dataset;
    dataset.AddSeries("bars");
    auto* series = dataset.GetSeries("bars");
    if (series) {
        series->x_data = x_vals;
        series->y_data = cat_data.values;
    }
    plot_mgr.AddDataset(plot_id_, "bars", dataset);
}

void PlotWindow::GenerateStemData() {
    auto data = plotting::TestDataGenerator::PlotSine(1.0, 2.0, 0.0, 0.0, 3.14159, num_points_);
    plotting::TestDataGenerator::AddNoise(data.y, noise_level_);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("stems");
    auto* series = dataset.GetSeries("stems");
    if (series) {
        series->x_data = data.x;
        series->y_data = data.y;
    }
    plot_mgr.AddDataset(plot_id_, "stems", dataset);
}

void PlotWindow::GenerateStairData() {
    std::vector<double> x, y;
    for (int i = 0; i < num_points_; ++i) {
        double t = static_cast<double>(i) / num_points_ * 10.0;
        x.push_back(t);
        y.push_back(std::floor(std::sin(t) * 5.0));
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("stairs");
    auto* series = dataset.GetSeries("stairs");
    if (series) {
        series->x_data = x;
        series->y_data = y;
    }
    plot_mgr.AddDataset(plot_id_, "stairs", dataset);
}

void PlotWindow::GenerateHistogramData() {
    auto data = plotting::TestDataGenerator::GenerateBimodal(num_points_, -2.0, 1.0, 2.0, 1.0);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    std::vector<double> x_dummy(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        x_dummy[i] = static_cast<double>(i);
    }
    plotting::PlotDataset dataset;
    dataset.AddSeries("histogram");
    auto* series = dataset.GetSeries("histogram");
    if (series) {
        series->x_data = x_dummy;
        series->y_data = data;
    }
    plot_mgr.AddDataset(plot_id_, "histogram", dataset);
}

void PlotWindow::GeneratePieData() {
    auto cat_data = plotting::TestDataGenerator::GenerateCategoricalData(6, 10.0, 100.0);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    std::vector<double> x_vals;
    for (size_t i = 0; i < cat_data.values.size(); ++i) {
        x_vals.push_back(static_cast<double>(i));
    }
    plotting::PlotDataset dataset;
    dataset.AddSeries("pie");
    auto* series = dataset.GetSeries("pie");
    if (series) {
        series->x_data = x_vals;
        series->y_data = cat_data.values;
    }
    plot_mgr.AddDataset(plot_id_, "pie", dataset);
}

void PlotWindow::GenerateBoxPlotData() {
    auto data = plotting::TestDataGenerator::GenerateDataWithOutliers(num_points_, 0.05);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    std::vector<double> x_vals(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        x_vals[i] = static_cast<double>(i);
    }
    plotting::PlotDataset dataset;
    dataset.AddSeries("boxplot");
    auto* series = dataset.GetSeries("boxplot");
    if (series) {
        series->x_data = x_vals;
        series->y_data = data;
    }
    plot_mgr.AddDataset(plot_id_, "boxplot", dataset);
}

void PlotWindow::GeneratePolarData() {
    std::vector<double> theta, r;
    for (int i = 0; i < num_points_; ++i) {
        double t = static_cast<double>(i) / num_points_ * 4 * 3.14159;
        theta.push_back(t);
        r.push_back(1.0 + 0.5 * std::sin(3 * t));
    }

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("polar");
    auto* series = dataset.GetSeries("polar");
    if (series) {
        series->x_data = theta;
        series->y_data = r;
    }
    plot_mgr.AddDataset(plot_id_, "polar", dataset);
}

void PlotWindow::GenerateHeatmapData() {
    int rows = 20, cols = 20;
    auto data = plotting::TestDataGenerator::GenerateHeatmapData(rows, cols, -1.0, 1.0);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    std::vector<double> x_vals, y_vals;
    for (int i = 0; i < rows * cols; ++i) {
        x_vals.push_back(static_cast<double>(i % cols));
        y_vals.push_back(data[i]);
    }
    plotting::PlotDataset dataset;
    dataset.AddSeries("heatmap");
    auto* series = dataset.GetSeries("heatmap");
    if (series) {
        series->x_data = x_vals;
        series->y_data = y_vals;
    }
    plot_mgr.AddDataset(plot_id_, "heatmap", dataset);
}

void PlotWindow::Generate3DSurfaceData() {
    // TODO: Implement when 3D plotting is fully integrated
    spdlog::info("3D Surface plot generation not yet implemented");
}

void PlotWindow::Generate3DScatterData() {
    // TODO: Implement when 3D plotting is fully integrated
    spdlog::info("3D Scatter plot generation not yet implemented");
}

void PlotWindow::Generate3DLineData() {
    // TODO: Implement when 3D plotting is fully integrated
    spdlog::info("3D Line plot generation not yet implemented");
}

void PlotWindow::GenerateParametricData() {
    auto data = plotting::TestDataGenerator::PlotLissajous(1.0, 1.0, 3.0, 2.0, 0.5, num_points_);

    auto& plot_mgr = plotting::PlotManager::GetInstance();
    plotting::PlotDataset dataset;
    dataset.AddSeries("parametric");
    auto* series = dataset.GetSeries("parametric");
    if (series) {
        series->x_data = data.x;
        series->y_data = data.y;
    }
    plot_mgr.AddDataset(plot_id_, "parametric", dataset);
}

// ============================================================================
// Matplotlib Image Loading
// ============================================================================

bool PlotWindow::LoadMatplotlibImage(const std::string& filepath) {
    // Unload previous texture if any
    UnloadMatplotlibTexture();

    // Check if file exists
    if (!std::filesystem::exists(filepath)) {
        spdlog::error("Matplotlib image file not found: {}", filepath);
        return false;
    }

    // Load image using stb_image
    int width, height, channels;
    unsigned char* image_data = stbi_load(filepath.c_str(), &width, &height, &channels, 4);  // Force RGBA

    if (!image_data) {
        spdlog::error("Failed to load matplotlib image: {}", filepath);
        return false;
    }

    matplotlib_image_width_ = width;
    matplotlib_image_height_ = height;

    // Create OpenGL texture
    glGenTextures(1, &matplotlib_texture_id_);
    glBindTexture(GL_TEXTURE_2D, matplotlib_texture_id_);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);

    // Free image data
    stbi_image_free(image_data);

    spdlog::info("Loaded matplotlib plot image: {} ({}x{})", filepath, width, height);
    return true;
}

void PlotWindow::UnloadMatplotlibTexture() {
    if (matplotlib_texture_id_ != 0) {
        glDeleteTextures(1, &matplotlib_texture_id_);
        matplotlib_texture_id_ = 0;
        matplotlib_image_width_ = 0;
        matplotlib_image_height_ = 0;
    }
}

} // namespace cyxwiz
