#include "plot_manager.h"
#include "backends/implot_backend.h"
#include "backends/matplotlib_backend.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz::plotting {

// ============================================================================
// Singleton Implementation
// ============================================================================

PlotManager& PlotManager::GetInstance() {
    static PlotManager instance;
    return instance;
}

PlotManager::PlotManager() {
    // Initialize Python backend for Matplotlib support
    InitializePythonBackend();
    spdlog::info("PlotManager initialized");
}

PlotManager::~PlotManager() {
    ClearAllPlots();
    if (python_initialized_) {
        ShutdownPythonBackend();
    }
    spdlog::info("PlotManager shutdown");
}

// ============================================================================
// Backend Management
// ============================================================================

void PlotManager::SetDefaultBackend(BackendType backend) {
    if (!IsBackendAvailable(backend)) {
        spdlog::warn("Backend {} not available, keeping current default",
                     static_cast<int>(backend));
        return;
    }
    default_backend_ = backend;
    spdlog::info("Default backend set to {}", static_cast<int>(backend));
}

bool PlotManager::IsBackendAvailable(BackendType backend) const {
    switch (backend) {
        case BackendType::ImPlot:
            return true;  // Always available (compiled in)
        case BackendType::Matplotlib:
            return python_initialized_;
        default:
            return false;
    }
}

std::unique_ptr<PlotBackend> PlotManager::CreateBackend(BackendType type) {
    switch (type) {
        case BackendType::ImPlot:
            return std::make_unique<ImPlotBackend>();
        case BackendType::Matplotlib:
            if (!python_initialized_) {
                spdlog::error("Python backend not initialized");
                return nullptr;
            }
            return std::make_unique<MatplotlibBackend>();
        default:
            spdlog::error("Unknown backend type: {}", static_cast<int>(type));
            return nullptr;
    }
}

// ============================================================================
// Plot Lifecycle
// ============================================================================

std::string PlotManager::CreatePlot(const PlotConfig& config) {
    std::string plot_id = GeneratePlotId();

    PlotInstance instance;
    instance.id = plot_id;
    instance.config = config;
    instance.backend = CreateBackend(config.backend);

    if (!instance.backend) {
        spdlog::error("Failed to create backend for plot {}", plot_id);
        return "";
    }

    instance.backend->Initialize(config.width, config.height);
    plots_[plot_id] = std::move(instance);

    spdlog::info("Created plot '{}' with backend {}", plot_id,
                 static_cast<int>(config.backend));
    return plot_id;
}

bool PlotManager::DeletePlot(const std::string& plot_id) {
    auto it = plots_.find(plot_id);
    if (it == plots_.end()) {
        spdlog::warn("Plot '{}' not found", plot_id);
        return false;
    }

    if (it->second.backend) {
        it->second.backend->Shutdown();
    }

    plots_.erase(it);
    spdlog::info("Deleted plot '{}'", plot_id);
    return true;
}

bool PlotManager::HasPlot(const std::string& plot_id) const {
    return plots_.find(plot_id) != plots_.end();
}

void PlotManager::ClearAllPlots() {
    for (auto& [id, plot] : plots_) {
        if (plot.backend) {
            plot.backend->Shutdown();
        }
    }
    plots_.clear();
    spdlog::info("Cleared all plots");
}

// ============================================================================
// Data Operations
// ============================================================================

bool PlotManager::AddDataset(const std::string& plot_id,
                             const std::string& dataset_name,
                             const PlotDataset& dataset) {
    auto* plot = GetPlot(plot_id);
    if (!plot) {
        spdlog::error("Plot '{}' not found", plot_id);
        return false;
    }

    plot->datasets[dataset_name] = dataset;
    plot->is_dirty = true;
    spdlog::debug("Added dataset '{}' to plot '{}'", dataset_name, plot_id);
    return true;
}

bool PlotManager::RemoveDataset(const std::string& plot_id,
                                const std::string& dataset_name) {
    auto* plot = GetPlot(plot_id);
    if (!plot) {
        return false;
    }

    auto it = plot->datasets.find(dataset_name);
    if (it == plot->datasets.end()) {
        spdlog::warn("Dataset '{}' not found in plot '{}'", dataset_name, plot_id);
        return false;
    }

    plot->datasets.erase(it);
    plot->is_dirty = true;
    return true;
}

PlotDataset* PlotManager::GetDataset(const std::string& plot_id,
                                      const std::string& dataset_name) {
    auto* plot = GetPlot(plot_id);
    if (!plot) {
        return nullptr;
    }

    auto it = plot->datasets.find(dataset_name);
    if (it == plot->datasets.end()) {
        return nullptr;
    }

    return &it->second;
}

// ============================================================================
// Real-time Plotting (ImPlot)
// ============================================================================

void PlotManager::RenderImPlot(const std::string& plot_id) {
    auto* plot = GetPlot(plot_id);
    if (!plot) {
        spdlog::error("Plot '{}' not found", plot_id);
        return;
    }

    if (!plot->backend) {
        spdlog::error("Backend not initialized for plot '{}'", plot_id);
        return;
    }

    // Convert datasets to backend format and render
    plot->backend->BeginPlot(plot->config.title.c_str());

    for (const auto& [name, dataset] : plot->datasets) {
        for (const auto& series : dataset.GetAllSeries()) {
            plot->backend->PlotLine(series.name.c_str(),
                                   series.x_data.data(),
                                   series.y_data.data(),
                                   static_cast<int>(series.x_data.size()));
        }
    }

    plot->backend->EndPlot();
    plot->is_dirty = false;
}

bool PlotManager::UpdateRealtimePlot(const std::string& plot_id, double x, double y,
                                     const std::string& series_name) {
    auto* dataset = GetDataset(plot_id, "realtime");
    if (!dataset) {
        // Create default realtime dataset
        PlotDataset new_dataset;
        new_dataset.AddSeries(series_name);
        AddDataset(plot_id, "realtime", new_dataset);
        dataset = GetDataset(plot_id, "realtime");
    }

    auto* series = dataset->GetSeries(series_name);
    if (!series) {
        dataset->AddSeries(series_name);
        series = dataset->GetSeries(series_name);
    }

    series->AddPoint(x, y);

    auto* plot = GetPlot(plot_id);
    if (plot) {
        plot->is_dirty = true;
    }

    return true;
}

// ============================================================================
// Offline Plotting (Matplotlib)
// ============================================================================

bool PlotManager::SavePlotToFile(const std::string& plot_id,
                                 const std::string& filepath) {
    auto* plot = GetPlot(plot_id);
    if (!plot || !plot->backend) {
        return false;
    }

    // Render the plot data to matplotlib before saving
    plot->backend->BeginPlot(plot->config.title.c_str());

    for (const auto& [name, dataset] : plot->datasets) {
        for (const auto& series : dataset.GetAllSeries()) {
            // Determine which plotting function to use based on plot type
            switch (plot->config.type) {
                case PlotType::Line:
                    plot->backend->PlotLine(series.name.c_str(),
                                           series.x_data.data(),
                                           series.y_data.data(),
                                           static_cast<int>(series.x_data.size()));
                    break;
                case PlotType::Scatter:
                    plot->backend->PlotScatter(series.name.c_str(),
                                              series.x_data.data(),
                                              series.y_data.data(),
                                              static_cast<int>(series.x_data.size()));
                    break;
                case PlotType::Bar:
                    plot->backend->PlotBars(series.name.c_str(),
                                           series.x_data.data(),
                                           series.y_data.data(),
                                           static_cast<int>(series.x_data.size()));
                    break;
                case PlotType::Histogram:
                    plot->backend->PlotHistogram(series.name.c_str(),
                                                series.y_data.data(),
                                                static_cast<int>(series.y_data.size()),
                                                20);  // Default 20 bins
                    break;
                default:
                    // Default to line plot
                    plot->backend->PlotLine(series.name.c_str(),
                                           series.x_data.data(),
                                           series.y_data.data(),
                                           static_cast<int>(series.x_data.size()));
                    break;
            }
        }
    }

    plot->backend->EndPlot();

    // Now save the rendered plot
    return plot->backend->SaveToFile(filepath.c_str());
}

bool PlotManager::ShowPlot(const std::string& plot_id) {
    auto* plot = GetPlot(plot_id);
    if (!plot || !plot->backend) {
        return false;
    }

    // This will display the plot using matplotlib (if using matplotlib backend)
    plot->backend->BeginPlot(plot->config.title.c_str());

    for (const auto& [name, dataset] : plot->datasets) {
        for (const auto& series : dataset.GetAllSeries()) {
            plot->backend->PlotLine(series.name.c_str(),
                                   series.x_data.data(),
                                   series.y_data.data(),
                                   static_cast<int>(series.x_data.size()));
        }
    }

    plot->backend->EndPlot();
    return true;
}

// ============================================================================
// Statistics
// ============================================================================

PlotManager::Statistics PlotManager::CalculateStatistics(
    const std::string& plot_id,
    const std::string& dataset_name) const {

    Statistics stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const auto* plot = GetPlot(plot_id);
    if (!plot) {
        return stats;
    }

    auto it = plot->datasets.find(dataset_name);
    if (it == plot->datasets.end() || it->second.GetAllSeries().empty()) {
        return stats;
    }

    // Collect all y values from all series
    std::vector<double> all_y_values;
    for (const auto& series : it->second.GetAllSeries()) {
        all_y_values.insert(all_y_values.end(),
                           series.y_data.begin(),
                           series.y_data.end());
    }

    if (all_y_values.empty()) {
        return stats;
    }

    // Sort for percentile calculations
    std::sort(all_y_values.begin(), all_y_values.end());

    // Min and max
    stats.min = all_y_values.front();
    stats.max = all_y_values.back();

    // Mean
    double sum = std::accumulate(all_y_values.begin(), all_y_values.end(), 0.0);
    stats.mean = sum / all_y_values.size();

    // Median
    size_t n = all_y_values.size();
    if (n % 2 == 0) {
        stats.median = (all_y_values[n/2 - 1] + all_y_values[n/2]) / 2.0;
    } else {
        stats.median = all_y_values[n/2];
    }

    // Standard deviation
    double sq_sum = 0.0;
    for (double val : all_y_values) {
        sq_sum += (val - stats.mean) * (val - stats.mean);
    }
    stats.std_dev = std::sqrt(sq_sum / all_y_values.size());

    // Quartiles
    stats.q1 = all_y_values[n / 4];
    stats.q3 = all_y_values[3 * n / 4];

    return stats;
}

// ============================================================================
// Plot Configuration
// ============================================================================

bool PlotManager::UpdatePlotConfig(const std::string& plot_id,
                                   const PlotConfig& config) {
    auto* plot = GetPlot(plot_id);
    if (!plot) {
        return false;
    }

    plot->config = config;
    plot->is_dirty = true;
    return true;
}

PlotManager::PlotConfig PlotManager::GetPlotConfig(const std::string& plot_id) const {
    const auto* plot = GetPlot(plot_id);
    if (plot) {
        return plot->config;
    }
    return PlotConfig{};
}

// ============================================================================
// Utility
// ============================================================================

std::vector<std::string> PlotManager::GetAllPlotIds() const {
    std::vector<std::string> ids;
    ids.reserve(plots_.size());
    for (const auto& [id, _] : plots_) {
        ids.push_back(id);
    }
    return ids;
}

// ============================================================================
// Python Integration
// ============================================================================

bool PlotManager::InitializePythonBackend() {
    // Python interpreter is already initialized by PythonEngine in application.cpp
    // We just need to mark it as available for Matplotlib backend
    spdlog::info("Python backend marked as available for Matplotlib");
    python_initialized_ = true;
    return true;
}

void PlotManager::ShutdownPythonBackend() {
    // TODO: Shutdown Python interpreter
    python_initialized_ = false;
}

// ============================================================================
// Internal Helpers
// ============================================================================

PlotManager::PlotInstance* PlotManager::GetPlot(const std::string& plot_id) {
    auto it = plots_.find(plot_id);
    return (it != plots_.end()) ? &it->second : nullptr;
}

const PlotManager::PlotInstance* PlotManager::GetPlot(const std::string& plot_id) const {
    auto it = plots_.find(plot_id);
    return (it != plots_.end()) ? &it->second : nullptr;
}

std::string PlotManager::GeneratePlotId() {
    return "plot_" + std::to_string(next_plot_id_++);
}

} // namespace cyxwiz::plotting
