#pragma once

#include "plot_dataset.h"
#include "backends/plot_backend.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz::plotting {

/**
 * PlotManager - Singleton manager for all plotting operations
 * Handles plot lifecycle, backend selection, and rendering coordination
 */
class PlotManager {
public:
    enum class BackendType {
        ImPlot,      // Real-time plotting (integrated into ImGui)
        Matplotlib   // Offline/scripting plotting
    };

    enum class PlotType {
        Line,
        Scatter,
        Histogram,
        BoxPlot,
        Violin,
        KDE,
        QQPlot,
        MosaicPlot,
        StemLeaf,
        DotChart,
        Heatmap,
        Bar
    };

    struct PlotConfig {
        std::string title;
        std::string x_label;
        std::string y_label;
        PlotType type;
        BackendType backend;
        bool auto_fit = true;
        bool show_legend = true;
        bool show_grid = true;
        int width = 800;
        int height = 600;
    };

    // Singleton access
    static PlotManager& GetInstance();

    // Delete copy/move constructors
    PlotManager(const PlotManager&) = delete;
    PlotManager& operator=(const PlotManager&) = delete;
    PlotManager(PlotManager&&) = delete;
    PlotManager& operator=(PlotManager&&) = delete;

    // Backend management
    void SetDefaultBackend(BackendType backend);
    BackendType GetDefaultBackend() const { return default_backend_; }
    bool IsBackendAvailable(BackendType backend) const;

    // Plot lifecycle
    std::string CreatePlot(const PlotConfig& config);
    bool DeletePlot(const std::string& plot_id);
    bool HasPlot(const std::string& plot_id) const;
    void ClearAllPlots();

    // Data operations
    bool AddDataset(const std::string& plot_id, const std::string& dataset_name,
                    const PlotDataset& dataset);
    bool RemoveDataset(const std::string& plot_id, const std::string& dataset_name);
    PlotDataset* GetDataset(const std::string& plot_id, const std::string& dataset_name);

    // Real-time plotting (ImPlot)
    void RenderImPlot(const std::string& plot_id);
    bool UpdateRealtimePlot(const std::string& plot_id, double x, double y,
                           const std::string& series_name = "default");

    // Offline plotting (Matplotlib)
    bool SavePlotToFile(const std::string& plot_id, const std::string& filepath);
    bool ShowPlot(const std::string& plot_id);  // Display using matplotlib

    // Statistics and analysis
    struct Statistics {
        double min;
        double max;
        double mean;
        double median;
        double std_dev;
        double q1;  // First quartile
        double q3;  // Third quartile
    };

    Statistics CalculateStatistics(const std::string& plot_id,
                                   const std::string& dataset_name) const;

    // Plot configuration
    bool UpdatePlotConfig(const std::string& plot_id, const PlotConfig& config);
    PlotConfig GetPlotConfig(const std::string& plot_id) const;

    // Utility
    std::vector<std::string> GetAllPlotIds() const;
    size_t GetPlotCount() const { return plots_.size(); }

    // Python integration (for matplotlib backend)
    bool InitializePythonBackend();
    void ShutdownPythonBackend();

private:
    PlotManager();
    ~PlotManager();

    struct PlotInstance {
        std::string id;
        PlotConfig config;
        std::unordered_map<std::string, PlotDataset> datasets;
        std::unique_ptr<PlotBackend> backend;
        bool is_dirty = true;  // Needs re-rendering
    };

    // Backend factories
    std::unique_ptr<PlotBackend> CreateBackend(BackendType type);

    // Internal helpers
    PlotInstance* GetPlot(const std::string& plot_id);
    const PlotInstance* GetPlot(const std::string& plot_id) const;
    std::string GeneratePlotId();

    // Data members
    std::unordered_map<std::string, PlotInstance> plots_;
    BackendType default_backend_ = BackendType::ImPlot;
    size_t next_plot_id_ = 0;
    bool python_initialized_ = false;
};

} // namespace cyxwiz::plotting
