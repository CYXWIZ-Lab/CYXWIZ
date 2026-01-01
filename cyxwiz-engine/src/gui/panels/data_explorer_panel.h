#pragma once

#include "../panel.h"
#include "../icons.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <cyxwiz/data_loader.h>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <deque>
#include <filesystem>
#include <thread>

namespace cyxwiz {

// Forward declarations for panel integration
class DataTable;
class DescriptiveStatsPanel;
class CorrelationMatrixPanel;
class RegressionPanel;
class OutlierDetectionPanel;
class MissingValuePanel;
class DataProfilerPanel;
class VisualizationPanel;

/**
 * DataExplorerPanel - All-in-one data analysis workspace for data science students
 *
 * Features:
 * - File browser with recent files and project-aware paths
 * - Schema viewer with column types
 * - SQL query editor with DuckDB backend
 * - Results table with pagination and export
 * - Quick Stats tab with descriptive statistics and histograms
 * - Visualize tab with scatter, histogram, bar, box charts
 * - Clean tab for missing values, outliers, type conversion
 * - Hub tab to send data to other analysis panels
 */

// Result area tabs
enum class DataExplorerTab {
    Results,
    QuickStats,
    Visualize,
    Clean,
    Hub
};

// Chart types for visualization
enum class ChartType {
    Histogram,
    Scatter,
    Bar,
    Box
};
class DataExplorerPanel : public Panel {
public:
    DataExplorerPanel();
    ~DataExplorerPanel() override;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_DATABASE; }
    void HandleKeyboardShortcuts() override;

    // External API
    void OpenFile(const std::string& path);
    void ExecuteQuery(const std::string& sql);

    // Panel integration for Hub
    void SetPanelReferences(
        DescriptiveStatsPanel* stats,
        CorrelationMatrixPanel* corr,
        RegressionPanel* reg,
        OutlierDetectionPanel* outlier,
        MissingValuePanel* missing,
        DataProfilerPanel* profiler);

    // Set visualization panel reference
    void SetVisualizationPanel(VisualizationPanel* viz) { visualization_panel_ = viz; }

private:
    // ===== Main Render Methods =====
    void RenderFileBrowserPane();
    void RenderSchemaViewerPane();
    void RenderQueryEditorPane();
    void RenderResultsPane();
    void RenderStatusBar();

    // ===== File Browser =====
    struct FileItem {
        std::string name;
        std::string path;
        bool is_directory = false;
        bool is_expanded = false;
        std::vector<std::unique_ptr<FileItem>> children;
        std::uintmax_t file_size = 0;
    };

    void BuildFileTree(const std::string& root_path);
    void RenderFileNode(FileItem& item, int depth = 0);
    void AddRecentFile(const std::string& path);
    void OpenFileDialog();
    bool IsDataFile(const std::string& path) const;
    const char* GetFileIcon(const std::string& ext) const;

    std::unique_ptr<FileItem> file_tree_root_;
    std::deque<std::string> recent_files_;
    static constexpr size_t MAX_RECENT_FILES = 20;
    std::string current_directory_;
    char path_buffer_[512] = "";

    // ===== Schema Viewer =====
    struct SchemaInfo {
        std::string file_path;
        std::vector<ColumnInfo> columns;
        size_t row_count = 0;
        bool is_loaded = false;
    };

    void LoadSchema(const std::string& path);
    ImVec4 GetTypeColor(const std::string& type) const;

    SchemaInfo current_schema_;
    std::atomic<bool> is_loading_schema_{false};
    std::mutex schema_mutex_;
    std::string schema_error_;

    // ===== SQL Query Editor =====
    struct QueryHistoryItem {
        std::string query;
        std::string timestamp;
        bool success = false;
        double execution_time_ms = 0.0;
    };

    struct ExampleQuery {
        const char* name;
        const char* query;
        const char* description;
    };

    static constexpr size_t QUERY_BUFFER_SIZE = 16384;
    char query_buffer_[QUERY_BUFFER_SIZE] = "";
    std::deque<QueryHistoryItem> query_history_;
    static constexpr size_t MAX_QUERY_HISTORY = 100;
    int history_nav_index_ = -1;

    void RenderQueryToolbar();
    void RenderExamplesPopup();
    void RenderHistoryPopup();
    void InsertFilePath(const std::string& path);
    std::string GetCurrentTimestamp() const;

    bool show_examples_popup_ = false;
    bool show_history_popup_ = false;
    static const std::vector<ExampleQuery> example_queries_;

    // ===== Results Table =====
    struct QueryResult {
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> rows;
        size_t total_rows = 0;
        double execution_time_ms = 0.0;
        bool success = false;
        std::string error_message;
    };

    void ExecuteCurrentQuery();
    void RenderResultsToolbar();
    void RenderResultsTable();
    void RenderPagination();
    void ExportResultsToCSV();
    void CopyResultsToClipboard();

    QueryResult current_result_;
    std::atomic<bool> is_executing_query_{false};
    std::mutex result_mutex_;

    int current_page_ = 0;
    int rows_per_page_ = 100;
    int sort_column_ = -1;
    bool sort_ascending_ = true;
    bool show_row_numbers_ = true;

    // ===== DataLoader Instance =====
    std::unique_ptr<DataLoader> data_loader_;
    bool duckdb_available_ = false;

    // ===== Split Pane State =====
    float left_pane_width_ = 220.0f;
    float schema_pane_height_ = 180.0f;
    float query_pane_height_ = 150.0f;

    // ===== Utility =====
    std::string FormatFileSize(std::uintmax_t bytes) const;
    std::string FormatNumber(size_t num) const;

    // ===== Smart Path (Phase 1) =====
    std::string SmartFormatPath(const std::string& absolute_path) const;
    bool IsInProjectFolder(const std::string& path) const;
    std::string GetProjectDatasetsPath() const;

    // ===== Tab System (Phase 1) =====
    DataExplorerTab current_tab_ = DataExplorerTab::Results;
    void RenderResultsTab();
    void RenderQuickStatsTab();
    void RenderVisualizeTab();
    void RenderCleanTab();
    void RenderHubTab();

    // ===== Quick Stats (Phase 2) =====
    struct QuickStatsCache {
        int column_index = -1;
        DescriptiveStats stats;
        std::vector<double> column_data;
        std::vector<std::pair<std::string, double>> top_correlations;
        bool is_valid = false;
        bool is_computing = false;
    };

    QuickStatsCache quick_stats_cache_;
    int selected_stats_column_ = 0;
    std::unique_ptr<std::thread> stats_thread_;
    std::mutex stats_mutex_;

    void ComputeQuickStats(int column_index);
    void RenderStatsTable();
    void RenderMiniHistogram();
    void RenderTopCorrelations();
    std::vector<double> GetColumnAsDoubles(int col_index) const;

    // ===== Visualize (Phase 3) =====
    ChartType chart_type_ = ChartType::Histogram;
    int viz_x_column_ = 0;
    int viz_y_column_ = 1;
    int viz_color_column_ = -1;

    void RenderChartSelector();
    void RenderHistogramChart();
    void RenderScatterChart();
    void RenderBarChart();
    void RenderBoxChart();

    // ===== Data Cleaning (Phase 5) =====
    struct CleaningPreview {
        std::vector<std::vector<std::string>> rows;
        int affected_count = 0;
        bool is_valid = false;
    };

    int clean_selected_column_ = 0;
    int missing_fill_method_ = 0;  // 0=mean, 1=median, 2=mode, 3=drop, 4=custom
    double missing_custom_value_ = 0.0;
    int outlier_method_ = 0;  // 0=IQR, 1=ZScore, 2=ModifiedZ
    float outlier_threshold_ = 1.5f;
    int outlier_action_ = 0;  // 0=remove, 1=cap, 2=replace
    CleaningPreview cleaning_preview_;

    void AnalyzeDataQuality();
    void RenderMissingValueSection();
    void RenderOutlierSection();
    void RenderTypeConversionSection();
    void PreviewMissingValueFix();
    void ApplyMissingValueFix();
    std::string GenerateCleaningSQL() const;

    // ===== Stats Export =====
    void ExportStatsToCSV();
    void ExportStatsToJSON();

    // ===== Integration Hub (Phase 4) =====
    void RenderHubButton(const char* icon, const char* label, const char* description,
                         std::function<void()> on_click);
    void SendToDescriptiveStats();
    void SendToCorrelationMatrix();
    void SendToRegression();
    void SendToOutlierDetection();
    void SendToMissingValuePanel();
    void SendToDataProfiler();

    // Data quality summary (computed on query)
    int missing_count_ = 0;
    int outlier_count_ = 0;
    std::vector<int> columns_with_missing_;

    // ===== Panel References (owned by MainWindow) =====
    DescriptiveStatsPanel* stats_panel_ = nullptr;
    CorrelationMatrixPanel* correlation_panel_ = nullptr;
    RegressionPanel* regression_panel_ = nullptr;
    OutlierDetectionPanel* outlier_panel_ = nullptr;
    MissingValuePanel* missing_panel_ = nullptr;
    DataProfilerPanel* profiler_panel_ = nullptr;
    VisualizationPanel* visualization_panel_ = nullptr;

    // Helper to convert QueryResult to DataTable for panel integration
    std::shared_ptr<DataTable> ConvertResultToDataTable() const;
};

} // namespace cyxwiz
