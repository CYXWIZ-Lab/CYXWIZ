#pragma once

#include <imgui.h>
#include <string>
#include <vector>
#include <map>

namespace cyxwiz {

/**
 * Analyzer - Statistical analysis and data profiling for Data Studio
 *
 * Phase 1, Week 1: Core Infrastructure
 *
 * This component provides statistical analysis and data quality profiling
 * for datasets loaded in Data Studio. It integrates with DuckDB for
 * efficient analytical queries on Arrow datasets.
 *
 * Features:
 * - Column statistics (mean, median, std, min, max, quartiles)
 * - Distribution analysis (histograms, density plots)
 * - Correlation matrix
 * - Missing value analysis
 * - Outlier detection
 * - Data quality score
 *
 * Architecture:
 *   Analyzer -> DuckDB -> Arrow Dataset -> Statistics Display
 */
class Analyzer {
public:
    Analyzer();
    ~Analyzer() = default;

    /**
     * Render the analyzer UI
     * Called from DataStudioPanel's Analyze tab
     */
    void Render();

    /**
     * Set the active dataset to analyze
     * Triggers background analysis computation
     */
    void SetActiveDataset(const std::string& dataset_name);

    /**
     * Run analysis on the current dataset
     * This may take time for large datasets (runs in background thread)
     */
    bool RunAnalysis();

    /**
     * Check if analysis is currently running
     */
    bool IsAnalysisRunning() const { return analysis_running_; }

    /**
     * Get progress of current analysis (0.0 to 1.0)
     */
    float GetAnalysisProgress() const { return analysis_progress_; }

private:
    // Analysis state
    std::string current_dataset_;
    bool analysis_running_;
    float analysis_progress_;
    std::string last_error_;

    // Analysis results
    struct ColumnStatistics {
        std::string name;
        std::string data_type;
        size_t non_null_count;
        size_t null_count;
        double mean;
        double std;
        double min;
        double max;
        double q25;  // 25th percentile
        double q50;  // Median
        double q75;  // 75th percentile
        std::vector<int> histogram;  // 20 bins
    };

    struct AnalysisReport {
        size_t total_rows;
        size_t total_columns;
        size_t numeric_columns;
        size_t categorical_columns;
        double missing_percentage;
        double data_quality_score;  // 0-100
        std::vector<ColumnStatistics> column_stats;
        std::map<std::string, std::map<std::string, double>> correlation_matrix;
    };

    AnalysisReport current_report_;

    // UI rendering helpers
    void RenderOverview();
    void RenderColumnStatistics();
    void RenderCorrelationMatrix();
    void RenderMissingValues();
    void RenderOutliers();
    void RenderDistributions();

    // Analysis computation (background thread)
    void ComputeStatistics();
    void ComputeCorrelations();
    void DetectOutliers();
    void CalculateDataQuality();
};

} // namespace cyxwiz
