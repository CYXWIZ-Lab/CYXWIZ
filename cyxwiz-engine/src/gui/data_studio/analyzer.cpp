#include "analyzer.h"
#include "../../core/duckdb_connector.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

Analyzer::Analyzer()
    : analysis_running_(false)
    , analysis_progress_(0.0f)
{
    spdlog::info("[Data Studio] Analyzer initialized");
}

void Analyzer::Render() {
    RenderOverview();
    ImGui::Separator();

    if (ImGui::BeginTabBar("AnalyzerTabs")) {
        if (ImGui::BeginTabItem("Statistics")) {
            RenderColumnStatistics();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Correlations")) {
            RenderCorrelationMatrix();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Quality")) {
            RenderMissingValues();
            ImGui::Spacing();
            RenderOutliers();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Distributions")) {
            RenderDistributions();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void Analyzer::RenderOverview() {
    ImGui::Text("Data Analysis");
    ImGui::Separator();

    // Dataset info
    if (!current_dataset_.empty()) {
        ImGui::Text("Dataset: %s", current_dataset_.c_str());
        ImGui::Text("Rows: %zu | Columns: %zu",
                   current_report_.total_rows,
                   current_report_.total_columns);
        ImGui::Text("Numeric: %zu | Categorical: %zu",
                   current_report_.numeric_columns,
                   current_report_.categorical_columns);

        // Data quality score with color
        float quality = current_report_.data_quality_score;
        ImVec4 color = quality > 80 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
                      quality > 60 ? ImVec4(0.8f, 0.8f, 0.2f, 1.0f) :
                                    ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
        ImGui::TextColored(color, "Data Quality: %.1f%%", quality);
    } else {
        ImGui::TextDisabled("No dataset selected");
    }

    ImGui::Spacing();

    // Analyze button
    if (ImGui::Button("Run Analysis")) {
        RunAnalysis();
    }
    ImGui::SameLine();
    if (ImGui::Button("Refresh")) {
        RunAnalysis();
    }

    // Progress bar
    if (analysis_running_) {
        ImGui::ProgressBar(analysis_progress_, ImVec2(-1, 0));
        ImGui::SameLine();
        ImGui::Text("Analyzing...");
    }

    // Error display
    if (!last_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Error: %s", last_error_.c_str());
    }
}

void Analyzer::RenderColumnStatistics() {
    if (current_report_.column_stats.empty()) {
        ImGui::TextDisabled("No statistics available. Click 'Run Analysis' to compute.");
        return;
    }

    // Table with statistics
    if (ImGui::BeginTable("##stats", 9,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                         ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {

        // Header
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Non-Null");
        ImGui::TableSetupColumn("Mean");
        ImGui::TableSetupColumn("Std");
        ImGui::TableSetupColumn("Min");
        ImGui::TableSetupColumn("Q50");
        ImGui::TableSetupColumn("Max");
        ImGui::TableSetupColumn("Missing %");
        ImGui::TableHeadersRow();

        // Rows
        for (const auto& stat : current_report_.column_stats) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(stat.name.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(stat.data_type.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%zu", stat.non_null_count);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.2f", stat.mean);

            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%.2f", stat.std);

            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.2f", stat.min);

            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.2f", stat.q50);

            ImGui::TableSetColumnIndex(7);
            ImGui::Text("%.2f", stat.max);

            ImGui::TableSetColumnIndex(8);
            double missing_pct = 100.0 * stat.null_count /
                               (stat.null_count + stat.non_null_count);
            ImGui::Text("%.1f%%", missing_pct);
        }

        ImGui::EndTable();
    }
}

void Analyzer::RenderCorrelationMatrix() {
    if (current_report_.correlation_matrix.empty()) {
        ImGui::TextDisabled("No correlation data. Run analysis first.");
        return;
    }

    // TODO: Phase 1 Week 2 - Implement correlation heatmap
    ImGui::Text("Correlation Matrix");
    ImGui::Separator();
    ImGui::TextDisabled("Heatmap visualization coming in Week 2");
}

void Analyzer::RenderMissingValues() {
    ImGui::Text("Missing Values Analysis");
    ImGui::Separator();

    if (current_report_.column_stats.empty()) {
        ImGui::TextDisabled("No data available");
        return;
    }

    // Overall missing percentage
    ImGui::Text("Overall Missing: %.2f%%", current_report_.missing_percentage);
    ImGui::Spacing();

    // Per-column missing values
    if (ImGui::BeginTable("##missing", 3,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {

        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Missing Count");
        ImGui::TableSetupColumn("Missing %");
        ImGui::TableHeadersRow();

        for (const auto& stat : current_report_.column_stats) {
            if (stat.null_count > 0) {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(stat.name.c_str());

                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%zu", stat.null_count);

                ImGui::TableSetColumnIndex(2);
                double pct = 100.0 * stat.null_count /
                           (stat.null_count + stat.non_null_count);
                ImGui::Text("%.1f%%", pct);
            }
        }

        ImGui::EndTable();
    }
}

void Analyzer::RenderOutliers() {
    ImGui::Text("Outlier Detection");
    ImGui::Separator();

    // TODO: Phase 1 Week 2 - Implement outlier detection using IQR method
    ImGui::TextDisabled("Outlier analysis coming in Week 2");
}

void Analyzer::RenderDistributions() {
    if (current_report_.column_stats.empty()) {
        ImGui::TextDisabled("No distribution data. Run analysis first.");
        return;
    }

    // TODO: Phase 1 Week 2 - Implement histogram plots using ImPlot
    ImGui::Text("Distribution Histograms");
    ImGui::Separator();
    ImGui::TextDisabled("Histogram visualization coming in Week 2");
}

void Analyzer::SetActiveDataset(const std::string& dataset_name) {
    current_dataset_ = dataset_name;
    spdlog::info("[Data Studio] Analyzer set active dataset: {}", dataset_name);

    // Auto-run analysis on dataset change
    RunAnalysis();
}

bool Analyzer::RunAnalysis() {
    if (current_dataset_.empty()) {
        last_error_ = "No dataset selected";
        return false;
    }

    spdlog::info("[Data Studio] Running analysis on: {}", current_dataset_);

    // TODO: Phase 1 Week 2 - Run actual analysis via DuckDB
    // For now, generate placeholder results

    analysis_running_ = true;
    analysis_progress_ = 0.0f;

    // Simulate analysis
    current_report_.total_rows = 1000;
    current_report_.total_columns = 5;
    current_report_.numeric_columns = 3;
    current_report_.categorical_columns = 2;
    current_report_.missing_percentage = 2.5;
    current_report_.data_quality_score = 87.5;

    // Placeholder column statistics
    current_report_.column_stats.clear();
    for (int i = 0; i < 5; i++) {
        ColumnStatistics stat;
        stat.name = "Column" + std::to_string(i);
        stat.data_type = (i < 3) ? "DOUBLE" : "VARCHAR";
        stat.non_null_count = 975;
        stat.null_count = 25;
        stat.mean = 50.0 + i * 10.0;
        stat.std = 10.0;
        stat.min = 0.0;
        stat.max = 100.0;
        stat.q25 = 25.0;
        stat.q50 = 50.0;
        stat.q75 = 75.0;
        current_report_.column_stats.push_back(stat);
    }

    analysis_running_ = false;
    analysis_progress_ = 1.0f;
    last_error_ = "";

    return true;
}

void Analyzer::ComputeStatistics() {
    // TODO: Phase 1 Week 2 - Implement via DuckDB SQL queries
}

void Analyzer::ComputeCorrelations() {
    // TODO: Phase 1 Week 2 - Compute correlation matrix
}

void Analyzer::DetectOutliers() {
    // TODO: Phase 1 Week 2 - IQR-based outlier detection
}

void Analyzer::CalculateDataQuality() {
    // TODO: Phase 1 Week 2 - Composite quality score
}

} // namespace cyxwiz
