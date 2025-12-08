#include "data_profiler_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>

namespace cyxwiz {

DataProfilerPanel::DataProfilerPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
    std::memset(column_filter_, 0, sizeof(column_filter_));
}

DataProfilerPanel::~DataProfilerPanel() {
    if (analysis_thread_ && analysis_thread_->joinable()) {
        is_analyzing_ = false;
        analysis_thread_->join();
    }
}

void DataProfilerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_BAR " Data Profiler###DataProfiler", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_analyzing_.load()) {
            RenderLoadingIndicator();
        } else if (has_profile_) {
            // Tab bar for Summary and Columns
            if (ImGui::BeginTabBar("ProfilerTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_LIST " Summary")) {
                    current_tab_ = 0;
                    RenderSummaryTab();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_TABLE_COLUMNS " Columns")) {
                    current_tab_ = 1;
                    RenderColumnsTab();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            // No data selected
            ImGui::Spacing();
            ImGui::TextDisabled("Select a data table to profile");
            ImGui::Spacing();
            ImGui::TextWrapped("Use the dropdown above to select a loaded table, "
                              "or load a CSV/Excel file in the Table Viewer first.");
        }
    }
    ImGui::End();

    // Export dialog
    RenderExportOptions();
}

void DataProfilerPanel::RenderToolbar() {
    // Data selector
    RenderDataSelector();

    ImGui::SameLine();

    // Refresh button
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh")) {
        if (current_table_) {
            StartProfileAsync(current_table_);
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Re-analyze the current table");
    }

    ImGui::SameLine();

    // Export button
    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        show_export_dialog_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Export profile to CSV or JSON");
    }

    ImGui::SameLine();

    // Settings
    if (ImGui::Button(ICON_FA_GEAR)) {
        ImGui::OpenPopup("ProfilerSettings");
    }

    if (ImGui::BeginPopup("ProfilerSettings")) {
        ImGui::Text("Settings");
        ImGui::Separator();

        ImGui::SliderInt("Histogram Bins", &histogram_bins_, 5, 50);
        ImGui::SliderInt("Top N Values", &top_n_values_, 5, 50);
        ImGui::Checkbox("Auto-refresh on change", &auto_refresh_);

        ImGui::EndPopup();
    }
}

void DataProfilerPanel::RenderDataSelector() {
    // Get available tables
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(250);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select a table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                auto table = registry.GetTable(name);
                if (table) {
                    AnalyzeTable(table);
                }
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void DataProfilerPanel::RenderLoadingIndicator() {
    ImGui::Spacing();

    float progress = analysis_progress_.load();
    ImGui::ProgressBar(progress, ImVec2(-1, 0), analysis_status_.c_str());

    ImGui::Spacing();
    ImGui::TextDisabled("Analyzing %zu rows and %zu columns...",
                        current_table_ ? current_table_->GetRowCount() : 0,
                        current_table_ ? current_table_->GetColumnCount() : 0);
}

void DataProfilerPanel::RenderSummaryTab() {
    std::lock_guard<std::mutex> lock(profile_mutex_);

    ImGui::Spacing();

    // Overview section
    if (ImGui::CollapsingHeader(ICON_FA_CIRCLE_INFO " Overview", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        ImGui::Text("Source: %s", profile_.source_name.c_str());
        ImGui::Text("Rows: %zu", profile_.row_count);
        ImGui::Text("Columns: %zu", profile_.column_count);
        ImGui::Text("Total Cells: %zu", profile_.row_count * profile_.column_count);

        ImGui::Spacing();

        // Memory estimate
        float mem_mb = static_cast<float>(profile_.memory_estimate) / (1024 * 1024);
        ImGui::Text("Est. Memory: %.2f MB", mem_mb);

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Missing values summary
    if (ImGui::CollapsingHeader(ICON_FA_CIRCLE_EXCLAMATION " Missing Values", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        ImGui::Text("Total Missing: %zu (%.1f%%)", profile_.total_nulls, profile_.null_percentage);

        // Show columns with missing values
        std::vector<const ColumnProfile*> cols_with_nulls;
        for (const auto& col : profile_.columns) {
            if (col.null_count > 0) {
                cols_with_nulls.push_back(&col);
            }
        }

        if (cols_with_nulls.empty()) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "No missing values!");
        } else {
            ImGui::Text("Columns with missing:");
            for (const auto* col : cols_with_nulls) {
                ImGui::BulletText("%s: %zu (%.1f%%)",
                                  col->name.c_str(), col->null_count, col->null_percentage);
            }
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Data types summary
    if (ImGui::CollapsingHeader(ICON_FA_DATABASE " Data Types", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        int numeric_count = 0, integer_count = 0, categorical_count = 0, other_count = 0;
        for (const auto& col : profile_.columns) {
            switch (col.dtype) {
                case ColumnDataType::Numeric: numeric_count++; break;
                case ColumnDataType::Integer: integer_count++; break;
                case ColumnDataType::Categorical: categorical_count++; break;
                default: other_count++; break;
            }
        }

        ImGui::Text("Numeric: %d", numeric_count);
        ImGui::Text("Integer: %d", integer_count);
        ImGui::Text("Categorical: %d", categorical_count);
        if (other_count > 0) {
            ImGui::Text("Other: %d", other_count);
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Quick stats table
    if (ImGui::CollapsingHeader(ICON_FA_TABLE " Quick Statistics")) {
        ImGui::Indent();

        if (ImGui::BeginTable("QuickStats", 6, ImGuiTableFlags_Borders |
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollY, ImVec2(0, 200))) {
            ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthFixed, 120);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Non-Null", ImGuiTableColumnFlags_WidthFixed, 70);
            ImGui::TableSetupColumn("Unique", ImGuiTableColumnFlags_WidthFixed, 70);
            ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableSetupColumn("Std Dev", ImGuiTableColumnFlags_WidthFixed, 80);
            ImGui::TableHeadersRow();

            for (const auto& col : profile_.columns) {
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%s", col.name.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%s", col.GetDTypeString().c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%zu", col.non_null_count);

                ImGui::TableNextColumn();
                ImGui::Text("%zu", col.unique_count);

                ImGui::TableNextColumn();
                if (col.IsNumeric()) {
                    ImGui::Text("%.3f", col.mean);
                } else {
                    ImGui::TextDisabled("-");
                }

                ImGui::TableNextColumn();
                if (col.IsNumeric()) {
                    ImGui::Text("%.3f", col.std_dev);
                } else {
                    ImGui::TextDisabled("-");
                }
            }

            ImGui::EndTable();
        }

        ImGui::Unindent();
    }
}

void DataProfilerPanel::RenderColumnsTab() {
    std::lock_guard<std::mutex> lock(profile_mutex_);

    // Column filter
    ImGui::SetNextItemWidth(200);
    ImGui::InputTextWithHint("##ColumnFilter", "Filter columns...", column_filter_, sizeof(column_filter_));

    ImGui::SameLine();
    ImGui::TextDisabled("(%zu columns)", profile_.columns.size());

    ImGui::Separator();

    // Two-pane layout: column list on left, details on right
    ImGui::BeginChild("ColumnList", ImVec2(200, 0), true);

    std::string filter_lower = column_filter_;
    std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

    for (size_t i = 0; i < profile_.columns.size(); i++) {
        const auto& col = profile_.columns[i];

        // Apply filter
        if (!filter_lower.empty()) {
            std::string name_lower = col.name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
            if (name_lower.find(filter_lower) == std::string::npos) {
                continue;
            }
        }

        // Icon based on type
        const char* icon = ICON_FA_QUESTION;
        switch (col.dtype) {
            case ColumnDataType::Numeric: icon = ICON_FA_HASHTAG; break;
            case ColumnDataType::Integer: icon = ICON_FA_HASHTAG; break;
            case ColumnDataType::Categorical: icon = ICON_FA_FONT; break;
            case ColumnDataType::Boolean: icon = ICON_FA_TOGGLE_ON; break;
            default: break;
        }

        std::string label = std::string(icon) + " " + col.name;

        if (ImGui::Selectable(label.c_str(), selected_column_ == static_cast<int>(i))) {
            selected_column_ = static_cast<int>(i);
        }

        // Show null indicator
        if (col.null_count > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%zu null)", col.null_count);
        }
    }

    ImGui::EndChild();

    ImGui::SameLine();

    // Column details pane
    ImGui::BeginChild("ColumnDetails", ImVec2(0, 0), true);

    if (selected_column_ >= 0 && selected_column_ < static_cast<int>(profile_.columns.size())) {
        RenderColumnDetail(profile_.columns[selected_column_]);
    } else {
        ImGui::TextDisabled("Select a column to view details");
    }

    ImGui::EndChild();
}

void DataProfilerPanel::RenderColumnDetail(const ColumnProfile& profile) {
    // Header
    ImGui::Text("%s", profile.name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", profile.GetDTypeString().c_str());

    ImGui::Separator();
    ImGui::Spacing();

    // Basic stats
    if (ImGui::CollapsingHeader("Basic Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        ImGui::Text("Total Values: %zu", profile.total_count);
        ImGui::Text("Non-Null: %zu (%.1f%%)", profile.non_null_count,
                    profile.total_count > 0 ? 100.0f - profile.null_percentage : 0.0f);
        ImGui::Text("Null: %zu (%.1f%%)", profile.null_count, profile.null_percentage);
        ImGui::Text("Unique Values: %zu", profile.unique_count);

        // Cardinality ratio
        float cardinality = profile.non_null_count > 0 ?
            static_cast<float>(profile.unique_count) / profile.non_null_count * 100.0f : 0.0f;
        ImGui::Text("Cardinality: %.1f%%", cardinality);

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Numeric statistics
    if (profile.IsNumeric()) {
        if (ImGui::CollapsingHeader("Numeric Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();

            // Use a table for nice alignment
            if (ImGui::BeginTable("NumericStats", 2, ImGuiTableFlags_None)) {
                auto row = [](const char* label, double value, const char* fmt = "%.4f") {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", label);
                    ImGui::TableNextColumn();
                    ImGui::Text(fmt, value);
                };

                row("Min", profile.min);
                row("Max", profile.max);
                row("Mean", profile.mean);
                row("Median", profile.median);
                row("Std Dev", profile.std_dev);
                row("Variance", profile.variance);
                row("Sum", profile.sum, "%.2f");

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Separator();
                ImGui::TableNextColumn();
                ImGui::Separator();

                row("Q1 (25%)", profile.q1);
                row("Q3 (75%)", profile.q3);
                row("IQR", profile.iqr);
                row("Skewness", profile.skewness);
                row("Kurtosis", profile.kurtosis);

                ImGui::EndTable();
            }

            ImGui::Unindent();
        }

        ImGui::Spacing();

        // Histogram
        RenderHistogram(profile);
    }

    // Categorical statistics
    if (profile.dtype == ColumnDataType::Categorical && !profile.top_values.empty()) {
        RenderTopValues(profile);
    }
}

void DataProfilerPanel::RenderHistogram(const ColumnProfile& profile) {
    if (profile.histogram.empty()) return;

    if (ImGui::CollapsingHeader("Distribution", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        // Prepare data for ImPlot
        std::vector<double> bin_centers;
        std::vector<double> counts;

        for (const auto& bin : profile.histogram) {
            bin_centers.push_back((bin.low + bin.high) / 2.0);
            counts.push_back(static_cast<double>(bin.count));
        }

        double bar_width = profile.histogram.size() > 1 ?
            (profile.histogram[1].low - profile.histogram[0].low) * 0.9 : 1.0;

        if (ImPlot::BeginPlot("##Histogram", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Value", "Count");
            ImPlot::PlotBars("##bars", bin_centers.data(), counts.data(),
                             static_cast<int>(bin_centers.size()), bar_width);
            ImPlot::EndPlot();
        }

        ImGui::Unindent();
    }
}

void DataProfilerPanel::RenderTopValues(const ColumnProfile& profile) {
    if (ImGui::CollapsingHeader("Top Values", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::BeginTable("TopValues", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70);
            ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthFixed, 60);
            ImGui::TableHeadersRow();

            for (const auto& tv : profile.top_values) {
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%s", tv.value.c_str());

                ImGui::TableNextColumn();
                ImGui::Text("%zu", tv.count);

                ImGui::TableNextColumn();
                ImGui::Text("%.1f%%", tv.percentage);
            }

            ImGui::EndTable();
        }

        ImGui::Unindent();
    }
}

void DataProfilerPanel::RenderExportOptions() {
    if (!show_export_dialog_) return;

    ImGui::OpenPopup("Export Profile");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Export Profile", &show_export_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Export Format:");
        ImGui::RadioButton("CSV", &export_format_, 0);
        ImGui::SameLine();
        ImGui::RadioButton("JSON", &export_format_, 1);

        ImGui::Spacing();

        ImGui::Text("File Path:");
        ImGui::InputText("##ExportPath", export_path_, sizeof(export_path_));

        ImGui::Spacing();

        if (ImGui::Button("Export", ImVec2(120, 0))) {
            if (export_format_ == 0) {
                ExportToCSV(export_path_);
            } else {
                ExportToJSON(export_path_);
            }
            show_export_dialog_ = false;
        }

        ImGui::SameLine();

        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_export_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void DataProfilerPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void DataProfilerPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;

    current_table_ = table;
    StartProfileAsync(table);
}

void DataProfilerPanel::StartProfileAsync(std::shared_ptr<DataTable> table) {
    if (is_analyzing_.load()) {
        spdlog::warn("Analysis already in progress");
        return;
    }

    // Wait for any previous thread
    if (analysis_thread_ && analysis_thread_->joinable()) {
        analysis_thread_->join();
    }

    is_analyzing_ = true;
    analysis_progress_ = 0.0f;
    analysis_status_ = "Starting analysis...";

    analysis_thread_ = std::make_unique<std::thread>([this, table]() {
        try {
            analysis_status_ = "Profiling columns...";

            DataAnalyzer analyzer;
            auto new_profile = analyzer.ProfileTable(*table, histogram_bins_, top_n_values_);

            analysis_progress_ = 1.0f;
            analysis_status_ = "Complete!";

            {
                std::lock_guard<std::mutex> lock(profile_mutex_);
                profile_ = std::move(new_profile);
                has_profile_ = true;
            }

            spdlog::info("Profile complete: {} rows, {} columns",
                         profile_.row_count, profile_.column_count);

        } catch (const std::exception& e) {
            spdlog::error("Profile error: {}", e.what());
            analysis_status_ = std::string("Error: ") + e.what();
        }

        is_analyzing_ = false;
    });
}

void DataProfilerPanel::ExportToCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(profile_mutex_);

    std::ofstream file(filepath);
    if (!file) {
        spdlog::error("Failed to open file for export: {}", filepath);
        return;
    }

    // Header
    file << "Column,Type,Total,NonNull,Null,NullPct,Unique,Mean,Median,StdDev,Min,Max,Q1,Q3\n";

    // Data
    for (const auto& col : profile_.columns) {
        file << "\"" << col.name << "\","
             << col.GetDTypeString() << ","
             << col.total_count << ","
             << col.non_null_count << ","
             << col.null_count << ","
             << col.null_percentage << ","
             << col.unique_count << ",";

        if (col.IsNumeric()) {
            file << col.mean << "," << col.median << "," << col.std_dev << ","
                 << col.min << "," << col.max << "," << col.q1 << "," << col.q3;
        } else {
            file << ",,,,,,";
        }

        file << "\n";
    }

    spdlog::info("Exported profile to CSV: {}", filepath);
}

void DataProfilerPanel::ExportToJSON(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(profile_mutex_);

    nlohmann::json j;

    j["source"] = profile_.source_name;
    j["rows"] = profile_.row_count;
    j["columns_count"] = profile_.column_count;
    j["total_nulls"] = profile_.total_nulls;
    j["null_percentage"] = profile_.null_percentage;

    for (const auto& col : profile_.columns) {
        nlohmann::json jcol;
        jcol["name"] = col.name;
        jcol["dtype"] = col.GetDTypeString();
        jcol["total"] = col.total_count;
        jcol["non_null"] = col.non_null_count;
        jcol["null"] = col.null_count;
        jcol["null_pct"] = col.null_percentage;
        jcol["unique"] = col.unique_count;

        if (col.IsNumeric()) {
            jcol["mean"] = col.mean;
            jcol["median"] = col.median;
            jcol["std_dev"] = col.std_dev;
            jcol["min"] = col.min;
            jcol["max"] = col.max;
            jcol["q1"] = col.q1;
            jcol["q3"] = col.q3;
            jcol["iqr"] = col.iqr;
            jcol["skewness"] = col.skewness;
            jcol["kurtosis"] = col.kurtosis;
        }

        if (!col.top_values.empty()) {
            nlohmann::json jtop = nlohmann::json::array();
            for (const auto& tv : col.top_values) {
                jtop.push_back({{"value", tv.value}, {"count", tv.count}, {"pct", tv.percentage}});
            }
            jcol["top_values"] = jtop;
        }

        j["columns"].push_back(jcol);
    }

    std::ofstream file(filepath);
    if (!file) {
        spdlog::error("Failed to open file for export: {}", filepath);
        return;
    }

    file << j.dump(2);
    spdlog::info("Exported profile to JSON: {}", filepath);
}

} // namespace cyxwiz
