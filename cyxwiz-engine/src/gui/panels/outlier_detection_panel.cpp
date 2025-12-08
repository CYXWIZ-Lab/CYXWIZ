#include "outlier_detection_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

OutlierDetectionPanel::OutlierDetectionPanel() = default;

OutlierDetectionPanel::~OutlierDetectionPanel() {
    if (analysis_thread_ && analysis_thread_->joinable()) {
        is_analyzing_ = false;
        analysis_thread_->join();
    }
}

void OutlierDetectionPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_TRIANGLE_EXCLAMATION " Outlier Detection###OutlierDetection", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_analyzing_.load()) {
            RenderLoadingIndicator();
        } else if (has_result_) {
            RenderResults();
        } else if (selected_column_idx_ >= 0) {
            ImGui::Spacing();
            ImGui::TextDisabled("Click 'Detect' to find outliers");
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Select a table and column to detect outliers");
        }
    }
    ImGui::End();
}

void OutlierDetectionPanel::RenderToolbar() {
    // Row 1: Data and column selection
    RenderDataSelector();

    ImGui::SameLine();
    RenderColumnSelector();

    // Row 2: Method selection and parameters
    ImGui::Spacing();
    RenderMethodSelector();

    ImGui::SameLine();

    // Detect button
    bool can_detect = current_table_ && selected_column_idx_ >= 0 && !is_analyzing_.load();
    if (!can_detect) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Detect")) {
        DetectAsync();
    }

    if (!can_detect) ImGui::EndDisabled();
}

void OutlierDetectionPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(150);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                if (name != selected_table_) {
                    selected_table_ = name;
                    auto table = registry.GetTable(name);
                    if (table) {
                        current_table_ = table;
                        // Find numeric columns
                        numeric_columns_.clear();
                        const auto& headers = table->GetHeaders();
                        for (size_t i = 0; i < headers.size(); i++) {
                            auto dtype = DataAnalyzer::DetectColumnType(*table, i);
                            if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                                numeric_columns_.push_back(headers[i]);
                            }
                        }
                        selected_column_idx_ = numeric_columns_.empty() ? -1 : 0;
                        has_result_ = false;
                    }
                }
            }
        }
        ImGui::EndCombo();
    }
}

void OutlierDetectionPanel::RenderColumnSelector() {
    if (numeric_columns_.empty()) {
        ImGui::TextDisabled("No numeric columns");
        return;
    }

    const char* current = selected_column_idx_ >= 0 ?
        numeric_columns_[selected_column_idx_].c_str() : "Select column...";

    ImGui::SetNextItemWidth(150);
    if (ImGui::BeginCombo("##ColumnSelect", current)) {
        for (int i = 0; i < static_cast<int>(numeric_columns_.size()); i++) {
            bool is_selected = (i == selected_column_idx_);
            if (ImGui::Selectable(numeric_columns_[i].c_str(), is_selected)) {
                if (i != selected_column_idx_) {
                    selected_column_idx_ = i;
                    has_result_ = false;
                }
            }
        }
        ImGui::EndCombo();
    }
}

void OutlierDetectionPanel::RenderMethodSelector() {
    ImGui::Text("Method:");
    ImGui::SameLine();

    ImGui::SetNextItemWidth(120);
    const char* method_names[] = {"IQR", "Z-Score", "Modified Z-Score"};
    int method_idx = static_cast<int>(method_);

    if (ImGui::Combo("##Method", &method_idx, method_names, 3)) {
        method_ = static_cast<OutlierMethod>(method_idx);
        has_result_ = false;
    }

    ImGui::SameLine();

    // Parameter based on method
    switch (method_) {
        case OutlierMethod::IQR:
            ImGui::SetNextItemWidth(80);
            if (ImGui::DragFloat("##IQRFactor", &iqr_factor_, 0.1f, 0.5f, 5.0f, "%.1fx")) {
                has_result_ = false;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("IQR multiplier (default: 1.5)");
            }
            break;

        case OutlierMethod::ZScore:
            ImGui::SetNextItemWidth(80);
            if (ImGui::DragFloat("##ZThreshold", &zscore_threshold_, 0.1f, 1.0f, 5.0f, "%.1f")) {
                has_result_ = false;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Z-score threshold (default: 3.0)");
            }
            break;

        case OutlierMethod::ModifiedZScore:
            ImGui::SetNextItemWidth(80);
            if (ImGui::DragFloat("##MZThreshold", &modified_zscore_threshold_, 0.1f, 1.0f, 5.0f, "%.1f")) {
                has_result_ = false;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Modified Z-score threshold (default: 3.5)");
            }
            break;
    }
}

void OutlierDetectionPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Detecting outliers...", ICON_FA_SPINNER);
}

void OutlierDetectionPanel::RenderResults() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    // Summary
    ImGui::Spacing();

    ImVec4 color = result_.outlier_percentage > 10 ?
        ImVec4(1.0f, 0.4f, 0.4f, 1.0f) :
        (result_.outlier_percentage > 5 ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) :
                                           ImVec4(0.4f, 1.0f, 0.4f, 1.0f));

    ImGui::Text("Column: %s", result_.column_name.c_str());
    ImGui::Text("Method: %s", OutlierMethodToString(result_.method));
    ImGui::TextColored(color, "Outliers: %zu (%.1f%% of %zu values)",
                       result_.outlier_count, result_.outlier_percentage, result_.total_valid);

    ImGui::Spacing();

    // Bounds info
    if (ImGui::CollapsingHeader("Detection Bounds", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent();

        if (ImGui::BeginTable("Bounds", 2, ImGuiTableFlags_None)) {
            auto row = [](const char* label, double value) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%s", label);
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", value);
            };

            row("Lower Bound", result_.lower_bound);
            row("Upper Bound", result_.upper_bound);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Separator();
            ImGui::TableNextColumn();
            ImGui::Separator();

            row("Mean", result_.mean);
            row("Median", result_.median);
            row("Std Dev", result_.std_dev);
            row("Q1", result_.q1);
            row("Q3", result_.q3);

            if (result_.method == OutlierMethod::ModifiedZScore) {
                row("MAD", result_.mad);
            }

            ImGui::EndTable();
        }

        ImGui::Unindent();
    }

    ImGui::Spacing();

    // Tabs for scatter plot and table
    if (ImGui::BeginTabBar("OutlierTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Distribution")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Outlier List")) {
            RenderOutlierTable();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void OutlierDetectionPanel::RenderScatterPlot() {
    if (!current_table_ || selected_column_idx_ < 0) return;

    // Get all values for plotting
    const auto& headers = current_table_->GetHeaders();
    size_t col_idx = 0;
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == result_.column_name) {
            col_idx = i;
            break;
        }
    }

    auto all_values = DataAnalyzer::GetNumericValues(*current_table_, col_idx);

    if (all_values.empty()) {
        ImGui::TextDisabled("No data to plot");
        return;
    }

    // Separate normal and outlier values
    std::vector<double> normal_x, normal_y;
    std::vector<double> outlier_x, outlier_y;

    // Create index-based plot
    for (size_t i = 0; i < all_values.size(); i++) {
        double val = all_values[i];
        bool is_outlier = false;

        for (const auto& o : result_.outliers) {
            if (std::abs(o.value - val) < 1e-10) {
                is_outlier = true;
                break;
            }
        }

        if (is_outlier) {
            outlier_x.push_back(static_cast<double>(i));
            outlier_y.push_back(val);
        } else {
            normal_x.push_back(static_cast<double>(i));
            normal_y.push_back(val);
        }
    }

    if (ImPlot::BeginPlot("##OutlierPlot", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Index", result_.column_name.c_str());

        // Plot normal points
        if (!normal_x.empty()) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 3, ImVec4(0.4f, 0.6f, 1.0f, 0.8f));
            ImPlot::PlotScatter("Normal", normal_x.data(), normal_y.data(),
                                static_cast<int>(normal_x.size()));
        }

        // Plot outliers
        if (!outlier_x.empty()) {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 6, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImPlot::PlotScatter("Outliers", outlier_x.data(), outlier_y.data(),
                                static_cast<int>(outlier_x.size()));
        }

        // Draw bounds lines
        double min_x = 0;
        double max_x = static_cast<double>(all_values.size());
        double bounds_x[] = {min_x, max_x};
        double upper_y[] = {result_.upper_bound, result_.upper_bound};
        double lower_y[] = {result_.lower_bound, result_.lower_bound};

        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.5f, 0.0f, 0.8f), 2);
        ImPlot::PlotLine("Upper", bounds_x, upper_y, 2);

        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.5f, 0.0f, 0.8f), 2);
        ImPlot::PlotLine("Lower", bounds_x, lower_y, 2);

        ImPlot::EndPlot();
    }
}

void OutlierDetectionPanel::RenderOutlierTable() {
    if (result_.outliers.empty()) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                          "%s No outliers detected!", ICON_FA_CIRCLE_CHECK);
        return;
    }

    // Pagination
    int total_pages = (static_cast<int>(result_.outliers.size()) + rows_per_page_ - 1) / rows_per_page_;
    page_ = std::clamp(page_, 0, total_pages - 1);

    ImGui::Text("Showing %d-%d of %zu outliers",
                page_ * rows_per_page_ + 1,
                std::min((page_ + 1) * rows_per_page_, static_cast<int>(result_.outliers.size())),
                result_.outliers.size());

    ImGui::SameLine();

    if (ImGui::Button("<<")) page_ = 0;
    ImGui::SameLine();
    if (ImGui::Button("<")) page_ = std::max(0, page_ - 1);
    ImGui::SameLine();
    ImGui::Text("Page %d/%d", page_ + 1, total_pages);
    ImGui::SameLine();
    if (ImGui::Button(">")) page_ = std::min(total_pages - 1, page_ + 1);
    ImGui::SameLine();
    if (ImGui::Button(">>")) page_ = total_pages - 1;

    ImGui::Spacing();

    if (ImGui::BeginTable("OutlierTable", 4,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 250))) {

        ImGui::TableSetupColumn("Row Index", ImGuiTableColumnFlags_DefaultSort, 80);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Direction", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        // Create sorted copy of outliers for display
        std::vector<OutlierEntry> sorted_outliers = result_.outliers;

        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && sort_specs->SpecsCount > 0) {
                sort_column_ = sort_specs->Specs[0].ColumnIndex;
                sort_ascending_ = sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
                sort_specs->SpecsDirty = false;
            }
        }

        std::sort(sorted_outliers.begin(), sorted_outliers.end(),
                  [this](const OutlierEntry& a, const OutlierEntry& b) {
                      bool result = false;
                      switch (sort_column_) {
                          case 0: result = a.row_index < b.row_index; break;
                          case 1: result = a.value < b.value; break;
                          case 2: result = a.score < b.score; break;
                          case 3: result = a.is_low < b.is_low; break;
                      }
                      return sort_ascending_ ? result : !result;
                  });

        // Display current page
        int start = page_ * rows_per_page_;
        int end = std::min(start + rows_per_page_, static_cast<int>(sorted_outliers.size()));

        for (int i = start; i < end; i++) {
            const auto& outlier = sorted_outliers[i];

            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%zu", outlier.row_index);

            ImGui::TableNextColumn();
            ImGui::Text("%.4f", outlier.value);

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", outlier.score);

            ImGui::TableNextColumn();
            if (outlier.is_low) {
                ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "Low");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "High");
            }
        }

        ImGui::EndTable();
    }
}

void OutlierDetectionPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void OutlierDetectionPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;

    current_table_ = table;

    // Find numeric columns
    numeric_columns_.clear();
    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < headers.size(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(headers[i]);
        }
    }

    selected_column_idx_ = numeric_columns_.empty() ? -1 : 0;
    has_result_ = false;
}

void OutlierDetectionPanel::DetectAsync() {
    if (is_analyzing_.load() || selected_column_idx_ < 0 || !current_table_) return;

    if (analysis_thread_ && analysis_thread_->joinable()) {
        analysis_thread_->join();
    }

    is_analyzing_ = true;

    std::string column_name = numeric_columns_[selected_column_idx_];
    OutlierMethod method = method_;
    double param = 0.0;

    switch (method) {
        case OutlierMethod::IQR: param = iqr_factor_; break;
        case OutlierMethod::ZScore: param = zscore_threshold_; break;
        case OutlierMethod::ModifiedZScore: param = modified_zscore_threshold_; break;
    }

    analysis_thread_ = std::make_unique<std::thread>([this, column_name, method, param]() {
        try {
            DataAnalyzer analyzer;
            auto result = analyzer.DetectOutliers(*current_table_, column_name, method, param);

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_ = std::move(result);
                has_result_ = true;
            }

            spdlog::info("Outlier detection complete: {} outliers found", result_.outlier_count);

        } catch (const std::exception& e) {
            spdlog::error("Outlier detection error: {}", e.what());
        }

        is_analyzing_ = false;
    });
}

} // namespace cyxwiz
