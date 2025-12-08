#include "missing_value_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

MissingValuePanel::MissingValuePanel() = default;

MissingValuePanel::~MissingValuePanel() {
    if (analysis_thread_ && analysis_thread_->joinable()) {
        is_analyzing_ = false;
        analysis_thread_->join();
    }
}

void MissingValuePanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CIRCLE_EXCLAMATION " Missing Values###MissingValues", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_analyzing_.load()) {
            RenderLoadingIndicator();
        } else if (has_analysis_) {
            RenderSummary();
            ImGui::Spacing();

            if (ImGui::BeginTabBar("MissingTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_TABLE " By Column")) {
                    RenderColumnTable();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Pattern")) {
                    RenderMissingPattern();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_LIGHTBULB " Suggestions")) {
                    RenderImputationSuggestions();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Spacing();
            ImGui::TextDisabled("Select a data table to analyze missing values");
        }
    }
    ImGui::End();
}

void MissingValuePanel::RenderToolbar() {
    RenderDataSelector();

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Analyze")) {
        if (current_table_) {
            AnalyzeAsync(current_table_);
        }
    }
}

void MissingValuePanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                auto table = registry.GetTable(name);
                if (table) {
                    AnalyzeTable(table);
                }
            }
        }
        ImGui::EndCombo();
    }
}

void MissingValuePanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Analyzing missing values...", ICON_FA_SPINNER);
}

void MissingValuePanel::RenderSummary() {
    std::lock_guard<std::mutex> lock(analysis_mutex_);

    // Summary cards
    float card_width = (ImGui::GetContentRegionAvail().x - 30) / 4;

    ImGui::BeginGroup();

    // Total cells
    ImGui::BeginChild("TotalCells", ImVec2(card_width, 60), true);
    ImGui::Text("Total Cells");
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "%zu", analysis_.total_cells);
    ImGui::EndChild();

    ImGui::SameLine();

    // Total missing
    ImGui::BeginChild("TotalMissing", ImVec2(card_width, 60), true);
    ImGui::Text("Missing Cells");
    ImVec4 color = analysis_.missing_percentage > 20 ?
        ImVec4(1.0f, 0.4f, 0.4f, 1.0f) :
        (analysis_.missing_percentage > 5 ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) :
                                             ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
    ImGui::TextColored(color, "%zu (%.1f%%)", analysis_.total_missing, analysis_.missing_percentage);
    ImGui::EndChild();

    ImGui::SameLine();

    // Complete rows
    ImGui::BeginChild("CompleteRows", ImVec2(card_width, 60), true);
    ImGui::Text("Complete Rows");
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%zu", analysis_.complete_rows);
    ImGui::EndChild();

    ImGui::SameLine();

    // Rows with missing
    ImGui::BeginChild("IncompleteRows", ImVec2(card_width, 60), true);
    ImGui::Text("Incomplete Rows");
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.4f, 1.0f), "%zu (%.1f%%)",
                       analysis_.rows_with_missing, analysis_.rows_with_missing_percentage);
    ImGui::EndChild();

    ImGui::EndGroup();
}

void MissingValuePanel::RenderColumnTable() {
    std::lock_guard<std::mutex> lock(analysis_mutex_);

    if (ImGui::BeginTable("MissingByColumn", 4,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable |
                          ImGuiTableFlags_ScrollY, ImVec2(0, 300))) {

        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_DefaultSort, 150);
        ImGui::TableSetupColumn("Missing", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        // Sort data if needed
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && sort_specs->SpecsCount > 0) {
                sort_column_ = sort_specs->Specs[0].ColumnIndex;
                sort_ascending_ = sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
                sort_specs->SpecsDirty = false;
            }
        }

        // Create sorted indices
        std::vector<size_t> indices(analysis_.columns.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;

        if (sort_column_ >= 0) {
            std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
                const auto& ca = analysis_.columns[a];
                const auto& cb = analysis_.columns[b];

                bool result = false;
                switch (sort_column_) {
                    case 0: result = ca.name < cb.name; break;
                    case 1: result = ca.missing_count < cb.missing_count; break;
                    case 2: result = ca.missing_percentage < cb.missing_percentage; break;
                    default: break;
                }

                return sort_ascending_ ? result : !result;
            });
        }

        for (size_t idx : indices) {
            const auto& col = analysis_.columns[idx];

            ImGui::TableNextRow();

            // Column name
            ImGui::TableNextColumn();
            bool selected = (selected_column_ == static_cast<int>(idx));
            if (ImGui::Selectable(col.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
                selected_column_ = static_cast<int>(idx);
            }

            // Missing count
            ImGui::TableNextColumn();
            ImGui::Text("%zu", col.missing_count);

            // Percentage
            ImGui::TableNextColumn();
            if (col.missing_count > 0) {
                ImVec4 color = col.missing_percentage > 20 ?
                    ImVec4(1.0f, 0.4f, 0.4f, 1.0f) :
                    (col.missing_percentage > 5 ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f) :
                                                   ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
                ImGui::TextColored(color, "%.1f%%", col.missing_percentage);
            } else {
                ImGui::TextDisabled("0%%");
            }

            // Visual bar
            ImGui::TableNextColumn();
            float bar_width = ImGui::GetContentRegionAvail().x;
            float fill_width = bar_width * (col.missing_percentage / 100.0f);

            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Background
            draw_list->AddRectFilled(pos, ImVec2(pos.x + bar_width, pos.y + 14),
                                     IM_COL32(60, 60, 60, 255));

            // Fill
            if (fill_width > 0) {
                ImU32 fill_color = col.missing_percentage > 20 ?
                    IM_COL32(200, 80, 80, 255) :
                    (col.missing_percentage > 5 ? IM_COL32(200, 180, 80, 255) :
                                                   IM_COL32(80, 200, 80, 255));
                draw_list->AddRectFilled(pos, ImVec2(pos.x + fill_width, pos.y + 14), fill_color);
            }

            ImGui::Dummy(ImVec2(bar_width, 14));
        }

        ImGui::EndTable();
    }

    // Show missing indices for selected column
    if (selected_column_ >= 0 && selected_column_ < static_cast<int>(analysis_.columns.size())) {
        const auto& col = analysis_.columns[selected_column_];

        if (!col.missing_indices.empty()) {
            ImGui::Spacing();
            ImGui::Text("Missing row indices for '%s':", col.name.c_str());

            ImGui::BeginChild("MissingIndices", ImVec2(0, 60), true);
            std::string indices_str;
            for (size_t i = 0; i < std::min(col.missing_indices.size(), size_t(50)); i++) {
                if (i > 0) indices_str += ", ";
                indices_str += std::to_string(col.missing_indices[i]);
            }
            if (col.missing_indices.size() > 50) {
                indices_str += "... (and " + std::to_string(col.missing_indices.size() - 50) + " more)";
            }
            ImGui::TextWrapped("%s", indices_str.c_str());
            ImGui::EndChild();
        }
    }
}

void MissingValuePanel::RenderMissingPattern() {
    std::lock_guard<std::mutex> lock(analysis_mutex_);

    // Bar chart of missing values per column
    std::vector<const char*> labels;
    std::vector<double> values;

    for (const auto& col : analysis_.columns) {
        labels.push_back(col.name.c_str());
        values.push_back(col.missing_percentage);
    }

    if (!labels.empty()) {
        if (ImPlot::BeginPlot("Missing Values by Column", ImVec2(-1, 300))) {
            ImPlot::SetupAxes("Column", "Missing %");
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0, static_cast<double>(labels.size() - 1),
                                    static_cast<int>(labels.size()), labels.data());

            std::vector<double> positions(labels.size());
            for (size_t i = 0; i < labels.size(); i++) positions[i] = static_cast<double>(i);

            ImPlot::PlotBars("Missing %", positions.data(), values.data(),
                              static_cast<int>(values.size()), 0.6);
            ImPlot::EndPlot();
        }
    }

    // Completeness pie chart
    ImGui::Spacing();

    float complete_pct = analysis_.total_cells > 0 ?
        100.0f - analysis_.missing_percentage : 100.0f;

    ImGui::Text("Data Completeness: %.1f%% complete, %.1f%% missing",
                complete_pct, analysis_.missing_percentage);

    // Simple visual indicator
    float bar_width = ImGui::GetContentRegionAvail().x;
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float complete_width = bar_width * (complete_pct / 100.0f);

    draw_list->AddRectFilled(pos, ImVec2(pos.x + complete_width, pos.y + 20),
                             IM_COL32(80, 180, 80, 255));
    draw_list->AddRectFilled(ImVec2(pos.x + complete_width, pos.y),
                             ImVec2(pos.x + bar_width, pos.y + 20),
                             IM_COL32(180, 80, 80, 255));

    ImGui::Dummy(ImVec2(bar_width, 20));
}

void MissingValuePanel::RenderImputationSuggestions() {
    std::lock_guard<std::mutex> lock(analysis_mutex_);

    ImGui::TextWrapped("Based on the missing value patterns, here are some suggestions:");
    ImGui::Spacing();

    if (analysis_.total_missing == 0) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f),
                          "%s No missing values detected! Your data is complete.",
                          ICON_FA_CIRCLE_CHECK);
        return;
    }

    // Analyze patterns and give suggestions
    for (const auto& col : analysis_.columns) {
        if (col.missing_count == 0) continue;

        ImGui::PushID(col.name.c_str());

        if (ImGui::CollapsingHeader(col.name.c_str())) {
            ImGui::Indent();

            ImGui::Text("Missing: %zu values (%.1f%%)", col.missing_count, col.missing_percentage);

            ImGui::Spacing();
            ImGui::Text("Suggested actions:");

            if (col.missing_percentage < 5) {
                ImGui::BulletText("Low missing rate - consider mean/median imputation");
                ImGui::BulletText("Or simply drop rows with missing values");
            } else if (col.missing_percentage < 20) {
                ImGui::BulletText("Moderate missing rate - consider:");
                ImGui::Indent();
                ImGui::BulletText("K-Nearest Neighbors imputation");
                ImGui::BulletText("Multiple imputation (MICE)");
                ImGui::BulletText("Model-based imputation");
                ImGui::Unindent();
            } else if (col.missing_percentage < 50) {
                ImGui::BulletText("High missing rate - consider:");
                ImGui::Indent();
                ImGui::BulletText("Creating a 'missing' indicator variable");
                ImGui::BulletText("Using algorithms that handle missing values natively");
                ImGui::BulletText("Investigating why data is missing (MCAR, MAR, MNAR)");
                ImGui::Unindent();
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                                  "Very high missing rate (>50%%)");
                ImGui::BulletText("Consider dropping this column entirely");
                ImGui::BulletText("Or investigate data collection issues");
            }

            ImGui::Unindent();
        }

        ImGui::PopID();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // General recommendations
    ImGui::Text("General Recommendations:");
    ImGui::BulletText("For numeric data: mean, median, or regression imputation");
    ImGui::BulletText("For categorical data: mode or 'Unknown' category");
    ImGui::BulletText("For time series: forward/backward fill or interpolation");
    ImGui::BulletText("Always document your imputation strategy!");
}

void MissingValuePanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void MissingValuePanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    AnalyzeAsync(table);
}

void MissingValuePanel::AnalyzeAsync(std::shared_ptr<DataTable> table) {
    if (is_analyzing_.load()) return;

    if (analysis_thread_ && analysis_thread_->joinable()) {
        analysis_thread_->join();
    }

    is_analyzing_ = true;

    analysis_thread_ = std::make_unique<std::thread>([this, table]() {
        try {
            DataAnalyzer analyzer;
            auto result = analyzer.AnalyzeMissingValues(*table);

            {
                std::lock_guard<std::mutex> lock(analysis_mutex_);
                analysis_ = std::move(result);
                has_analysis_ = true;
            }

            spdlog::info("Missing value analysis complete: {} total missing",
                         analysis_.total_missing);

        } catch (const std::exception& e) {
            spdlog::error("Missing value analysis error: {}", e.what());
        }

        is_analyzing_ = false;
    });
}

} // namespace cyxwiz
