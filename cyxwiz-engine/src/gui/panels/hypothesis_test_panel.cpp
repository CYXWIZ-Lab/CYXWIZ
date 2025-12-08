#include "hypothesis_test_panel.h"
#include "../../data/data_table.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace cyxwiz {

HypothesisTestPanel::HypothesisTestPanel() = default;

HypothesisTestPanel::~HypothesisTestPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void HypothesisTestPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(550, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_FLASK " Hypothesis Testing###HypothesisTest", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderTestConfiguration();

            if (has_result_) {
                ImGui::Separator();
                RenderResults();
            }
        }
    }
    ImGui::End();
}

void HypothesisTestPanel::RenderToolbar() {
    RenderDataSelector();
    ImGui::SameLine();
    RenderTestTypeSelector();
}

void HypothesisTestPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::SetNextItemWidth(150);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                current_table_ = registry.GetTable(name);
                sample1_col_ = -1;
                sample2_col_ = -1;
                group_col_ = -1;
                has_result_ = false;

                // Refresh column lists
                numeric_columns_.clear();
                all_columns_.clear();

                if (current_table_) {
                    const auto& headers = current_table_->GetHeaders();
                    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
                        std::string col_name = i < headers.size() ? headers[i] : "Column " + std::to_string(i);
                        all_columns_.push_back(col_name);

                        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
                        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                            numeric_columns_.push_back(col_name);
                        }
                    }
                }
            }
        }
        ImGui::EndCombo();
    }
}

void HypothesisTestPanel::RenderTestTypeSelector() {
    const char* test_types[] = {
        "One-Sample t-Test",
        "Two-Sample t-Test",
        "Paired t-Test",
        "One-Way ANOVA",
        "Chi-Square Test"
    };

    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("Test", &test_type_, test_types, IM_ARRAYSIZE(test_types))) {
        has_result_ = false;
    }
}

void HypothesisTestPanel::RenderTestConfiguration() {
    if (!current_table_) {
        ImGui::TextDisabled("Select a data table to configure the test");
        return;
    }

    ImGui::Spacing();
    ImGui::Text("%s Test Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    switch (test_type_) {
        case 0: RenderOneSampleConfig(); break;
        case 1: RenderTwoSampleConfig(); break;
        case 2: RenderPairedConfig(); break;
        case 3: RenderANOVAConfig(); break;
        case 4: RenderChiSquareConfig(); break;
    }

    ImGui::Spacing();

    // Alpha level (common to all tests)
    ImGui::Text("Significance Level:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##Alpha", &alpha_, 0.01f, 0.05f, "%.2f")) {
        alpha_ = std::clamp(alpha_, 0.001f, 0.5f);
        has_result_ = false;
    }

    ImGui::Spacing();

    // Run test button
    bool can_run = false;
    switch (test_type_) {
        case 0: can_run = sample1_col_ >= 0; break;
        case 1:
        case 2: can_run = sample1_col_ >= 0 && sample2_col_ >= 0; break;
        case 3: can_run = sample1_col_ >= 0 && group_col_ >= 0; break;
        case 4: can_run = chi_square_cols_.size() >= 2; break;
    }

    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run Test", ImVec2(120, 0))) {
        RunTest();
    }

    if (!can_run) ImGui::EndDisabled();
}

void HypothesisTestPanel::RenderOneSampleConfig() {
    ImGui::Text("Compare sample mean to hypothesized value");
    ImGui::Spacing();

    // Sample column
    ImGui::Text("Sample Column:");
    ImGui::SetNextItemWidth(200);
    std::string preview = sample1_col_ >= 0 && sample1_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample1_col_] : "Select...";

    if (ImGui::BeginCombo("##Sample", preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample1_col_ == static_cast<int>(i))) {
                sample1_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Hypothesized mean
    ImGui::Text("Hypothesized Mean (H0: mu = ):");
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputFloat("##HypMean", &hypothesized_mean_, 0.1f, 1.0f, "%.2f")) {
        has_result_ = false;
    }
}

void HypothesisTestPanel::RenderTwoSampleConfig() {
    ImGui::Text("Compare means of two independent samples");
    ImGui::Spacing();

    // Sample 1
    ImGui::Text("Sample 1 Column:");
    ImGui::SetNextItemWidth(200);
    std::string preview1 = sample1_col_ >= 0 && sample1_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample1_col_] : "Select...";

    if (ImGui::BeginCombo("##Sample1", preview1.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample1_col_ == static_cast<int>(i))) {
                sample1_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Sample 2
    ImGui::Text("Sample 2 Column:");
    ImGui::SetNextItemWidth(200);
    std::string preview2 = sample2_col_ >= 0 && sample2_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample2_col_] : "Select...";

    if (ImGui::BeginCombo("##Sample2", preview2.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample2_col_ == static_cast<int>(i))) {
                sample2_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Equal variance assumption
    if (ImGui::Checkbox("Assume equal variances", &equal_variance_)) {
        has_result_ = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Uncheck for Welch's t-test (unequal variances)");
    }
}

void HypothesisTestPanel::RenderPairedConfig() {
    ImGui::Text("Compare paired observations (before/after)");
    ImGui::Spacing();

    // Sample 1 (Before)
    ImGui::Text("Before/Time 1 Column:");
    ImGui::SetNextItemWidth(200);
    std::string preview1 = sample1_col_ >= 0 && sample1_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample1_col_] : "Select...";

    if (ImGui::BeginCombo("##Before", preview1.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample1_col_ == static_cast<int>(i))) {
                sample1_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Sample 2 (After)
    ImGui::Text("After/Time 2 Column:");
    ImGui::SetNextItemWidth(200);
    std::string preview2 = sample2_col_ >= 0 && sample2_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample2_col_] : "Select...";

    if (ImGui::BeginCombo("##After", preview2.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample2_col_ == static_cast<int>(i))) {
                sample2_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }
}

void HypothesisTestPanel::RenderANOVAConfig() {
    ImGui::Text("Compare means across multiple groups");
    ImGui::Spacing();

    // Value column
    ImGui::Text("Values Column (numeric):");
    ImGui::SetNextItemWidth(200);
    std::string preview1 = sample1_col_ >= 0 && sample1_col_ < static_cast<int>(numeric_columns_.size())
        ? numeric_columns_[sample1_col_] : "Select...";

    if (ImGui::BeginCombo("##Values", preview1.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            if (ImGui::Selectable(numeric_columns_[i].c_str(), sample1_col_ == static_cast<int>(i))) {
                sample1_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Group column
    ImGui::Text("Group Column (categorical):");
    ImGui::SetNextItemWidth(200);
    std::string preview2 = group_col_ >= 0 && group_col_ < static_cast<int>(all_columns_.size())
        ? all_columns_[group_col_] : "Select...";

    if (ImGui::BeginCombo("##Groups", preview2.c_str())) {
        for (size_t i = 0; i < all_columns_.size(); i++) {
            if (ImGui::Selectable(all_columns_[i].c_str(), group_col_ == static_cast<int>(i))) {
                group_col_ = static_cast<int>(i);
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }
}

void HypothesisTestPanel::RenderChiSquareConfig() {
    ImGui::Text("Test for independence between categorical variables");
    ImGui::Spacing();
    ImGui::TextDisabled("Select two or more categorical columns to create contingency table");

    // Multi-select columns
    for (size_t i = 0; i < all_columns_.size(); i++) {
        bool selected = std::find(chi_square_cols_.begin(), chi_square_cols_.end(), static_cast<int>(i)) != chi_square_cols_.end();
        if (ImGui::Checkbox(all_columns_[i].c_str(), &selected)) {
            if (selected) {
                chi_square_cols_.push_back(static_cast<int>(i));
            } else {
                chi_square_cols_.erase(std::remove(chi_square_cols_.begin(), chi_square_cols_.end(), static_cast<int>(i)), chi_square_cols_.end());
            }
            has_result_ = false;
        }
    }
}

void HypothesisTestPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Running statistical test...", ICON_FA_SPINNER);
}

void HypothesisTestPanel::RenderResults() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    ImGui::Text("%s Test Results", ICON_FA_CHART_LINE);
    ImGui::Spacing();

    // Test type
    ImGui::Text("Test: %s", TestTypeToString(result_.test_type));

    if (ImGui::BeginTable("ResultsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Statistic", ImGuiTableColumnFlags_WidthFixed, 180);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto AddRow = [](const char* name, const char* value) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text("%s", value);
        };

        char buf[64];

        snprintf(buf, sizeof(buf), "%.4f", result_.test_statistic);
        AddRow(result_.test_type == TestType::OneWayANOVA ? "F-statistic" :
               result_.test_type == TestType::ChiSquare ? "Chi-Square" : "t-statistic", buf);

        snprintf(buf, sizeof(buf), "%.0f", result_.df);
        if (result_.test_type == TestType::OneWayANOVA) {
            snprintf(buf, sizeof(buf), "(%.0f, %.0f)", result_.df, result_.df2);
        }
        AddRow("Degrees of Freedom", buf);

        snprintf(buf, sizeof(buf), "%.6f", result_.p_value);
        AddRow("p-value", buf);

        if (result_.test_type != TestType::ChiSquare && result_.test_type != TestType::OneWayANOVA) {
            snprintf(buf, sizeof(buf), "%.4f", result_.mean_diff);
            AddRow("Mean Difference", buf);

            snprintf(buf, sizeof(buf), "[%.4f, %.4f]",
                    result_.confidence_interval_low, result_.confidence_interval_high);
            AddRow("95% Confidence Interval", buf);
        }

        snprintf(buf, sizeof(buf), "%.4f", result_.effect_size);
        const char* effect_name = result_.test_type == TestType::OneWayANOVA ? "Eta-squared" :
                                  result_.test_type == TestType::ChiSquare ? "Cramer's V" : "Cohen's d";
        AddRow(effect_name, buf);

        ImGui::EndTable();
    }

    ImGui::Spacing();
    RenderPValueVisualization();
    ImGui::Spacing();
    RenderDecision();
    ImGui::Spacing();
    RenderEffectSizeInterpretation();
}

void HypothesisTestPanel::RenderPValueVisualization() {
    ImGui::Text("P-Value Interpretation:");

    float p = static_cast<float>(result_.p_value);
    float bar_width = ImGui::GetContentRegionAvail().x - 100;

    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Draw significance regions
    float x = pos.x;
    float y = pos.y;

    // Very significant (p < 0.01)
    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + bar_width * 0.01f, y + 20),
                             IM_COL32(0, 150, 0, 255));
    // Significant (0.01 <= p < 0.05)
    draw_list->AddRectFilled(ImVec2(x + bar_width * 0.01f, y), ImVec2(x + bar_width * 0.05f, y + 20),
                             IM_COL32(100, 200, 100, 255));
    // Marginally significant (0.05 <= p < 0.10)
    draw_list->AddRectFilled(ImVec2(x + bar_width * 0.05f, y), ImVec2(x + bar_width * 0.10f, y + 20),
                             IM_COL32(255, 200, 100, 255));
    // Not significant (p >= 0.10)
    draw_list->AddRectFilled(ImVec2(x + bar_width * 0.10f, y), ImVec2(x + bar_width, y + 20),
                             IM_COL32(200, 100, 100, 255));

    // Alpha line
    float alpha_x = x + bar_width * alpha_;
    draw_list->AddLine(ImVec2(alpha_x, y - 5), ImVec2(alpha_x, y + 25),
                       IM_COL32(255, 255, 255, 255), 2.0f);

    // P-value marker
    float p_x = x + bar_width * std::min(p, 1.0f);
    draw_list->AddTriangleFilled(ImVec2(p_x, y + 25), ImVec2(p_x - 5, y + 32), ImVec2(p_x + 5, y + 32),
                                  IM_COL32(255, 255, 0, 255));

    ImGui::Dummy(ImVec2(bar_width, 40));

    // Labels
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::TextColored(ImVec4(0, 0.6f, 0, 1), "p<0.01");
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1), "p<0.05");
    ImGui::EndGroup();
}

void HypothesisTestPanel::RenderDecision() {
    if (result_.reject_null) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
        ImGui::Text("%s REJECT H0", ICON_FA_CIRCLE_CHECK);
        ImGui::PopStyleColor();
        ImGui::TextWrapped("The result is statistically significant at alpha = %.2f", alpha_);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
        ImGui::Text("%s FAIL TO REJECT H0", ICON_FA_CIRCLE_XMARK);
        ImGui::PopStyleColor();
        ImGui::TextWrapped("The result is not statistically significant at alpha = %.2f", alpha_);
    }

    ImGui::Spacing();
    ImGui::TextWrapped("%s", result_.interpretation.c_str());
}

void HypothesisTestPanel::RenderEffectSizeInterpretation() {
    ImGui::Text("Effect Size Interpretation:");

    double d = std::abs(result_.effect_size);
    const char* interpretation = "Negligible";

    if (result_.test_type == TestType::OneWayANOVA) {
        // Eta-squared
        if (d >= 0.14) interpretation = "Large";
        else if (d >= 0.06) interpretation = "Medium";
        else if (d >= 0.01) interpretation = "Small";
    } else if (result_.test_type == TestType::ChiSquare) {
        // Cramer's V
        if (d >= 0.5) interpretation = "Large";
        else if (d >= 0.3) interpretation = "Medium";
        else if (d >= 0.1) interpretation = "Small";
    } else {
        // Cohen's d
        if (d >= 0.8) interpretation = "Large";
        else if (d >= 0.5) interpretation = "Medium";
        else if (d >= 0.2) interpretation = "Small";
    }

    ImGui::SameLine();
    ImGui::TextColored(d >= 0.5 ? ImVec4(0.2f, 0.8f, 0.2f, 1) :
                       d >= 0.2 ? ImVec4(1.0f, 0.8f, 0.2f, 1) :
                                  ImVec4(0.6f, 0.6f, 0.6f, 1),
                       "%s (%.3f)", interpretation, result_.effect_size);
}

void HypothesisTestPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void HypothesisTestPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    sample1_col_ = -1;
    sample2_col_ = -1;
    group_col_ = -1;
    has_result_ = false;

    // Refresh columns
    numeric_columns_.clear();
    all_columns_.clear();

    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < table->GetColumnCount(); i++) {
        std::string col_name = i < headers.size() ? headers[i] : "Column " + std::to_string(i);
        all_columns_.push_back(col_name);

        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(col_name);
        }
    }
}

void HypothesisTestPanel::RunTest() {
    if (is_computing_.load() || !current_table_) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;

    auto table = current_table_;
    int test = test_type_;
    int col1 = sample1_col_;
    int col2 = sample2_col_;
    int grp = group_col_;
    float mu0 = hypothesized_mean_;
    float a = alpha_;
    bool eq_var = equal_variance_;
    auto chi_cols = chi_square_cols_;

    // Find actual column indices for numeric columns
    auto GetNumericColumnIndex = [&](int numeric_idx) -> size_t {
        int count = 0;
        for (size_t i = 0; i < table->GetColumnCount(); i++) {
            auto dtype = DataAnalyzer::DetectColumnType(*table, i);
            if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                if (count == numeric_idx) return i;
                count++;
            }
        }
        return 0;
    };

    compute_thread_ = std::make_unique<std::thread>([this, table, test, col1, col2, grp, mu0, a, eq_var, chi_cols, GetNumericColumnIndex]() {
        try {
            HypothesisTestResult res;

            switch (test) {
                case 0: {  // One-sample t-test
                    size_t idx = GetNumericColumnIndex(col1);
                    auto vals = DataAnalyzer::GetNumericValues(*table, idx);
                    res = DataAnalyzer::OneSampleTTest(vals, mu0, a);
                    break;
                }
                case 1: {  // Two-sample t-test
                    size_t idx1 = GetNumericColumnIndex(col1);
                    size_t idx2 = GetNumericColumnIndex(col2);
                    auto vals1 = DataAnalyzer::GetNumericValues(*table, idx1);
                    auto vals2 = DataAnalyzer::GetNumericValues(*table, idx2);
                    res = DataAnalyzer::TwoSampleTTest(vals1, vals2, eq_var, a);
                    break;
                }
                case 2: {  // Paired t-test
                    size_t idx1 = GetNumericColumnIndex(col1);
                    size_t idx2 = GetNumericColumnIndex(col2);
                    auto vals1 = DataAnalyzer::GetNumericValues(*table, idx1);
                    auto vals2 = DataAnalyzer::GetNumericValues(*table, idx2);
                    res = DataAnalyzer::PairedTTest(vals1, vals2, a);
                    break;
                }
                case 3: {  // ANOVA
                    size_t val_idx = GetNumericColumnIndex(col1);

                    // Group values by the group column
                    std::unordered_map<std::string, std::vector<double>> groups;
                    for (size_t i = 0; i < table->GetRowCount(); i++) {
                        std::string group_key = table->GetCellAsString(i, grp);
                        auto opt = DataAnalyzer::ToDouble(table->GetCell(i, val_idx));
                        if (opt.has_value()) {
                            groups[group_key].push_back(opt.value());
                        }
                    }

                    std::vector<std::vector<double>> group_data;
                    for (auto& pair : groups) {
                        if (!pair.second.empty()) {
                            group_data.push_back(std::move(pair.second));
                        }
                    }

                    res = DataAnalyzer::OneWayANOVA(group_data, a);
                    break;
                }
                case 4: {  // Chi-square
                    if (chi_cols.size() >= 2) {
                        // Build contingency table from first two columns
                        std::unordered_map<std::string, std::unordered_map<std::string, double>> contingency;
                        std::unordered_set<std::string> row_vals, col_vals;

                        for (size_t i = 0; i < table->GetRowCount(); i++) {
                            std::string row_key = table->GetCellAsString(i, chi_cols[0]);
                            std::string col_key = table->GetCellAsString(i, chi_cols[1]);
                            contingency[row_key][col_key] += 1.0;
                            row_vals.insert(row_key);
                            col_vals.insert(col_key);
                        }

                        // Convert to 2D vector
                        std::vector<std::string> rows(row_vals.begin(), row_vals.end());
                        std::vector<std::string> cols(col_vals.begin(), col_vals.end());

                        std::vector<std::vector<double>> table_data(rows.size(), std::vector<double>(cols.size(), 0.0));
                        for (size_t r = 0; r < rows.size(); r++) {
                            for (size_t c = 0; c < cols.size(); c++) {
                                table_data[r][c] = contingency[rows[r]][cols[c]];
                            }
                        }

                        res = DataAnalyzer::ChiSquareTest(table_data, a);
                    }
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                result_ = res;
                has_result_ = true;
            }

            spdlog::info("Hypothesis test completed: p = {}", res.p_value);

        } catch (const std::exception& e) {
            spdlog::error("Hypothesis test error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
