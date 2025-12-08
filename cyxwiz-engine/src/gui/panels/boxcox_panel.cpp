#include "boxcox_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include <cyxwiz/data_transform.h>
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

BoxCoxPanel::BoxCoxPanel() : Panel("Box-Cox Transform", true) {}

void BoxCoxPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(850, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_WAND_MAGIC_SPARKLES) + " Box-Cox Transform###BoxCoxPanel").c_str(), &visible_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::Separator();
        RenderSettings();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
        if (transform_result_.success) {
            if (ImGui::BeginTabBar("BoxCoxTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Lambda")) {
                    RenderLambdaPlot();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Results")) {
                    RenderResults();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        } else {
            ImGui::Text("Select data and configure settings, then click 'Apply'.");
            ImGui::Spacing();
            ImGui::TextWrapped("Box-Cox transform achieves normality for positive data.");
            ImGui::Spacing();
            ImGui::Text("Formula:");
            ImGui::BulletText("y = (x^lambda - 1) / lambda  if lambda != 0");
            ImGui::BulletText("y = log(x)                   if lambda == 0");
            ImGui::Spacing();
            ImGui::Text("Common lambda values:");
            ImGui::BulletText("lambda = -1: Reciprocal (1/x)");
            ImGui::BulletText("lambda = -0.5: Reciprocal sqrt");
            ImGui::BulletText("lambda = 0: Log transform");
            ImGui::BulletText("lambda = 0.5: Square root");
            ImGui::BulletText("lambda = 1: No transform");
            ImGui::BulletText("lambda = 2: Square");

            if (!can_transform_) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1, 0.5f, 0, 1),
                    "Warning: Data contains non-positive values.");
                ImGui::TextWrapped("Box-Cox requires strictly positive data. Try Yeo-Johnson instead.");
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void BoxCoxPanel::RenderDataSelector() {
    ImGui::Text(ICON_FA_TABLE " Data Selection");
    ImGui::Spacing();

    if (ImGui::BeginCombo("Dataset", selected_table_idx_ >= 0 && selected_table_idx_ < static_cast<int>(available_tables_.size())
                                     ? available_tables_[selected_table_idx_].c_str() : "Select...")) {
        for (int i = 0; i < static_cast<int>(available_tables_.size()); ++i) {
            if (ImGui::Selectable(available_tables_[i].c_str(), i == selected_table_idx_)) {
                selected_table_idx_ = i;
                LoadSelectedData();
            }
        }
        ImGui::EndCombo();
    }

    if (!column_names_.empty()) {
        ImGui::Spacing();

        std::vector<const char*> col_names_c;
        for (const auto& name : column_names_) col_names_c.push_back(name.c_str());

        ImGui::Combo("Column", &selected_column_, col_names_c.data(), static_cast<int>(col_names_c.size()));
    }
}

void BoxCoxPanel::RenderSettings() {
    ImGui::Text(ICON_FA_COG " Settings");
    ImGui::Spacing();

    ImGui::Checkbox("Use Yeo-Johnson", &use_yeo_johnson_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Yeo-Johnson supports negative values.\nBox-Cox requires strictly positive data.");
    }

    ImGui::Spacing();
    ImGui::Checkbox("Auto-find optimal lambda", &auto_lambda_);

    if (!auto_lambda_) {
        ImGui::SliderFloat("Lambda", &manual_lambda_, -5.0f, 5.0f, "%.2f");

        // Quick presets
        if (ImGui::Button("-1", ImVec2(30, 0))) manual_lambda_ = -1.0f;
        ImGui::SameLine();
        if (ImGui::Button("-0.5", ImVec2(35, 0))) manual_lambda_ = -0.5f;
        ImGui::SameLine();
        if (ImGui::Button("0", ImVec2(30, 0))) manual_lambda_ = 0.0f;
        ImGui::SameLine();
        if (ImGui::Button("0.5", ImVec2(35, 0))) manual_lambda_ = 0.5f;
        ImGui::SameLine();
        if (ImGui::Button("1", ImVec2(30, 0))) manual_lambda_ = 1.0f;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    bool can_apply = current_table_ != nullptr && (can_transform_ || use_yeo_johnson_);

    if (!can_apply) ImGui::BeginDisabled();

    if (auto_lambda_) {
        if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Find Optimal Lambda", ImVec2(-1, 25))) {
            FindOptimalLambda();
        }
    }

    if (ImGui::Button(ICON_FA_PLAY " Apply Transform", ImVec2(-1, 30))) {
        ApplyBoxCox();
    }

    if (!can_apply) ImGui::EndDisabled();

    if (!status_message_.empty()) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", status_message_.c_str());
    }
}

void BoxCoxPanel::RenderLambdaPlot() {
    if (lambda_result_.lambdas_tested.empty()) {
        ImGui::Text("Click 'Find Optimal Lambda' to see the lambda profile.");
        return;
    }

    ImGui::Text("Log-Likelihood Profile:");
    ImGui::Spacing();

    ImGui::Text("Optimal Lambda: %.4f", lambda_result_.optimal_lambda);

    if (ImPlot::BeginPlot("##LambdaProfile", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Lambda", "Log-Likelihood");

        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), 2.0f);
        ImPlot::PlotLine("Profile",
                        lambda_result_.lambdas_tested.data(),
                        lambda_result_.log_likelihoods.data(),
                        static_cast<int>(lambda_result_.lambdas_tested.size()));

        // Mark optimal point
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 8, ImVec4(1, 0, 0, 1), 2);
        ImPlot::PlotScatter("Optimal", &lambda_result_.optimal_lambda, &lambda_result_.best_log_likelihood, 1);

        ImPlot::EndPlot();
    }

    // Distribution comparison
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Distribution Comparison:");

    if (transform_result_.success && !transform_result_.transformed_data.empty()) {
        std::vector<double> original_data;
        if (current_table_) {
            int n_rows = static_cast<int>(current_table_->GetRowCount());
            for (int row = 0; row < n_rows && row < 10000; ++row) {
                auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, selected_column_));
                if (val.has_value()) {
                    original_data.push_back(val.value());
                }
            }
        }

        const auto& transformed_data = transform_result_.transformed_data[0];

        if (!original_data.empty()) {
            if (ImPlot::BeginPlot("##OrigHist", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 180))) {
                ImPlot::SetupAxes("Value", "Count");
                ImPlot::PlotHistogram("Original", original_data.data(), static_cast<int>(original_data.size()), 30);
                ImPlot::EndPlot();
            }

            ImGui::SameLine();

            if (ImPlot::BeginPlot("##TransHist", ImVec2(ImGui::GetContentRegionAvail().x, 180))) {
                ImPlot::SetupAxes("Transformed", "Count");
                ImPlot::SetNextFillStyle(ImVec4(0.2f, 0.8f, 0.4f, 0.7f));
                ImPlot::PlotHistogram("Box-Cox", transformed_data.data(), static_cast<int>(transformed_data.size()), 30);
                ImPlot::EndPlot();
            }
        }
    }

    // Normality test results
    ImGui::Spacing();
    ImGui::Text("Normality Assessment:");
    ImGui::BulletText("Original: %s (p=%.4f)",
                     orig_normality_.is_normal ? "Normal" : "Non-normal",
                     orig_normality_.p_value);
    ImGui::BulletText("Transformed: %s (p=%.4f)",
                     trans_normality_.is_normal ? "Normal" : "Non-normal",
                     trans_normality_.p_value);
}

void BoxCoxPanel::RenderResults() {
    if (!transform_result_.success) return;

    ImGui::Text(ICON_FA_CHART_BAR " Transform Results");
    ImGui::Separator();
    ImGui::Spacing();

    // Lambda used
    double lambda_used = transform_result_.params.count("lambda") ?
                         transform_result_.params.at("lambda") : manual_lambda_;

    ImGui::Text("Lambda Used: %.4f", lambda_used);

    // Interpretation
    const char* interp = "Custom transform";
    if (std::abs(lambda_used - (-1)) < 0.1) interp = "Reciprocal (1/x)";
    else if (std::abs(lambda_used - (-0.5)) < 0.1) interp = "Reciprocal square root";
    else if (std::abs(lambda_used) < 0.1) interp = "Log transform";
    else if (std::abs(lambda_used - 0.5) < 0.1) interp = "Square root";
    else if (std::abs(lambda_used - 1) < 0.1) interp = "No transform";
    else if (std::abs(lambda_used - 2) < 0.1) interp = "Square";

    ImGui::Text("Interpretation: %s", interp);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Statistics:");

    if (ImGui::BeginTable("BoxCoxStats", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Statistic");
        ImGui::TableSetupColumn("Original");
        ImGui::TableSetupColumn("Transformed");
        ImGui::TableHeadersRow();

        auto add_row = [](const char* name, double orig, double trans) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", name);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", orig);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", trans);
        };

        add_row("Mean", original_stats_.mean, transformed_stats_.mean);
        add_row("Std Dev", original_stats_.std_dev, transformed_stats_.std_dev);
        add_row("Min", original_stats_.min, transformed_stats_.min);
        add_row("Max", original_stats_.max, transformed_stats_.max);
        add_row("Median", original_stats_.median, transformed_stats_.median);

        ImGui::EndTable();
    }

    // Data preview
    ImGui::Spacing();
    ImGui::Text("Data Preview (first 15 rows):");

    if (!transform_result_.transformed_data.empty()) {
        const auto& trans = transform_result_.transformed_data[0];
        int show_rows = std::min(static_cast<int>(trans.size()), 15);

        if (ImGui::BeginTable("DataPreview", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Row");
            ImGui::TableSetupColumn("Original");
            ImGui::TableSetupColumn("Transformed");
            ImGui::TableHeadersRow();

            for (int i = 0; i < show_rows; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%d", i);
                ImGui::TableNextColumn();

                if (current_table_) {
                    auto val = DataAnalyzer::ToDouble(current_table_->GetCell(i, selected_column_));
                    if (val.has_value()) {
                        ImGui::Text("%.4f", val.value());
                    }
                }

                ImGui::TableNextColumn();
                ImGui::Text("%.4f", trans[i]);
            }

            ImGui::EndTable();
        }
    }
}

void BoxCoxPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_column_ = 0;

    transform_result_ = TransformResult();
    lambda_result_ = BoxCoxLambdaResult();
    original_stats_ = ColumnStats();
    transformed_stats_ = ColumnStats();

    // Check if data is valid for Box-Cox
    std::vector<double> col_data;
    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, selected_column_));
        if (val.has_value()) {
            col_data.push_back(val.value());
        }
    }

    can_transform_ = DataTransform::CanApplyBoxCox(col_data);
    status_message_ = can_transform_ ?
        "Data loaded. Configure and apply transform." :
        "Warning: Data has non-positive values. Use Yeo-Johnson.";
}

void BoxCoxPanel::FindOptimalLambda() {
    if (!current_table_) return;

    std::vector<double> col_data;
    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, selected_column_));
        if (val.has_value()) {
            col_data.push_back(val.value());
        }
    }

    if (col_data.empty()) {
        status_message_ = "No valid numeric data.";
        return;
    }

    if (!use_yeo_johnson_ && !DataTransform::CanApplyBoxCox(col_data)) {
        status_message_ = "Cannot apply Box-Cox to non-positive data.";
        return;
    }

    lambda_result_ = DataTransform::FindOptimalLambda(col_data, -5.0, 5.0, 100);

    if (lambda_result_.success) {
        manual_lambda_ = static_cast<float>(lambda_result_.optimal_lambda);
        status_message_ = "Optimal lambda found: " + std::to_string(lambda_result_.optimal_lambda);
    } else {
        status_message_ = "Failed to find optimal lambda.";
    }
}

void BoxCoxPanel::ApplyBoxCox() {
    if (!current_table_) return;

    std::vector<double> col_data;
    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, selected_column_));
        if (val.has_value()) {
            col_data.push_back(val.value());
        }
    }

    if (col_data.empty()) {
        status_message_ = "No valid numeric data.";
        return;
    }

    original_stats_ = DataTransform::ComputeColumnStats(col_data);
    orig_normality_ = DataTransform::DAgostinoPearsonTest(col_data);

    if (use_yeo_johnson_) {
        transform_result_ = DataTransform::YeoJohnsonColumn(col_data, manual_lambda_, auto_lambda_);
    } else {
        if (!DataTransform::CanApplyBoxCox(col_data)) {
            status_message_ = "Cannot apply Box-Cox to non-positive data.";
            can_transform_ = false;
            return;
        }
        transform_result_ = DataTransform::BoxCoxColumn(col_data, manual_lambda_, auto_lambda_);
    }

    if (transform_result_.success && !transform_result_.transformed_data.empty()) {
        transformed_stats_ = DataTransform::ComputeColumnStats(transform_result_.transformed_data[0]);
        trans_normality_ = DataTransform::DAgostinoPearsonTest(transform_result_.transformed_data[0]);

        // Also run lambda search if auto was enabled
        if (auto_lambda_ && !use_yeo_johnson_) {
            lambda_result_ = DataTransform::FindOptimalLambda(col_data, -5.0, 5.0, 100);
        }

        status_message_ = "Transform applied successfully.";
    } else {
        status_message_ = "Error: " + transform_result_.error_message;
    }
}

} // namespace cyxwiz
