/**
 * Dataset Panel - Prepare Tab
 * Context-aware data preparation wired to DataAnalyzer and DataTransform backends.
 * Tabular: missing values, outliers, scaling, encoding, feature selection, split.
 * Image: preprocessing, quality analysis, normalization, split.
 */

#include "dataset_panel.h"
#include "../icons.h"
#include "../../core/data_analyzer.h"
#include <cyxwiz/data_transform.h>
#include "../../data/data_table.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <map>
#include <spdlog/spdlog.h>

namespace gui {

// Helper: extract numeric column data from DataTable
static std::vector<double> ExtractColumnDoubles(const cyxwiz::DataTable& table, size_t col) {
    std::vector<double> data;
    data.reserve(table.GetRowCount());
    for (size_t r = 0; r < table.GetRowCount(); r++) {
        auto cell = table.GetCell(r, col);
        if (auto* d = std::get_if<double>(&cell)) {
            data.push_back(*d);
        } else if (auto* i = std::get_if<int64_t>(&cell)) {
            data.push_back(static_cast<double>(*i));
        }
        // skip monostate/string (missing/non-numeric)
    }
    return data;
}

// Helper: extract all numeric columns as 2D data for DataTransform
static std::vector<std::vector<double>> ExtractAllNumericColumns(const cyxwiz::DataTable& table) {
    std::vector<std::vector<double>> columns;
    for (size_t c = 0; c < table.GetColumnCount(); c++) {
        auto col_data = ExtractColumnDoubles(table, c);
        if (!col_data.empty()) {
            columns.push_back(std::move(col_data));
        }
    }
    return columns;
}

// Helper: try to get a DataTable for the currently loaded dataset
static std::shared_ptr<cyxwiz::DataTable> GetDataTableForDataset(const cyxwiz::DatasetInfo& info) {
    auto& registry = cyxwiz::DataTableRegistry::Instance();
    // Try the dataset name first, then the source path
    auto table = registry.GetTable(info.name);
    if (!table && !info.path.empty()) {
        // Try filename as key
        auto pos = info.path.find_last_of("/\\");
        if (pos != std::string::npos) {
            table = registry.GetTable(info.path.substr(pos + 1));
        }
    }
    return table;
}

void TrainingEvaluationPanel::RenderPrepareContent() {
    if (!IsDatasetLoaded()) {
        ImGui::TextDisabled("No dataset loaded. Load a dataset first.");
        return;
    }

    // Detect data type
    bool is_image = (cached_info_.type == cyxwiz::DatasetType::ImageFolder ||
                     cached_info_.type == cyxwiz::DatasetType::ImageCSV ||
                     cached_info_.type == cyxwiz::DatasetType::MNIST ||
                     cached_info_.type == cyxwiz::DatasetType::CIFAR10 ||
                     cached_info_.type == cyxwiz::DatasetType::FashionMNIST ||
                     cached_info_.type == cyxwiz::DatasetType::CIFAR100);

    bool is_tabular = (cached_info_.type == cyxwiz::DatasetType::CSV ||
                       cached_info_.type == cyxwiz::DatasetType::TSV ||
                       cached_info_.type == cyxwiz::DatasetType::TXT ||
                       cached_info_.type == cyxwiz::DatasetType::ARFF);

    if (is_tabular) {
        RenderTabularPrepareContent();
    } else if (is_image) {
        RenderImagePrepareContent();
    } else {
        // Generic: show split config + basic preprocessing
        RenderSplitConfigSection();
        ImGui::Spacing();
        if (ImGui::CollapsingHeader(ICON_FA_SLIDERS " Preprocessing")) {
            ImGui::Indent(8);
            RenderNormalizationSection();
            RenderScalingSection();
            ImGui::Unindent(8);
        }
    }
}

void TrainingEvaluationPanel::RenderTabularPrepareContent() {
    auto table = GetDataTableForDataset(cached_info_);

    // 1. Missing Values
    if (ImGui::CollapsingHeader(ICON_FA_CIRCLE_QUESTION " Missing Values", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);

        if (table) {
            // Analyze button
            if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Analyze Missing Values")) {
                cyxwiz::DataAnalyzer analyzer;
                auto result = analyzer.AnalyzeMissingValues(*table);
                prepare_missing_analyzed_ = true;
                prepare_missing_columns_.clear();
                for (auto& col : result.columns) {
                    if (col.missing_count > 0) {
                        prepare_missing_columns_.emplace_back(col.name, (int)col.missing_count);
                    }
                }
                char buf[256];
                snprintf(buf, sizeof(buf), "Total: %zu missing cells (%.1f%%) across %zu columns",
                         result.total_missing, result.missing_percentage * 100.0f,
                         prepare_missing_columns_.size());
                prepare_missing_summary_ = buf;
            }

            if (prepare_missing_analyzed_) {
                ImGui::TextWrapped("%s", prepare_missing_summary_.c_str());

                if (!prepare_missing_columns_.empty()) {
                    ImGui::Spacing();
                    if (ImGui::BeginTable("##MissingCols", 2,
                                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                        ImGui::TableSetupColumn("Column", 0, 2.0f);
                        ImGui::TableSetupColumn("Missing", 0, 1.0f);
                        ImGui::TableHeadersRow();
                        for (auto& [name, count] : prepare_missing_columns_) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", name.c_str());
                            ImGui::TableNextColumn();
                            ImGui::Text("%d", count);
                        }
                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    ImGui::Combo("Strategy", &prepare_missing_strategy_,
                                 "Drop rows\0Fill with Mean\0Fill with Median\0Fill with Mode\0Fill with Constant\0");
                    if (prepare_missing_strategy_ == 4) {
                        ImGui::InputFloat("Fill Value", &prepare_missing_fill_value_);
                    }
                    if (ImGui::Button(ICON_FA_CHECK " Apply##MissingValues")) {
                        size_t filled = 0;
                        for (size_t c = 0; c < table->GetColumnCount(); c++) {
                            // Compute fill value for this column
                            auto col_data = ExtractColumnDoubles(*table, c);
                            if (col_data.empty()) continue;

                            double fill_val = 0.0;
                            if (prepare_missing_strategy_ == 1) { // Mean
                                double sum = 0; for (auto v : col_data) sum += v;
                                fill_val = sum / col_data.size();
                            } else if (prepare_missing_strategy_ == 2) { // Median
                                std::sort(col_data.begin(), col_data.end());
                                fill_val = col_data[col_data.size() / 2];
                            } else if (prepare_missing_strategy_ == 3) { // Mode
                                std::map<int64_t, int> freq;
                                for (auto v : col_data) freq[(int64_t)(v * 1000)]++;
                                int max_f = 0;
                                for (auto& [k, f] : freq) { if (f > max_f) { max_f = f; fill_val = k / 1000.0; } }
                            } else if (prepare_missing_strategy_ == 4) { // Constant
                                fill_val = (double)prepare_missing_fill_value_;
                            }

                            // Apply: fill missing cells or drop rows
                            if (prepare_missing_strategy_ == 0) {
                                // Drop rows handled separately below
                                continue;
                            }
                            for (size_t r = 0; r < table->GetRowCount(); r++) {
                                auto cell = table->GetCell(r, c);
                                if (std::holds_alternative<std::monostate>(cell)) {
                                    table->SetCell(r, c, cyxwiz::DataTable::CellValue(fill_val));
                                    filled++;
                                }
                            }
                        }
                        if (prepare_missing_strategy_ == 0) {
                            spdlog::info("Prepare: Drop rows with missing values (not yet supported for in-place)");
                        } else {
                            spdlog::info("Prepare: Filled {} missing cells with strategy {}", filled, prepare_missing_strategy_);
                        }
                        // Re-analyze
                        prepare_missing_analyzed_ = false;
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f),
                                       ICON_FA_CHECK " No missing values found!");
                }
            }
        } else {
            ImGui::TextDisabled("Load dataset in Table Viewer for missing value analysis.");
            ImGui::Spacing();
            ImGui::Combo("Strategy", &prepare_missing_strategy_,
                         "Drop rows\0Fill with Mean\0Fill with Median\0Fill with Mode\0Fill with Constant\0");
            if (prepare_missing_strategy_ == 4) {
                ImGui::InputFloat("Fill Value", &prepare_missing_fill_value_);
            }
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 2. Outlier Detection
    if (ImGui::CollapsingHeader(ICON_FA_TRIANGLE_EXCLAMATION " Outlier Detection")) {
        ImGui::Indent(8);

        ImGui::Combo("Method", &prepare_outlier_method_,
                     "IQR (1.5x)\0Z-Score (>3)\0Modified Z-Score\0");

        // Method-specific parameter
        if (prepare_outlier_method_ == 0) {
            ImGui::SliderFloat("IQR Factor", &prepare_outlier_param_, 1.0f, 3.0f, "%.1f");
        } else {
            ImGui::SliderFloat("Threshold", &prepare_outlier_param_, 1.0f, 5.0f, "%.1f");
        }

        ImGui::Combo("Action", &prepare_outlier_action_,
                     "Flag only\0Remove\0Clip to bounds\0");

        if (table) {
            // Column selector
            auto col_names = table->GetHeaders();
            std::vector<const char*> col_cstrs;
            for (auto& n : col_names) col_cstrs.push_back(n.c_str());
            if (!col_cstrs.empty()) {
                ImGui::Combo("Column##Outlier", &prepare_outlier_col_idx_,
                             col_cstrs.data(), (int)col_cstrs.size());
            }

            if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Detect Outliers")) {
                cyxwiz::DataAnalyzer analyzer;
                cyxwiz::OutlierMethod method = static_cast<cyxwiz::OutlierMethod>(prepare_outlier_method_);
                auto result = analyzer.DetectOutliers(*table, (size_t)prepare_outlier_col_idx_,
                                                      method, (double)prepare_outlier_param_);
                prepare_outlier_analyzed_ = true;
                prepare_outlier_count_ = (int)result.outlier_count;
                char buf[256];
                snprintf(buf, sizeof(buf), "%d outliers found (%.1f%%) in '%s' [bounds: %.3f - %.3f]",
                         (int)result.outlier_count, result.outlier_percentage * 100.0f,
                         result.column_name.c_str(), result.lower_bound, result.upper_bound);
                prepare_outlier_summary_ = buf;
            }

            if (prepare_outlier_analyzed_) {
                ImGui::Spacing();
                if (prepare_outlier_count_ > 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "%s", prepare_outlier_summary_.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f),
                                       ICON_FA_CHECK " No outliers detected.");
                }
            }
        } else {
            ImGui::TextDisabled("Load dataset in Table Viewer for outlier detection.");
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 3. Feature Scaling
    if (ImGui::CollapsingHeader(ICON_FA_ARROWS_LEFT_RIGHT " Feature Scaling")) {
        ImGui::Indent(8);

        ImGui::Combo("Method##Scaling", &prepare_scaling_method_,
                     "Standard (Z-score)\0Min-Max [0,1]\0Robust (IQR)\0MaxAbs\0"
                     "Log Transform\0Box-Cox\0Yeo-Johnson\0");

        // Method descriptions
        const char* descriptions[] = {
            "Centers to mean=0, scales to std=1. Best for normally distributed features.",
            "Scales to [0, 1] range. Best when you need bounded values.",
            "Uses median and IQR. Robust to outliers.",
            "Scales by maximum absolute value to [-1, 1].",
            "Applies log transform. Reduces right-skew. Requires positive values.",
            "Power transform for positive data. Auto-finds optimal lambda.",
            "Like Box-Cox but supports negative values."
        };
        ImGui::TextDisabled("%s", descriptions[prepare_scaling_method_]);

        ImGui::Spacing();
        if (table) {
            if (ImGui::Button(ICON_FA_CHECK " Apply Scaling")) {
                auto data = ExtractAllNumericColumns(*table);
                if (!data.empty()) {
                    cyxwiz::TransformResult result;
                    switch (prepare_scaling_method_) {
                        case 0: result = cyxwiz::DataTransform::Standardize(data); break;
                        case 1: result = cyxwiz::DataTransform::Normalize(data); break;
                        case 2: result = cyxwiz::DataTransform::RobustScale(data); break;
                        case 3: result = cyxwiz::DataTransform::MaxAbsScale(data); break;
                        case 4: result = cyxwiz::DataTransform::LogTransform(data); break;
                        case 5: result = cyxwiz::DataTransform::BoxCox(data); break;
                        case 6: result = cyxwiz::DataTransform::YeoJohnson(data); break;
                    }
                    if (result.success) {
                        // Write transformed data back to table
                        size_t col_idx = 0;
                        for (size_t c = 0; c < table->GetColumnCount() && col_idx < result.transformed_data.size(); c++) {
                            auto test = ExtractColumnDoubles(*table, c);
                            if (test.empty()) continue; // skip non-numeric columns
                            auto& transformed = result.transformed_data[col_idx];
                            for (size_t r = 0; r < std::min(table->GetRowCount(), transformed.size()); r++) {
                                table->SetCell(r, c, cyxwiz::DataTable::CellValue(transformed[r]));
                            }
                            col_idx++;
                        }
                        spdlog::info("Prepare: Applied {} scaling to {} columns", result.method, col_idx);
                    } else {
                        spdlog::warn("Prepare: Scaling failed - {}", result.error_message);
                    }
                }
            }
        } else {
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_CHECK " Apply Scaling");
            ImGui::EndDisabled();
            ImGui::TextDisabled("Load dataset in Table Viewer to apply scaling.");
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 4. Categorical Encoding
    if (ImGui::CollapsingHeader(ICON_FA_TAGS " Categorical Encoding")) {
        ImGui::Indent(8);

        ImGui::Combo("Encoding", &prepare_encoding_method_, "One-Hot\0Label\0Ordinal\0");

        const char* enc_descriptions[] = {
            "Creates binary columns for each category. Best for nominal data with few categories.",
            "Assigns integer labels. Best for tree-based models.",
            "Assigns ordered integers. Best for ordinal data (e.g., low/medium/high)."
        };
        ImGui::TextDisabled("%s", enc_descriptions[prepare_encoding_method_]);

        ImGui::Spacing();
        if (table) {
            // Detect categorical columns (columns with string values)
            std::vector<size_t> cat_cols;
            for (size_t c = 0; c < table->GetColumnCount(); c++) {
                for (size_t r = 0; r < std::min((size_t)100, table->GetRowCount()); r++) {
                    auto cell = table->GetCell(r, c);
                    if (std::holds_alternative<std::string>(cell)) {
                        cat_cols.push_back(c);
                        break;
                    }
                }
            }
            if (cat_cols.empty()) {
                ImGui::TextDisabled("No categorical columns detected.");
            } else {
                ImGui::Text("%d categorical column(s): ", (int)cat_cols.size());
                for (size_t i = 0; i < std::min(cat_cols.size(), (size_t)5); i++) {
                    auto& headers = table->GetHeaders();
                    if (i > 0) ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "%s", headers[cat_cols[i]].c_str());
                }
                if (cat_cols.size() > 5) ImGui::Text("  ... and %d more", (int)(cat_cols.size() - 5));
            }

            if (ImGui::Button(ICON_FA_CHECK " Apply Encoding")) {
                if (prepare_encoding_method_ == 1 || prepare_encoding_method_ == 2) {
                    // Label / Ordinal encoding: map unique strings to integers
                    for (size_t c : cat_cols) {
                        std::map<std::string, int> label_map;
                        int next_label = 0;
                        for (size_t r = 0; r < table->GetRowCount(); r++) {
                            auto cell = table->GetCell(r, c);
                            if (auto* s = std::get_if<std::string>(&cell)) {
                                auto it = label_map.find(*s);
                                if (it == label_map.end()) {
                                    label_map[*s] = next_label++;
                                }
                                table->SetCell(r, c, cyxwiz::DataTable::CellValue((int64_t)label_map[*s]));
                            }
                        }
                    }
                    spdlog::info("Prepare: Applied label encoding to {} columns", cat_cols.size());
                } else {
                    // One-hot: more complex, log as not fully supported inline
                    spdlog::info("Prepare: One-hot encoding requires column expansion (use Pipeline for full support)");
                }
            }
        } else {
            ImGui::TextDisabled("Load dataset in Table Viewer for encoding.");
            ImGui::BeginDisabled();
            ImGui::Button(ICON_FA_CHECK " Apply Encoding");
            ImGui::EndDisabled();
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 5. Feature Selection
    if (ImGui::CollapsingHeader(ICON_FA_FILTER " Feature Selection")) {
        ImGui::Indent(8);

        ImGui::Combo("Method##Selection", &prepare_selection_method_,
                     "Correlation Threshold\0Variance Threshold\0");

        ImGui::SliderFloat("Threshold", &prepare_selection_threshold_, 0.0f, 1.0f, "%.2f");

        if (prepare_selection_method_ == 0) {
            ImGui::TextDisabled("Remove one of each pair of features with correlation above threshold.");
        } else {
            ImGui::TextDisabled("Remove features with variance below threshold.");
        }

        if (table && ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Analyze##Selection")) {
            if (prepare_selection_method_ == 0) {
                cyxwiz::DataAnalyzer analyzer;
                auto corr = analyzer.ComputeCorrelationMatrix(*table);
                prepare_high_corr_pairs_.clear();
                for (size_t i = 0; i < corr.column_names.size(); i++) {
                    for (size_t j = i + 1; j < corr.column_names.size(); j++) {
                        float val = (float)std::abs(corr.Get(i, j));
                        if (val >= prepare_selection_threshold_) {
                            std::string pair = corr.column_names[i] + " <-> " + corr.column_names[j];
                            prepare_high_corr_pairs_.emplace_back(pair, val);
                        }
                    }
                }
                std::sort(prepare_high_corr_pairs_.begin(), prepare_high_corr_pairs_.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });
                prepare_correlation_computed_ = true;
            }
        }

        if (prepare_correlation_computed_ && prepare_selection_method_ == 0) {
            ImGui::Spacing();
            if (prepare_high_corr_pairs_.empty()) {
                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f),
                                   ICON_FA_CHECK " No highly correlated pairs found.");
            } else {
                ImGui::Text("%d pairs above threshold %.2f:",
                            (int)prepare_high_corr_pairs_.size(), prepare_selection_threshold_);
                if (ImGui::BeginTable("##CorrPairs", 2,
                                      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Feature Pair", 0, 3.0f);
                    ImGui::TableSetupColumn("Correlation", 0, 1.0f);
                    ImGui::TableHeadersRow();
                    int show = std::min((int)prepare_high_corr_pairs_.size(), 20);
                    for (int i = 0; i < show; i++) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", prepare_high_corr_pairs_[i].first.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", prepare_high_corr_pairs_[i].second);
                    }
                    ImGui::EndTable();
                }
            }
        }

        if (!table) {
            ImGui::TextDisabled("Load dataset in Table Viewer for feature selection.");
        }

        ImGui::Unindent(8);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 6. Train/Val/Test Split
    RenderSplitConfigSection();
}

void TrainingEvaluationPanel::RenderImagePrepareContent() {
    // 1. Image Preprocessing
    if (ImGui::CollapsingHeader(ICON_FA_IMAGE " Image Preprocessing", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);
        RenderImagePreprocessingSection();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 2. Quality Analysis
    if (ImGui::CollapsingHeader(ICON_FA_MAGNIFYING_GLASS " Quality Analysis")) {
        ImGui::Indent(8);
        RenderQualityAnalysisSection();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 3. Normalization
    if (ImGui::CollapsingHeader(ICON_FA_SLIDERS " Normalization")) {
        ImGui::Indent(8);
        RenderNormalizationSection();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();

    // 4. Duplicate Detection
    if (ImGui::CollapsingHeader(ICON_FA_COPY " Duplicate Detection")) {
        ImGui::Indent(8);
        RenderDuplicateDetectionSection();
        ImGui::Unindent(8);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 5. Train/Val/Test Split
    RenderSplitConfigSection();
}

void TrainingEvaluationPanel::RenderSplitConfigSection() {
    if (ImGui::CollapsingHeader(ICON_FA_SCISSORS " Train / Validation / Test Split", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(8);
        RenderSplitConfiguration();
        ImGui::Unindent(8);
    }
}

} // namespace gui
