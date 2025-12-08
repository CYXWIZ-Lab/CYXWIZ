#include "cluster_eval_panel.h"
#include "../icons.h"
#include "../../core/data_registry.h"
#include "../../core/data_analyzer.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <set>

namespace cyxwiz {

ClusterEvalPanel::ClusterEvalPanel() : Panel("Cluster Evaluation", true) {}

ClusterEvalPanel::~ClusterEvalPanel() {
    if (compute_thread_.joinable()) compute_thread_.join();
}

void ClusterEvalPanel::Render() {
    if (!is_open_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin((std::string(ICON_FA_CHART_COLUMN) + " Cluster Evaluation###ClusterEvalPanel").c_str(), &is_open_)) {
        if (data_registry_) available_tables_ = data_registry_->GetTableNames();

        ImGui::BeginChild("ConfigPanel", ImVec2(280, 0), true);
        RenderDataSelector();
        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);

        if (ImGui::BeginTabBar("EvalTabs")) {
            if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Metrics")) {
                RenderMetrics();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem(ICON_FA_CHART_LINE " Silhouette Plot")) {
                RenderSilhouettePlot();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem(ICON_FA_TABLE " Comparison")) {
                RenderComparison();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        ImGui::EndChild();
    }
    ImGui::End();
}

void ClusterEvalPanel::RenderDataSelector() {
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
        ImGui::Text("Feature Columns:");
        ImGui::BeginChild("ColumnSelect", ImVec2(0, 100), true);
        for (size_t i = 0; i < column_names_.size(); ++i) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(column_names_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
            }
        }
        ImGui::EndChild();

        // Label column selector
        std::vector<const char*> label_options = {"None"};
        for (const auto& name : column_names_) label_options.push_back(name.c_str());
        int label_idx = label_column_idx_ + 1;
        if (ImGui::Combo("Label Column", &label_idx, label_options.data(), static_cast<int>(label_options.size()))) {
            label_column_idx_ = label_idx - 1;
        }
    }

    ImGui::Spacing();

    bool can_run = !is_computing_ && !input_data_.empty() && label_column_idx_ >= 0;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_CALCULATOR " Compute Metrics", ImVec2(-1, 30))) {
        ComputeMetrics();
    }

    if (!can_run) ImGui::EndDisabled();

    if (is_computing_) {
        ImGui::Text("Computing...");
    }

    if (!status_message_.empty()) {
        ImGui::TextWrapped("%s", status_message_.c_str());
    }

    // Save to history
    if (metrics_.success) {
        ImGui::Spacing();
        static char name_buf[64] = "";
        ImGui::InputText("Name", name_buf, sizeof(name_buf));

        if (ImGui::Button(ICON_FA_PLUS " Add to Comparison", ImVec2(-1, 25))) {
            EvalRecord record;
            record.name = strlen(name_buf) > 0 ? name_buf : ("Run " + std::to_string(eval_history_.size() + 1));
            record.n_clusters = metrics_.n_clusters;
            record.silhouette = metrics_.silhouette_score;
            record.davies_bouldin = metrics_.davies_bouldin_index;
            record.calinski_harabasz = metrics_.calinski_harabasz_score;
            eval_history_.push_back(record);
            name_buf[0] = '\0';
        }
    }
}

void ClusterEvalPanel::RenderMetrics() {
    if (!metrics_.success) {
        ImGui::Text("Load data with cluster labels and compute metrics.");
        return;
    }

    ImGui::Text("Cluster Quality Metrics");
    ImGui::Separator();
    ImGui::Spacing();

    // Metrics cards
    ImGui::BeginGroup();
    {
        ImGui::BeginChild("Silhouette", ImVec2(200, 100), true);
        ImGui::Text(ICON_FA_CHART_AREA " Silhouette Score");
        ImGui::Separator();

        float silhouette = static_cast<float>(metrics_.silhouette_score);
        ImVec4 color;
        if (silhouette > 0.5) color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
        else if (silhouette > 0.25) color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
        else color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);

        ImGui::TextColored(color, "%.4f", silhouette);
        ImGui::TextWrapped("Range: -1 to 1. Higher is better.");
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    ImGui::SameLine();

    ImGui::BeginGroup();
    {
        ImGui::BeginChild("DaviesBouldin", ImVec2(200, 100), true);
        ImGui::Text(ICON_FA_BULLSEYE " Davies-Bouldin");
        ImGui::Separator();

        float db = static_cast<float>(metrics_.davies_bouldin_index);
        ImVec4 color;
        if (db < 0.5) color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
        else if (db < 1.0) color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
        else color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);

        ImGui::TextColored(color, "%.4f", db);
        ImGui::TextWrapped("Lower is better.");
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    ImGui::SameLine();

    ImGui::BeginGroup();
    {
        ImGui::BeginChild("CalinskiHarabasz", ImVec2(200, 100), true);
        ImGui::Text(ICON_FA_CHART_COLUMN " Calinski-Harabasz");
        ImGui::Separator();

        float ch = static_cast<float>(metrics_.calinski_harabasz_score);
        ImGui::Text("%.2f", ch);
        ImGui::TextWrapped("Higher is better.");
        ImGui::EndChild();
    }
    ImGui::EndGroup();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Info
    ImGui::Text("Samples: %d", metrics_.n_samples);
    ImGui::Text("Clusters: %d", metrics_.n_clusters);

    // Interpretation
    ImGui::Spacing();
    ImGui::Text("Interpretation:");
    ImGui::BulletText("Silhouette > 0.5: Strong clustering structure");
    ImGui::BulletText("Silhouette 0.25-0.5: Reasonable structure");
    ImGui::BulletText("Silhouette < 0.25: Weak or artificial structure");
}

void ClusterEvalPanel::RenderSilhouettePlot() {
    if (!metrics_.success || metrics_.per_sample_silhouette.empty()) {
        ImGui::Text("Compute metrics to see silhouette plot.");

        // Show placeholder explanation
        ImGui::Spacing();
        ImGui::TextWrapped("The silhouette plot shows the silhouette coefficient for each sample, "
                          "grouped by cluster. A wider positive area indicates better clustering.");
        return;
    }

    // Sort samples by cluster and silhouette
    struct SampleInfo {
        int idx;
        int cluster;
        double silhouette;
    };

    std::vector<SampleInfo> samples;
    for (size_t i = 0; i < metrics_.per_sample_silhouette.size(); ++i) {
        samples.push_back({static_cast<int>(i), cluster_labels_[i], metrics_.per_sample_silhouette[i]});
    }

    std::sort(samples.begin(), samples.end(), [](const SampleInfo& a, const SampleInfo& b) {
        if (a.cluster != b.cluster) return a.cluster < b.cluster;
        return a.silhouette > b.silhouette;
    });

    // Prepare data for plotting
    std::vector<double> y_vals(samples.size());
    std::vector<double> silhouette_vals(samples.size());

    for (size_t i = 0; i < samples.size(); ++i) {
        y_vals[i] = static_cast<double>(i);
        silhouette_vals[i] = samples[i].silhouette;
    }

    if (ImPlot::BeginPlot("##SilhouettePlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Silhouette Coefficient", "Sample Index");
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.2, 1.0, ImGuiCond_FirstUseEver);

        ImPlot::PlotBars("Silhouette", silhouette_vals.data(), y_vals.data(),
                        static_cast<int>(samples.size()), 1.0, ImPlotBarsFlags_Horizontal);

        // Draw average line
        double avg = metrics_.silhouette_score;
        ImPlot::PlotInfLines("Average", &avg, 1);

        ImPlot::EndPlot();
    }
}

void ClusterEvalPanel::RenderComparison() {
    if (eval_history_.empty()) {
        ImGui::Text("Add evaluation results to compare different clustering runs.");
        return;
    }

    // Comparison table
    if (ImGui::BeginTable("ComparisonTable", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Clusters");
        ImGui::TableSetupColumn("Silhouette");
        ImGui::TableSetupColumn("Davies-Bouldin");
        ImGui::TableSetupColumn("Calinski-Harabasz");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < eval_history_.size(); ++i) {
            const auto& record = eval_history_[i];
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%s", record.name.c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%d", record.n_clusters);

            ImGui::TableNextColumn();
            ImVec4 color;
            if (record.silhouette > 0.5) color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            else if (record.silhouette > 0.25) color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
            else color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            ImGui::TextColored(color, "%.4f", record.silhouette);

            ImGui::TableNextColumn();
            ImGui::Text("%.4f", record.davies_bouldin);

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", record.calinski_harabasz);
        }

        ImGui::EndTable();
    }

    // Plot comparison
    if (eval_history_.size() >= 2) {
        ImGui::Spacing();

        if (ImPlot::BeginPlot("##ComparisonPlot", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("Run", "Silhouette Score");

            std::vector<double> x_pos(eval_history_.size());
            std::vector<double> silhouettes(eval_history_.size());
            for (size_t i = 0; i < eval_history_.size(); ++i) {
                x_pos[i] = static_cast<double>(i);
                silhouettes[i] = eval_history_[i].silhouette;
            }

            ImPlot::PlotBars("Silhouette", x_pos.data(), silhouettes.data(),
                            static_cast<int>(x_pos.size()), 0.6);

            ImPlot::EndPlot();
        }
    }

    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_TRASH " Clear History")) {
        eval_history_.clear();
    }
}

void ClusterEvalPanel::LoadSelectedData() {
    if (selected_table_idx_ < 0 || !data_registry_) return;

    current_table_ = data_registry_->GetTable(available_tables_[selected_table_idx_]);
    if (!current_table_) return;

    column_names_ = current_table_->GetHeaders();
    selected_columns_.assign(column_names_.size(), false);
    label_column_idx_ = -1;

    int selected_count = 0;
    for (size_t i = 0; i < column_names_.size() && selected_count < 4; ++i) {
        if (current_table_->GetRowCount() > 0) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(0, static_cast<int>(i)));
            if (val.has_value()) {
                selected_columns_[i] = true;
                selected_count++;
            }
        }
    }

    metrics_ = ClusterMetrics();
    input_data_.clear();
    cluster_labels_.clear();
    status_message_ = "Data loaded. Select label column.";
}

void ClusterEvalPanel::ComputeMetrics() {
    if (!current_table_ || is_computing_ || label_column_idx_ < 0) return;

    std::vector<int> col_indices;
    for (size_t i = 0; i < selected_columns_.size(); ++i) {
        if (selected_columns_[i] && static_cast<int>(i) != label_column_idx_) {
            col_indices.push_back(static_cast<int>(i));
        }
    }

    if (col_indices.empty()) {
        status_message_ = "Select at least 1 feature column.";
        return;
    }

    input_data_.clear();
    cluster_labels_.clear();

    int n_rows = static_cast<int>(current_table_->GetRowCount());
    for (int row = 0; row < n_rows; ++row) {
        std::vector<double> row_data;
        bool valid = true;

        for (int col : col_indices) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, col));
            if (val.has_value()) row_data.push_back(val.value());
            else { valid = false; break; }
        }

        // Get label
        auto label_val = DataAnalyzer::ToDouble(current_table_->GetCell(row, label_column_idx_));
        if (!label_val.has_value()) valid = false;

        if (valid) {
            input_data_.push_back(row_data);
            cluster_labels_.push_back(static_cast<int>(label_val.value()));
        }
    }

    if (input_data_.empty()) {
        status_message_ = "No valid data.";
        return;
    }

    is_computing_ = true;
    status_message_ = "Computing metrics...";

    if (compute_thread_.joinable()) compute_thread_.join();

    compute_thread_ = std::thread([this]() {
        metrics_ = Clustering::EvaluateClustering(input_data_, cluster_labels_);
        is_computing_ = false;

        if (metrics_.success) {
            status_message_ = "Metrics computed successfully.";
            current_name_ = available_tables_[selected_table_idx_];
        } else {
            status_message_ = "Failed: " + metrics_.error_message;
        }
    });
}

void ClusterEvalPanel::SetClusteringData(const std::vector<std::vector<double>>& data,
                                         const std::vector<int>& labels,
                                         const std::string& name) {
    input_data_ = data;
    cluster_labels_ = labels;
    current_name_ = name;

    // Automatically compute metrics
    if (!input_data_.empty() && !cluster_labels_.empty()) {
        metrics_ = Clustering::EvaluateClustering(input_data_, cluster_labels_);
        if (metrics_.success) {
            status_message_ = "Metrics computed for: " + name;
        }
    }
}

} // namespace cyxwiz
