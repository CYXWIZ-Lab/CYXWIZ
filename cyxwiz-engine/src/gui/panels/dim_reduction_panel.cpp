#include "dim_reduction_panel.h"
#include "../../data/data_table.h"
#include "../../core/data_analyzer.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>

namespace cyxwiz {

DimReductionPanel::DimReductionPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
}

DimReductionPanel::~DimReductionPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void DimReductionPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 750), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_SIMPLE " Dimensionality Reduction###DimReduction", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderDataSelector();
            ImGui::Spacing();
            RenderAlgorithmConfig();

            if (has_result_) {
                ImGui::Separator();
                RenderResults();
            }
        }
    }
    ImGui::End();
}

void DimReductionPanel::RenderToolbar() {
    if (!has_result_) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportEmbeddings");
    }

    if (!has_result_) ImGui::EndDisabled();

    RenderExportOptions();
}

void DimReductionPanel::RenderDataSelector() {
    auto& registry = DataTableRegistry::Instance();
    auto table_names = registry.GetTableNames();

    ImGui::Text("%s Data Selection", ICON_FA_DATABASE);
    ImGui::Spacing();

    // Table selector
    ImGui::Text("Table:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##TableSelect", selected_table_.empty() ?
                          "Select table..." : selected_table_.c_str())) {
        for (const auto& name : table_names) {
            bool is_selected = (name == selected_table_);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                selected_table_ = name;
                current_table_ = registry.GetTable(name);
                selected_columns_.clear();
                label_column_ = -1;
                has_result_ = false;

                // Find numeric columns
                numeric_columns_.clear();
                if (current_table_) {
                    const auto& headers = current_table_->GetHeaders();
                    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
                        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
                        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
                            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
                        }
                    }
                    // Default: select all numeric columns
                    for (size_t i = 0; i < numeric_columns_.size(); i++) {
                        selected_columns_.push_back(static_cast<int>(i));
                    }
                }
            }
        }
        ImGui::EndCombo();
    }

    if (!current_table_) {
        ImGui::TextDisabled("Select a data table to continue");
        return;
    }

    // Feature columns multi-select
    ImGui::Text("Features:");
    ImGui::SameLine();

    std::string preview = std::to_string(selected_columns_.size()) + " columns selected";
    ImGui::SetNextItemWidth(200);
    if (ImGui::BeginCombo("##Features", preview.c_str())) {
        for (size_t i = 0; i < numeric_columns_.size(); i++) {
            bool is_selected = std::find(selected_columns_.begin(), selected_columns_.end(),
                                        static_cast<int>(i)) != selected_columns_.end();
            if (ImGui::Checkbox(numeric_columns_[i].c_str(), &is_selected)) {
                if (is_selected) {
                    selected_columns_.push_back(static_cast<int>(i));
                } else {
                    selected_columns_.erase(
                        std::remove(selected_columns_.begin(), selected_columns_.end(), static_cast<int>(i)),
                        selected_columns_.end());
                }
                has_result_ = false;
            }
        }
        ImGui::EndCombo();
    }

    // Label column for coloring
    ImGui::Text("Label (color):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);

    const auto& headers = current_table_->GetHeaders();
    std::string label_preview = label_column_ >= 0 && label_column_ < static_cast<int>(headers.size())
        ? headers[label_column_] : "None";

    if (ImGui::BeginCombo("##LabelCol", label_preview.c_str())) {
        if (ImGui::Selectable("None", label_column_ < 0)) {
            label_column_ = -1;
        }
        for (size_t i = 0; i < headers.size(); i++) {
            if (ImGui::Selectable(headers[i].c_str(), label_column_ == static_cast<int>(i))) {
                label_column_ = static_cast<int>(i);
            }
        }
        ImGui::EndCombo();
    }
}

void DimReductionPanel::RenderAlgorithmConfig() {
    ImGui::Text("%s Algorithm Configuration", ICON_FA_GEAR);
    ImGui::Spacing();

    // Algorithm selector
    const char* algorithms[] = {"PCA", "t-SNE", "UMAP"};
    ImGui::Text("Algorithm:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##Algorithm", &algorithm_, algorithms, IM_ARRAYSIZE(algorithms))) {
        has_result_ = false;
    }

    ImGui::Spacing();

    // Algorithm-specific config
    switch (algorithm_) {
        case 0: RenderPCAConfig(); break;
        case 1: RenderTSNEConfig(); break;
        case 2: RenderUMAPConfig(); break;
    }

    ImGui::Spacing();

    // Run button
    bool can_run = current_table_ && selected_columns_.size() >= 2;
    if (!can_run) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_PLAY " Run Reduction", ImVec2(150, 0))) {
        RunReduction();
    }

    if (!can_run) ImGui::EndDisabled();

    if (selected_columns_.size() < 2) {
        ImGui::SameLine();
        ImGui::TextDisabled("Select at least 2 features");
    }
}

void DimReductionPanel::RenderPCAConfig() {
    ImGui::Text("Components:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##PCAComponents", &pca_n_components_, 2, 3)) {
        has_result_ = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(2 or 3 for visualization)");
}

void DimReductionPanel::RenderTSNEConfig() {
    ImGui::Text("Dimensions:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##TSNEDims", &tsne_n_dims_, 2, 3)) {
        has_result_ = false;
    }

    ImGui::Text("Perplexity:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##Perplexity", &tsne_perplexity_, 5, 100)) {
        has_result_ = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(5-50 typical)");

    ImGui::Text("Iterations:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##Iterations", &tsne_iterations_, 250, 2000)) {
        has_result_ = false;
    }

    ImGui::Text("Learning Rate:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    double lr = tsne_learning_rate_;
    if (ImGui::InputDouble("##LR", &lr, 10.0, 50.0, "%.1f")) {
        tsne_learning_rate_ = std::max(10.0, std::min(1000.0, lr));
        has_result_ = false;
    }
}

void DimReductionPanel::RenderUMAPConfig() {
    ImGui::Text("Dimensions:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##UMAPDims", &umap_n_dims_, 2, 3)) {
        has_result_ = false;
    }

    ImGui::Text("Neighbors:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##Neighbors", &umap_n_neighbors_, 5, 50)) {
        has_result_ = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(15 typical)");

    ImGui::Text("Min Distance:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    double dist = umap_min_dist_;
    if (ImGui::InputDouble("##MinDist", &dist, 0.01, 0.1, "%.2f")) {
        umap_min_dist_ = std::max(0.01, std::min(1.0, dist));
        has_result_ = false;
    }
}

void DimReductionPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing dimensionality reduction...", ICON_FA_SPINNER);

    if (algorithm_ == 1) {
        // t-SNE progress
        int iter = progress_iteration_.load();
        double kl = progress_kl_.load();
        ImGui::Text("Iteration: %d / %d", iter, tsne_iterations_);
        ImGui::Text("KL Divergence: %.4f", kl);

        float progress = static_cast<float>(iter) / tsne_iterations_;
        ImGui::ProgressBar(progress, ImVec2(-1, 0));
    }
}

void DimReductionPanel::RenderResults() {
    if (ImGui::BeginTabBar("DimRedTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Scatter Plot")) {
            RenderScatterPlot();
            ImGui::EndTabItem();
        }

        if (last_algorithm_ == 0 && ImGui::BeginTabItem(ICON_FA_CHART_BAR " Variance")) {
            RenderExplainedVarianceChart();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_TABLE " Embeddings")) {
            RenderEmbeddingsTable();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void DimReductionPanel::RenderScatterPlot() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    // Get embeddings based on algorithm
    const std::vector<std::vector<double>>* embeddings = nullptr;

    if (last_algorithm_ == 0) embeddings = &pca_result_.transformed;
    else if (last_algorithm_ == 1) embeddings = &tsne_result_.embeddings;
    else if (last_algorithm_ == 2) embeddings = &umap_result_.embeddings;

    if (!embeddings || embeddings->empty()) {
        ImGui::TextDisabled("No embeddings to display");
        return;
    }

    // Prepare data for plotting
    std::vector<double> x_vals, y_vals;
    for (const auto& point : *embeddings) {
        if (point.size() >= 2) {
            x_vals.push_back(point[0]);
            y_vals.push_back(point[1]);
        }
    }

    if (x_vals.empty()) {
        ImGui::TextDisabled("No valid points to plot");
        return;
    }

    // Create scatter plot
    if (ImPlot::BeginPlot("2D Projection", ImVec2(-1, 450))) {
        ImPlot::SetupAxes("Dimension 1", "Dimension 2");

        if (labels_.empty() || label_column_ < 0) {
            // Single color
            ImPlot::PlotScatter("Data", x_vals.data(), y_vals.data(),
                               static_cast<int>(x_vals.size()));
        } else {
            // Color by label - group points by label
            std::map<std::string, std::vector<std::pair<double, double>>> label_groups;
            for (size_t i = 0; i < x_vals.size() && i < labels_.size(); i++) {
                label_groups[labels_[i]].push_back({x_vals[i], y_vals[i]});
            }

            for (const auto& [label, points] : label_groups) {
                std::vector<double> gx, gy;
                for (const auto& p : points) {
                    gx.push_back(p.first);
                    gy.push_back(p.second);
                }
                ImPlot::PlotScatter(label.c_str(), gx.data(), gy.data(),
                                   static_cast<int>(gx.size()));
            }
        }

        ImPlot::EndPlot();
    }

    // Statistics
    ImGui::Text("Points: %zu", x_vals.size());
    if (last_algorithm_ == 0) {
        ImGui::SameLine();
        ImGui::Text("| Variance Explained: %.1f%%", pca_result_.total_variance_explained * 100);
    } else if (last_algorithm_ == 1) {
        ImGui::SameLine();
        ImGui::Text("| Final KL: %.4f", tsne_result_.final_kl_divergence);
    }
}

void DimReductionPanel::RenderExplainedVarianceChart() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    if (pca_result_.explained_variance_ratio.empty()) {
        ImGui::TextDisabled("No variance data available");
        return;
    }

    // Cumulative variance
    std::vector<double> cumulative(pca_result_.explained_variance_ratio.size());
    double sum = 0;
    for (size_t i = 0; i < pca_result_.explained_variance_ratio.size(); i++) {
        sum += pca_result_.explained_variance_ratio[i];
        cumulative[i] = sum;
    }

    // Component indices
    std::vector<double> indices(pca_result_.explained_variance_ratio.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = static_cast<double>(i + 1);
    }

    if (ImPlot::BeginPlot("Explained Variance", ImVec2(-1, 300))) {
        ImPlot::SetupAxes("Principal Component", "Variance Ratio");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.1);

        ImPlot::PlotBars("Individual", indices.data(),
                        pca_result_.explained_variance_ratio.data(),
                        static_cast<int>(indices.size()), 0.5);

        ImPlot::SetNextLineStyle(ImVec4(1, 0.5f, 0, 1), 2);
        ImPlot::PlotLine("Cumulative", indices.data(), cumulative.data(),
                        static_cast<int>(indices.size()));

        ImPlot::EndPlot();
    }

    // Table of variance
    if (ImGui::BeginTable("VarianceTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Component");
        ImGui::TableSetupColumn("Variance Ratio");
        ImGui::TableSetupColumn("Cumulative");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < pca_result_.explained_variance_ratio.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("PC%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f (%.1f%%)", pca_result_.explained_variance_ratio[i],
                       pca_result_.explained_variance_ratio[i] * 100);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f (%.1f%%)", cumulative[i], cumulative[i] * 100);
        }

        ImGui::EndTable();
    }
}

void DimReductionPanel::RenderEmbeddingsTable() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    const std::vector<std::vector<double>>* embeddings = nullptr;
    if (last_algorithm_ == 0) embeddings = &pca_result_.transformed;
    else if (last_algorithm_ == 1) embeddings = &tsne_result_.embeddings;
    else if (last_algorithm_ == 2) embeddings = &umap_result_.embeddings;

    if (!embeddings || embeddings->empty()) {
        ImGui::TextDisabled("No embeddings available");
        return;
    }

    int n_dims = embeddings->front().size();
    int n_cols = n_dims + (labels_.empty() ? 0 : 1);

    if (ImGui::BeginTable("EmbeddingsTable", n_cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY, ImVec2(0, 300))) {

        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 60);
        for (int d = 0; d < n_dims; d++) {
            char buf[16];
            snprintf(buf, sizeof(buf), "Dim %d", d + 1);
            ImGui::TableSetupColumn(buf);
        }
        if (!labels_.empty()) {
            ImGui::TableSetupColumn("Label");
        }
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(embeddings->size()));

        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                ImGui::TableNextRow();

                ImGui::TableNextColumn();
                ImGui::Text("%d", row);

                for (int d = 0; d < n_dims; d++) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", (*embeddings)[row][d]);
                }

                if (!labels_.empty() && row < static_cast<int>(labels_.size())) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", labels_[row].c_str());
                }
            }
        }

        ImGui::EndTable();
    }
}

void DimReductionPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportEmbeddings")) {
        ImGui::Text("Export Embeddings");
        ImGui::Separator();

        ImGui::InputText("File Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Save CSV")) {
            std::lock_guard<std::mutex> lock(result_mutex_);

            const std::vector<std::vector<double>>* embeddings = nullptr;
            if (last_algorithm_ == 0) embeddings = &pca_result_.transformed;
            else if (last_algorithm_ == 1) embeddings = &tsne_result_.embeddings;
            else if (last_algorithm_ == 2) embeddings = &umap_result_.embeddings;

            if (embeddings && !embeddings->empty()) {
                std::ofstream file(export_path_);
                if (file) {
                    int n_dims = embeddings->front().size();

                    // Header
                    for (int d = 0; d < n_dims; d++) {
                        file << "dim_" << (d + 1);
                        if (d < n_dims - 1 || !labels_.empty()) file << ",";
                    }
                    if (!labels_.empty()) file << "label";
                    file << "\n";

                    // Data
                    for (size_t i = 0; i < embeddings->size(); i++) {
                        for (int d = 0; d < n_dims; d++) {
                            file << (*embeddings)[i][d];
                            if (d < n_dims - 1 || !labels_.empty()) file << ",";
                        }
                        if (!labels_.empty() && i < labels_.size()) {
                            file << labels_[i];
                        }
                        file << "\n";
                    }

                    spdlog::info("Exported embeddings to: {}", export_path_);
                }
            }

            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void DimReductionPanel::AnalyzeTable(const std::string& table_name) {
    auto& registry = DataTableRegistry::Instance();
    auto table = registry.GetTable(table_name);
    if (table) {
        selected_table_ = table_name;
        AnalyzeTable(table);
    }
}

void DimReductionPanel::AnalyzeTable(std::shared_ptr<DataTable> table) {
    if (!table) return;
    current_table_ = table;
    selected_columns_.clear();
    label_column_ = -1;
    has_result_ = false;

    numeric_columns_.clear();
    const auto& headers = table->GetHeaders();
    for (size_t i = 0; i < table->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*table, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            numeric_columns_.push_back(i < headers.size() ? headers[i] : "Column " + std::to_string(i));
            selected_columns_.push_back(static_cast<int>(numeric_columns_.size()) - 1);
        }
    }
}

void DimReductionPanel::PrepareData() {
    if (!current_table_ || selected_columns_.empty()) return;

    input_data_.clear();
    labels_.clear();

    // Get numeric column indices
    std::vector<size_t> col_indices;
    int count = 0;
    for (size_t i = 0; i < current_table_->GetColumnCount(); i++) {
        auto dtype = DataAnalyzer::DetectColumnType(*current_table_, i);
        if (dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer) {
            if (std::find(selected_columns_.begin(), selected_columns_.end(), count) != selected_columns_.end()) {
                col_indices.push_back(i);
            }
            count++;
        }
    }

    // Extract data
    size_t n_rows = current_table_->GetRowCount();
    input_data_.resize(n_rows);

    for (size_t row = 0; row < n_rows; row++) {
        input_data_[row].resize(col_indices.size());
        for (size_t j = 0; j < col_indices.size(); j++) {
            auto val = DataAnalyzer::ToDouble(current_table_->GetCell(row, col_indices[j]));
            input_data_[row][j] = val.value_or(0.0);
        }
    }

    // Extract labels if specified
    if (label_column_ >= 0 && label_column_ < static_cast<int>(current_table_->GetColumnCount())) {
        labels_.resize(n_rows);
        for (size_t row = 0; row < n_rows; row++) {
            labels_[row] = current_table_->GetCellAsString(row, label_column_);
        }
    }
}

void DimReductionPanel::RunReduction() {
    if (is_computing_.load() || !current_table_ || selected_columns_.size() < 2) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    // Prepare data on main thread
    PrepareData();

    if (input_data_.empty()) {
        spdlog::error("No data to reduce");
        return;
    }

    is_computing_ = true;
    progress_iteration_ = 0;
    progress_kl_ = 0.0;

    int algo = algorithm_;
    auto data = input_data_;

    // Algorithm parameters
    int pca_comp = pca_n_components_;
    int tsne_dims = tsne_n_dims_;
    int tsne_perp = tsne_perplexity_;
    int tsne_iter = tsne_iterations_;
    double tsne_lr = tsne_learning_rate_;
    int umap_dims = umap_n_dims_;
    int umap_neigh = umap_n_neighbors_;
    double umap_dist = umap_min_dist_;

    compute_thread_ = std::make_unique<std::thread>([this, algo, data,
                                                      pca_comp, tsne_dims, tsne_perp, tsne_iter, tsne_lr,
                                                      umap_dims, umap_neigh, umap_dist]() {
        try {
            switch (algo) {
                case 0: {
                    // PCA
                    spdlog::info("Running PCA with {} components", pca_comp);
                    auto result = DimensionalityReduction::ComputePCA(data, pca_comp);

                    std::lock_guard<std::mutex> lock(result_mutex_);
                    pca_result_ = std::move(result);
                    last_algorithm_ = 0;
                    has_result_ = true;

                    spdlog::info("PCA complete: explained variance = {:.2f}%",
                                pca_result_.total_variance_explained * 100);
                    break;
                }

                case 1: {
                    // t-SNE
                    spdlog::info("Running t-SNE: dims={}, perplexity={}, iterations={}",
                                tsne_dims, tsne_perp, tsne_iter);

                    auto callback = [this](int iter, double kl) {
                        progress_iteration_ = iter;
                        progress_kl_ = kl;
                    };

                    auto result = DimensionalityReduction::ComputetSNE(
                        data, tsne_dims, tsne_perp, tsne_iter, tsne_lr, callback);

                    std::lock_guard<std::mutex> lock(result_mutex_);
                    tsne_result_ = std::move(result);
                    last_algorithm_ = 1;
                    has_result_ = true;

                    spdlog::info("t-SNE complete: final KL = {:.4f}", tsne_result_.final_kl_divergence);
                    break;
                }

                case 2: {
                    // UMAP
                    spdlog::info("Running UMAP: dims={}, neighbors={}, min_dist={}",
                                umap_dims, umap_neigh, umap_dist);

                    auto result = DimensionalityReduction::ComputeUMAP(
                        data, umap_dims, umap_neigh, umap_dist);

                    std::lock_guard<std::mutex> lock(result_mutex_);
                    umap_result_ = std::move(result);
                    last_algorithm_ = 2;
                    has_result_ = true;

                    spdlog::info("UMAP complete");
                    break;
                }
            }
        } catch (const std::exception& e) {
            spdlog::error("Dimensionality reduction error: {}", e.what());
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
