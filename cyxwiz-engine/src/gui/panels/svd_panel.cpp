#include "svd_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cyxwiz {

SVDPanel::SVDPanel() {
    ResizeMatrix(rows_, cols_);
    // Initialize with sample data
    matrix_[0] = {1, 2, 3};
    matrix_[1] = {4, 5, 6};
    matrix_[2] = {7, 8, 9};
    matrix_[3] = {10, 11, 12};
}

SVDPanel::~SVDPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void SVDPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_PIE " SVD (Singular Value Decomposition)###SVDPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.35f, 0), true);
            RenderMatrixInput();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void SVDPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute SVD")) {
        ComputeAsync();
    }

    ImGui::SameLine();
    ImGui::Checkbox("Full matrices", &full_matrices_);

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        has_approx_ = false;
        error_message_.clear();
    }
}

void SVDPanel::RenderMatrixInput() {
    ImGui::Text(ICON_FA_TABLE_CELLS " Input Matrix");
    ImGui::Separator();

    ImGui::Text("Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    if (ImGui::InputInt("##rows", &rows_, 0, 0)) {
        rows_ = std::clamp(rows_, 2, 10);
        ResizeMatrix(rows_, cols_);
        has_result_ = false;
    }
    ImGui::SameLine();
    ImGui::Text("x");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(50);
    if (ImGui::InputInt("##cols", &cols_, 0, 0)) {
        cols_ = std::clamp(cols_, 2, 10);
        ResizeMatrix(rows_, cols_);
        has_result_ = false;
    }

    ImGui::Spacing();

    if (ImGui::BeginTable("MatrixInput", cols_ + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < cols_; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("c%d", j + 1);
        }

        for (int i = 0; i < rows_; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("r%d", i + 1);

            for (int j = 0; j < cols_; j++) {
                ImGui::TableNextColumn();
                ImGui::PushID(i * cols_ + j);
                ImGui::SetNextItemWidth(40);
                ImGui::InputDouble("##val", &matrix_[i][j], 0, 0, "%.1f");
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

void SVDPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing SVD...", ICON_FA_SPINNER);
}

void SVDPanel::RenderResults() {
    ImGui::Text(ICON_FA_SQUARE_CHECK " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click Compute to perform SVD: A = U * S * V^T");
        return;
    }

    if (ImGui::BeginTabBar("SVDTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Singular Values")) {
            RenderSingularValues();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("U Matrix")) {
            RenderMatrixU();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("V^T Matrix")) {
            RenderMatrixVt();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_COMPRESS " Low-Rank")) {
            RenderLowRankApprox();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void SVDPanel::RenderSingularValues() {
    ImGui::Text("Singular Values (sigma):");
    ImGui::Spacing();

    // Bar plot of singular values
    if (ImPlot::BeginPlot("##SingularValues", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Index", "Value");

        std::vector<double> indices(result_.S.size());
        std::iota(indices.begin(), indices.end(), 1);

        ImPlot::PlotBars("Singular Values", indices.data(), result_.S.data(), static_cast<int>(result_.S.size()), 0.7);
        ImPlot::EndPlot();
    }

    // Table of values
    if (ImGui::BeginTable("SingularValuesTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("% Variance", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        // Compute total variance (sum of squared singular values)
        double total_var = 0;
        for (double s : result_.S) {
            total_var += s * s;
        }

        double cumulative = 0;
        for (size_t i = 0; i < result_.S.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.S[i]);
            ImGui::TableNextColumn();
            double var_pct = (result_.S[i] * result_.S[i] / total_var) * 100.0;
            cumulative += var_pct;
            ImGui::Text("%.2f%% (cum: %.2f%%)", var_pct, cumulative);
        }
        ImGui::EndTable();
    }
}

void SVDPanel::RenderMatrixU() {
    ImGui::Text("U Matrix (%zu x %zu):", result_.U.size(),
                result_.U.empty() ? 0 : result_.U[0].size());

    if (result_.U.empty()) {
        ImGui::TextDisabled("U matrix not computed");
        return;
    }

    int rows = static_cast<int>(result_.U.size());
    int cols = static_cast<int>(result_.U[0].size());

    if (ImGui::BeginTable("MatrixU", cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 200))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", j + 1);
        }

        for (int i = 0; i < rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < cols; j++) {
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", result_.U[i][j]);
            }
        }
        ImGui::EndTable();
    }
}

void SVDPanel::RenderMatrixVt() {
    ImGui::Text("V^T Matrix (%zu x %zu):", result_.Vt.size(),
                result_.Vt.empty() ? 0 : result_.Vt[0].size());

    if (result_.Vt.empty()) {
        ImGui::TextDisabled("V^T matrix not computed");
        return;
    }

    int rows = static_cast<int>(result_.Vt.size());
    int cols = static_cast<int>(result_.Vt[0].size());

    if (ImGui::BeginTable("MatrixVt", cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 200))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", j + 1);
        }

        for (int i = 0; i < rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < cols; j++) {
                ImGui::TableNextColumn();
                ImGui::Text("%.4f", result_.Vt[i][j]);
            }
        }
        ImGui::EndTable();
    }
}

void SVDPanel::RenderLowRankApprox() {
    ImGui::Text("Low-Rank Approximation:");
    ImGui::Spacing();

    int max_rank = static_cast<int>(result_.S.size());
    ImGui::Text("Rank (k):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("##rank", &approx_rank_, 1, max_rank)) {
        has_approx_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button("Compute Approximation")) {
        approx_matrix_ = LinearAlgebra::LowRankApproximation(matrix_, approx_rank_);
        has_approx_ = approx_matrix_.success;
    }

    if (has_approx_) {
        ImGui::Spacing();
        ImGui::Text("Approximated Matrix (rank %d):", approx_rank_);

        if (ImGui::BeginTable("ApproxMatrix", approx_matrix_.cols + 1,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX,
                              ImVec2(0, 150))) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled(" ");
            for (int j = 0; j < approx_matrix_.cols; j++) {
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%d", j + 1);
            }

            for (int i = 0; i < approx_matrix_.rows; i++) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%d", i + 1);

                for (int j = 0; j < approx_matrix_.cols; j++) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%.3f", approx_matrix_.matrix[i][j]);
                }
            }
            ImGui::EndTable();
        }

        // Compute approximation error
        double frobenius_error = 0;
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                double diff = matrix_[i][j] - approx_matrix_.matrix[i][j];
                frobenius_error += diff * diff;
            }
        }
        frobenius_error = std::sqrt(frobenius_error);

        ImGui::Text("Frobenius norm error: %.6f", frobenius_error);
    }
}

void SVDPanel::ResizeMatrix(int rows, int cols) {
    matrix_.resize(rows);
    for (auto& row : matrix_) {
        row.resize(cols, 0.0);
    }
}

void SVDPanel::ComputeAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    has_approx_ = false;
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = LinearAlgebra::SVD(matrix_, full_matrices_);
            if (result_.success) {
                has_result_ = true;
                approx_rank_ = std::min(approx_rank_, static_cast<int>(result_.S.size()));
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

} // namespace cyxwiz
