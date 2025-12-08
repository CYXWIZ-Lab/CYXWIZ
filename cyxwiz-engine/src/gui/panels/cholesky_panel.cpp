#include "cholesky_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

CholeskyPanel::CholeskyPanel() {
    ResizeMatrix(size_);
    // Initialize with a positive definite matrix
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    matrix_[0] = {4, 12, -16};
    matrix_[1] = {12, 37, -43};
    matrix_[2] = {-16, -43, 98};
}

CholeskyPanel::~CholeskyPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void CholeskyPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_SQUARE_ROOT_VARIABLE " Cholesky Decomposition###CholeskyPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.4f, 0), true);
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

void CholeskyPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_WAND_MAGIC_SPARKLES " Make Pos. Def.")) {
        MakePositiveDefinite();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Convert to positive definite: A = A^T * A + I");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
    }
}

void CholeskyPanel::RenderMatrixInput() {
    ImGui::Text(ICON_FA_TABLE_CELLS " Input Matrix A");
    ImGui::Separator();

    ImGui::TextDisabled("Must be symmetric positive definite");
    ImGui::Spacing();

    ImGui::Text("Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##size", &size_, 1, 1)) {
        size_ = std::clamp(size_, 2, 8);
        ResizeMatrix(size_);
        has_result_ = false;
    }

    ImGui::Spacing();

    // Check symmetry and positive definiteness
    bool is_symmetric = LinearAlgebra::IsSymmetric(matrix_);
    bool is_pos_def = LinearAlgebra::IsPositiveDefinite(matrix_);

    if (is_symmetric) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_CHECK " Symmetric");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), ICON_FA_XMARK " Not symmetric");
    }
    ImGui::SameLine();
    if (is_pos_def) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_CHECK " Pos. Def.");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), ICON_FA_XMARK " Not pos. def.");
    }

    ImGui::Spacing();

    // Matrix grid
    if (ImGui::BeginTable("MatrixInput", size_ + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < size_; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("c%d", j + 1);
        }

        for (int i = 0; i < size_; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("r%d", i + 1);

            for (int j = 0; j < size_; j++) {
                ImGui::TableNextColumn();
                ImGui::PushID(i * size_ + j);
                ImGui::SetNextItemWidth(50);
                if (ImGui::InputDouble("##val", &matrix_[i][j], 0, 0, "%.2f")) {
                    has_result_ = false;
                }
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_ROTATE " Symmetrize")) {
        // Make symmetric: A = (A + A^T) / 2
        for (int i = 0; i < size_; i++) {
            for (int j = i + 1; j < size_; j++) {
                double avg = (matrix_[i][j] + matrix_[j][i]) / 2.0;
                matrix_[i][j] = avg;
                matrix_[j][i] = avg;
            }
        }
        has_result_ = false;
    }
}

void CholeskyPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing Cholesky decomposition...", ICON_FA_SPINNER);
}

void CholeskyPanel::RenderResults() {
    ImGui::Text(ICON_FA_SQUARE_CHECK " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click Compute to perform Cholesky: A = L * L^T");
        return;
    }

    // Status
    if (result_.is_positive_definite) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_CHECK " Matrix is positive definite");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), ICON_FA_XMARK " Matrix is NOT positive definite");
    }

    ImGui::Spacing();

    if (ImGui::BeginTabBar("CholeskyTabs")) {
        if (ImGui::BeginTabItem("L (Lower Tri.)")) {
            RenderMatrixL();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHECK " Verify")) {
            RenderVerification();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void CholeskyPanel::RenderMatrixL() {
    ImGui::Text("L Matrix (%d x %d) - Lower Triangular:", result_.n, result_.n);
    ImGui::TextDisabled("A = L * L^T");
    ImGui::Spacing();

    if (result_.L.empty()) {
        ImGui::TextDisabled("L matrix not computed");
        return;
    }

    int n = result_.n;

    if (ImGui::BeginTable("MatrixL", n + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 200))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < n; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", j + 1);
        }

        for (int i = 0; i < n; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < n; j++) {
                ImGui::TableNextColumn();
                double val = result_.L[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;

                // Color upper triangular zeros
                if (j > i && std::abs(val) < 1e-10) {
                    ImGui::TextDisabled("0");
                } else {
                    ImGui::Text("%.4f", val);
                }
            }
        }
        ImGui::EndTable();
    }
}

void CholeskyPanel::RenderVerification() {
    ImGui::Text("Verification: A = L * L^T");
    ImGui::Spacing();

    if (result_.L.empty()) {
        ImGui::TextDisabled("Cannot verify - no L matrix");
        return;
    }

    // Compute L^T
    auto Lt = LinearAlgebra::Transpose(result_.L);
    if (!Lt.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Could not compute L^T");
        return;
    }

    // Compute L * L^T
    auto LLt = LinearAlgebra::Multiply(result_.L, Lt.matrix);
    if (!LLt.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Could not compute L * L^T");
        return;
    }

    // Show L * L^T matrix
    ImGui::Text("L * L^T:");
    if (ImGui::BeginTable("LLtProduct", LLt.cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX,
                          ImVec2(0, 150))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < LLt.cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", j + 1);
        }

        for (int i = 0; i < LLt.rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < LLt.cols; j++) {
                ImGui::TableNextColumn();
                double val = LLt.matrix[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;
                ImGui::Text("%.3f", val);
            }
        }
        ImGui::EndTable();
    }

    // Compute reconstruction error
    double max_error = 0;
    for (int i = 0; i < size_ && i < LLt.rows; i++) {
        for (int j = 0; j < size_ && j < LLt.cols; j++) {
            double err = std::abs(matrix_[i][j] - LLt.matrix[i][j]);
            max_error = std::max(max_error, err);
        }
    }

    ImGui::Spacing();
    if (max_error < 1e-6) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_CHECK " Verified: max error = %.2e", max_error);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Max error = %.6f", max_error);
    }
}

void CholeskyPanel::ResizeMatrix(int n) {
    matrix_.resize(n);
    for (auto& row : matrix_) {
        row.resize(n, 0.0);
    }
}

void CholeskyPanel::MakePositiveDefinite() {
    // Create A = A^T * A + epsilon * I (guaranteed positive definite)
    auto At = LinearAlgebra::Transpose(matrix_);
    if (!At.success) return;

    auto AtA = LinearAlgebra::Multiply(At.matrix, matrix_);
    if (!AtA.success) return;

    // Add small diagonal for numerical stability
    for (int i = 0; i < size_; i++) {
        AtA.matrix[i][i] += 0.1;
    }

    matrix_ = AtA.matrix;
    has_result_ = false;
}

void CholeskyPanel::ComputeAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = LinearAlgebra::Cholesky(matrix_);
            if (result_.success) {
                has_result_ = true;
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
