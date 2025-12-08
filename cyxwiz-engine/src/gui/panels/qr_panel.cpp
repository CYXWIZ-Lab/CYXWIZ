#include "qr_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

QRPanel::QRPanel() {
    ResizeMatrix(rows_, cols_);
    // Initialize with sample data
    matrix_[0] = {12, -51, 4};
    matrix_[1] = {6, 167, -68};
    matrix_[2] = {-4, 24, -41};
    matrix_[3] = {-1, 1, 0};
}

QRPanel::~QRPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void QRPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(750, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_TABLE_COLUMNS " QR Decomposition###QRPanel", &visible_)) {
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

void QRPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute QR")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
    }
}

void QRPanel::RenderMatrixInput() {
    ImGui::Text(ICON_FA_TABLE_CELLS " Input Matrix A");
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
    ImGui::TextDisabled("QR: A = Q * R");
    ImGui::TextDisabled("Q: orthogonal (m x m)");
    ImGui::TextDisabled("R: upper triangular");
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
                ImGui::SetNextItemWidth(45);
                ImGui::InputDouble("##val", &matrix_[i][j], 0, 0, "%.1f");
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
}

void QRPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing QR decomposition...", ICON_FA_SPINNER);
}

void QRPanel::RenderResults() {
    ImGui::Text(ICON_FA_SQUARE_CHECK " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click Compute to perform QR decomposition");
        return;
    }

    if (ImGui::BeginTabBar("QRTabs")) {
        if (ImGui::BeginTabItem("Q (Orthogonal)")) {
            RenderMatrixQ();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("R (Upper Tri.)")) {
            RenderMatrixR();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHECK " Verify")) {
            RenderVerification();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void QRPanel::RenderMatrixQ() {
    ImGui::Text("Q Matrix (%zu x %zu) - Orthogonal:", result_.Q.size(),
                result_.Q.empty() ? 0 : result_.Q[0].size());
    ImGui::Spacing();

    if (result_.Q.empty()) {
        ImGui::TextDisabled("Q matrix not computed");
        return;
    }

    int rows = static_cast<int>(result_.Q.size());
    int cols = static_cast<int>(result_.Q[0].size());

    if (ImGui::BeginTable("MatrixQ", cols + 1,
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
                double val = result_.Q[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;
                ImGui::Text("%.4f", val);
            }
        }
        ImGui::EndTable();
    }

    // Check orthogonality
    bool is_orthogonal = LinearAlgebra::IsOrthogonal(result_.Q);
    if (is_orthogonal) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), ICON_FA_CHECK " Q is orthogonal (Q^T * Q = I)");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Q orthogonality check failed");
    }
}

void QRPanel::RenderMatrixR() {
    ImGui::Text("R Matrix (%zu x %zu) - Upper Triangular:", result_.R.size(),
                result_.R.empty() ? 0 : result_.R[0].size());
    ImGui::Spacing();

    if (result_.R.empty()) {
        ImGui::TextDisabled("R matrix not computed");
        return;
    }

    int rows = static_cast<int>(result_.R.size());
    int cols = static_cast<int>(result_.R[0].size());

    if (ImGui::BeginTable("MatrixR", cols + 1,
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
                double val = result_.R[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;

                // Color lower triangular zeros
                if (i > j && std::abs(val) < 1e-10) {
                    ImGui::TextDisabled("0");
                } else {
                    ImGui::Text("%.4f", val);
                }
            }
        }
        ImGui::EndTable();
    }
}

void QRPanel::RenderVerification() {
    ImGui::Text("Verification: A = Q * R");
    ImGui::Spacing();

    // Compute Q * R
    auto qr_product = LinearAlgebra::Multiply(result_.Q, result_.R);

    if (!qr_product.success) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Could not verify: %s", qr_product.error_message.c_str());
        return;
    }

    // Show Q * R matrix
    ImGui::Text("Q * R:");
    if (ImGui::BeginTable("QRProduct", qr_product.cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX,
                          ImVec2(0, 150))) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < qr_product.cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", j + 1);
        }

        for (int i = 0; i < qr_product.rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < qr_product.cols; j++) {
                ImGui::TableNextColumn();
                double val = qr_product.matrix[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;
                ImGui::Text("%.3f", val);
            }
        }
        ImGui::EndTable();
    }

    // Compute reconstruction error
    double max_error = 0;
    for (int i = 0; i < rows_ && i < qr_product.rows; i++) {
        for (int j = 0; j < cols_ && j < qr_product.cols; j++) {
            double err = std::abs(matrix_[i][j] - qr_product.matrix[i][j]);
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

void QRPanel::ResizeMatrix(int rows, int cols) {
    matrix_.resize(rows);
    for (auto& row : matrix_) {
        row.resize(cols, 0.0);
    }
}

void QRPanel::ComputeAsync() {
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
            result_ = LinearAlgebra::QR(matrix_);
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
