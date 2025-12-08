#include "eigen_decomp_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

namespace cyxwiz {

EigenDecompPanel::EigenDecompPanel() {
    ResizeMatrix(size_);
    // Initialize with sample matrix
    matrix_[0] = {4, -2, 1};
    matrix_[1] = {-2, 4, -2};
    matrix_[2] = {1, -2, 4};
}

EigenDecompPanel::~EigenDecompPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void EigenDecompPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_LINE " Eigenvalue Decomposition###EigenDecomp", &visible_)) {
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

void EigenDecompPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
    }
}

void EigenDecompPanel::RenderMatrixInput() {
    ImGui::Text(ICON_FA_TABLE_CELLS " Input Matrix (Square)");
    ImGui::Separator();

    ImGui::Text("Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##size", &size_, 1, 1)) {
        size_ = std::clamp(size_, 2, 8);
        ResizeMatrix(size_);
        has_result_ = false;
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
                ImGui::SetNextItemWidth(45);
                ImGui::InputDouble("##val", &matrix_[i][j], 0, 0, "%.2f");
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_EYE " Identity")) {
        for (int i = 0; i < size_; i++) {
            for (int j = 0; j < size_; j++) {
                matrix_[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Symmetric")) {
        // Make symmetric: A = (A + A^T) / 2
        for (int i = 0; i < size_; i++) {
            for (int j = i + 1; j < size_; j++) {
                double avg = (matrix_[i][j] + matrix_[j][i]) / 2.0;
                matrix_[i][j] = avg;
                matrix_[j][i] = avg;
            }
        }
    }
}

void EigenDecompPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing eigendecomposition...", ICON_FA_SPINNER);
}

void EigenDecompPanel::RenderResults() {
    ImGui::Text(ICON_FA_SQUARE_CHECK " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Click Compute to find eigenvalues and eigenvectors");
        return;
    }

    if (ImGui::BeginTabBar("EigenTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_LIST " Eigenvalues")) {
            RenderEigenvalues();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Eigenvectors")) {
            RenderEigenvectors();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_SIMPLE " Spectrum")) {
            RenderEigenspectrum();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void EigenDecompPanel::RenderEigenvalues() {
    ImGui::Text("Eigenvalues (lambda):");
    ImGui::Spacing();

    if (ImGui::BeginTable("Eigenvalues", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Real", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Imaginary", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.eigenvalues.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", i + 1);
            ImGui::TableNextColumn();
            ImGui::Text("%.6f", result_.eigenvalues[i].real());
            ImGui::TableNextColumn();
            double imag = result_.eigenvalues[i].imag();
            if (std::abs(imag) > 1e-10) {
                ImGui::Text("%.6fi", imag);
            } else {
                ImGui::TextDisabled("0");
            }
        }
        ImGui::EndTable();
    }
}

void EigenDecompPanel::RenderEigenvectors() {
    ImGui::Text("Eigenvectors (columns):");
    ImGui::Spacing();

    if (result_.eigenvectors.empty()) {
        ImGui::TextDisabled("Eigenvectors not computed");
        return;
    }

    int n = result_.n;
    if (ImGui::BeginTable("Eigenvectors", n + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < n; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("v%d", j + 1);
        }

        for (int i = 0; i < n; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", i + 1);

            for (int j = 0; j < n; j++) {
                ImGui::TableNextColumn();
                auto& val = result_.eigenvectors[i][j];
                if (std::abs(val.imag()) > 1e-10) {
                    ImGui::Text("%.3f%+.3fi", val.real(), val.imag());
                } else {
                    ImGui::Text("%.4f", val.real());
                }
            }
        }
        ImGui::EndTable();
    }
}

void EigenDecompPanel::RenderEigenspectrum() {
    ImGui::Text("Eigenvalue Spectrum (Complex Plane):");

    if (ImPlot::BeginPlot("##EigenSpectrum", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Real", "Imaginary");

        std::vector<double> real_parts, imag_parts;
        for (const auto& ev : result_.eigenvalues) {
            real_parts.push_back(ev.real());
            imag_parts.push_back(ev.imag());
        }

        ImPlot::PlotScatter("Eigenvalues", real_parts.data(), imag_parts.data(), static_cast<int>(real_parts.size()));

        // Draw unit circle for reference
        std::vector<double> circle_x, circle_y;
        for (int i = 0; i <= 100; i++) {
            double theta = 2.0 * 3.14159265 * i / 100.0;
            circle_x.push_back(std::cos(theta));
            circle_y.push_back(std::sin(theta));
        }
        ImPlot::PlotLine("Unit Circle", circle_x.data(), circle_y.data(), static_cast<int>(circle_x.size()));

        ImPlot::EndPlot();
    }

    // Eigenvalue magnitudes
    ImGui::Spacing();
    ImGui::Text("Magnitudes:");
    for (size_t i = 0; i < result_.eigenvalues.size(); i++) {
        double mag = std::abs(result_.eigenvalues[i]);
        ImGui::BulletText("lambda_%zu: %.6f", i + 1, mag);
    }
}

void EigenDecompPanel::ResizeMatrix(int n) {
    matrix_.resize(n);
    for (auto& row : matrix_) {
        row.resize(n, 0.0);
    }
}

void EigenDecompPanel::ComputeAsync() {
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
            result_ = LinearAlgebra::Eigen(matrix_);
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
