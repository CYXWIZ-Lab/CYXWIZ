#include "matrix_calculator_panel.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

MatrixCalculatorPanel::MatrixCalculatorPanel() {
    std::memset(export_path_, 0, sizeof(export_path_));
    ResizeMatrix(matrix_a_, rows_a_, cols_a_);
    ResizeMatrix(matrix_b_, rows_b_, cols_b_);

    // Initialize with identity matrices
    for (int i = 0; i < rows_a_ && i < cols_a_; i++) {
        matrix_a_[i][i] = 1.0;
    }
    for (int i = 0; i < rows_b_ && i < cols_b_; i++) {
        matrix_b_[i][i] = 1.0;
    }
}

MatrixCalculatorPanel::~MatrixCalculatorPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        is_computing_ = false;
        compute_thread_->join();
    }
}

void MatrixCalculatorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CALCULATOR " Matrix Calculator###MatrixCalc", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            // Split view: Left for input, Right for results
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.5f, 0), true);
            {
                if (ImGui::CollapsingHeader(ICON_FA_TABLE_CELLS " Matrix A", ImGuiTreeNodeFlags_DefaultOpen)) {
                    RenderMatrixA();
                }

                if (show_matrix_b_) {
                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader(ICON_FA_TABLE " Matrix B", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RenderMatrixB();
                    }
                }

                ImGui::Spacing();
                if (ImGui::CollapsingHeader(ICON_FA_SLIDERS " Operation", ImGuiTreeNodeFlags_DefaultOpen)) {
                    RenderOperationSelector();
                    if (selected_op_ == Operation::ScalarMultiply) {
                        RenderScalarInput();
                    }
                }
            }
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            {
                RenderResults();
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void MatrixCalculatorPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute")) {
        ComputeAsync();
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_matrix_result_ = false;
        has_scalar_result_ = false;
        error_message_.clear();
    }

    ImGui::SameLine();

    bool can_export = has_matrix_result_ || has_scalar_result_;
    if (!can_export) ImGui::BeginDisabled();

    if (ImGui::Button(ICON_FA_FILE_EXPORT " Export")) {
        ImGui::OpenPopup("ExportMatrix");
    }

    if (!can_export) ImGui::EndDisabled();

    RenderExportOptions();
}

void MatrixCalculatorPanel::RenderMatrixA() {
    RenderMatrixEditor("Matrix A", matrix_a_, rows_a_, cols_a_, "A");
}

void MatrixCalculatorPanel::RenderMatrixB() {
    RenderMatrixEditor("Matrix B", matrix_b_, rows_b_, cols_b_, "B");
}

void MatrixCalculatorPanel::RenderMatrixEditor(const char* label, std::vector<std::vector<double>>& matrix,
                                                int& rows, int& cols, const char* id_prefix) {
    ImGui::PushID(id_prefix);

    // Size controls
    ImGui::Text("Size:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    if (ImGui::InputInt("##rows", &rows, 0, 0)) {
        rows = std::clamp(rows, 1, 10);
        ResizeMatrix(matrix, rows, cols);
    }
    ImGui::SameLine();
    ImGui::Text("x");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    if (ImGui::InputInt("##cols", &cols, 0, 0)) {
        cols = std::clamp(cols, 1, 10);
        ResizeMatrix(matrix, rows, cols);
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_EYE " Identity")) {
        int n = std::min(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (i == j && i < n) ? 1.0 : 0.0;
            }
        }
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Zero")) {
        for (auto& row : matrix) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }

    ImGui::Spacing();

    // Matrix grid
    if (ImGui::BeginTable("MatrixGrid", cols + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        // Header row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("c%d", j + 1);
        }

        // Data rows
        for (int i = 0; i < rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("r%d", i + 1);

            for (int j = 0; j < cols; j++) {
                ImGui::TableNextColumn();
                ImGui::PushID(i * cols + j);
                ImGui::SetNextItemWidth(50);
                ImGui::InputDouble("##val", &matrix[i][j], 0, 0, "%.2f");
                ImGui::PopID();
            }
        }

        ImGui::EndTable();
    }

    ImGui::PopID();
}

void MatrixCalculatorPanel::RenderOperationSelector() {
    const char* operations[] = {
        "Add (A + B)",
        "Subtract (A - B)",
        "Multiply (A * B)",
        "Scalar Multiply (k * A)",
        "Transpose (A^T)",
        "Inverse (A^-1)",
        "Determinant",
        "Trace",
        "Rank",
        "Frobenius Norm",
        "Condition Number"
    };

    int op_index = static_cast<int>(selected_op_);
    ImGui::SetNextItemWidth(200);
    if (ImGui::Combo("##Operation", &op_index, operations, IM_ARRAYSIZE(operations))) {
        selected_op_ = static_cast<Operation>(op_index);
        has_matrix_result_ = false;
        has_scalar_result_ = false;

        // Show matrix B for binary operations
        show_matrix_b_ = (selected_op_ == Operation::Add ||
                          selected_op_ == Operation::Subtract ||
                          selected_op_ == Operation::Multiply);
    }
}

void MatrixCalculatorPanel::RenderScalarInput() {
    ImGui::Text("Scalar (k):");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputDouble("##scalar", &scalar_value_, 0.1, 1.0, "%.4f");
}

void MatrixCalculatorPanel::RenderLoadingIndicator() {
    ImGui::Spacing();
    ImGui::Text("%s Computing...", ICON_FA_SPINNER);
}

void MatrixCalculatorPanel::RenderResults() {
    ImGui::Text(ICON_FA_SQUARE_CHECK " Results");
    ImGui::Separator();

    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (has_scalar_result_) {
        RenderScalarResults();
    }

    if (has_matrix_result_) {
        RenderMatrixResult();
    }

    if (!has_matrix_result_ && !has_scalar_result_) {
        ImGui::TextDisabled("Select an operation and click Compute");
    }
}

void MatrixCalculatorPanel::RenderMatrixResult() {
    ImGui::Text("Result Matrix (%d x %d):", matrix_result_.rows, matrix_result_.cols);
    ImGui::Spacing();

    if (ImGui::BeginTable("ResultMatrix", matrix_result_.cols + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                          ImVec2(0, 200))) {
        // Header row
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled(" ");
        for (int j = 0; j < matrix_result_.cols; j++) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("c%d", j + 1);
        }

        // Data rows
        for (int i = 0; i < matrix_result_.rows; i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("r%d", i + 1);

            for (int j = 0; j < matrix_result_.cols; j++) {
                ImGui::TableNextColumn();
                double val = matrix_result_.matrix[i][j];
                if (std::abs(val) < 1e-10) val = 0.0;  // Clean up near-zero
                ImGui::Text("%.4f", val);
            }
        }

        ImGui::EndTable();
    }

    // Copy result to Matrix A button
    ImGui::Spacing();
    if (ImGui::Button(ICON_FA_COPY " Copy to Matrix A")) {
        matrix_a_ = matrix_result_.matrix;
        rows_a_ = matrix_result_.rows;
        cols_a_ = matrix_result_.cols;
    }
}

void MatrixCalculatorPanel::RenderScalarResults() {
    ImGui::Text("Result:");
    ImGui::Spacing();

    const char* label = "";
    switch (selected_op_) {
        case Operation::Determinant: label = "Determinant"; break;
        case Operation::Trace: label = "Trace"; break;
        case Operation::Rank: label = "Rank"; break;
        case Operation::FrobeniusNorm: label = "Frobenius Norm"; break;
        case Operation::ConditionNumber: label = "Condition Number"; break;
        default: label = "Value"; break;
    }

    ImGui::BulletText("%s: %.6g", label, scalar_result_.value);
}

void MatrixCalculatorPanel::RenderExportOptions() {
    if (ImGui::BeginPopup("ExportMatrix")) {
        ImGui::Text(ICON_FA_FILE_EXPORT " Export Result");
        ImGui::Separator();

        ImGui::InputText("Path", export_path_, sizeof(export_path_));

        if (ImGui::Button("Export CSV")) {
            std::ofstream file(export_path_);
            if (file.is_open()) {
                if (has_matrix_result_) {
                    for (int i = 0; i < matrix_result_.rows; i++) {
                        for (int j = 0; j < matrix_result_.cols; j++) {
                            if (j > 0) file << ",";
                            file << std::setprecision(10) << matrix_result_.matrix[i][j];
                        }
                        file << "\n";
                    }
                } else if (has_scalar_result_) {
                    file << std::setprecision(10) << scalar_result_.value << "\n";
                }
                spdlog::info("[MatrixCalculator] Exported to {}", export_path_);
            }
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

void MatrixCalculatorPanel::ResizeMatrix(std::vector<std::vector<double>>& matrix, int rows, int cols) {
    matrix.resize(rows);
    for (auto& row : matrix) {
        row.resize(cols, 0.0);
    }
}

void MatrixCalculatorPanel::ComputeAsync() {
    if (is_computing_.load()) return;

    // Join previous thread if exists
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_matrix_result_ = false;
    has_scalar_result_ = false;
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        ComputeOperation();
        is_computing_ = false;
    });
}

void MatrixCalculatorPanel::ComputeOperation() {
    std::lock_guard<std::mutex> lock(result_mutex_);

    try {
        switch (selected_op_) {
            case Operation::Add: {
                matrix_result_ = LinearAlgebra::Add(matrix_a_, matrix_b_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::Subtract: {
                matrix_result_ = LinearAlgebra::Subtract(matrix_a_, matrix_b_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::Multiply: {
                matrix_result_ = LinearAlgebra::Multiply(matrix_a_, matrix_b_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::ScalarMultiply: {
                matrix_result_ = LinearAlgebra::ScalarMultiply(matrix_a_, scalar_value_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::Transpose: {
                matrix_result_ = LinearAlgebra::Transpose(matrix_a_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::Inverse: {
                matrix_result_ = LinearAlgebra::Inverse(matrix_a_);
                if (matrix_result_.success) {
                    has_matrix_result_ = true;
                } else {
                    error_message_ = matrix_result_.error_message;
                }
                break;
            }
            case Operation::Determinant: {
                scalar_result_ = LinearAlgebra::Determinant(matrix_a_);
                if (scalar_result_.success) {
                    has_scalar_result_ = true;
                } else {
                    error_message_ = scalar_result_.error_message;
                }
                break;
            }
            case Operation::Trace: {
                scalar_result_ = LinearAlgebra::Trace(matrix_a_);
                if (scalar_result_.success) {
                    has_scalar_result_ = true;
                } else {
                    error_message_ = scalar_result_.error_message;
                }
                break;
            }
            case Operation::Rank: {
                scalar_result_ = LinearAlgebra::Rank(matrix_a_);
                if (scalar_result_.success) {
                    has_scalar_result_ = true;
                } else {
                    error_message_ = scalar_result_.error_message;
                }
                break;
            }
            case Operation::FrobeniusNorm: {
                scalar_result_ = LinearAlgebra::FrobeniusNorm(matrix_a_);
                if (scalar_result_.success) {
                    has_scalar_result_ = true;
                } else {
                    error_message_ = scalar_result_.error_message;
                }
                break;
            }
            case Operation::ConditionNumber: {
                scalar_result_ = LinearAlgebra::ConditionNumber(matrix_a_);
                if (scalar_result_.success) {
                    has_scalar_result_ = true;
                } else {
                    error_message_ = scalar_result_.error_message;
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        error_message_ = std::string("Exception: ") + e.what();
    }
}

} // namespace cyxwiz
