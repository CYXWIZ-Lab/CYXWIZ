#pragma once

#include <cyxwiz/linear_algebra.h>
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

namespace cyxwiz {

/**
 * MatrixCalculatorPanel - Interactive Matrix Operations Tool
 *
 * Features:
 * - Manual matrix entry (resizable grid)
 * - Load matrices from datasets
 * - Operations: Add, Subtract, Multiply, Scalar Multiply, Transpose, Inverse
 * - Properties: Determinant, Trace, Rank, Frobenius Norm, Condition Number
 * - Result display and export
 * - GPU-accelerated via LinearAlgebra backend
 */
class MatrixCalculatorPanel {
public:
    MatrixCalculatorPanel();
    ~MatrixCalculatorPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderMatrixA();
    void RenderMatrixB();
    void RenderOperationSelector();
    void RenderScalarInput();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderMatrixResult();
    void RenderScalarResults();
    void RenderExportOptions();

    void ResizeMatrix(std::vector<std::vector<double>>& matrix, int rows, int cols);
    void RenderMatrixEditor(const char* label, std::vector<std::vector<double>>& matrix,
                            int& rows, int& cols, const char* id_prefix);

    void ComputeAsync();
    void ComputeOperation();

    bool visible_ = false;

    // Matrix A (primary)
    std::vector<std::vector<double>> matrix_a_;
    int rows_a_ = 3;
    int cols_a_ = 3;

    // Matrix B (secondary for binary operations)
    std::vector<std::vector<double>> matrix_b_;
    int rows_b_ = 3;
    int cols_b_ = 3;
    bool show_matrix_b_ = false;

    // Scalar for scalar multiply
    double scalar_value_ = 1.0;

    // Operation selection
    enum class Operation {
        Add,
        Subtract,
        Multiply,
        ScalarMultiply,
        Transpose,
        Inverse,
        Determinant,
        Trace,
        Rank,
        FrobeniusNorm,
        ConditionNumber
    };
    Operation selected_op_ = Operation::Add;

    // Results
    MatrixResult matrix_result_;
    ScalarResult scalar_result_;
    bool has_matrix_result_ = false;
    bool has_scalar_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;

    // Export
    char export_path_[256] = "";
};

} // namespace cyxwiz
