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
 * SVDPanel - Singular Value Decomposition Tool
 *
 * Features:
 * - Input any matrix (m x n)
 * - Compute U, S, V^T matrices
 * - Singular value plot
 * - Rank approximation
 * - Explained variance
 */
class SVDPanel {
public:
    SVDPanel();
    ~SVDPanel();

    void Render();

    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    void Toggle() { visible_ = !visible_; }
    bool* GetVisiblePtr() { return &visible_; }

private:
    void RenderToolbar();
    void RenderMatrixInput();
    void RenderLoadingIndicator();
    void RenderResults();
    void RenderSingularValues();
    void RenderMatrixU();
    void RenderMatrixVt();
    void RenderLowRankApprox();

    void ResizeMatrix(int rows, int cols);
    void ComputeAsync();

    bool visible_ = false;

    // Input matrix
    std::vector<std::vector<double>> matrix_;
    int rows_ = 4;
    int cols_ = 3;
    bool full_matrices_ = false;

    // Results
    SVDResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Low-rank approximation
    int approx_rank_ = 2;
    MatrixResult approx_matrix_;
    bool has_approx_ = false;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
