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
 * CholeskyPanel - Cholesky Decomposition Tool
 *
 * Features:
 * - Input positive definite matrix
 * - Compute L where A = L * L^T
 * - Verify positive definiteness
 * - Verify reconstruction
 */
class CholeskyPanel {
public:
    CholeskyPanel();
    ~CholeskyPanel();

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
    void RenderMatrixL();
    void RenderVerification();

    void ResizeMatrix(int n);
    void MakePositiveDefinite();
    void ComputeAsync();

    bool visible_ = false;

    // Input matrix (must be square, symmetric, positive definite)
    std::vector<std::vector<double>> matrix_;
    int size_ = 3;

    // Results
    CholeskyResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
