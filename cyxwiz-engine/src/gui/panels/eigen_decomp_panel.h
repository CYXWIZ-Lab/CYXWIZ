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
 * EigenDecompPanel - Eigenvalue Decomposition Tool
 *
 * Features:
 * - Input square matrix
 * - Compute eigenvalues and eigenvectors
 * - Eigenspectrum visualization
 * - Complex eigenvalue display
 * - Export results
 */
class EigenDecompPanel {
public:
    EigenDecompPanel();
    ~EigenDecompPanel();

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
    void RenderEigenvalues();
    void RenderEigenvectors();
    void RenderEigenspectrum();

    void ResizeMatrix(int n);
    void ComputeAsync();

    bool visible_ = false;

    // Input matrix (square)
    std::vector<std::vector<double>> matrix_;
    int size_ = 3;

    // Results
    EigenResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
