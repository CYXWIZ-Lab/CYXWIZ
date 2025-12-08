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
 * QRPanel - QR Decomposition Tool
 *
 * Features:
 * - Input any matrix
 * - Compute Q (orthogonal) and R (upper triangular)
 * - Verify orthogonality of Q
 * - Verify A = Q * R
 */
class QRPanel {
public:
    QRPanel();
    ~QRPanel();

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
    void RenderMatrixQ();
    void RenderMatrixR();
    void RenderVerification();

    void ResizeMatrix(int rows, int cols);
    void ComputeAsync();

    bool visible_ = false;

    // Input matrix
    std::vector<std::vector<double>> matrix_;
    int rows_ = 4;
    int cols_ = 3;

    // Results
    QRResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async computation
    std::atomic<bool> is_computing_{false};
    std::unique_ptr<std::thread> compute_thread_;
    std::mutex result_mutex_;
};

} // namespace cyxwiz
