#pragma once

#include <imgui.h>
#include <cyxwiz/text_processing.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>

namespace cyxwiz {

class SentimentPanel {
public:
    SentimentPanel();
    ~SentimentPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderResults();
    void RenderScoreView();
    void RenderWordContributions();
    void RenderBatchAnalysis();
    void RenderLoadingIndicator();

    void AnalyzeAsync();
    void AnalyzeBatchAsync();
    void GenerateSampleText();

private:
    bool visible_ = false;

    // Input
    char text_buffer_[8192] = {0};
    int sample_type_idx_ = 2;  // Positive review

    // Batch input
    std::vector<std::string> batch_texts_;
    std::vector<SentimentResult> batch_results_;

    // Settings
    int method_idx_ = 0;  // 0=simple, 1=afinn

    // Results
    SentimentResult result_;
    bool has_result_ = false;
    bool has_batch_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
