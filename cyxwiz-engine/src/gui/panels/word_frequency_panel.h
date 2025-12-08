#pragma once

#include <imgui.h>
#include <cyxwiz/text_processing.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace cyxwiz {

class WordFrequencyPanel {
public:
    WordFrequencyPanel();
    ~WordFrequencyPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderResults();
    void RenderBarChart();
    void RenderFrequencyTable();
    void RenderLengthDistribution();
    void RenderLoadingIndicator();

    void AnalyzeAsync();
    void GenerateSampleText();

private:
    bool visible_ = false;

    // Input
    char text_buffer_[16384] = {0};
    int sample_type_idx_ = 1;  // News

    // Settings
    int top_n_ = 30;
    int min_word_length_ = 2;
    bool remove_stopwords_ = true;

    // Results
    WordFrequencyResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
