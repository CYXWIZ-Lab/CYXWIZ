#pragma once

#include <imgui.h>
#include <cyxwiz/text_processing.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace cyxwiz {

class TokenizationPanel {
public:
    TokenizationPanel();
    ~TokenizationPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderResults();
    void RenderTokenList();
    void RenderStatistics();
    void RenderLoadingIndicator();

    void TokenizeAsync();
    void GenerateSampleText();
    void CopyToClipboard();

private:
    bool visible_ = false;

    // Input
    std::string input_text_;
    char text_buffer_[8192] = {0};

    // Settings
    int method_idx_ = 1;  // 0=whitespace, 1=word, 2=sentence, 3=ngram
    int ngram_n_ = 2;
    bool lowercase_ = true;
    bool remove_punctuation_ = true;
    bool remove_stopwords_ = false;
    bool apply_stemming_ = false;
    int sample_type_idx_ = 0;

    // Results
    TokenizationResult result_;
    std::vector<std::string> processed_tokens_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
