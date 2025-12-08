#pragma once

#include <imgui.h>
#include <cyxwiz/text_processing.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace cyxwiz {

class EmbeddingsPanel {
public:
    EmbeddingsPanel();
    ~EmbeddingsPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderResults();
    void RenderVectorView();
    void RenderSimilarWords();
    void Render2DPlot();
    void RenderLoadingIndicator();

    void CreateEmbeddingsAsync();
    void FindSimilarAsync();
    void GenerateSampleVocabulary();

private:
    bool visible_ = false;

    // Input
    char vocab_buffer_[4096] = {0};
    char query_word_[256] = {0};
    int vocab_domain_idx_ = 0;

    // Settings
    int method_idx_ = 1;  // 0=onehot, 1=random
    int embedding_dim_ = 50;
    int top_n_similar_ = 10;

    // Results
    EmbeddingResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
