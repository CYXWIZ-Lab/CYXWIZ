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

class TFIDFPanel {
public:
    TFIDFPanel();
    ~TFIDFPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputPanel();
    void RenderResults();
    void RenderTopTerms();
    void RenderSimilarityMatrix();
    void RenderVocabulary();
    void RenderLoadingIndicator();

    void ComputeAsync();
    void GenerateSampleDocuments();
    void AddDocument();
    void RemoveDocument(int index);

private:
    bool visible_ = false;

    // Documents
    std::vector<std::string> documents_;
    std::vector<char[4096]> doc_buffers_;
    int doc_type_idx_ = 0;

    // Settings
    bool use_idf_ = true;
    bool smooth_idf_ = true;
    int norm_idx_ = 1;  // 0=none, 1=l2, 2=l1
    int selected_doc_ = 0;

    // Results
    TFIDFResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
