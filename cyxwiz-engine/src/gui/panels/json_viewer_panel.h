#pragma once

#include <imgui.h>
#include <cyxwiz/utilities.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace cyxwiz {

class JSONViewerPanel {
public:
    JSONViewerPanel();
    ~JSONViewerPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputSection();
    void RenderFormattedView();
    void RenderTreeView();
    void RenderStatistics();
    void RenderPathQuery();
    void RenderLoadingIndicator();

    void ValidateAsync();
    void FormatAsync();
    void MinifyAsync();
    void QueryPathAsync();
    void CopyFormatted();
    void CopyMinified();
    void ClearAll();
    void LoadSample();

private:
    bool visible_ = false;

    // Input
    char input_buffer_[65536] = {0};
    int indent_size_ = 2;

    // Path query
    char path_buffer_[256] = {0};
    std::string path_result_;

    // Results
    JSONResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
