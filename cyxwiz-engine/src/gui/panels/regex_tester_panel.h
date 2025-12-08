#pragma once

#include <imgui.h>
#include <cyxwiz/utilities.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>
#include <map>

namespace cyxwiz {

class RegexTesterPanel {
public:
    RegexTesterPanel();
    ~RegexTesterPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderPatternInput();
    void RenderTextInput();
    void RenderResults();
    void RenderMatches();
    void RenderReplacement();
    void RenderCommonPatterns();
    void RenderLoadingIndicator();

    void TestAsync();
    void ReplaceAsync();
    void CopyResult();
    void ClearAll();
    void InsertPattern(const std::string& name, const std::string& pattern);
    void GenerateSampleText();

private:
    bool visible_ = false;

    // Pattern
    char pattern_buffer_[1024] = {0};

    // Flags
    bool flag_case_insensitive_ = false;
    bool flag_multiline_ = false;
    bool flag_global_ = true;

    // Input text
    char text_buffer_[16384] = {0};

    // Replacement
    char replacement_buffer_[1024] = {0};
    bool show_replacement_ = false;

    // Results
    RegexResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Common patterns
    std::map<std::string, std::string> common_patterns_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
