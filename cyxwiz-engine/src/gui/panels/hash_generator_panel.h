#pragma once

#include <imgui.h>
#include <cyxwiz/utilities.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>

namespace cyxwiz {

class HashGeneratorPanel {
public:
    HashGeneratorPanel();
    ~HashGeneratorPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderInputSection();
    void RenderResults();
    void RenderVerification();
    void RenderLoadingIndicator();

    void HashAsync();
    void VerifyAsync();
    void CopyHash(const std::string& hash);
    void ClearAll();
    void BrowseFile();

private:
    bool visible_ = false;

    // Input mode
    int input_mode_ = 0;  // 0=text, 1=file
    char text_buffer_[8192] = {0};
    char file_path_buffer_[512] = {0};

    // Algorithm selection
    int algorithm_idx_ = 2;  // 0=MD5, 1=SHA-1, 2=SHA-256, 3=SHA-512, 4=All

    // Verification
    char expected_hash_buffer_[256] = {0};
    bool verification_result_ = false;
    bool has_verification_result_ = false;

    // Results
    HashResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
