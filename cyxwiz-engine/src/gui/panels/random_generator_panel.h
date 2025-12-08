#pragma once

#include <imgui.h>
#include <cyxwiz/utilities.h>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>

namespace cyxwiz {

class RandomGeneratorPanel {
public:
    RandomGeneratorPanel();
    ~RandomGeneratorPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderSettings();
    void RenderResults();
    void RenderHistogram();
    void RenderStatistics();
    void RenderUUIDs();
    void RenderLoadingIndicator();

    void GenerateAsync();
    void GenerateUUIDsAsync();
    void CopyValues();
    void ClearResults();

private:
    bool visible_ = false;

    // Distribution settings
    int distribution_idx_ = 0;  // 0=uniform, 1=normal, 2=exponential, 3=poisson, 4=integer
    int count_ = 100;

    // Distribution parameters
    double uniform_min_ = 0.0;
    double uniform_max_ = 1.0;
    double normal_mean_ = 0.0;
    double normal_stddev_ = 1.0;
    double exponential_lambda_ = 1.0;
    double poisson_lambda_ = 5.0;
    int integer_min_ = 0;
    int integer_max_ = 100;

    // Seed
    bool use_custom_seed_ = false;
    int64_t custom_seed_ = 42;

    // UUID settings
    int uuid_count_ = 5;
    std::vector<std::string> generated_uuids_;

    // Results
    RandomNumberResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
