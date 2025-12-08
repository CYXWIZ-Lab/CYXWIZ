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

class CalculatorPanel {
public:
    CalculatorPanel();
    ~CalculatorPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderExpressionInput();
    void RenderResults();
    void RenderVariables();
    void RenderFunctions();
    void RenderHistory();
    void RenderLoadingIndicator();

    void EvaluateAsync();
    void ClearAll();
    void CopyResult();
    void AddToHistory();
    void InsertFunction(const std::string& func);
    void InsertConstant(const std::string& name, double value);

private:
    bool visible_ = false;

    // Input
    char expression_buffer_[1024] = {0};
    std::string current_expression_;

    // Settings
    int angle_mode_idx_ = 0;  // 0=radians, 1=degrees
    int precision_ = 6;

    // Variables
    std::vector<std::pair<std::string, double>> user_variables_;
    char var_name_buffer_[64] = {0};
    char var_value_buffer_[64] = {0};

    // Results
    CalculatorResult result_;
    bool has_result_ = false;
    std::string error_message_;

    // History
    std::vector<std::pair<std::string, double>> history_;
    static const size_t MAX_HISTORY = 50;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
