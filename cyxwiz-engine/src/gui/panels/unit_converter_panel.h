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

class UnitConverterPanel {
public:
    UnitConverterPanel();
    ~UnitConverterPanel();

    void Render();
    void SetVisible(bool visible) { visible_ = visible; }
    bool IsVisible() const { return visible_; }

private:
    void RenderToolbar();
    void RenderCategorySelector();
    void RenderConversionInput();
    void RenderResults();
    void RenderAllConversions();
    void RenderLoadingIndicator();

    void ConvertAsync();
    void ConvertToAllAsync();
    void UpdateUnitsForCategory();
    void SwapUnits();
    void CopyResult();

private:
    bool visible_ = false;

    // Category and units
    std::vector<std::string> categories_;
    std::vector<std::string> units_in_category_;
    int category_idx_ = 0;
    int from_unit_idx_ = 0;
    int to_unit_idx_ = 1;

    // Input
    char value_buffer_[64] = "1.0";
    double input_value_ = 1.0;

    // Results
    UnitConversionResult result_;
    bool has_result_ = false;
    bool show_all_conversions_ = true;
    std::string error_message_;

    // Async
    std::unique_ptr<std::thread> compute_thread_;
    std::atomic<bool> is_computing_{false};
    std::mutex result_mutex_;
};

} // namespace cyxwiz
