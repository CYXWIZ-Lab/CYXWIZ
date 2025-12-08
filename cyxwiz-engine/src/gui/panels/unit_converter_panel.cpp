#include "unit_converter_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace cyxwiz {

UnitConverterPanel::UnitConverterPanel() {
    // Load available categories
    categories_ = Utilities::GetUnitCategories();
    if (!categories_.empty()) {
        UpdateUnitsForCategory();
    }
    spdlog::info("UnitConverterPanel initialized with {} categories", categories_.size());
}

UnitConverterPanel::~UnitConverterPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void UnitConverterPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_SCALE_BALANCED " Unit Converter###UnitConverterPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            RenderCategorySelector();
            ImGui::Spacing();
            RenderConversionInput();
            ImGui::Separator();
            RenderResults();

            if (show_all_conversions_ && has_result_) {
                ImGui::Separator();
                RenderAllConversions();
            }
        }
    }
    ImGui::End();
}

void UnitConverterPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Convert")) {
        ConvertAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Swap")) {
        SwapUnits();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Copy")) {
        CopyResult();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    ImGui::Checkbox("Show All", &show_all_conversions_);

    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Show conversions to all units in category");
    }
}

void UnitConverterPanel::RenderCategorySelector() {
    ImGui::Text(ICON_FA_FOLDER " Category:");
    ImGui::SameLine();

    // Category buttons
    float button_width = 80.0f;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float available_width = ImGui::GetContentRegionAvail().x;
    int buttons_per_row = static_cast<int>(available_width / (button_width + spacing));
    if (buttons_per_row < 1) buttons_per_row = 1;

    int col = 0;
    for (size_t i = 0; i < categories_.size(); ++i) {
        if (col > 0) ImGui::SameLine();

        bool is_selected = (static_cast<int>(i) == category_idx_);
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
        }

        if (ImGui::Button(categories_[i].c_str(), ImVec2(button_width, 0))) {
            if (category_idx_ != static_cast<int>(i)) {
                category_idx_ = static_cast<int>(i);
                UpdateUnitsForCategory();
                has_result_ = false;
            }
        }

        if (is_selected) {
            ImGui::PopStyleColor();
        }

        col++;
        if (col >= buttons_per_row) col = 0;
    }
}

void UnitConverterPanel::RenderConversionInput() {
    float panel_width = ImGui::GetContentRegionAvail().x;

    ImGui::BeginChild("ConversionInput", ImVec2(0, 100), true);

    // From section
    ImGui::Columns(3, "conversion_cols", false);
    ImGui::SetColumnWidth(0, panel_width * 0.4f);
    ImGui::SetColumnWidth(1, panel_width * 0.2f);
    ImGui::SetColumnWidth(2, panel_width * 0.4f);

    // From unit
    ImGui::Text(ICON_FA_RIGHT_FROM_BRACKET " From:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##Value", value_buffer_, sizeof(value_buffer_),
                         ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_EnterReturnsTrue)) {
        ConvertAsync();
    }
    ImGui::SetNextItemWidth(-1);
    if (!units_in_category_.empty()) {
        std::vector<const char*> items;
        for (const auto& u : units_in_category_) {
            items.push_back(u.c_str());
        }
        if (ImGui::Combo("##FromUnit", &from_unit_idx_, items.data(),
                        static_cast<int>(items.size()))) {
            if (has_result_) ConvertAsync();
        }
    }

    // Arrow
    ImGui::NextColumn();
    ImGui::Spacing();
    ImGui::Spacing();
    float arrow_x = ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - ImGui::CalcTextSize(ICON_FA_ARROW_RIGHT).x) * 0.5f;
    ImGui::SetCursorPosX(arrow_x);
    ImGui::Text(ICON_FA_ARROW_RIGHT);

    // To unit
    ImGui::NextColumn();
    ImGui::Text(ICON_FA_RIGHT_TO_BRACKET " To:");

    // Result display
    if (has_result_) {
        std::ostringstream oss;
        oss << std::setprecision(10) << result_.output_value;
        std::string result_str = oss.str();

        // Clean up trailing zeros
        if (result_str.find('.') != std::string::npos) {
            size_t last_non_zero = result_str.find_last_not_of('0');
            if (last_non_zero != std::string::npos && result_str[last_non_zero] == '.') {
                result_str = result_str.substr(0, last_non_zero);
            } else if (last_non_zero != std::string::npos) {
                result_str = result_str.substr(0, last_non_zero + 1);
            }
        }

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##Result", const_cast<char*>(result_str.c_str()),
                        result_str.size() + 1, ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleColor();
    } else {
        ImGui::TextDisabled("---");
    }

    ImGui::SetNextItemWidth(-1);
    if (!units_in_category_.empty()) {
        std::vector<const char*> items;
        for (const auto& u : units_in_category_) {
            items.push_back(u.c_str());
        }
        if (ImGui::Combo("##ToUnit", &to_unit_idx_, items.data(),
                        static_cast<int>(items.size()))) {
            if (has_result_) ConvertAsync();
        }
    }

    ImGui::Columns(1);
    ImGui::EndChild();
}

void UnitConverterPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter a value and click 'Convert'");
        return;
    }

    // Show formula
    if (!result_.formula.empty()) {
        ImGui::Text(ICON_FA_INFO " Formula: %s", result_.formula.c_str());
    }

    // Summary
    ImGui::Text("%g %s = %g %s",
               result_.input_value, result_.input_unit.c_str(),
               result_.output_value, result_.output_unit.c_str());
}

void UnitConverterPanel::RenderAllConversions() {
    ImGui::Text(ICON_FA_LIST " All Conversions:");

    if (result_.all_conversions.empty()) {
        ImGui::TextDisabled("No conversions available");
        return;
    }

    ImGui::BeginChild("AllConversions", ImVec2(0, 0), true);

    ImGui::Columns(2, "all_conv_cols", true);

    for (const auto& conv : result_.all_conversions) {
        // Highlight current selection
        bool is_selected = (conv.first == result_.output_unit);
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
        }

        ImGui::Text("%s", conv.first.c_str());
        ImGui::NextColumn();

        std::ostringstream oss;
        oss << std::setprecision(10) << conv.second;
        ImGui::Text("%s", oss.str().c_str());
        ImGui::NextColumn();

        if (is_selected) {
            ImGui::PopStyleColor();
        }
    }

    ImGui::Columns(1);
    ImGui::EndChild();
}

void UnitConverterPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Converting...");
}

void UnitConverterPanel::ConvertAsync() {
    if (is_computing_.load()) return;

    try {
        input_value_ = std::stod(value_buffer_);
    } catch (...) {
        error_message_ = "Invalid input value";
        return;
    }

    if (units_in_category_.empty() ||
        from_unit_idx_ < 0 || from_unit_idx_ >= static_cast<int>(units_in_category_.size()) ||
        to_unit_idx_ < 0 || to_unit_idx_ >= static_cast<int>(units_in_category_.size())) {
        error_message_ = "Invalid unit selection";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    std::string from_unit = units_in_category_[from_unit_idx_];
    std::string to_unit = units_in_category_[to_unit_idx_];

    compute_thread_ = std::make_unique<std::thread>([this, from_unit, to_unit]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            if (show_all_conversions_) {
                result_ = Utilities::ConvertToAllUnits(input_value_, from_unit);
                // Also get the specific conversion for the selected to_unit
                auto specific = Utilities::ConvertUnit(input_value_, from_unit, to_unit);
                if (specific.success) {
                    result_.output_value = specific.output_value;
                    result_.output_unit = specific.output_unit;
                    result_.formula = specific.formula;
                }
            } else {
                result_ = Utilities::ConvertUnit(input_value_, from_unit, to_unit);
            }

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Converted {} {} to {} {}",
                            input_value_, from_unit, result_.output_value, to_unit);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void UnitConverterPanel::ConvertToAllAsync() {
    // This is now integrated into ConvertAsync when show_all_conversions_ is true
    ConvertAsync();
}

void UnitConverterPanel::UpdateUnitsForCategory() {
    if (category_idx_ < 0 || category_idx_ >= static_cast<int>(categories_.size())) {
        units_in_category_.clear();
        return;
    }

    units_in_category_ = Utilities::GetUnitsForCategory(categories_[category_idx_]);
    from_unit_idx_ = 0;
    to_unit_idx_ = units_in_category_.size() > 1 ? 1 : 0;
}

void UnitConverterPanel::SwapUnits() {
    std::swap(from_unit_idx_, to_unit_idx_);
    if (has_result_) {
        // Put the result as the new input
        std::ostringstream oss;
        oss << std::setprecision(10) << result_.output_value;
        strcpy(value_buffer_, oss.str().c_str());
        ConvertAsync();
    }
}

void UnitConverterPanel::CopyResult() {
    if (!has_result_) return;

    std::ostringstream oss;
    oss << std::setprecision(10) << result_.output_value << " " << result_.output_unit;
    ImGui::SetClipboardText(oss.str().c_str());
    spdlog::info("Result copied to clipboard");
}

} // namespace cyxwiz
