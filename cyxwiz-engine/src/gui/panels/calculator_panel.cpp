#include "calculator_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace cyxwiz {

CalculatorPanel::CalculatorPanel() {
    // Initialize with a sample expression
    strcpy(expression_buffer_, "2 * pi + sqrt(16)");
    spdlog::info("CalculatorPanel initialized");
}

CalculatorPanel::~CalculatorPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void CalculatorPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CALCULATOR " Calculator###CalculatorPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;
            float panel_height = ImGui::GetContentRegionAvail().y;

            // Top section: Expression + Result
            ImGui::BeginChild("TopSection", ImVec2(0, panel_height * 0.4f), true);
            RenderExpressionInput();
            ImGui::Separator();
            RenderResults();
            ImGui::EndChild();

            // Bottom section: Variables, Functions, History
            if (ImGui::BeginTabBar("CalculatorTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_SUBSCRIPT " Variables")) {
                    RenderVariables();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_SQUARE_ROOT_VARIABLE " Functions")) {
                    RenderFunctions();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CLOCK_ROTATE_LEFT " History")) {
                    RenderHistory();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        }
    }
    ImGui::End();
}

void CalculatorPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Evaluate")) {
        EvaluateAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        ClearAll();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Copy")) {
        CopyResult();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Angle mode
    const char* angle_modes[] = { "Radians", "Degrees" };
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##AngleMode", &angle_mode_idx_, angle_modes, IM_ARRAYSIZE(angle_modes));

    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::DragInt("Precision", &precision_, 0.1f, 1, 15);
}

void CalculatorPanel::RenderExpressionInput() {
    ImGui::Text(ICON_FA_KEYBOARD " Expression:");

    ImGui::SetNextItemWidth(-1);
    bool enter_pressed = ImGui::InputText("##Expression", expression_buffer_, sizeof(expression_buffer_),
                                          ImGuiInputTextFlags_EnterReturnsTrue);

    if (enter_pressed) {
        EvaluateAsync();
    }

    // Quick insert buttons
    ImGui::Spacing();
    ImGui::Text("Quick Insert:");
    ImGui::SameLine();

    if (ImGui::SmallButton("(")) { strcat(expression_buffer_, "("); }
    ImGui::SameLine();
    if (ImGui::SmallButton(")")) { strcat(expression_buffer_, ")"); }
    ImGui::SameLine();
    if (ImGui::SmallButton("^")) { strcat(expression_buffer_, "^"); }
    ImGui::SameLine();
    if (ImGui::SmallButton("sqrt")) { strcat(expression_buffer_, "sqrt("); }
    ImGui::SameLine();
    if (ImGui::SmallButton("sin")) { strcat(expression_buffer_, "sin("); }
    ImGui::SameLine();
    if (ImGui::SmallButton("cos")) { strcat(expression_buffer_, "cos("); }
    ImGui::SameLine();
    if (ImGui::SmallButton("log")) { strcat(expression_buffer_, "log("); }
    ImGui::SameLine();
    if (ImGui::SmallButton("exp")) { strcat(expression_buffer_, "exp("); }
    ImGui::SameLine();
    if (ImGui::SmallButton("pi")) { strcat(expression_buffer_, "pi"); }
    ImGui::SameLine();
    if (ImGui::SmallButton("e")) { strcat(expression_buffer_, "e"); }
}

void CalculatorPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter an expression and press Enter or click 'Evaluate'");
        return;
    }

    ImGui::Text(ICON_FA_EQUALS " Result:");

    // Display result with large font
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));

    std::ostringstream oss;
    oss << std::setprecision(precision_) << std::fixed << result_.result;
    std::string result_str = oss.str();

    // Remove trailing zeros
    size_t dot_pos = result_str.find('.');
    if (dot_pos != std::string::npos) {
        size_t last_non_zero = result_str.find_last_not_of('0');
        if (last_non_zero > dot_pos) {
            result_str = result_str.substr(0, last_non_zero + 1);
        } else {
            result_str = result_str.substr(0, dot_pos);
        }
    }

    ImGui::Text("%s", result_str.c_str());
    ImGui::PopStyleColor();

    // Show parsed expression if different
    if (!result_.parsed_expression.empty() && result_.parsed_expression != result_.expression) {
        ImGui::TextDisabled("Parsed: %s", result_.parsed_expression.c_str());
    }

    // Show variables used
    if (!result_.variables.empty()) {
        ImGui::TextDisabled("Variables used:");
        for (const auto& var : result_.variables) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s=%.4g", var.first.c_str(), var.second);
        }
    }
}

void CalculatorPanel::RenderVariables() {
    ImGui::Text(ICON_FA_SUBSCRIPT " User Variables");
    ImGui::Separator();

    // Add new variable
    ImGui::SetNextItemWidth(100);
    ImGui::InputText("Name", var_name_buffer_, sizeof(var_name_buffer_));
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputText("Value", var_value_buffer_, sizeof(var_value_buffer_));
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PLUS " Add")) {
        if (strlen(var_name_buffer_) > 0 && strlen(var_value_buffer_) > 0) {
            try {
                double value = std::stod(var_value_buffer_);
                // Check if variable already exists
                bool found = false;
                for (auto& var : user_variables_) {
                    if (var.first == var_name_buffer_) {
                        var.second = value;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    user_variables_.emplace_back(var_name_buffer_, value);
                }
                var_name_buffer_[0] = '\0';
                var_value_buffer_[0] = '\0';
            } catch (...) {
                // Invalid number
            }
        }
    }

    ImGui::Spacing();

    // Show existing variables
    if (user_variables_.empty()) {
        ImGui::TextDisabled("No custom variables defined");
    } else {
        ImGui::BeginChild("VariablesList", ImVec2(0, 0), false);
        for (size_t i = 0; i < user_variables_.size(); ++i) {
            const auto& var = user_variables_[i];

            ImGui::PushID(static_cast<int>(i));

            if (ImGui::SmallButton(ICON_FA_TRASH)) {
                user_variables_.erase(user_variables_.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Insert")) {
                strcat(expression_buffer_, var.first.c_str());
            }
            ImGui::SameLine();
            ImGui::Text("%s = %.6g", var.first.c_str(), var.second);

            ImGui::PopID();
        }
        ImGui::EndChild();
    }

    // Built-in constants section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_PI " Built-in Constants:");

    auto constants = Utilities::GetSupportedConstants();
    int col = 0;
    for (const auto& c : constants) {
        if (col > 0) ImGui::SameLine();

        ImGui::PushID(c.first.c_str());
        if (ImGui::SmallButton(c.first.c_str())) {
            strcat(expression_buffer_, c.first.c_str());
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s = %.10g", c.first.c_str(), c.second);
        }
        ImGui::PopID();

        col++;
        if (col >= 6) col = 0;
    }
}

void CalculatorPanel::RenderFunctions() {
    ImGui::Text(ICON_FA_SQUARE_ROOT_VARIABLE " Available Functions");
    ImGui::Separator();

    auto functions = Utilities::GetSupportedFunctions();

    // Group by category
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> categories;
    for (const auto& f : functions) {
        std::string category = "Other";
        if (f.first == "sin" || f.first == "cos" || f.first == "tan" ||
            f.first == "asin" || f.first == "acos" || f.first == "atan" ||
            f.first == "atan2") {
            category = "Trigonometric";
        } else if (f.first == "sinh" || f.first == "cosh" || f.first == "tanh") {
            category = "Hyperbolic";
        } else if (f.first == "sqrt" || f.first == "cbrt" || f.first == "pow") {
            category = "Power/Root";
        } else if (f.first == "exp" || f.first == "log" || f.first == "log10" || f.first == "log2") {
            category = "Exponential";
        } else if (f.first == "floor" || f.first == "ceil" || f.first == "round" ||
                   f.first == "abs" || f.first == "sign") {
            category = "Rounding";
        } else if (f.first == "min" || f.first == "max" || f.first == "mod") {
            category = "Comparison";
        }
        categories[category].emplace_back(f.first, f.second);
    }

    ImGui::BeginChild("FunctionsList", ImVec2(0, 0), false);

    for (const auto& cat : categories) {
        if (ImGui::CollapsingHeader(cat.first.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            for (const auto& f : cat.second) {
                ImGui::PushID(f.first.c_str());
                if (ImGui::SmallButton("Insert")) {
                    std::string insert_text = f.first + "(";
                    strcat(expression_buffer_, insert_text.c_str());
                }
                ImGui::SameLine();
                ImGui::Text("%s - %s", f.first.c_str(), f.second.c_str());
                ImGui::PopID();
            }
        }
    }

    ImGui::EndChild();
}

void CalculatorPanel::RenderHistory() {
    ImGui::Text(ICON_FA_CLOCK_ROTATE_LEFT " Calculation History");
    ImGui::Separator();

    if (history_.empty()) {
        ImGui::TextDisabled("No calculations yet");
        return;
    }

    if (ImGui::Button(ICON_FA_TRASH " Clear History")) {
        history_.clear();
    }

    ImGui::Spacing();

    ImGui::BeginChild("HistoryList", ImVec2(0, 0), false);

    for (size_t i = history_.size(); i > 0; --i) {
        const auto& entry = history_[i - 1];

        ImGui::PushID(static_cast<int>(i));

        if (ImGui::SmallButton("Use")) {
            strcpy(expression_buffer_, entry.first.c_str());
        }
        ImGui::SameLine();

        ImGui::Text("%s = ", entry.first.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
        ImGui::Text("%.10g", entry.second);
        ImGui::PopStyleColor();

        ImGui::PopID();
    }

    ImGui::EndChild();
}

void CalculatorPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Evaluating...");
}

void CalculatorPanel::EvaluateAsync() {
    if (is_computing_.load()) return;

    current_expression_ = expression_buffer_;
    if (current_expression_.empty()) {
        error_message_ = "No expression entered";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    compute_thread_ = std::make_unique<std::thread>([this]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            // Build variables map
            std::map<std::string, double> vars;
            for (const auto& v : user_variables_) {
                vars[v.first] = v.second;
            }

            const char* angle_modes[] = { "radians", "degrees" };
            std::string angle_mode = angle_modes[angle_mode_idx_];

            result_ = Utilities::Evaluate(current_expression_, vars, angle_mode);

            if (result_.success) {
                has_result_ = true;
                AddToHistory();
                spdlog::info("Calculator: {} = {}", current_expression_, result_.result);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void CalculatorPanel::ClearAll() {
    expression_buffer_[0] = '\0';
    has_result_ = false;
    error_message_.clear();
    result_ = CalculatorResult();
}

void CalculatorPanel::CopyResult() {
    if (!has_result_) return;

    std::ostringstream oss;
    oss << std::setprecision(precision_) << result_.result;
    ImGui::SetClipboardText(oss.str().c_str());
    spdlog::info("Result copied to clipboard");
}

void CalculatorPanel::AddToHistory() {
    // Add to history if not duplicate of last entry
    if (history_.empty() || history_.back().first != current_expression_) {
        history_.emplace_back(current_expression_, result_.result);

        // Limit history size
        while (history_.size() > MAX_HISTORY) {
            history_.erase(history_.begin());
        }
    }
}

void CalculatorPanel::InsertFunction(const std::string& func) {
    std::string insert_text = func + "(";
    strcat(expression_buffer_, insert_text.c_str());
}

void CalculatorPanel::InsertConstant(const std::string& name, double value) {
    strcat(expression_buffer_, name.c_str());
}

} // namespace cyxwiz
