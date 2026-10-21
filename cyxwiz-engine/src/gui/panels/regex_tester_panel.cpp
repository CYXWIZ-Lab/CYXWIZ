#include "regex_tester_panel.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace cyxwiz {

RegexTesterPanel::RegexTesterPanel() {
    // Load common patterns
    common_patterns_ = Utilities::GetCommonPatterns();

    // Sample pattern and text
    strcpy(pattern_buffer_, R"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)");
    GenerateSampleText();

    spdlog::debug("RegexTesterPanel initialized");
}

RegexTesterPanel::~RegexTesterPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void RegexTesterPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(850, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_ASTERISK " Regex Tester###RegexTesterPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            // Left: pattern + test text, right: results
            ui::BeginSettingsColumn("regex_tester", ImGui::GetContentRegionAvail().x * 0.45f);
            RenderPatternInput();
            ImGui::Spacing();
            RenderTextInput();
            ui::BeginResultsArea();

            if (ImGui::BeginTabBar("RegexTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_LIST " Matches")) {
                    RenderMatches();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_RIGHT_LEFT " Replace")) {
                    RenderReplacement();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_BOOKMARK " Patterns")) {
                    RenderCommonPatterns();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ui::EndResultsArea();
        }
    }
    ImGui::End();
}

void RegexTesterPanel::RenderToolbar() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ImGui::Button(ICON_FA_PLAY " Test")) {
        TestAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        ClearAll();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FILE_LINES " Sample")) {
        GenerateSampleText();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    // Flags
    ImGui::Checkbox("Case Insensitive", &flag_case_insensitive_);
    ImGui::SameLine();
    ImGui::Checkbox("Multiline", &flag_multiline_);

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        if (result_.is_valid_pattern) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.success);
            ImGui::Text(ICON_FA_CHECK " %d matches", result_.match_count);
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, t.error);
            ImGui::Text(ICON_FA_XMARK " Invalid pattern");
            ImGui::PopStyleColor();
        }
    }
}

void RegexTesterPanel::RenderPatternInput() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::FieldLabel("Regular expression");

    // Pattern validity indicator
    if (has_result_) {
        ImGui::SameLine();
        if (result_.is_valid_pattern) {
            ImGui::TextColored(t.success, ICON_FA_CHECK);
        } else {
            ImGui::TextColored(t.error, ICON_FA_XMARK);
        }
    }

    ImGui::SetNextItemWidth(-FLT_MIN);
    bool enter_pressed = ImGui::InputText("##Pattern", pattern_buffer_, sizeof(pattern_buffer_),
                                          ImGuiInputTextFlags_EnterReturnsTrue);

    if (enter_pressed) {
        TestAsync();
    }

    // Show pattern error if any
    if (has_result_ && !result_.is_valid_pattern && !result_.pattern_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, t.error);
        ImGui::TextWrapped("%s", result_.pattern_error.c_str());
        ImGui::PopStyleColor();
    }

    // Quick insert chips (wrap to the next line when the column is narrow)
    ImGui::Spacing();
    ui::FieldLabel("Quick insert");
    static const char* const kQuickInserts[] = {
        "\\d", "\\w", "\\s", ".", "*", "+", "?", "()", "[]", "^", "$"};
    ui::FlowRow row(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX());
    for (const char* insert : kQuickInserts) {
        row.Next(ui::ChipButtonWidth(insert));
        if (ui::ChipButton(insert)) { strcat(pattern_buffer_, insert); }
    }
}

void RegexTesterPanel::RenderTextInput() {
    ui::FieldLabel("Test text");

    float available_height = ImGui::GetContentRegionAvail().y - 30;
    ImGui::InputTextMultiline("##TestText", text_buffer_, sizeof(text_buffer_),
                              ImVec2(-1, available_height),
                              ImGuiInputTextFlags_AllowTabInput);

    // Character count
    size_t len = strlen(text_buffer_);
    ImGui::TextDisabled("%zu characters", len);
}

void RegexTesterPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().error);
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ui::EmptyState(ICON_FA_ASTERISK, "No matches yet", "Enter a pattern and test text, then Test.");
        return;
    }
}

void RegexTesterPanel::RenderMatches() {
    const ui::Tokens& t = ui::CurrentTokens();
    RenderResults();

    if (!has_result_ || !result_.is_valid_pattern) {
        return;
    }

    ImGui::Text(ICON_FA_LIST " Matches (%d)", result_.match_count);
    ImGui::Separator();

    if (result_.matches.empty()) {
        ImGui::TextDisabled("No matches found");
        return;
    }

    ImGui::BeginChild("MatchesList", ImVec2(0, 0), false);

    for (size_t i = 0; i < result_.matches.size(); ++i) {
        const auto& match = result_.matches[i];

        ImGui::PushID(static_cast<int>(i));

        // Match header
        bool node_open = ImGui::TreeNode("##match",
            "Match %d (pos %d-%d)", static_cast<int>(i + 1), match.start_pos, match.end_pos);

        if (node_open) {
            // Full match text
            ImGui::PushStyleColor(ImGuiCol_Text, t.series[0]);
            ImGui::TextWrapped("\"%s\"", match.match_text.c_str());
            ImGui::PopStyleColor();

            // Position info
            ImGui::TextDisabled("Position: %d to %d | Line: %d",
                               match.start_pos, match.end_pos, match.line_number);

            // Capture groups
            if (!match.groups.empty()) {
                ImGui::Text("Capture Groups:");
                for (size_t g = 0; g < match.groups.size(); ++g) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ui::SeriesColour(t, g + 1));
                    ImGui::BulletText("$%d: \"%s\"", static_cast<int>(g + 1), match.groups[g].c_str());
                    ImGui::PopStyleColor();
                }
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    ImGui::EndChild();
}

void RegexTesterPanel::RenderReplacement() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!has_result_ || !result_.is_valid_pattern) {
        ImGui::TextDisabled("Valid pattern required for replacement");
        return;
    }

    ImGui::Text(ICON_FA_RIGHT_LEFT " Find & Replace");
    ImGui::Separator();

    ImGui::Text("Replacement pattern:");
    ImGui::SetNextItemWidth(-80);
    ImGui::InputText("##Replacement", replacement_buffer_, sizeof(replacement_buffer_));
    ImGui::SameLine();
    if (ImGui::Button("Replace")) {
        ReplaceAsync();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Use $1, $2, etc. for capture groups");
    ImGui::Spacing();

    if (!result_.replaced_text.empty()) {
        ImGui::Text("Result:");
        ImGui::Separator();

        if (ImGui::Button(ICON_FA_COPY " Copy")) {
            ImGui::SetClipboardText(result_.replaced_text.c_str());
        }

        ImGui::Spacing();

        ui::BeginInsetList("ReplacedText", ImGui::GetContentRegionAvail().y);
        ImGui::PushStyleColor(ImGuiCol_Text, t.success);
        ImGui::TextWrapped("%s", result_.replaced_text.c_str());
        ImGui::PopStyleColor();
        ui::EndInsetList();
    }
}

void RegexTesterPanel::RenderCommonPatterns() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Text(ICON_FA_BOOKMARK " Common Patterns");
    ImGui::Separator();

    ImGui::BeginChild("PatternsList", ImVec2(0, 0), false);

    for (const auto& p : common_patterns_) {
        ImGui::PushID(p.first.c_str());

        if (ImGui::SmallButton("Use")) {
            strcpy(pattern_buffer_, p.second.c_str());
            has_result_ = false;
        }
        ImGui::SameLine();

        ImGui::Text("%s:", p.first.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, t.accent_text);
        ImGui::TextWrapped("%s", p.second.c_str());
        ImGui::PopStyleColor();

        ImGui::PopID();
    }

    ImGui::EndChild();
}

void RegexTesterPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Testing pattern...");
}

void RegexTesterPanel::TestAsync() {
    if (is_computing_.load()) return;

    std::string pattern = pattern_buffer_;
    if (pattern.empty()) {
        error_message_ = "No pattern entered";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    std::string text = text_buffer_;
    std::string flags;
    if (flag_case_insensitive_) flags += "i";
    if (flag_multiline_) flags += "m";

    compute_thread_ = std::make_unique<std::thread>([this, pattern, text, flags]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = Utilities::TestRegex(pattern, text, flags);
            has_result_ = true;

            if (result_.success) {
                spdlog::info("Regex test: {} matches found", result_.match_count);
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void RegexTesterPanel::ReplaceAsync() {
    if (is_computing_.load()) return;

    std::string pattern = pattern_buffer_;
    std::string text = text_buffer_;
    std::string replacement = replacement_buffer_;

    if (pattern.empty()) {
        error_message_ = "No pattern entered";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    error_message_.clear();

    std::string flags;
    if (flag_case_insensitive_) flags += "i";
    if (flag_multiline_) flags += "m";

    compute_thread_ = std::make_unique<std::thread>([this, pattern, text, replacement, flags]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = Utilities::ReplaceRegex(pattern, text, replacement, flags);
            has_result_ = true;

            if (result_.success) {
                spdlog::info("Regex replacement: {} matches replaced", result_.match_count);
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void RegexTesterPanel::CopyResult() {
    if (!result_.replaced_text.empty()) {
        ImGui::SetClipboardText(result_.replaced_text.c_str());
        spdlog::info("Result copied to clipboard");
    }
}

void RegexTesterPanel::ClearAll() {
    pattern_buffer_[0] = '\0';
    text_buffer_[0] = '\0';
    replacement_buffer_[0] = '\0';
    has_result_ = false;
    error_message_.clear();
    result_ = RegexResult();
}

void RegexTesterPanel::GenerateSampleText() {
    const char* sample = R"(Contact Information:
John Doe - john.doe@example.com - (555) 123-4567
Jane Smith - jane_smith123@company.org - 555.987.6543
Support Team - support@help-desk.io

Website URLs:
https://www.example.com/products?id=123
http://api.service.net/v2/data
ftp://files.server.com/docs/file.pdf

IP Addresses:
Server 1: 192.168.1.100
Server 2: 10.0.0.1
External: 203.0.113.50

Dates and Times:
Meeting: 2024-03-15 at 14:30
Deadline: 12/31/2024
Updated: 03-15-2024 09:00:00

Product Codes:
SKU-12345-AB
ITEM_67890_XY
REF:11111:ZZ)";

    strcpy(text_buffer_, sample);
}

} // namespace cyxwiz
