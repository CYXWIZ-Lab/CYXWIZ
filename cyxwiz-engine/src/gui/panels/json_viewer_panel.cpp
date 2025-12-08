#include "json_viewer_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace cyxwiz {

JSONViewerPanel::JSONViewerPanel() {
    LoadSample();
    spdlog::info("JSONViewerPanel initialized");
}

JSONViewerPanel::~JSONViewerPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void JSONViewerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CODE " JSON Viewer###JSONViewerPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            // Left panel: Input
            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.4f, 0), true);
            RenderInputSection();
            ImGui::EndChild();

            ImGui::SameLine();

            // Right panel: Results
            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);

            if (ImGui::BeginTabBar("JSONTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_ALIGN_LEFT " Formatted")) {
                    RenderFormattedView();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_SITEMAP " Tree View")) {
                    RenderTreeView();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_CHART_PIE " Statistics")) {
                    RenderStatistics();
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem(ICON_FA_MAGNIFYING_GLASS " Path Query")) {
                    RenderPathQuery();
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void JSONViewerPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_CHECK " Validate")) {
        ValidateAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ALIGN_LEFT " Format")) {
        FormatAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COMPRESS " Minify")) {
        MinifyAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        ClearAll();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_FILE_CODE " Sample")) {
        LoadSample();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    ImGui::SetNextItemWidth(80);
    ImGui::DragInt("Indent", &indent_size_, 0.1f, 1, 8);

    ImGui::SameLine();

    if (has_result_) {
        if (result_.is_valid) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
            ImGui::Text(ICON_FA_CHECK " Valid JSON");
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
            ImGui::Text(ICON_FA_XMARK " Invalid JSON");
            ImGui::PopStyleColor();
        }
    }
}

void JSONViewerPanel::RenderInputSection() {
    ImGui::Text(ICON_FA_KEYBOARD " Input JSON");
    ImGui::Separator();

    float available_height = ImGui::GetContentRegionAvail().y - 30;
    ImGui::InputTextMultiline("##JSONInput", input_buffer_, sizeof(input_buffer_),
                              ImVec2(-1, available_height),
                              ImGuiInputTextFlags_AllowTabInput);

    // Show input stats
    size_t len = strlen(input_buffer_);
    ImGui::TextDisabled("%zu characters", len);
}

void JSONViewerPanel::RenderFormattedView() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter JSON and click 'Validate' or 'Format'");
        return;
    }

    if (!result_.is_valid) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::Text("Error at line %d, column %d:", result_.error_line, result_.error_column);
        ImGui::TextWrapped("%s", result_.error_detail.c_str());
        ImGui::PopStyleColor();
        return;
    }

    // Copy buttons
    if (ImGui::Button(ICON_FA_COPY " Copy Formatted")) {
        CopyFormatted();
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Copy Minified")) {
        CopyMinified();
    }

    ImGui::Separator();

    // Display formatted JSON
    ImGui::BeginChild("FormattedJSON", ImVec2(0, 0), false);

    if (!result_.formatted_json.empty()) {
        // Simple syntax highlighting
        const std::string& json = result_.formatted_json;
        size_t i = 0;

        while (i < json.length()) {
            char c = json[i];

            if (c == '"') {
                // Find end of string
                size_t start = i;
                i++;
                while (i < json.length() && (json[i] != '"' || json[i-1] == '\\')) {
                    i++;
                }
                i++; // Include closing quote

                std::string str_segment = json.substr(start, i - start);

                // Check if it's a key (followed by :)
                size_t next_non_space = i;
                while (next_non_space < json.length() && (json[next_non_space] == ' ' || json[next_non_space] == '\t')) {
                    next_non_space++;
                }

                if (next_non_space < json.length() && json[next_non_space] == ':') {
                    // Key - blue
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.8f, 1.0f, 1.0f));
                } else {
                    // String value - green
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 1.0f, 0.6f, 1.0f));
                }
                ImGui::TextUnformatted(str_segment.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine(0, 0);
            }
            else if (c == ':' || c == ',' || c == '{' || c == '}' || c == '[' || c == ']') {
                // Punctuation - white
                char buf[2] = { c, '\0' };
                ImGui::TextUnformatted(buf);
                ImGui::SameLine(0, 0);
                i++;
            }
            else if (c == '\n') {
                ImGui::NewLine();
                i++;
            }
            else if (isdigit(c) || c == '-' || c == '.') {
                // Number - yellow
                size_t start = i;
                while (i < json.length() && (isdigit(json[i]) || json[i] == '.' || json[i] == '-' || json[i] == 'e' || json[i] == 'E' || json[i] == '+')) {
                    i++;
                }
                std::string num_segment = json.substr(start, i - start);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.4f, 1.0f));
                ImGui::TextUnformatted(num_segment.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine(0, 0);
            }
            else if (json.substr(i, 4) == "true" || json.substr(i, 5) == "false" || json.substr(i, 4) == "null") {
                // Boolean/null - orange
                size_t len = (json[i] == 't' || json[i] == 'n') ? 4 : 5;
                std::string kw = json.substr(i, len);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.6f, 0.4f, 1.0f));
                ImGui::TextUnformatted(kw.c_str());
                ImGui::PopStyleColor();
                ImGui::SameLine(0, 0);
                i += len;
            }
            else {
                // Whitespace or other
                if (c == ' ') {
                    ImGui::TextUnformatted(" ");
                    ImGui::SameLine(0, 0);
                }
                i++;
            }
        }
    }

    ImGui::EndChild();
}

void JSONViewerPanel::RenderTreeView() {
    if (!has_result_ || !result_.is_valid) {
        ImGui::TextDisabled("Valid JSON required for tree view");
        return;
    }

    ImGui::Text(ICON_FA_SITEMAP " JSON Structure");
    ImGui::Separator();

    ImGui::BeginChild("TreeView", ImVec2(0, 0), false);

    // Display top-level keys
    if (!result_.keys.empty()) {
        ImGui::Text("Top-level keys:");
        for (const auto& key : result_.keys) {
            ImGui::BulletText("%s", key.c_str());
        }
    }

    ImGui::Spacing();
    ImGui::Text("Depth: %d", result_.depth);

    ImGui::EndChild();
}

void JSONViewerPanel::RenderStatistics() {
    if (!has_result_) {
        ImGui::TextDisabled("No statistics available");
        return;
    }

    ImGui::Text(ICON_FA_CHART_PIE " JSON Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "stats_cols", false);

    ImGui::Text("Valid:");
    ImGui::NextColumn();
    ImGui::Text("%s", result_.is_valid ? "Yes" : "No");
    ImGui::NextColumn();

    ImGui::Text("Max Depth:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.depth);
    ImGui::NextColumn();

    ImGui::Text("Objects:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.object_count);
    ImGui::NextColumn();

    ImGui::Text("Arrays:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.array_count);
    ImGui::NextColumn();

    ImGui::Text("Strings:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.string_count);
    ImGui::NextColumn();

    ImGui::Text("Numbers:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.number_count);
    ImGui::NextColumn();

    ImGui::Text("Booleans:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.bool_count);
    ImGui::NextColumn();

    ImGui::Text("Nulls:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.null_count);
    ImGui::NextColumn();

    ImGui::Columns(1);

    // Size comparison
    if (result_.is_valid && !result_.formatted_json.empty() && !result_.minified_json.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text(ICON_FA_ARROWS_LEFT_RIGHT " Size Comparison:");
        ImGui::Text("Original: %zu bytes", strlen(input_buffer_));
        ImGui::Text("Formatted: %zu bytes", result_.formatted_json.size());
        ImGui::Text("Minified: %zu bytes", result_.minified_json.size());

        if (!result_.minified_json.empty()) {
            float compression = 1.0f - (static_cast<float>(result_.minified_json.size()) / strlen(input_buffer_));
            ImGui::Text("Space saved: %.1f%%", compression * 100.0f);
        }
    }
}

void JSONViewerPanel::RenderPathQuery() {
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS " JSON Path Query");
    ImGui::Separator();

    ImGui::Text("Enter path (e.g., data.items[0].name):");
    ImGui::SetNextItemWidth(-80);
    bool enter_pressed = ImGui::InputText("##PathInput", path_buffer_, sizeof(path_buffer_),
                                          ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ImGui::Button("Query") || enter_pressed) {
        QueryPathAsync();
    }

    ImGui::Spacing();

    if (!path_result_.empty()) {
        ImGui::Text("Result:");
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s", path_result_.c_str());
        ImGui::PopStyleColor();
    }
}

void JSONViewerPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Processing...");
}

void JSONViewerPanel::ValidateAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    std::string json_text = input_buffer_;

    compute_thread_ = std::make_unique<std::thread>([this, json_text]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = Utilities::ValidateJSON(json_text);
            has_result_ = true;

            if (result_.success && result_.is_valid) {
                spdlog::info("JSON validated successfully");
            } else {
                spdlog::info("JSON validation: invalid");
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void JSONViewerPanel::FormatAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    std::string json_text = input_buffer_;
    int indent = indent_size_;

    compute_thread_ = std::make_unique<std::thread>([this, json_text, indent]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = Utilities::FormatJSON(json_text, indent);
            has_result_ = true;

            if (result_.success) {
                spdlog::info("JSON formatted");
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void JSONViewerPanel::MinifyAsync() {
    if (is_computing_.load()) return;

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    std::string json_text = input_buffer_;

    compute_thread_ = std::make_unique<std::thread>([this, json_text]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = Utilities::MinifyJSON(json_text);
            has_result_ = true;

            if (result_.success) {
                spdlog::info("JSON minified");
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void JSONViewerPanel::QueryPathAsync() {
    std::string path = path_buffer_;
    if (path.empty()) {
        path_result_ = "Enter a path to query";
        return;
    }

    path_result_ = Utilities::GetJSONValue(input_buffer_, path);
    if (path_result_.empty()) {
        path_result_ = "(not found or null)";
    }
}

void JSONViewerPanel::CopyFormatted() {
    if (!result_.formatted_json.empty()) {
        ImGui::SetClipboardText(result_.formatted_json.c_str());
        spdlog::info("Formatted JSON copied to clipboard");
    }
}

void JSONViewerPanel::CopyMinified() {
    if (!result_.minified_json.empty()) {
        ImGui::SetClipboardText(result_.minified_json.c_str());
        spdlog::info("Minified JSON copied to clipboard");
    }
}

void JSONViewerPanel::ClearAll() {
    input_buffer_[0] = '\0';
    path_buffer_[0] = '\0';
    path_result_.clear();
    has_result_ = false;
    error_message_.clear();
    result_ = JSONResult();
}

void JSONViewerPanel::LoadSample() {
    const char* sample = R"({
  "name": "CyxWiz",
  "version": "1.0.0",
  "description": "Decentralized ML Compute Platform",
  "features": [
    "Visual Node Editor",
    "GPU Acceleration",
    "Distributed Training"
  ],
  "config": {
    "backend": "ArrayFire",
    "gpu_enabled": true,
    "max_workers": 10
  },
  "stats": {
    "models_trained": 1234,
    "total_compute_hours": 5678.9
  }
})";

    strcpy(input_buffer_, sample);
    has_result_ = false;
}

} // namespace cyxwiz
