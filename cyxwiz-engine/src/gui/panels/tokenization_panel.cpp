#include "tokenization_panel.h"
#include "../icons.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace cyxwiz {

TokenizationPanel::TokenizationPanel() {
    GenerateSampleText();
    spdlog::info("TokenizationPanel initialized");
}

TokenizationPanel::~TokenizationPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void TokenizationPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_SCISSORS " Tokenization###TokenizationPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.35f, 0), true);
            RenderInputPanel();
            ImGui::EndChild();

            ImGui::SameLine();

            ImGui::BeginChild("ResultsPanel", ImVec2(0, 0), true);
            RenderResults();
            ImGui::EndChild();
        }
    }
    ImGui::End();
}

void TokenizationPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Tokenize")) {
        TokenizeAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateSampleText();
        has_result_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
        result_ = TokenizationResult();
        processed_tokens_.clear();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_COPY " Copy")) {
        CopyToClipboard();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("Tokens: %d | Unique: %d | Avg Length: %.1f",
                    result_.token_count, result_.unique_count, result_.avg_token_length);
    } else {
        ImGui::TextDisabled("No results");
    }
}

void TokenizationPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_SLIDERS " Settings");
    ImGui::Separator();

    // Tokenization method
    const char* methods[] = { "Whitespace", "Word", "Sentence", "N-gram" };
    ImGui::Combo("Method", &method_idx_, methods, IM_ARRAYSIZE(methods));

    if (method_idx_ == 3) {
        ImGui::DragInt("N-gram size", &ngram_n_, 1.0f, 2, 5);
    }

    ImGui::Spacing();

    // Options
    ImGui::Checkbox("Lowercase", &lowercase_);
    ImGui::Checkbox("Remove Punctuation", &remove_punctuation_);
    ImGui::Checkbox("Remove Stopwords", &remove_stopwords_);
    ImGui::Checkbox("Apply Stemming", &apply_stemming_);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_DATABASE " Sample Text");
    ImGui::Separator();

    const char* sample_types[] = { "Lorem Ipsum", "News Article", "Positive Review",
                                   "Negative Review", "Technical" };
    if (ImGui::Combo("Type", &sample_type_idx_, sample_types, IM_ARRAYSIZE(sample_types))) {
        GenerateSampleText();
        has_result_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_KEYBOARD " Input Text");
    ImGui::Separator();

    float available_height = ImGui::GetContentRegionAvail().y - 40;
    ImGui::InputTextMultiline("##InputText", text_buffer_, sizeof(text_buffer_),
                              ImVec2(-1, available_height),
                              ImGuiInputTextFlags_AllowTabInput);

    if (ImGui::Button(ICON_FA_PLAY " Tokenize", ImVec2(-1, 0))) {
        TokenizeAsync();
    }
}

void TokenizationPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter text and click 'Tokenize' to see results");
        return;
    }

    if (ImGui::BeginTabBar("TokenizationTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_LIST " Tokens")) {
            RenderTokenList();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_PIE " Statistics")) {
            RenderStatistics();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void TokenizationPanel::RenderTokenList() {
    ImGui::Text("Tokens (%d):", static_cast<int>(processed_tokens_.size()));
    ImGui::Separator();

    // Token display
    ImGui::BeginChild("TokenList", ImVec2(0, 0), false);

    // Display tokens with wrapping
    float wrap_width = ImGui::GetContentRegionAvail().x;
    float current_x = 0;

    for (size_t i = 0; i < processed_tokens_.size(); ++i) {
        const auto& token = processed_tokens_[i];
        ImVec2 text_size = ImGui::CalcTextSize(token.c_str());
        float token_width = text_size.x + 20;  // padding

        if (current_x + token_width > wrap_width && current_x > 0) {
            current_x = 0;
        } else if (i > 0 && current_x > 0) {
            ImGui::SameLine();
        }

        // Token chip
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.5f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.6f, 1.0f));
        ImGui::SmallButton(token.c_str());
        ImGui::PopStyleColor(2);

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Token %d: \"%s\" (length: %d)",
                             static_cast<int>(i + 1), token.c_str(),
                             static_cast<int>(token.length()));
        }

        current_x += token_width;
    }

    ImGui::EndChild();
}

void TokenizationPanel::RenderStatistics() {
    ImGui::Text(ICON_FA_CHART_PIE " Token Statistics");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Columns(2, "stats_cols", false);

    ImGui::Text("Total Tokens:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.token_count);
    ImGui::NextColumn();

    ImGui::Text("Unique Tokens:");
    ImGui::NextColumn();
    ImGui::Text("%d", result_.unique_count);
    ImGui::NextColumn();

    ImGui::Text("Average Length:");
    ImGui::NextColumn();
    ImGui::Text("%.2f", result_.avg_token_length);
    ImGui::NextColumn();

    ImGui::Text("Type-Token Ratio:");
    ImGui::NextColumn();
    float ttr = result_.token_count > 0 ?
                static_cast<float>(result_.unique_count) / result_.token_count : 0.0f;
    ImGui::Text("%.3f", ttr);
    ImGui::NextColumn();

    ImGui::Text("Method:");
    ImGui::NextColumn();
    ImGui::Text("%s", result_.method.c_str());
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Token length distribution
    ImGui::Text(ICON_FA_CHART_BAR " Length Distribution:");

    if (!processed_tokens_.empty()) {
        std::map<int, int> length_dist;
        for (const auto& token : processed_tokens_) {
            length_dist[static_cast<int>(token.length())]++;
        }

        // Simple bar chart using text
        int max_count = 0;
        for (const auto& pair : length_dist) {
            max_count = std::max(max_count, pair.second);
        }

        for (const auto& pair : length_dist) {
            int bar_width = max_count > 0 ? (pair.second * 30) / max_count : 0;
            std::string bar(bar_width, '#');
            ImGui::Text("Len %2d: %s %d", pair.first, bar.c_str(), pair.second);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Most common tokens
    ImGui::Text(ICON_FA_ARROW_UP " Most Common Tokens:");

    std::map<std::string, int> freq;
    for (const auto& token : processed_tokens_) {
        freq[token]++;
    }

    std::vector<std::pair<std::string, int>> sorted_freq(freq.begin(), freq.end());
    std::sort(sorted_freq.begin(), sorted_freq.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    int show_count = std::min(10, static_cast<int>(sorted_freq.size()));
    for (int i = 0; i < show_count; ++i) {
        ImGui::Text("%d. \"%s\" (%d)", i + 1,
                   sorted_freq[i].first.c_str(), sorted_freq[i].second);
    }
}

void TokenizationPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Tokenizing...");
}

void TokenizationPanel::TokenizeAsync() {
    if (is_computing_.load()) return;

    input_text_ = text_buffer_;
    if (input_text_.empty()) {
        error_message_ = "No input text";
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
            const char* methods[] = { "whitespace", "word", "sentence", "ngram" };
            std::string method = methods[method_idx_];

            result_ = TextProcessing::Tokenize(
                input_text_,
                method,
                ngram_n_,
                lowercase_,
                remove_punctuation_
            );

            if (result_.success) {
                processed_tokens_ = result_.tokens;

                // Apply additional processing
                if (remove_stopwords_) {
                    processed_tokens_ = TextProcessing::RemoveStopwords(processed_tokens_, "english");
                }

                if (apply_stemming_) {
                    processed_tokens_ = TextProcessing::StemWords(processed_tokens_);
                }

                // Update counts after processing
                result_.token_count = static_cast<int>(processed_tokens_.size());
                std::set<std::string> unique_set(processed_tokens_.begin(), processed_tokens_.end());
                result_.unique_count = static_cast<int>(unique_set.size());

                // Recalculate average length
                if (!processed_tokens_.empty()) {
                    double total_len = 0;
                    for (const auto& t : processed_tokens_) {
                        total_len += t.length();
                    }
                    result_.avg_token_length = total_len / processed_tokens_.size();
                }

                has_result_ = true;
                spdlog::info("Tokenization complete: {} tokens ({} unique)",
                            result_.token_count, result_.unique_count);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void TokenizationPanel::GenerateSampleText() {
    const char* types[] = { "lorem", "news", "review_positive", "review_negative", "technical" };
    std::string sample = TextProcessing::GenerateSampleText(types[sample_type_idx_]);

    strncpy(text_buffer_, sample.c_str(), sizeof(text_buffer_) - 1);
    text_buffer_[sizeof(text_buffer_) - 1] = '\0';
}

void TokenizationPanel::CopyToClipboard() {
    if (!has_result_ || processed_tokens_.empty()) return;

    std::ostringstream oss;
    for (size_t i = 0; i < processed_tokens_.size(); ++i) {
        if (i > 0) oss << "\n";
        oss << processed_tokens_[i];
    }

    ImGui::SetClipboardText(oss.str().c_str());
    spdlog::info("Tokens copied to clipboard");
}

} // namespace cyxwiz
