#include "sentiment_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

SentimentPanel::SentimentPanel() {
    GenerateSampleText();
    spdlog::info("SentimentPanel initialized");
}

SentimentPanel::~SentimentPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void SentimentPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_FACE_SMILE " Sentiment Analysis###SentimentPanel", &visible_)) {
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

void SentimentPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Analyze")) {
        AnalyzeAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateSampleText();
        has_result_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        has_batch_result_ = false;
        error_message_.clear();
        result_ = SentimentResult();
        batch_results_.clear();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImVec4 color;
        if (result_.label == "positive") {
            color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
        } else if (result_.label == "negative") {
            color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
        } else {
            color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
        }

        ImGui::TextColored(color, "%s (%.0f%% confidence)",
                          result_.label.c_str(), result_.confidence * 100);
    } else {
        ImGui::TextDisabled("No results");
    }
}

void SentimentPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_SLIDERS " Settings");
    ImGui::Separator();

    const char* methods[] = { "Simple Lexicon", "AFINN" };
    ImGui::Combo("Method", &method_idx_, methods, IM_ARRAYSIZE(methods));

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

    if (ImGui::Button(ICON_FA_PLAY " Analyze Sentiment", ImVec2(-1, 0))) {
        AnalyzeAsync();
    }
}

void SentimentPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter text and click 'Analyze Sentiment'");
        return;
    }

    if (ImGui::BeginTabBar("SentimentTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_GAUGE " Score")) {
            RenderScoreView();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_LIST " Word Analysis")) {
            RenderWordContributions();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void SentimentPanel::RenderScoreView() {
    ImGui::Spacing();

    // Overall sentiment label
    ImVec4 label_color;
    const char* icon;
    if (result_.label == "positive") {
        label_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
        icon = ICON_FA_FACE_SMILE;
    } else if (result_.label == "negative") {
        label_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
        icon = ICON_FA_FACE_FROWN;
    } else {
        label_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
        icon = ICON_FA_FACE_MEH;
    }

    ImGui::PushFont(nullptr);  // Use default font
    float old_scale = ImGui::GetFontSize();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::TextColored(label_color, "%s %s", icon, result_.label.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopFont();

    ImGui::SameLine();
    ImGui::Text("(%.0f%% confidence)", result_.confidence * 100);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Polarity gauge
    ImGui::Text(ICON_FA_ARROW_RIGHT " Polarity:");
    ImGui::SameLine();

    float polarity_normalized = (static_cast<float>(result_.polarity) + 1.0f) / 2.0f;

    // Custom progress bar for polarity
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
        result_.polarity > 0 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
        result_.polarity < 0 ? ImVec4(0.9f, 0.3f, 0.3f, 1.0f) :
        ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

    char polarity_label[32];
    snprintf(polarity_label, sizeof(polarity_label), "%.2f", result_.polarity);
    ImGui::ProgressBar(polarity_normalized, ImVec2(-1, 0), polarity_label);
    ImGui::PopStyleColor();

    ImGui::Text("(-1.0 = Negative)");
    ImGui::SameLine(200);
    ImGui::Text("(0 = Neutral)");
    ImGui::SameLine(400);
    ImGui::Text("(+1.0 = Positive)");

    ImGui::Spacing();

    // Subjectivity gauge
    ImGui::Text(ICON_FA_ARROW_RIGHT " Subjectivity:");
    ImGui::SameLine();

    char subj_label[32];
    snprintf(subj_label, sizeof(subj_label), "%.2f", result_.subjectivity);
    ImGui::ProgressBar(static_cast<float>(result_.subjectivity), ImVec2(-1, 0), subj_label);

    ImGui::Text("(0.0 = Objective)");
    ImGui::SameLine(300);
    ImGui::Text("(1.0 = Subjective)");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Word counts
    ImGui::Text(ICON_FA_CHART_PIE " Sentiment Word Distribution:");

    ImGui::Columns(3, "word_counts", false);

    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_PLUS " Positive");
    ImGui::Text("%d words", result_.positive_count);
    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), ICON_FA_MINUS " Negative");
    ImGui::Text("%d words", result_.negative_count);
    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), ICON_FA_CIRCLE " Neutral");
    ImGui::Text("%d words", result_.neutral_count);
    ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Analysis text
    ImGui::Text(ICON_FA_CIRCLE_INFO " Analysis:");
    ImGui::TextWrapped("%s", result_.analysis.c_str());
}

void SentimentPanel::RenderWordContributions() {
    ImGui::Text(ICON_FA_LIST " Word-Level Sentiment Contributions:");
    ImGui::Separator();
    ImGui::Spacing();

    if (result_.word_scores.empty()) {
        ImGui::TextDisabled("No sentiment words found in text");
        return;
    }

    // Sort by absolute contribution
    std::vector<std::pair<std::string, double>> sorted_scores = result_.word_scores;
    std::sort(sorted_scores.begin(), sorted_scores.end(),
        [](const auto& a, const auto& b) {
            return std::abs(a.second) > std::abs(b.second);
        });

    if (ImGui::BeginTable("WordScores", 3, ImGuiTableFlags_Borders |
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupColumn("Word", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Sentiment", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableHeadersRow();

        for (const auto& word_score : sorted_scores) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", word_score.first.c_str());

            ImGui::TableNextColumn();
            double score = word_score.second;
            if (score > 0) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "+%.2f", score);
            } else if (score < 0) {
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%.2f", score);
            } else {
                ImGui::TextDisabled("0.00");
            }

            ImGui::TableNextColumn();
            if (score > 0.1) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "Positive");
            } else if (score < -0.1) {
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "Negative");
            } else {
                ImGui::TextDisabled("Neutral");
            }
        }

        ImGui::EndTable();
    }
}

void SentimentPanel::RenderBatchAnalysis() {
    ImGui::Text(ICON_FA_LIST " Batch Analysis Results:");
    ImGui::Separator();

    if (batch_results_.empty()) {
        ImGui::TextDisabled("No batch results");
        return;
    }

    if (ImGui::BeginTable("BatchResults", 4, ImGuiTableFlags_Borders |
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30);
        ImGui::TableSetupColumn("Text", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sentiment", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Polarity", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < batch_results_.size(); ++i) {
            const auto& res = batch_results_[i];
            ImGui::TableNextRow();

            ImGui::TableNextColumn();
            ImGui::Text("%d", static_cast<int>(i + 1));

            ImGui::TableNextColumn();
            std::string preview = batch_texts_[i].substr(0, 50);
            if (batch_texts_[i].length() > 50) preview += "...";
            ImGui::Text("%s", preview.c_str());

            ImGui::TableNextColumn();
            ImVec4 color;
            if (res.label == "positive") {
                color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            } else if (res.label == "negative") {
                color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);
            } else {
                color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
            }
            ImGui::TextColored(color, "%s", res.label.c_str());

            ImGui::TableNextColumn();
            ImGui::Text("%.2f", res.polarity);
        }

        ImGui::EndTable();
    }
}

void SentimentPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Analyzing sentiment...");
}

void SentimentPanel::AnalyzeAsync() {
    if (is_computing_.load()) return;

    std::string input_text = text_buffer_;
    if (input_text.empty()) {
        error_message_ = "No input text";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    const char* methods[] = { "simple", "afinn" };
    std::string method = methods[method_idx_];

    compute_thread_ = std::make_unique<std::thread>([this, input_text, method]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = TextProcessing::AnalyzeSentiment(input_text, method);

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Sentiment analysis: {} (polarity={:.2f})",
                            result_.label, result_.polarity);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void SentimentPanel::AnalyzeBatchAsync() {
    if (is_computing_.load()) return;

    if (batch_texts_.empty()) {
        error_message_ = "No batch texts";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_batch_result_ = false;
    error_message_.clear();
    batch_results_.clear();

    const char* methods[] = { "simple", "afinn" };
    std::string method = methods[method_idx_];

    compute_thread_ = std::make_unique<std::thread>([this, method]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            for (const auto& text : batch_texts_) {
                auto res = TextProcessing::AnalyzeSentiment(text, method);
                batch_results_.push_back(res);
            }
            has_batch_result_ = true;
            spdlog::info("Batch sentiment analysis: {} texts", batch_results_.size());
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void SentimentPanel::GenerateSampleText() {
    const char* types[] = { "lorem", "news", "review_positive", "review_negative", "technical" };
    std::string sample = TextProcessing::GenerateSampleText(types[sample_type_idx_]);

    strncpy(text_buffer_, sample.c_str(), sizeof(text_buffer_) - 1);
    text_buffer_[sizeof(text_buffer_) - 1] = '\0';
}

} // namespace cyxwiz
