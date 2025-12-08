#include "word_frequency_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>

namespace cyxwiz {

WordFrequencyPanel::WordFrequencyPanel() {
    GenerateSampleText();
    spdlog::info("WordFrequencyPanel initialized");
}

WordFrequencyPanel::~WordFrequencyPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void WordFrequencyPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_BAR " Word Frequency###WordFrequencyPanel", &visible_)) {
        RenderToolbar();
        ImGui::Separator();

        if (is_computing_.load()) {
            RenderLoadingIndicator();
        } else {
            float panel_width = ImGui::GetContentRegionAvail().x;

            ImGui::BeginChild("InputPanel", ImVec2(panel_width * 0.3f, 0), true);
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

void WordFrequencyPanel::RenderToolbar() {
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
        error_message_.clear();
        result_ = WordFrequencyResult();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("Words: %d | Unique: %d | TTR: %.3f",
                    result_.total_words, result_.unique_words, result_.type_token_ratio);
    } else {
        ImGui::TextDisabled("No results");
    }
}

void WordFrequencyPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_SLIDERS " Settings");
    ImGui::Separator();

    ImGui::DragInt("Top N", &top_n_, 1.0f, 5, 100);
    ImGui::DragInt("Min Length", &min_word_length_, 1.0f, 1, 10);
    ImGui::Checkbox("Remove Stopwords", &remove_stopwords_);

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

    if (ImGui::Button(ICON_FA_PLAY " Analyze", ImVec2(-1, 0))) {
        AnalyzeAsync();
    }
}

void WordFrequencyPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter text and click 'Analyze' to see results");
        return;
    }

    if (ImGui::BeginTabBar("FrequencyTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_CHART_BAR " Chart")) {
            RenderBarChart();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Table")) {
            RenderFrequencyTable();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_AREA " Length Dist")) {
            RenderLengthDistribution();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void WordFrequencyPanel::RenderBarChart() {
    if (result_.frequencies.empty()) {
        ImGui::TextDisabled("No frequency data");
        return;
    }

    // Prepare data for horizontal bar chart
    int num_words = std::min(static_cast<int>(result_.frequencies.size()), 20);

    std::vector<double> counts(num_words);
    std::vector<const char*> labels(num_words);

    for (int i = 0; i < num_words; ++i) {
        counts[i] = static_cast<double>(result_.frequencies[i].second);
        labels[i] = result_.frequencies[i].first.c_str();
    }

    if (ImPlot::BeginPlot("##FreqChart", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Frequency", "Word");

        // Plot horizontal bars
        std::vector<double> positions(num_words);
        for (int i = 0; i < num_words; ++i) {
            positions[i] = static_cast<double>(num_words - 1 - i);
        }

        ImPlot::SetupAxisTicks(ImAxis_Y1, positions.data(), num_words, labels.data());
        ImPlot::PlotBars("Count", counts.data(), num_words, 0.6, 0, ImPlotBarsFlags_Horizontal);

        ImPlot::EndPlot();
    }
}

void WordFrequencyPanel::RenderFrequencyTable() {
    // Statistics summary
    ImGui::Text(ICON_FA_CIRCLE_INFO " Summary:");
    ImGui::Columns(2, "summary_cols", false);
    ImGui::Text("Total Words:"); ImGui::NextColumn();
    ImGui::Text("%d", result_.total_words); ImGui::NextColumn();
    ImGui::Text("Unique Words:"); ImGui::NextColumn();
    ImGui::Text("%d", result_.unique_words); ImGui::NextColumn();
    ImGui::Text("Type-Token Ratio:"); ImGui::NextColumn();
    ImGui::Text("%.4f", result_.type_token_ratio); ImGui::NextColumn();
    ImGui::Text("Avg Word Length:"); ImGui::NextColumn();
    ImGui::Text("%.2f", result_.avg_word_length); ImGui::NextColumn();
    ImGui::Text("Most Common:"); ImGui::NextColumn();
    ImGui::Text("\"%s\" (%d)", result_.most_common_word.c_str(), result_.max_frequency); ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Frequency table
    ImGui::Text(ICON_FA_LIST " Frequency Table:");

    if (ImGui::BeginTable("FreqTable", 3, ImGuiTableFlags_Borders |
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Word", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.frequencies.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", static_cast<int>(i + 1));
            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.frequencies[i].first.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", result_.frequencies[i].second);
        }

        ImGui::EndTable();
    }
}

void WordFrequencyPanel::RenderLengthDistribution() {
    if (result_.length_distribution.empty()) {
        ImGui::TextDisabled("No length distribution data");
        return;
    }

    // Prepare data
    std::vector<double> lengths;
    std::vector<double> counts;

    for (const auto& pair : result_.length_distribution) {
        lengths.push_back(static_cast<double>(pair.first));
        counts.push_back(static_cast<double>(pair.second));
    }

    ImGui::Text("Word Length Distribution:");

    if (ImPlot::BeginPlot("##LengthDist", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Word Length", "Count");
        ImPlot::PlotBars("Count", lengths.data(), counts.data(),
                        static_cast<int>(lengths.size()), 0.6);
        ImPlot::EndPlot();
    }
}

void WordFrequencyPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Analyzing...");
}

void WordFrequencyPanel::AnalyzeAsync() {
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

    compute_thread_ = std::make_unique<std::thread>([this, input_text]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = TextProcessing::ComputeWordFrequency(
                input_text,
                top_n_,
                remove_stopwords_,
                min_word_length_
            );

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Word frequency analysis: {} total, {} unique",
                            result_.total_words, result_.unique_words);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void WordFrequencyPanel::GenerateSampleText() {
    const char* types[] = { "lorem", "news", "review_positive", "review_negative", "technical" };
    std::string sample = TextProcessing::GenerateSampleText(types[sample_type_idx_]);

    strncpy(text_buffer_, sample.c_str(), sizeof(text_buffer_) - 1);
    text_buffer_[sizeof(text_buffer_) - 1] = '\0';
}

} // namespace cyxwiz
