#include "embeddings_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace cyxwiz {

EmbeddingsPanel::EmbeddingsPanel() {
    GenerateSampleVocabulary();
    spdlog::info("EmbeddingsPanel initialized");
}

EmbeddingsPanel::~EmbeddingsPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void EmbeddingsPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(900, 600), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CUBE " Word Embeddings###EmbeddingsPanel", &visible_)) {
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

void EmbeddingsPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Create Embeddings")) {
        CreateEmbeddingsAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateSampleVocabulary();
        has_result_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
        result_ = EmbeddingResult();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("Words: %d | Dim: %d | Method: %s",
                    static_cast<int>(result_.words.size()),
                    result_.embedding_dim,
                    result_.method.c_str());
    } else {
        ImGui::TextDisabled("No embeddings");
    }
}

void EmbeddingsPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_SLIDERS " Settings");
    ImGui::Separator();

    const char* methods[] = { "One-Hot", "Random" };
    ImGui::Combo("Method", &method_idx_, methods, IM_ARRAYSIZE(methods));

    if (method_idx_ == 1) {
        ImGui::DragInt("Dimensions", &embedding_dim_, 1.0f, 10, 300);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_DATABASE " Sample Vocabulary");
    ImGui::Separator();

    const char* domains[] = { "General", "Tech", "Science" };
    if (ImGui::Combo("Domain", &vocab_domain_idx_, domains, IM_ARRAYSIZE(domains))) {
        GenerateSampleVocabulary();
        has_result_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_KEYBOARD " Vocabulary (one word per line)");
    ImGui::Separator();

    float vocab_height = ImGui::GetContentRegionAvail().y - 120;
    ImGui::InputTextMultiline("##VocabInput", vocab_buffer_, sizeof(vocab_buffer_),
                              ImVec2(-1, vocab_height),
                              ImGuiInputTextFlags_AllowTabInput);

    if (ImGui::Button(ICON_FA_PLAY " Create", ImVec2(-1, 0))) {
        CreateEmbeddingsAsync();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS " Find Similar Words");
    ImGui::Separator();

    ImGui::InputText("Query", query_word_, sizeof(query_word_));
    ImGui::DragInt("Top N", &top_n_similar_, 1.0f, 1, 20);

    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Find", ImVec2(-1, 0))) {
        FindSimilarAsync();
    }
}

void EmbeddingsPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Enter vocabulary and click 'Create Embeddings'");
        return;
    }

    if (ImGui::BeginTabBar("EmbeddingTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_TABLE " Vectors")) {
            RenderVectorView();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_MAGNIFYING_GLASS " Similar")) {
            RenderSimilarWords();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_CHART_SCATTER " 2D Plot")) {
            Render2DPlot();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void EmbeddingsPanel::RenderVectorView() {
    ImGui::Text(ICON_FA_TABLE " Word Vectors (%d words, %d dims):",
               static_cast<int>(result_.words.size()), result_.embedding_dim);
    ImGui::Separator();

    // Show a table of words and their embedding vectors (first few dimensions)
    int show_dims = std::min(10, result_.embedding_dim);

    if (ImGui::BeginTable("VectorTable", show_dims + 1,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupColumn("Word", ImGuiTableColumnFlags_WidthFixed, 100);
        for (int i = 0; i < show_dims; ++i) {
            char header[16];
            snprintf(header, sizeof(header), "D%d", i + 1);
            ImGui::TableSetupColumn(header, ImGuiTableColumnFlags_WidthFixed, 60);
        }
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.words.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.words[i].c_str());

            for (int j = 0; j < show_dims && j < static_cast<int>(result_.embeddings[i].size()); ++j) {
                ImGui::TableNextColumn();
                double val = result_.embeddings[i][j];
                if (val > 0) {
                    ImGui::TextColored(ImVec4(0.2f, 0.6f, 0.2f, 1.0f), "%.2f", val);
                } else if (val < 0) {
                    ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%.2f", val);
                } else {
                    ImGui::TextDisabled("0.00");
                }
            }
        }

        ImGui::EndTable();
    }
}

void EmbeddingsPanel::RenderSimilarWords() {
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS " Similar Words:");
    ImGui::Separator();
    ImGui::Spacing();

    if (result_.similar_words.empty()) {
        ImGui::TextDisabled("Enter a query word and click 'Find' to see similar words");
        return;
    }

    ImGui::Text("Similar to \"%s\":", query_word_);
    ImGui::Spacing();

    if (ImGui::BeginTable("SimilarTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Word", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Similarity", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.similar_words.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", static_cast<int>(i + 1));
            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.similar_words[i].first.c_str());
            ImGui::TableNextColumn();

            double sim = result_.similar_words[i].second;
            if (sim > 0.7) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.4f", sim);
            } else if (sim > 0.3) {
                ImGui::Text("%.4f", sim);
            } else {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%.4f", sim);
            }
        }

        ImGui::EndTable();
    }
}

void EmbeddingsPanel::Render2DPlot() {
    if (result_.embeddings.empty() || result_.embedding_dim < 2) {
        ImGui::TextDisabled("Need at least 2D embeddings for plot");
        return;
    }

    ImGui::Text(ICON_FA_CHART_SCATTER " 2D Projection (First 2 Dimensions):");
    ImGui::Separator();

    // Extract first 2 dimensions
    std::vector<double> x_vals, y_vals;
    for (const auto& emb : result_.embeddings) {
        x_vals.push_back(emb[0]);
        y_vals.push_back(emb.size() > 1 ? emb[1] : 0.0);
    }

    if (ImPlot::BeginPlot("##EmbeddingPlot", ImVec2(-1, -1))) {
        ImPlot::SetupAxes("Dimension 1", "Dimension 2");

        // Plot points
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6);
        ImPlot::PlotScatter("Words", x_vals.data(), y_vals.data(),
                           static_cast<int>(x_vals.size()));

        // Add word labels
        for (size_t i = 0; i < result_.words.size(); ++i) {
            ImPlot::PlotText(result_.words[i].c_str(),
                            x_vals[i], y_vals[i] + 0.05);
        }

        ImPlot::EndPlot();
    }
}

void EmbeddingsPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Creating embeddings...");
}

void EmbeddingsPanel::CreateEmbeddingsAsync() {
    if (is_computing_.load()) return;

    // Parse vocabulary from buffer
    std::vector<std::string> vocabulary;
    std::istringstream iss(vocab_buffer_);
    std::string word;
    while (std::getline(iss, word)) {
        // Trim whitespace
        size_t start = word.find_first_not_of(" \t\r\n");
        size_t end = word.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            vocabulary.push_back(word.substr(start, end - start + 1));
        }
    }

    if (vocabulary.empty()) {
        error_message_ = "No vocabulary words provided";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    int method = method_idx_;
    int dim = embedding_dim_;

    compute_thread_ = std::make_unique<std::thread>([this, vocabulary, method, dim]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            if (method == 0) {
                result_ = TextProcessing::CreateOneHotEmbeddings(vocabulary);
            } else {
                result_ = TextProcessing::CreateRandomEmbeddings(vocabulary, dim);
            }

            if (result_.success) {
                has_result_ = true;
                spdlog::info("Created {} embeddings with {} dimensions",
                            result_.words.size(), result_.embedding_dim);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void EmbeddingsPanel::FindSimilarAsync() {
    if (!has_result_ || result_.embeddings.empty()) {
        error_message_ = "Create embeddings first";
        return;
    }

    std::string query = query_word_;
    if (query.empty()) {
        error_message_ = "Enter a query word";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    error_message_.clear();

    int top_n = top_n_similar_;
    EmbeddingResult current_result = result_;

    compute_thread_ = std::make_unique<std::thread>([this, query, top_n, current_result]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            EmbeddingResult similar_result = TextProcessing::FindSimilarWords(
                query, current_result, top_n);

            if (similar_result.success) {
                result_.similar_words = similar_result.similar_words;
                spdlog::info("Found {} similar words to '{}'",
                            result_.similar_words.size(), query);
            } else {
                error_message_ = similar_result.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void EmbeddingsPanel::GenerateSampleVocabulary() {
    const char* domains[] = { "general", "tech", "science" };
    auto words = TextProcessing::GenerateSampleVocabulary(30, domains[vocab_domain_idx_]);

    std::ostringstream oss;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) oss << "\n";
        oss << words[i];
    }

    strncpy(vocab_buffer_, oss.str().c_str(), sizeof(vocab_buffer_) - 1);
    vocab_buffer_[sizeof(vocab_buffer_) - 1] = '\0';

    if (!words.empty()) {
        strncpy(query_word_, words[0].c_str(), sizeof(query_word_) - 1);
        query_word_[sizeof(query_word_) - 1] = '\0';
    }
}

} // namespace cyxwiz
