#include "tfidf_panel.h"
#include "../icons.h"
#include <implot.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

TFIDFPanel::TFIDFPanel() {
    GenerateSampleDocuments();
    spdlog::info("TFIDFPanel initialized");
}

TFIDFPanel::~TFIDFPanel() {
    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }
}

void TFIDFPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(950, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_TABLE " TF-IDF Analysis###TFIDFPanel", &visible_)) {
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

void TFIDFPanel::RenderToolbar() {
    if (ImGui::Button(ICON_FA_PLAY " Compute TF-IDF")) {
        ComputeAsync();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Generate")) {
        GenerateSampleDocuments();
        has_result_ = false;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ERASER " Clear")) {
        has_result_ = false;
        error_message_.clear();
        result_ = TFIDFResult();
    }

    ImGui::SameLine();
    ImGui::Separator();
    ImGui::SameLine();

    if (has_result_) {
        ImGui::Text("Docs: %d | Vocab: %d", result_.num_documents, result_.vocab_size);
    } else {
        ImGui::Text("Documents: %d", static_cast<int>(documents_.size()));
    }
}

void TFIDFPanel::RenderInputPanel() {
    ImGui::Text(ICON_FA_SLIDERS " Settings");
    ImGui::Separator();

    ImGui::Checkbox("Use IDF", &use_idf_);
    ImGui::Checkbox("Smooth IDF", &smooth_idf_);

    const char* norms[] = { "None", "L2", "L1" };
    ImGui::Combo("Normalization", &norm_idx_, norms, IM_ARRAYSIZE(norms));

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_DATABASE " Sample Documents");
    ImGui::Separator();

    const char* doc_types[] = { "News", "Reviews", "Technical" };
    if (ImGui::Combo("Type", &doc_type_idx_, doc_types, IM_ARRAYSIZE(doc_types))) {
        GenerateSampleDocuments();
        has_result_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text(ICON_FA_FILE_LINES " Documents (%d)", static_cast<int>(documents_.size()));
    ImGui::Separator();

    // Document list
    float list_height = ImGui::GetContentRegionAvail().y - 80;
    ImGui::BeginChild("DocList", ImVec2(-1, list_height), true);

    for (size_t i = 0; i < documents_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));

        bool selected = (selected_doc_ == static_cast<int>(i));
        std::string preview = documents_[i].substr(0, 40);
        if (documents_[i].length() > 40) preview += "...";

        char label[64];
        snprintf(label, sizeof(label), "Doc %d: %s", static_cast<int>(i + 1), preview.c_str());

        if (ImGui::Selectable(label, selected)) {
            selected_doc_ = static_cast<int>(i);
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", documents_[i].c_str());
        }

        ImGui::PopID();
    }

    ImGui::EndChild();

    // Document controls
    if (ImGui::Button(ICON_FA_PLUS " Add", ImVec2(-1, 0))) {
        AddDocument();
    }

    if (selected_doc_ >= 0 && selected_doc_ < static_cast<int>(documents_.size())) {
        if (ImGui::Button(ICON_FA_TRASH " Remove Selected", ImVec2(-1, 0))) {
            RemoveDocument(selected_doc_);
        }
    }
}

void TFIDFPanel::RenderResults() {
    if (!error_message_.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::PopStyleColor();
        return;
    }

    if (!has_result_) {
        ImGui::TextDisabled("Add documents and click 'Compute TF-IDF' to see results");
        return;
    }

    if (ImGui::BeginTabBar("TFIDFTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_ARROW_UP " Top Terms")) {
            RenderTopTerms();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_GRIP " Similarity")) {
            RenderSimilarityMatrix();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(ICON_FA_BOOK " Vocabulary")) {
            RenderVocabulary();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void TFIDFPanel::RenderTopTerms() {
    ImGui::Text(ICON_FA_CIRCLE_INFO " Top Terms by TF-IDF Score:");
    ImGui::Separator();
    ImGui::Spacing();

    for (size_t doc_idx = 0; doc_idx < result_.doc_top_terms.size(); ++doc_idx) {
        ImGui::PushID(static_cast<int>(doc_idx));

        if (ImGui::CollapsingHeader(("Document " + std::to_string(doc_idx + 1)).c_str(),
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& top_terms = result_.doc_top_terms[doc_idx];

            if (ImGui::BeginTable("TopTerms", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Term", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("TF-IDF", ImGuiTableColumnFlags_WidthFixed, 80);
                ImGui::TableHeadersRow();

                for (const auto& term_score : top_terms) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", term_score.first.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4f", term_score.second);
                }

                ImGui::EndTable();
            }
        }

        ImGui::PopID();
    }
}

void TFIDFPanel::RenderSimilarityMatrix() {
    if (result_.similarity_matrix.empty()) {
        ImGui::TextDisabled("No similarity data");
        return;
    }

    ImGui::Text(ICON_FA_GRIP " Document Similarity Matrix (Cosine):");
    ImGui::Separator();
    ImGui::Spacing();

    int n = result_.num_documents;

    // Create heatmap data
    std::vector<double> heatmap_data;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            heatmap_data.push_back(result_.similarity_matrix[i][j]);
        }
    }

    // Create labels
    std::vector<std::string> label_strs;
    std::vector<const char*> labels;
    for (int i = 0; i < n; ++i) {
        label_strs.push_back("D" + std::to_string(i + 1));
    }
    for (const auto& s : label_strs) {
        labels.push_back(s.c_str());
    }

    if (ImPlot::BeginPlot("##SimilarityHeatmap", ImVec2(-1, 300))) {
        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_Lock, ImPlotAxisFlags_Lock);
        ImPlot::SetupAxisTicks(ImAxis_X1, 0 + 0.5, n - 0.5, n, labels.data());
        ImPlot::SetupAxisTicks(ImAxis_Y1, 0 + 0.5, n - 0.5, n, labels.data());
        ImPlot::PlotHeatmap("Similarity", heatmap_data.data(), n, n, 0, 1,
                           "%.2f", ImPlotPoint(0, 0), ImPlotPoint(n, n));
        ImPlot::EndPlot();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Table view
    ImGui::Text("Similarity Values:");

    if (ImGui::BeginTable("SimTable", n + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 50);
        for (int i = 0; i < n; ++i) {
            ImGui::TableSetupColumn(("D" + std::to_string(i + 1)).c_str());
        }
        ImGui::TableHeadersRow();

        for (int i = 0; i < n; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("D%d", i + 1);
            for (int j = 0; j < n; ++j) {
                ImGui::TableNextColumn();
                double sim = result_.similarity_matrix[i][j];
                if (i == j) {
                    ImGui::TextDisabled("1.00");
                } else if (sim > 0.7) {
                    ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "%.2f", sim);
                } else if (sim > 0.3) {
                    ImGui::Text("%.2f", sim);
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%.2f", sim);
                }
            }
        }

        ImGui::EndTable();
    }
}

void TFIDFPanel::RenderVocabulary() {
    ImGui::Text(ICON_FA_BOOK " Vocabulary (%d terms):", result_.vocab_size);
    ImGui::Separator();

    // Show IDF scores
    if (ImGui::BeginTable("VocabTable", 3, ImGuiTableFlags_Borders |
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                          ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 40);
        ImGui::TableSetupColumn("Term", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("IDF", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < result_.vocabulary.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d", static_cast<int>(i + 1));
            ImGui::TableNextColumn();
            ImGui::Text("%s", result_.vocabulary[i].c_str());
            ImGui::TableNextColumn();
            if (i < result_.idf_scores.size()) {
                ImGui::Text("%.3f", result_.idf_scores[i]);
            }
        }

        ImGui::EndTable();
    }
}

void TFIDFPanel::RenderLoadingIndicator() {
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() / 2 - 20);
    float width = ImGui::GetWindowWidth();
    ImGui::SetCursorPosX(width / 2 - 80);
    ImGui::Text(ICON_FA_SPINNER " Computing TF-IDF...");
}

void TFIDFPanel::ComputeAsync() {
    if (is_computing_.load()) return;

    if (documents_.empty()) {
        error_message_ = "No documents to analyze";
        return;
    }

    if (compute_thread_ && compute_thread_->joinable()) {
        compute_thread_->join();
    }

    is_computing_ = true;
    has_result_ = false;
    error_message_.clear();

    // Copy documents for thread safety
    std::vector<std::string> docs_copy = documents_;
    bool use_idf = use_idf_;
    bool smooth_idf = smooth_idf_;
    const char* norms[] = { "none", "l2", "l1" };
    std::string norm = norms[norm_idx_];

    compute_thread_ = std::make_unique<std::thread>([this, docs_copy, use_idf, smooth_idf, norm]() {
        std::lock_guard<std::mutex> lock(result_mutex_);

        try {
            result_ = TextProcessing::ComputeTFIDF(docs_copy, use_idf, smooth_idf, norm);

            if (result_.success) {
                has_result_ = true;
                spdlog::info("TF-IDF computed: {} docs, {} terms",
                            result_.num_documents, result_.vocab_size);
            } else {
                error_message_ = result_.error_message;
            }
        } catch (const std::exception& e) {
            error_message_ = std::string("Exception: ") + e.what();
        }

        is_computing_ = false;
    });
}

void TFIDFPanel::GenerateSampleDocuments() {
    const char* types[] = { "news", "reviews", "technical" };
    documents_ = TextProcessing::GenerateSampleDocuments(5, types[doc_type_idx_]);
    selected_doc_ = 0;
}

void TFIDFPanel::AddDocument() {
    documents_.push_back("New document. Enter your text here.");
    selected_doc_ = static_cast<int>(documents_.size()) - 1;
}

void TFIDFPanel::RemoveDocument(int index) {
    if (index >= 0 && index < static_cast<int>(documents_.size())) {
        documents_.erase(documents_.begin() + index);
        if (selected_doc_ >= static_cast<int>(documents_.size())) {
            selected_doc_ = std::max(0, static_cast<int>(documents_.size()) - 1);
        }
        has_result_ = false;
    }
}

} // namespace cyxwiz
