// Data Input dialog: "Sequence (token tagging)" category (TOFIX112). A
// token-tagging dataset is a tabular file whose token, POS and tag columns
// hold space-separated per-token values; the sequence Data Input carries
// the tagging contract the compiler and sequence batcher read.
#include "node_config_dialog.h"

#include "../core/sequence_fusion_presentation.h"
#include "ui_buttons.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace gui {

namespace {

constexpr const char* kSequenceCategory = "sequence_text";

// Every key the sequence contract owns, canonical and older aliases.
constexpr const char* kSequenceKeys[] = {
    "token_column", "tokens_column", "token_sequence_column",
    "pos_column", "pos_sequence_column",
    "tag_column", "tags_column", "tag_sequence_column",
    "sentence_id_column", "sequence_id_column",
    "max_sequence_length", "create_attention_mask",
};

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

// First column whose lower-cased name is one of `names`.
std::string FindColumn(const std::vector<std::string>& columns,
                       std::initializer_list<const char*> names) {
    for (const char* name : names) {
        for (const auto& column : columns) {
            if (Lower(column) == name) return column;
        }
    }
    return {};
}

}  // namespace

void DataInputDialog::LoadSequenceSettings() {
    const auto contract = cyxwiz::ReadSequenceTaggingContract(*node_);
    sequence_tagging_ = contract.is_sequence;
    sequence_loaded_ = contract.is_sequence;
    sequence_token_column_ = contract.token_column;
    sequence_pos_column_ = contract.pos_column;
    sequence_tag_column_ = contract.tag_column;
    sequence_sentence_column_ = contract.sentence_id_column;
    sequence_max_length_ = contract.max_sequence_length;
    sequence_attention_mask_ = contract.is_sequence ? contract.create_attention_mask : true;
}

void DataInputDialog::RenderSequenceColumns() {
    if (!ImGui::CollapsingHeader("Sequence columns", ImGuiTreeNodeFlags_DefaultOpen)) return;
    const ImVec4 muted = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
    ImGui::TextColored(muted,
        "One row per sentence; each column holds space-separated values, one per token.");
    ImGui::Spacing();

    // Suggest the conventional column names when they are present and differ.
    const std::string token = FindColumn(available_columns_, {"tokens", "token", "words", "word"});
    const std::string pos = FindColumn(available_columns_, {"pos_tags", "pos", "pos_tag"});
    const std::string tag = FindColumn(available_columns_, {"ner_tags", "tags", "tag", "labels", "ner"});
    const std::string sentence = FindColumn(available_columns_, {"sentence_id", "sentence", "sequence_id"});
    const bool suggestion_differs = (!token.empty() && token != sequence_token_column_) ||
                                    (!pos.empty() && pos != sequence_pos_column_) ||
                                    (!tag.empty() && tag != sequence_tag_column_) ||
                                    (!sentence.empty() && sentence != sequence_sentence_column_);
    if (!token.empty() && !tag.empty() && suggestion_differs) {
        std::string found = token;
        if (!pos.empty()) found += ", " + pos;
        found += ", " + tag;
        if (!sentence.empty()) found += ", " + sentence;
        ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x -
                               cyxwiz::ui::ButtonWidth("Use", cyxwiz::ui::ButtonSize::Small) - 12.0f);
        ImGui::TextColored(ImVec4(0.81f, 0.78f, 1.0f, 1.0f),
                           "Suggestion: %s look like the token, POS, tag and sentence columns.",
                           found.c_str());
        ImGui::PopTextWrapPos();
        ImGui::SameLine();
        if (cyxwiz::ui::SecondaryButton("Use##sequence_suggestion")) {
            sequence_token_column_ = token;
            if (!pos.empty()) sequence_pos_column_ = pos;
            sequence_tag_column_ = tag;
            if (!sentence.empty()) sequence_sentence_column_ = sentence;
            has_changes_ = true;
        }
        ImGui::Spacing();
    }

    const auto column_combo = [this, &muted](const char* label, const char* id, std::string& value,
                                              bool optional, const char* hint_set, const char* hint_none) {
        ImGui::TextUnformatted(label);
        ImGui::SetNextItemWidth(260.0f);
        const std::string preview = value.empty() ? (optional ? "(None)" : "(select a column)") : value;
        if (ImGui::BeginCombo(id, preview.c_str())) {
            if (optional && ImGui::Selectable("(None)", value.empty())) {
                value.clear();
                has_changes_ = true;
            }
            for (const auto& column : available_columns_) {
                if (ImGui::Selectable(column.c_str(), column == value)) {
                    value = column;
                    has_changes_ = true;
                }
            }
            ImGui::EndCombo();
        }
        const bool missing = !value.empty() && !available_columns_.empty() &&
                             std::find(available_columns_.begin(), available_columns_.end(), value) ==
                                 available_columns_.end();
        if (missing) {
            ImGui::TextColored(ImVec4(0.96f, 0.63f, 0.29f, 1.0f), "'%s' is not a column of this file.",
                               value.c_str());
        } else if (value.empty() && hint_none[0] != '\0') {
            ImGui::TextColored(ImVec4(0.90f, 0.75f, 0.29f, 1.0f), "%s", hint_none);
        } else {
            ImGui::TextColored(muted, "%s", hint_set);
        }
    };

    if (ImGui::BeginTable("##sequence_columns", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        column_combo("Token column", "##seq_token", sequence_token_column_, false,
                     "Word ids come from this column.", "Required: choose the token column.");
        ImGui::TableNextColumn();
        column_combo("POS column (optional)", "##seq_pos", sequence_pos_column_, true,
                     "POS ids feed a POS Embedding (Concatenate Input 2).",
                     "No POS ids: a word + POS Concatenate will not compile.");
        ImGui::TableNextColumn();
        column_combo("Tag column", "##seq_tag", sequence_tag_column_, false,
                     "Per-token labels (BIO tags).", "Required: choose the tag column.");
        ImGui::TableNextColumn();
        column_combo("Sentence id column (optional)", "##seq_sentence", sequence_sentence_column_, true,
                     "Groups rows into sentences.", "Each row is one sentence.");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("Max sequence length");
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("##seq_max_length", &sequence_max_length_, 0, 0)) {
            sequence_max_length_ = std::max(0, sequence_max_length_);
            has_changes_ = true;
        }
        ImGui::TextColored(muted, "Longer sentences are cut; 0 = the longest in the data.");
        ImGui::TableNextColumn();
        ImGui::Dummy(ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing()));
        if (ImGui::Checkbox("Create attention mask", &sequence_attention_mask_)) has_changes_ = true;
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(muted,
        "Labels: the tag column becomes the per-token labels. The batcher builds the word, "
        "POS and tag vocabularies from this data.");
    ImGui::PopTextWrapPos();
}

void DataInputDialog::RenderSequenceSummary() {
    if (!sequence_tagging_) return;
    ImGui::Spacing();
    ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_HeaderActive], "SEQUENCE");
    ImGui::Separator();
    const auto row = [](const char* key, const std::string& value) {
        ImGui::TextUnformatted(key);
        ImGui::SameLine(120);
        ImGui::TextUnformatted(value.empty() ? "-" : value.c_str());
    };
    row("Tokens:", sequence_token_column_);
    row("POS:", sequence_pos_column_);
    row("Tags:", sequence_tag_column_);
    row("Max length:", sequence_max_length_ > 0 ? std::to_string(sequence_max_length_) : "longest");
    row("Attention mask:", sequence_attention_mask_ ? "yes" : "no");
}

void DataInputDialog::ApplySequenceSettings() {
    auto& params = node_->parameters;
    if (!sequence_tagging_) {
        // Switched away from Sequence: the contract keys would otherwise keep
        // the compiler treating this dataset as token tagging.
        if (sequence_loaded_) {
            for (const char* key : kSequenceKeys) params.erase(key);
        }
        return;
    }
    params["file_category"] = kSequenceCategory;
    for (const char* key : kSequenceKeys) params.erase(key);
    params["token_column"] = sequence_token_column_;
    params["pos_column"] = sequence_pos_column_;
    params["tag_column"] = sequence_tag_column_;
    params["sentence_id_column"] = sequence_sentence_column_;
    params["max_sequence_length"] = std::to_string(sequence_max_length_);
    params["create_attention_mask"] = sequence_attention_mask_ ? "true" : "false";
    // Per-token labels come from the tag column, not a table label column.
    params["label_column"] = "";
    sequence_loaded_ = true;
}

}  // namespace gui
