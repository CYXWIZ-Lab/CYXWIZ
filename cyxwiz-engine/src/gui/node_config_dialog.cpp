#include "node_config_dialog.h"
#include "quality_analyzer_dialog.h"
#include "word_embeddings_dialog.h"
#include "embedding_dialog.h"
#include "node_editor.h"
#include "separate_windows.h"
#include "sql_workbench.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/project_data_path.h"
#include "../core/project_manager.h"
#include <cyxwiz/tokenizer.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <set>
#include <unordered_map>

namespace gui {

// Alias to avoid any potential macro conflicts
using NT = NodeType;

// ==================== NodeConfigDialog Base ====================

NodeConfigDialog::NodeConfigDialog(const std::string& title, MLNode* node)
    : title_(title), node_(node)
{
    if (node_) {
        original_params_ = node_->parameters;
    }
}

void NodeConfigDialog::Open() {
    is_open_ = true;
    first_open_ = true;
    // In front also when its window is already open behind another one (a
    // dock tab, the canvas raised by the same double-click).
    ::gui::BringToFront(title_);
    has_changes_ = false;
    if (node_) {
        original_params_ = node_->parameters;
    }
}

void NodeConfigDialog::Close() {
    is_open_ = false;
    OnClose();
}

bool NodeConfigDialog::Render() {
    if (!is_open_) return false;

    bool should_close = false;

    // Set initial size on first open
    if (first_open_) {
        ImVec2 size = GetDefaultSize();
        ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
        first_open_ = false;
    }

    // Window flags
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse;

    // Begin the dialog window
    const bool visible = ImGui::Begin(title_.c_str(), &is_open_, flags);
    if (visible && !UsesStandardButtons()) {
        RenderContent();
    } else if (visible) {
        // Content area
        ImVec2 avail = ImGui::GetContentRegionAvail();

        // Main content (excluding button area)
        ImGui::BeginChild("DialogContent", ImVec2(0, avail.y - FooterHeight()), false);
        RenderContent();
        ImGui::EndChild();

        RenderFooter(should_close);
    }
    ImGui::End();

    if (should_close || !is_open_) {
        Close();
    }

    return is_open_;
}

const char* NodeConfigDialog::ApplyBlockedReason() const {
    if (IsBusy()) return "Wait for it to finish";
    if (!CanApply()) return "Nothing to apply here";
    return nullptr;
}

float NodeConfigDialog::FooterHeight() const {
    const auto& t = cyxwiz::ui::CurrentTokens();
    return ImGui::GetFontSize() + t.button_padding.y * 2.0f + ImGui::GetStyle().ItemSpacing.y * 3.0f;
}

void NodeConfigDialog::RenderFooter(bool& should_close) {
    namespace ui = cyxwiz::ui;
    ImGui::Dummy(ImVec2(0, 2.0f));
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float width = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular) +
                        ui::ButtonWidth("Apply", ui::ButtonSize::Regular) +
                        ui::ButtonWidth("OK", ui::ButtonSize::Regular) + spacing * 2.0f;
    ImGui::SetCursorPosX((std::max)(ImGui::GetCursorPosX(), ImGui::GetContentRegionMax().x - width));
    // Cancel stays on while busy, so the user can always leave.
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Regular)) {
        Reset();
        should_close = true;
    }
    const char* blocked = ApplyBlockedReason();
    const bool unchanged = TracksChanges() && !has_changes_;
    ImGui::SameLine();
    if (ui::SecondaryButton("Apply", !blocked && !unchanged, blocked ? blocked : "Nothing changed",
                            ui::ButtonSize::Regular))
        Apply();
    ImGui::SameLine();
    if (ui::PrimaryButton("OK", !blocked, blocked)) {
        Apply();
        should_close = true;
    }
}

void NodeConfigDialog::ValidationMessage(const std::string& message, bool is_error) {
    const cyxwiz::ui::Tokens& t = cyxwiz::ui::CurrentTokens();
    ImVec4 color = is_error ? t.error : t.success;
    ImGui::TextColored(color, "%s", message.c_str());
}

// ==================== TokenizerDialog ====================

namespace {

std::string TextDialogTitle(const MLNode* node) {
    if (!node) return "Text Preprocessing Configuration";
    switch (node->type) {
        case NodeType::TextVocabulary:
            return "Text Vocabulary Configuration";
        case NodeType::TextPadding:
            return "Text Padding Configuration";
        case NodeType::TextTokenizer:
        default:
            return "Text Tokenizer Configuration";
    }
}

void CopyParam(const MLNode* node, const char* key, char* dest, size_t dest_size) {
    if (!node || dest_size == 0) return;
    auto it = node->parameters.find(key);
    if (it == node->parameters.end()) return;
    std::strncpy(dest, it->second.c_str(), dest_size - 1);
    dest[dest_size - 1] = '\0';
}

void ReadIntParam(const MLNode* node, const char* key, int& value) {
    if (!node) return;
    auto it = node->parameters.find(key);
    if (it == node->parameters.end() || it->second.empty()) return;
    try {
        value = std::stoi(it->second);
    } catch (...) {
    }
}

void ReadBoolParam(const MLNode* node, const char* key, bool& value) {
    if (!node) return;
    auto it = node->parameters.find(key);
    if (it == node->parameters.end() || it->second.empty()) return;
    value = (it->second == "true" || it->second == "1");
}

const char* MethodNameFromType(int tokenizer_type) {
    switch (tokenizer_type) {
        case 0: return "whitespace";
        case 2: return "character";
        case 3: return "byte_bpe";
        case 4: return "wordpiece";
        case 1:
        default:
            return "word";
    }
}

std::vector<std::string> TokenizePreviewText(const std::string& text,
                                             int tokenizer_type,
                                             bool lowercase) {
    std::string normalized = text;
    if (lowercase) {
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
    }

    std::vector<std::string> tokens;
    if (tokenizer_type == 2) {
        for (char c : normalized) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                tokens.emplace_back(1, c);
            }
        }
        return tokens;
    }

    if (tokenizer_type == 0) {
        std::istringstream iss(normalized);
        std::string token;
        while (iss >> token) tokens.push_back(token);
        return tokens;
    }

    std::string current;
    for (char c : normalized) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_') {
            current.push_back(c);
        } else {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            if (!std::isspace(uc)) {
                tokens.emplace_back(1, c);
            }
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

std::vector<std::string> SplitCsvLineSimple(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

bool CsvRecordHasOpenQuote(const std::string& record) {
    bool in_quotes = false;
    for (size_t i = 0; i < record.size(); ++i) {
        if (record[i] != '"') continue;
        if (in_quotes && i + 1 < record.size() && record[i + 1] == '"') {
            ++i;
            continue;
        }
        in_quotes = !in_quotes;
    }
    return in_quotes;
}

bool ReadCsvRecord(std::istream& in, std::string& record) {
    record.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (!record.empty()) {
            record.push_back('\n');
        }
        record += line;
        if (!CsvRecordHasOpenQuote(record)) {
            return true;
        }
    }
    return !record.empty();
}

int FindColumnIndex(const std::vector<std::string>& headers,
                    const std::string& name) {
    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == name) return static_cast<int>(i);
    }
    return -1;
}

} // namespace

TokenizerDialog::TokenizerDialog(MLNode* node)
    : NodeConfigDialog(TextDialogTitle(node), node)
{
    LoadFromNode();
}

bool TokenizerDialog::IsTokenizerNode() const {
    return !node_ || node_->type == NodeType::TextTokenizer;
}

bool TokenizerDialog::IsVocabularyNode() const {
    return node_ && node_->type == NodeType::TextVocabulary;
}

bool TokenizerDialog::IsPaddingNode() const {
    return node_ && node_->type == NodeType::TextPadding;
}

void TokenizerDialog::LoadFromNode() {
    if (!node_) return;

    tokenizer_type_ = 1;
    bpe_piece_policy_.clear();
    bpe_initial_unit_.clear();
    token_output_mode_ = 0;
    std::copy_n("document_id", sizeof("document_id"), document_id_col_);
    std::copy_n("split", sizeof("split"), split_col_);
    target_text_col_[0] = '\0';
    std::copy_n("token_ids", sizeof("token_ids"), token_ids_col_);
    max_length_ = IsPaddingNode() ? 512 : 256;
    max_vocab_size_ = IsVocabularyNode() ? -1 : 10000;
    min_word_freq_ = IsVocabularyNode() ? 1 : 2;
    pad_value_ = 0;
    lowercase_ = true;
    padding_ = true;
    truncation_ = true;
    vocab_build_if_missing_ = false;
    std::strncpy(text_col_, "text", sizeof(text_col_) - 1);
    text_col_[sizeof(text_col_) - 1] = '\0';
    label_col_[0] = '\0';
    vocab_file_[0] = '\0';
    source_csv_[0] = '\0';
    status_message_.clear();
    status_is_error_ = false;

    auto method = node_->parameters.find("method");
    if (method != node_->parameters.end()) {
        if (method->second == "whitespace") tokenizer_type_ = 0;
        else if (method->second == "word") tokenizer_type_ = 1;
        else if (method->second == "character") tokenizer_type_ = 2;
        else if (method->second == "byte_bpe") tokenizer_type_ = 3;
        else if (method->second == "wordpiece") tokenizer_type_ = 4;
        else if (method->second == "sentencepiece_bpe") tokenizer_type_ = 5;
        else if (method->second == "sentencepiece_unigram") tokenizer_type_ = 6;
    }

    if (node_->parameters.count("text_col")) {
        CopyParam(node_, "text_col", text_col_, sizeof(text_col_));
    } else {
        CopyParam(node_, "text_column", text_col_, sizeof(text_col_));
    }
    CopyParam(node_, "label_col", label_col_, sizeof(label_col_));
    CopyParam(node_, "vocab_file", vocab_file_, sizeof(vocab_file_));
    CopyParam(node_, "source_csv", source_csv_, sizeof(source_csv_));

    if (auto mode=node_->parameters.find("output_mode"); mode!=node_->parameters.end()) {
        if (mode->second == "causal_windows") token_output_mode_ = 1;
        else if (mode->second == "decode") token_output_mode_ = 2;
        else if (mode->second == "roundtrip") token_output_mode_ = 3;
        else token_output_mode_ = 0;
    }
    CopyParam(node_, "document_id_col", document_id_col_, sizeof(document_id_col_));
    CopyParam(node_, "split_col", split_col_, sizeof(split_col_));
    CopyParam(node_, "target_text_col", target_text_col_, sizeof(target_text_col_));
    CopyParam(node_, "token_ids_col", token_ids_col_, sizeof(token_ids_col_));
    ReadIntParam(node_, "tokenizer_type", tokenizer_type_);
    if (const auto policy = node_->parameters.find("bpe_piece_policy"); policy != node_->parameters.end())
        bpe_piece_policy_ = policy->second;
    if (const auto unit = node_->parameters.find("bpe_initial_unit"); unit != node_->parameters.end())
        bpe_initial_unit_ = unit->second;
    ReadIntParam(node_, "max_length", max_length_);
    ReadIntParam(node_, "max_vocab_size", max_vocab_size_);
    ReadIntParam(node_, "min_word_freq", min_word_freq_);
    ReadIntParam(node_, "min_freq", min_word_freq_);
    ReadIntParam(node_, "min_frequency", min_word_freq_);
    ReadIntParam(node_, "pad_value", pad_value_);
    if (tokenizer_type_ == 3) lowercase_ = false;
    ReadBoolParam(node_, "lowercase", lowercase_);
    ReadBoolParam(node_, "padding", padding_);
    ReadBoolParam(node_, "truncation", truncation_);
    ReadBoolParam(node_, "vocab_build_if_missing", vocab_build_if_missing_);

    if (tokenizer_type_ < 0 || tokenizer_type_ > 6) tokenizer_type_ = 1;
    if (max_length_ < 1) max_length_ = 1;
    if (min_word_freq_ < 1) min_word_freq_ = 1;
}

void TokenizerDialog::Apply() {
    if (!node_) return;

    if (IsTokenizerNode()) {
        const char* output_mode = "wide";
        if (token_output_mode_ == 1) output_mode = "causal_windows";
        else if (token_output_mode_ == 2) output_mode = "decode";
        else if (token_output_mode_ == 3) output_mode = "roundtrip";
        node_->parameters["output_mode"] = output_mode;
        if (tokenizer_type_ == 3 && !bpe_piece_policy_.empty())
            node_->parameters["bpe_piece_policy"] = bpe_piece_policy_;
        else
            node_->parameters.erase("bpe_piece_policy");
        if (tokenizer_type_ == 3 && !bpe_initial_unit_.empty())
            node_->parameters["bpe_initial_unit"] = bpe_initial_unit_;
        else
            node_->parameters.erase("bpe_initial_unit");
        node_->parameters["document_id_col"] = document_id_col_;
        node_->parameters["split_col"] = split_col_;
        if (token_output_mode_ == 1 && target_text_col_[0] != '\0')
            node_->parameters["target_text_col"] = target_text_col_;
        else
            node_->parameters.erase("target_text_col");
        node_->parameters["token_ids_col"] = token_ids_col_;
        node_->parameters["text_col"] = text_col_;
        node_->parameters["label_col"] = label_col_;
        node_->parameters["text_column"] = text_col_;
        node_->parameters["method"] = MethodNameFromType(tokenizer_type_);
        node_->parameters["tokenizer_type"] = std::to_string(tokenizer_type_);
        node_->parameters["max_length"] = std::to_string(max_length_);
        node_->parameters["lowercase"] = lowercase_ ? "true" : "false";
        node_->parameters["padding"] = padding_ ? "true" : "false";
        node_->parameters["truncation"] = truncation_ ? "true" : "false";
        node_->parameters["min_word_freq"] = std::to_string(min_word_freq_);
        node_->parameters["min_freq"] = std::to_string(min_word_freq_);
        node_->parameters["max_vocab_size"] = std::to_string(max_vocab_size_);
        node_->parameters["vocab_file"] = vocab_file_;
        node_->parameters["source_csv"] = source_csv_;
        node_->parameters["pad_value"] = std::to_string(pad_value_);
        node_->parameters["vocab_build_if_missing"] = vocab_build_if_missing_ ? "true" : "false";
    }

    if (IsVocabularyNode()) {
        node_->parameters["min_freq"] = std::to_string(min_word_freq_);
        node_->parameters["min_word_freq"] = std::to_string(min_word_freq_);
        node_->parameters["max_vocab_size"] = std::to_string(max_vocab_size_);
        node_->parameters["vocab_file"] = vocab_file_;
        node_->parameters["source_csv"] = source_csv_;
        node_->parameters["text_col"] = text_col_;
        node_->parameters["method"] = MethodNameFromType(tokenizer_type_);
        node_->parameters["tokenizer_type"] = std::to_string(tokenizer_type_);
        node_->parameters["lowercase"] = lowercase_ ? "true" : "false";
    }

    if (IsPaddingNode()) {
        node_->parameters["max_length"] = std::to_string(max_length_);
        node_->parameters["pad_value"] = std::to_string(pad_value_);
        node_->parameters["padding"] = "true";
        node_->parameters["truncation"] = truncation_ ? "true" : "false";
    }

    has_changes_ = false;
    spdlog::info("TokenizerDialog: Applied text preprocessing settings");
}

bool TokenizerDialog::BuildVocabularyFile() {
    if (tokenizer_type_ == 3) {
        status_message_ = "Fit Byte BPE using Run Pipeline with vocab_build_if_missing enabled. This keeps fitting off the GUI thread.";
        status_is_error_ = true;
        return false;
    }
    namespace fs = std::filesystem;
    const fs::path csv_path(source_csv_);
    const fs::path out_path(vocab_file_);
    if (source_csv_[0] == '\0') {
        status_message_ = "Choose a source CSV before building the vocabulary.";
        status_is_error_ = true;
        return false;
    }
    if (vocab_file_[0] == '\0') {
        status_message_ = "Choose an output vocab file before building.";
        status_is_error_ = true;
        return false;
    }

    std::ifstream in(csv_path);
    if (!in.is_open()) {
        status_message_ = "Could not open source CSV: " + csv_path.string();
        status_is_error_ = true;
        return false;
    }

    std::string header_line;
    if (!ReadCsvRecord(in, header_line)) {
        status_message_ = "Source CSV is empty: " + csv_path.string();
        status_is_error_ = true;
        return false;
    }

    const auto headers = SplitCsvLineSimple(header_line);
    const int text_idx = FindColumnIndex(headers, text_col_);
    if (text_idx < 0) {
        status_message_ = "Text column '" + std::string(text_col_) +
                          "' was not found in the CSV header.";
        status_is_error_ = true;
        return false;
    }

    std::vector<std::string> documents;
    std::string line;
    while (ReadCsvRecord(in, line)) {
        const auto fields = SplitCsvLineSimple(line);
        if (text_idx >= static_cast<int>(fields.size())) continue;
        const auto& document = fields[static_cast<size_t>(text_idx)];
        if (!document.empty()) documents.push_back(document);
    }
    if (in.bad()) {
        status_message_ = "Failed to read source CSV: " + csv_path.string();
        status_is_error_ = true;
        return false;
    }
    cyxwiz::Tokenizer tokenizer(static_cast<cyxwiz::TokenizerType>(tokenizer_type_));
    tokenizer.SetLowercase(lowercase_);
    tokenizer.Train(documents, min_word_freq_, max_vocab_size_);
    const auto& vocabulary = tokenizer.GetVocabulary();
    const size_t rows = documents.size();

    if (out_path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
    }
    if (!vocabulary.SaveToFile(out_path.string())) {
        status_message_ = "Could not write vocab file: " + out_path.string();
        status_is_error_ = true;
        return false;
    }
    max_vocab_size_ = static_cast<int>(vocabulary.Size());
    if (node_) {
        node_->parameters["max_vocab_size"] = std::to_string(max_vocab_size_);
        node_->parameters["vocab_file"] = vocab_file_;
        node_->parameters["source_csv"] = source_csv_;
    }
    status_message_ = "Built " + std::to_string(vocabulary.Size()) +
                      " vocabulary entries from " + std::to_string(rows) +
                      " non-empty text rows.";
    status_is_error_ = false;
    return true;
}

bool TokenizerDialog::InspectVocabularyFile() {
    if (vocab_file_[0] == '\0') {
        status_message_ = "Choose a vocab file to inspect.";
        status_is_error_ = true;
        return false;
    }
    cyxwiz::Vocabulary vocabulary;
    if (!vocabulary.LoadFromFile(vocab_file_)) {
        status_message_ = "Could not read valid vocabulary: " + std::string(vocab_file_);
        status_is_error_ = true;
        return false;
    }
    status_message_ = "Vocab file contains " + std::to_string(vocabulary.Size()) + " entries.";
    status_is_error_ = false;
    return true;
}

void TokenizerDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    preview_tokens_.clear();
    preview_token_ids_.clear();
    preview_decoded_text_.clear();
    preview_roundtrip_ok_ = false;
    preview_has_roundtrip_ = false;
    has_changes_ = false;
}

void TokenizerDialog::TakeColumnsFromInput() {
    columns_from_input_checked_ = true;
    if (!node_ || !node_editor_ || !IsTokenizerNode()) return;
    if (node_->parameters.count("text_col") || node_->parameters.count("text_column")) return;
    const auto& nodes = node_editor_->GetNodes();
    const auto& links = node_editor_->GetLinks();
    std::set<int> seen{node_->id};
    std::vector<int> frontier{node_->id};
    while (!frontier.empty()) {
        const int to = frontier.back();
        frontier.pop_back();
        for (const auto& link : links) {
            if (link.to_node != to || !seen.insert(link.from_node).second) continue;
            const auto it = std::find_if(nodes.begin(), nodes.end(),
                                         [&](const MLNode& n) { return n.id == link.from_node; });
            if (it == nodes.end()) continue;
            const auto text = it->parameters.find("text_column");
            if (it->type != NodeType::DataInput || text == it->parameters.end() || text->second.empty()) {
                frontier.push_back(it->id);
                continue;
            }
            std::strncpy(text_col_, text->second.c_str(), sizeof(text_col_) - 1);
            text_col_[sizeof(text_col_) - 1] = '\0';
            if (const auto label = it->parameters.find("text_label_column");
                label != it->parameters.end() && label_col_[0] == '\0') {
                std::strncpy(label_col_, label->second.c_str(), sizeof(label_col_) - 1);
                label_col_[sizeof(label_col_) - 1] = '\0';
            }
            has_changes_ = true;
            return;
        }
    }
}

namespace {

constexpr float kTokLabel = 170.0f;

// A muted label whose hover text is the help (the old "(?)" marks), then the
// control at the label column.
void TokLabel(const char* label, const char* help) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "%s", label);
    if (help && ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(help);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    ImGui::SameLine(kTokLabel);
}

// Lines of a vocabulary file and how many are not valid UTF-8 (byte tokens
// saved before the 2026-10-16 tokenizer fix, TOFIX155).
struct VocabFacts {
    size_t entries = 0;
    size_t broken = 0;
    bool found = false;
};

bool ValidUtf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (n == 0 || i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k)
            if ((static_cast<unsigned char>(s[i + k]) >> 6) != 0x2) return false;
        i += n;
    }
    return true;
}

VocabFacts ReadVocabFacts(const std::string& path) {
    VocabFacts facts;
    std::ifstream in(path, std::ios::binary);
    if (!in) return facts;
    facts.found = true;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        ++facts.entries;
        if (!ValidUtf8(line)) ++facts.broken;
    }
    return facts;
}

// The dialog's tokenizer_type index for each "Split into" choice, in the
// order shown (SentencePiece only appears for a node that already uses it).
constexpr int kSplitTypes[] = {0, 1, 2, 4, 3};
const char* kSplitLabels[] = {"Spaces", "Words", "Characters", "WordPiece", "Byte BPE"};

const char* SplitDescription(int type) {
    switch (type) {
        case 0: return "Spaces: splits on whitespace only; punctuation stays on the word.";
        case 2: return "Characters: one token per character.";
        case 3: return "Byte BPE: learned sub-word pieces over bytes; keeps case and spaces.";
        case 4: return "WordPiece: words split into ## continuation pieces, longest match first.";
        case 5:
        case 6: return "SentencePiece is the optional provider path, not in this build: Run Pipeline refuses it.";
        default:
            return "Words: runs of letters and digits of any script (\"caf\xC3\xA9\" stays whole); punctuation and "
                   "emoji end a word.";
    }
}

}  // namespace

void TokenizerDialog::RenderContent() {
    if (!columns_from_input_checked_) TakeColumnsFromInput();
    // Tabs named by what is decided (board 4): Words, Vocabulary, Length, Try it.
    if (ImGui::BeginTabBar("TextPreprocessingTabs")) {
        if (IsTokenizerNode() && ImGui::BeginTabItem("Words")) {
            RenderWordsTab();
            ImGui::EndTabItem();
        }
        if ((IsTokenizerNode() || IsVocabularyNode()) && ImGui::BeginTabItem("Vocabulary")) {
            RenderVocabularyTab();
            ImGui::EndTabItem();
        }
        if ((IsTokenizerNode() || IsPaddingNode()) && ImGui::BeginTabItem("Length")) {
            RenderLengthTab();
            ImGui::EndTabItem();
        }
        if (IsTokenizerNode() && ImGui::BeginTabItem("Try it")) {
            RenderTryItTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void TokenizerDialog::ClearTryIt() {
    try_message_.clear();
    preview_tokens_.clear();
    preview_token_ids_.clear();
    preview_decoded_text_.clear();
    preview_has_roundtrip_ = false;
}

void TokenizerDialog::RenderWordsTab() {
    ImGui::Spacing();
    TokLabel("Text column", "The column holding the sentences or documents; for the sentiment graph 'statement'.");
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("##text_col", text_col_, sizeof(text_col_))) has_changes_ = true;
    TokLabel("Label column", "Optional target column for supervised training; empty for unlabeled text. For "
                             "sentiment 'status'.");
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("##label_col", label_col_, sizeof(label_col_))) has_changes_ = true;

    ImGui::Spacing();
    TokLabel("Split into", "How raw text is split before the vocabulary lookup.");
    int choice = -1;
    for (int i = 0; i < 5; ++i)
        if (kSplitTypes[i] == tokenizer_type_) choice = i;
    if (choice < 0) {
        // A node using SentencePiece keeps it; the choice shows as none.
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning, "SentencePiece (type %d)", tokenizer_type_);
    } else if (cyxwiz::ui::SegmentedControl("##split_into", kSplitLabels, 5, &choice)) {
        tokenizer_type_ = kSplitTypes[choice];
        if (tokenizer_type_ == 3) lowercase_ = false;
        has_changes_ = true;
        ClearTryIt();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "%s", SplitDescription(tokenizer_type_));
    ImGui::PopTextWrapPos();

    ImGui::BeginDisabled(tokenizer_type_ == 3);
    if (ImGui::Checkbox("Lowercase first", &lowercase_)) {
        has_changes_ = true;
        ClearTryIt();
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(tokenizer_type_ == 3 ? "Byte BPE keeps case."
                                               : "Merges 'Happy' and 'happy'. Off when capitalization carries "
                                                 "meaning.");

    if (tokenizer_type_ == 3) RenderBpeOptions();

    ImGui::Spacing();
    ImGui::Separator();
    RenderVocabularyCard();
}

void TokenizerDialog::RenderBpeOptions() {
    ImGui::Spacing();
    TokLabel("Starting units", "Bytes starts with all 256 byte values. Unicode characters starts with each observed "
                               "code point. Unseen characters become [UNK]; invalid UTF-8 is refused. A saved "
                               "vocabulary owns its encoding; use a new file to refit.");
    const char* unit_values[] = {"", "byte", "unicode_character"};
    const char* unit_labels[] = {"Follow saved vocabulary", "Bytes (256)", "Unicode characters"};
    const char* unit_label = "Unsupported unit - select a valid option";
    for (int i = 0; i < 3; ++i) if (bpe_initial_unit_ == unit_values[i]) unit_label = unit_labels[i];
    ImGui::SetNextItemWidth(280.0f);
    if (ImGui::BeginCombo("##bpe_initial_unit", unit_label)) {
        for (int i = 0; i < 3; ++i) {
            if (ImGui::Selectable(unit_labels[i], bpe_initial_unit_ == unit_values[i])) {
                bpe_initial_unit_ = unit_values[i];
                has_changes_ = true;
                ClearTryIt();
            }
        }
        ImGui::EndCombo();
    }
    TokLabel("Space handling", "Where BPE may learn merges. A saved vocabulary owns its policy; an explicit choice "
                               "must match it. Changing token ids needs new windows and a compatible model.");
    const char* values[] = {"", "whitespace_v1", "leading_space_v2"};
    const char* labels[] = {"Follow saved vocabulary", "Separate spaces (v1)", "Merge leading space (v2)"};
    const char* selected = "Unsupported policy - select a valid option";
    for (int i = 0; i < 3; ++i) if (bpe_piece_policy_ == values[i]) selected = labels[i];
    ImGui::SetNextItemWidth(280.0f);
    if (ImGui::BeginCombo("##bpe_piece_policy", selected)) {
        for (int i = 0; i < 3; ++i) {
            if (ImGui::Selectable(labels[i], bpe_piece_policy_ == values[i])) {
                bpe_piece_policy_ = values[i];
                has_changes_ = true;
                ClearTryIt();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Byte BPE preserves case and whitespace. Fit and save its vocabulary through Run Pipeline; "
                        "reuse that file for validation, test and generation. Without a saved vocabulary it starts "
                        "from bytes with separate spaces.");
    ImGui::PopTextWrapPos();
}

void TokenizerDialog::RenderVocabularyCard() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextUnformatted("Vocabulary");
    const std::string path =
        cyxwiz::ResolveProjectDataPath(vocab_file_, cyxwiz::ProjectManager::Instance().GetProjectRoot());
    if (path != vocab_card_path_) {
        vocab_card_path_ = path;
        const auto facts = path.empty() ? VocabFacts{} : ReadVocabFacts(path);
        vocab_card_found_ = facts.found;
        vocab_card_entries_ = facts.entries;
        vocab_card_broken_ = facts.broken;
    }
    if (vocab_file_[0] == '\0') {
        ImGui::TextColored(t.text_dim, "No vocabulary file: it is built from the training text when the graph runs.");
    } else {
        ImGui::TextColored(t.text_dim, "file");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::filesystem::path(vocab_file_).filename().string().c_str());
        ImGui::SameLine(0.0f, 16.0f);
        if (vocab_card_found_) {
            ImGui::Text("%s", std::to_string(vocab_card_entries_).c_str());
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "tokens");
        } else {
            ImGui::TextColored(vocab_build_if_missing_ ? t.text_dim : t.warning,
                               vocab_build_if_missing_ ? "not built yet (built on the first run)" : "file not found");
        }
        ImGui::SameLine(0.0f, 16.0f);
        ImGui::TextColored(t.text_dim, "min frequency %d", min_word_freq_);
        if (vocab_card_broken_ > 0) {
            ImGui::TextColored(t.warning, "%zu tokens are broken bytes from before 2026-10-16: rebuild it",
                               vocab_card_broken_);
        }
    }
    if (cyxwiz::ui::SecondaryButton("Build and save vocabulary", source_csv_[0] != '\0',
                                    "Set the source CSV in the Vocabulary tab", cyxwiz::ui::ButtonSize::Small)) {
        BuildVocabularyFile();
        has_changes_ = true;
        vocab_card_path_.clear();
    }
    ImGui::SameLine();
    if (cyxwiz::ui::SecondaryButton("Inspect file", vocab_file_[0] != '\0', "No vocabulary file",
                                    cyxwiz::ui::ButtonSize::Small)) {
        InspectVocabularyFile();
        vocab_card_path_.clear();
    }
    if (source_csv_[0]) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "from %s in %s", text_col_,
                           std::filesystem::path(source_csv_).filename().string().c_str());
    }
    if (!status_message_.empty()) ValidationMessage(status_message_, status_is_error_);
}

void TokenizerDialog::RenderVocabularyTab() {
    ImGui::Spacing();
    if (IsTokenizerNode() || IsVocabularyNode()) {
        TokLabel("Source CSV", "File scanned by Build and save vocabulary. A run can also build from the loaded data.");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##source_csv", source_csv_, sizeof(source_csv_))) has_changes_ = true;
        TokLabel("Text column", "Column of the source CSV whose tokens are counted.");
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##vocab_text_col", text_col_, sizeof(text_col_))) has_changes_ = true;
    }

    TokLabel(tokenizer_type_ == 3 ? "Minimum pair frequency" : "Minimum frequency",
             tokenizer_type_ == 3 ? "Minimum count for a BPE merge. Every starting byte or observed character is kept."
                                  : "Tokens seen fewer times are left out (they read as [UNK]); higher shrinks the "
                                    "vocabulary.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##min_word_freq", &min_word_freq_)) {
        if (min_word_freq_ < 1) min_word_freq_ = 1;
        has_changes_ = true;
    }
    TokLabel("Maximum size", tokenizer_type_ == 3
                                 ? "BPE needs an explicit cap covering the starting alphabet plus 4 specials (bytes: "
                                   "at least 260). The cap includes learned merges."
                                 : "Upper bound including [PAD] and [UNK]; -1 for no cap.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_vocab", &max_vocab_size_)) {
        if (max_vocab_size_ < -1) max_vocab_size_ = -1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled(tokenizer_type_ == 3 ? "explicit cap required" : "-1 = unlimited");

    if (IsTokenizerNode() || IsVocabularyNode()) {
        TokLabel("Vocabulary file", "One token per line. The tokenizer loads it, or builds it from the training text "
                                    "when it is missing.");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##vocab_file", vocab_file_, sizeof(vocab_file_))) has_changes_ = true;
        if (ImGui::Checkbox("Build and save it when the file is missing", &vocab_build_if_missing_))
            has_changes_ = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A run with a missing vocabulary file trains it from the input text and writes it "
                              "before tokenizing.");
        RenderVocabularyCard();
    }
}

void TokenizerDialog::RenderLengthTab() {
    ImGui::Spacing();
    TokLabel("Output", "What the node emits.");
    const char* outputs[] = {"Padded token columns", "Causal document windows", "Decode token IDs",
                             "Round-trip check"};
    ImGui::SetNextItemWidth(260.0f);
    if (ImGui::Combo("##output", &token_output_mode_, outputs, 4)) has_changes_ = true;

    ImGui::PushTextWrapPos(0.0f);
    if (token_output_mode_ == 1) {
        TokLabel("Document id column", nullptr);
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##doc_id", document_id_col_, sizeof(document_id_col_))) has_changes_ = true;
        TokLabel("Split column", nullptr);
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##split_col", split_col_, sizeof(split_col_))) has_changes_ = true;
        TokLabel("Response column", "Optional (supervised fine-tuning).");
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##response_col", target_text_col_, sizeof(target_text_col_))) has_changes_ = true;
        ImGui::TextDisabled("Response column: each row becomes one window, the prompt then the answer + EOS, and the "
                            "loss counts only the answer's tokens. Needs an existing vocabulary file; empty = "
                            "ordinary windows where every token is a target. Max length is the model context; each "
                            "window carries up to context + 1 ids. Leave Label column empty.");
    } else if (token_output_mode_ == 2) {
        TokLabel("Token ids column", nullptr);
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##token_ids_col", token_ids_col_, sizeof(token_ids_col_))) has_changes_ = true;
        ImGui::TextDisabled("Decodes a token id column back to text with the vocabulary file (or wide tok_0..tok_n "
                            "columns up to the maximum length). Leave Label column empty.");
    } else if (token_output_mode_ == 3) {
        ImGui::TextDisabled("Encodes the text column, decodes it with the same vocabulary and emits input_text, "
                            "encoded_ids, decoded_text and roundtrip_ok. Leave Label column empty.");
    }
    ImGui::PopTextWrapPos();

    TokLabel("Maximum length", "Padded output: the token count. Causal windows: the context length. Decode: bounds "
                               "the tok_0..tok_n columns read.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_length", &max_length_)) {
        if (max_length_ < 1) max_length_ = 1;
        has_changes_ = true;
        ClearTryIt();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("tokens per row");
    if (token_output_mode_ == 1 || token_output_mode_ == 2) return;

    TokLabel("Pad value", "Token id filling short rows: 0 when the vocabulary starts with [PAD] and the Embedding's "
                          "padding index is 0.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##pad_value", &pad_value_)) has_changes_ = true;
    if (IsTokenizerNode() && ImGui::Checkbox("Pad short rows to the maximum length", &padding_)) has_changes_ = true;
    if (ImGui::Checkbox("Cut long rows at the maximum length", &truncation_)) has_changes_ = true;
}

void TokenizerDialog::RunTryIt() {
    ClearTryIt();
    preview_roundtrip_ok_ = false;
    try_message_.clear();
    try_is_error_ = false;
    try {
        cyxwiz::TokenizerType tokenizer_type = cyxwiz::TokenizerType::Word;
        if (tokenizer_type_ == 0) tokenizer_type = cyxwiz::TokenizerType::Whitespace;
        else if (tokenizer_type_ == 2) tokenizer_type = cyxwiz::TokenizerType::Character;
        else if (tokenizer_type_ == 3) tokenizer_type = cyxwiz::TokenizerType::ByteBPE;
        else if (tokenizer_type_ == 4) tokenizer_type = cyxwiz::TokenizerType::WordPiece;
        else if (tokenizer_type_ == 5) tokenizer_type = cyxwiz::TokenizerType::SentencePieceBPE;
        else if (tokenizer_type_ == 6) tokenizer_type = cyxwiz::TokenizerType::SentencePieceUnigram;

        cyxwiz::Tokenizer tokenizer(tokenizer_type);
        tokenizer.SetLowercase(lowercase_);
        tokenizer.SetMaxLength(max_length_);
        tokenizer.SetPadding(false);
        tokenizer.SetTruncation(truncation_);

        const std::string vocab_file = cyxwiz::ResolveProjectDataPath(
            vocab_file_, cyxwiz::ProjectManager::Instance().GetProjectRoot());
        if (!vocab_file.empty() && std::filesystem::exists(vocab_file)) {
            if (!tokenizer.GetVocabulary().LoadFromFile(vocab_file)) {
                throw std::runtime_error("Could not load tokenizer vocabulary: " + vocab_file);
            }
            tokenizer.ValidateVocabulary();
            if (tokenizer_type_ == 3 && !bpe_initial_unit_.empty()) {
                const auto actual = tokenizer.GetVocabulary().GetBPEInitialUnit();
                if (!((bpe_initial_unit_ == "byte" && actual == cyxwiz::BPEInitialUnit::Byte) ||
                      (bpe_initial_unit_ == "unicode_character" && actual == cyxwiz::BPEInitialUnit::UnicodeCharacter)))
                    throw std::runtime_error("BPE starting units differ from the saved vocabulary. Follow the saved "
                                             "vocabulary or select a new file and refit through Run Pipeline.");
            }
            if (tokenizer_type_ == 3 && !bpe_piece_policy_.empty()) {
                const auto actual = tokenizer.GetVocabulary().GetBPEPiecePolicy();
                const bool matches =
                    (bpe_piece_policy_ == "whitespace_v1" && actual == cyxwiz::ByteBPEPiecePolicy::WhitespaceV1) ||
                    (bpe_piece_policy_ == "leading_space_v2" && actual == cyxwiz::ByteBPEPiecePolicy::LeadingSpaceV2);
                if (!matches)
                    throw std::runtime_error("Space handling differs from the saved vocabulary. Follow the saved "
                                             "vocabulary or select a new file and refit through Run Pipeline.");
            }
        } else {
            if (tokenizer_type_ == 3) {
                throw std::runtime_error("Build and select a BPE vocabulary first: Try it uses the exact saved "
                                         "file.");
            }
            tokenizer.Train({std::string(sample_text_)}, min_word_freq_, max_vocab_size_);
            tokenizer.ValidateVocabulary();
            try_message_ = "No vocabulary file yet: the ids come from a vocabulary fitted to this sample.";
            try_is_error_ = false;
        }

        const auto tokenized = tokenizer.Tokenize(sample_text_);
        preview_tokens_ = tokenized.tokens;
        preview_token_ids_ = tokenized.token_ids;
        preview_decoded_text_ = tokenizer.Decode(preview_token_ids_);
        preview_roundtrip_ok_ = preview_decoded_text_ == std::string(sample_text_);
        preview_has_roundtrip_ = true;
    } catch (const std::exception& e) {
        try_message_ = e.what();
        try_is_error_ = true;
    }
}

void TokenizerDialog::RenderTryItTab() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::SetNextItemWidth(-1);
    // It tokenizes as you type; nothing in the dataset changes.
    if (ImGui::InputTextMultiline("##sample", sample_text_, sizeof(sample_text_), ImVec2(0, 80)) ||
        (!preview_has_roundtrip_ && try_message_.empty()))
        RunTryIt();
    if (!try_message_.empty()) ValidationMessage(try_message_, try_is_error_);
    if (!preview_has_roundtrip_) return;

    // Tokens as text chips, [UNK] in orange; hover for the id.
    ImGui::Spacing();
    cyxwiz::ui::FlowRow row;
    for (size_t i = 0; i < preview_tokens_.size(); ++i) {
        const std::string& token = preview_tokens_[i];
        const ImVec2 size = ImGui::CalcTextSize(token.c_str());
        row.Next(size.x + 12.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddRectFilled(at, ImVec2(at.x + size.x + 12.0f, at.y + size.y + 6.0f),
                                                  ImGui::GetColorU32(t.bg_raised), 4.0f);
        ImGui::SetCursorScreenPos(ImVec2(at.x + 6.0f, at.y + 3.0f));
        ImGui::TextColored(token == "[UNK]" ? t.warning : t.text, "%s", token.c_str());
        if (ImGui::IsItemHovered() && i < preview_token_ids_.size())
            ImGui::SetTooltip("id %d", preview_token_ids_[i]);
        ImGui::SetCursorScreenPos(ImVec2(at.x + size.x + 12.0f, at.y));
        ImGui::Dummy(ImVec2(0.0f, size.y + 6.0f));
    }
    ImGui::Spacing();
    const int count = static_cast<int>(preview_tokens_.size());
    if (padding_ && count < max_length_)
        ImGui::TextColored(t.text_dim, "%d tokens of %d, padded with %d to %d. Hover a token for its id.", count,
                           max_length_, pad_value_, max_length_);
    else
        ImGui::TextColored(t.text_dim, "%d tokens (maximum %d). Hover a token for its id.", count, max_length_);

    if (ImGui::TreeNodeEx("Ids and the round trip", ImGuiTreeNodeFlags_SpanAvailWidth)) {
        std::ostringstream ids;
        for (size_t i = 0; i < preview_token_ids_.size(); ++i) ids << (i ? " " : "") << preview_token_ids_[i];
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Ids");
        ImGui::TextUnformatted(ids.str().c_str());
        ImGui::TextColored(t.text_dim, "Decoded");
        ImGui::TextUnformatted(preview_decoded_text_.c_str());
        ImGui::PopTextWrapPos();
        ValidationMessage(preview_roundtrip_ok_ ? "Round trip exact: the decoded text matches the sample."
                                                : "Round trip differs: the decoded text does not match the sample.",
                          !preview_roundtrip_ok_);
        ImGui::TreePop();
    }
}

// ==================== FilterDialog ====================

FilterDialog::FilterDialog(MLNode* node)
    : NodeConfigDialog("Filter Configuration", node)
{
    conditions_.push_back(FilterCondition{});
}

void FilterDialog::Apply() {
    if (!node_) return;

    // Build filter expression from conditions
    std::string expression;
    for (size_t i = 0; i < conditions_.size(); i++) {
        const auto& cond = conditions_[i];
        if (i > 0) {
            expression += (cond.join_type == 0) ? " AND " : " OR ";
        }
        expression += "col" + std::to_string(cond.column_idx);
        const char* ops[] = { "==", "!=", ">", "<", ">=", "<=", "CONTAINS" };
        expression += " " + std::string(ops[cond.operator_idx]) + " ";
        expression += "'" + std::string(cond.value) + "'";
    }

    node_->parameters["filter_expression"] = expression;
    has_changes_ = false;
}

void FilterDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    conditions_.clear();
    conditions_.push_back(FilterCondition{});
    has_changes_ = false;
}

void FilterDialog::RenderContent() {
    ImGui::Spacing();
    ImGui::Text("Filter Conditions:");
    ImGui::Separator();
    ImGui::Spacing();

    const char* operators[] = { "equals", "not equals", ">", "<", ">=", "<=", "contains" };
    const char* joins[] = { "AND", "OR" };

    for (size_t i = 0; i < conditions_.size(); i++) {
        auto& cond = conditions_[i];
        ImGui::PushID(static_cast<int>(i));

        if (i > 0) {
            ImGui::SetNextItemWidth(60.0f);
            ImGui::Combo("##join", &cond.join_type, joins, 2);
            ImGui::SameLine();
        }

        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputInt("Column", &cond.column_idx);
        ImGui::SameLine();

        ImGui::SetNextItemWidth(100.0f);
        ImGui::Combo("##op", &cond.operator_idx, operators, 7);
        ImGui::SameLine();

        ImGui::SetNextItemWidth(150.0f);
        ImGui::InputText("Value", cond.value, sizeof(cond.value));
        ImGui::SameLine();

        if (ImGui::Button("X") && conditions_.size() > 1) {
            conditions_.erase(conditions_.begin() + i);
            i--;
        }

        ImGui::PopID();
    }

    ImGui::Spacing();
    if (ImGui::Button("+ Add Condition")) {
        conditions_.push_back(FilterCondition{});
    }
}

// ==================== NodeConfigDialogFactory ====================

NodeConfigDialogFactory& NodeConfigDialogFactory::Instance() {
    static NodeConfigDialogFactory instance;
    return instance;
}

NodeConfigDialogFactory::NodeConfigDialogFactory() {
    // Register default dialogs - use correct NodeType enum values
    RegisterDialog(NT::TextTokenizer, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });
    RegisterDialog(NT::TextVocabulary, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });
    RegisterDialog(NT::TextPadding, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });
    RegisterDialog(NT::Embedding, [](MLNode* node) {
        return std::make_unique<EmbeddingDialog>(node);
    });

    RegisterDialog(NT::FilterRows, [](MLNode* node) {
        return std::make_unique<FilterDialog>(node);
    });

    // Smart I/O Nodes - Universal dialogs
    RegisterDialog(NT::DataInput, [](MLNode* node) {
        return std::make_unique<DataInputDialog>(node);
    });

    RegisterDialog(NT::DataOutput, [](MLNode* node) {
        return std::make_unique<DataOutputDialog>(node);
    });

    RegisterDialog(NT::DataConvert, [](MLNode* node) {
        return std::make_unique<DataConvertDialog>(node);
    });

    // Data pipeline nodes - single source of truth for batching and splitting
    RegisterDialog(NT::DataLoader, [](MLNode* node) {
        return std::make_unique<DataLoaderDialog>(node);
    });

    RegisterDialog(NT::DataSplit, [](MLNode* node) {
        return std::make_unique<DataSplitDialog>(node);
    });

    // SQL step: the SQL Workbench (TOFIX146), a dockable window.
    RegisterDialog(NT::SQLQuery, [](MLNode* node) {
        return std::make_unique<SqlWorkbenchDialog>(node);
    });

    RegisterDialog(NT::QualityAnalyzer, [](MLNode* node) {
        return std::make_unique<QualityAnalyzerDialog>(node);
    });

    RegisterDialog(NT::WordEmbeddings, [](MLNode* node) {
        return std::make_unique<WordEmbeddingsDialog>(node);
    });
}

void NodeConfigDialogFactory::RegisterDialog(NodeType type, DialogCreator creator) {
    creators_[type] = std::move(creator);
}

bool NodeConfigDialogFactory::HasDialog(NodeType type) const {
    return creators_.count(type) > 0;
}

std::unique_ptr<NodeConfigDialog> NodeConfigDialogFactory::CreateDialog(MLNode* node) {
    if (!node) return nullptr;

    auto it = creators_.find(node->type);
    if (it != creators_.end()) {
        return it->second(node);
    }
    return nullptr;
}

// ==================== Helper Functions ====================

bool ShouldShowOpenDialogButton(NT type) {
    // The button shows iff a dialog factory is actually registered for
    // this NodeType. Previously this function maintained a hand-coded
    // whitelist of 19 types that drifted from the real factory (9
    // registrations), so clicking the button on ~10 node types did
    // silently nothing (CreateDialog returned null). Deriving from
    // NodeConfigDialogFactory::HasDialog eliminates the drift — adding
    // a new dialog via RegisterDialog now also makes the button appear
    // automatically, and the reverse drift (button shown but no
    // factory) can't happen.
    return NodeConfigDialogFactory::Instance().HasDialog(type);
}

} // namespace gui
