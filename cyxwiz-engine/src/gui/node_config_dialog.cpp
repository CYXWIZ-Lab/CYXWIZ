// Include Windows headers BEFORE undefining macros
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

// Undefine Windows macros that conflict with our method names
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif

#include "node_config_dialog.h"
#include "node_editor.h"
#include "visualization/bar_chart_dialog.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>

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
    has_changes_ = false;
    if (node_) {
        original_params_ = node_->parameters;
    }
}

void NodeConfigDialog::Close() {
    is_open_ = false;
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
    if (ImGui::Begin(title_.c_str(), &is_open_, flags)) {
        // Content area
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float button_height = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y * 2;

        // Main content (excluding button area)
        ImGui::BeginChild("DialogContent", ImVec2(0, avail.y - button_height), false);
        RenderContent();
        ImGui::EndChild();

        ImGui::Separator();

        // Bottom buttons (OK, Cancel, Apply)
        float button_width = 80.0f;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float total_width = button_width * 3 + spacing * 2;
        ImGui::SetCursorPosX(avail.x - total_width);

        // Grey out OK/Apply while a subclass reports it's busy (e.g. async
        // data load in DataInputDialog). Cancel stays enabled so the user
        // can always bail out.
        bool busy = IsBusy();

        ImGui::BeginDisabled(busy);
        if (ImGui::Button("OK", ImVec2(button_width, 0))) {
            Apply();
            should_close = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            Reset();
            should_close = true;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Apply", ImVec2(button_width, 0))) {
            Apply();
        }
        ImGui::EndDisabled();
    }
    ImGui::End();

    if (should_close) {
        Close();
    }

    return is_open_;
}

bool NodeConfigDialog::FileSelector(const char* label, std::string& path, const char* filter) {
    bool changed = false;

    char buf[512];
    strncpy(buf, path.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    ImGui::Text("%s", label);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90.0f);
    if (ImGui::InputText("##filepath", buf, sizeof(buf))) {
        path = buf;
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse...", ImVec2(80.0f, 0))) {
#ifdef _WIN32
        OPENFILENAMEA ofn = {};
        char file[512] = {};
        strncpy(file, path.c_str(), sizeof(file) - 1);
        ofn.lStructSize = sizeof(ofn);
        ofn.lpstrFilter = filter;
        ofn.lpstrFile = file;
        ofn.nMaxFile = sizeof(file);
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
        if (GetOpenFileNameA(&ofn)) {
            path = file;
            changed = true;
        }
#endif
    }

    return changed;
}

void NodeConfigDialog::ValidationMessage(const std::string& message, bool is_error) {
    ImVec4 color = is_error ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f) : ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
    ImGui::TextColored(color, "%s", message.c_str());
}

// ==================== CSVReaderDialog ====================

CSVReaderDialog::CSVReaderDialog(MLNode* node)
    : NodeConfigDialog("CSV Reader Configuration", node)
{
    if (node_) {
        // Initialize from node parameters
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
        }
        if (node_->parameters.count("delimiter")) {
            std::string delim = node_->parameters["delimiter"];
            if (delim == ",") delimiter_idx_ = 0;
            else if (delim == ";") delimiter_idx_ = 1;
            else if (delim == "\\t" || delim == "\t") delimiter_idx_ = 2;
            else {
                delimiter_idx_ = 3;
                strncpy(custom_delimiter_, delim.c_str(), sizeof(custom_delimiter_) - 1);
            }
        }
        if (node_->parameters.count("has_header")) {
            has_header_ = (node_->parameters["has_header"] == "true");
        }
        if (node_->parameters.count("skip_first_n")) {
            try { skip_first_n_ = std::stoi(node_->parameters["skip_first_n"]); } catch (...) {}
        }
        if (node_->parameters.count("limit_rows")) {
            try { limit_rows_ = std::stoi(node_->parameters["limit_rows"]); } catch (...) {}
        }
    }
}

void CSVReaderDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;

    switch (delimiter_idx_) {
        case 0: node_->parameters["delimiter"] = ","; break;
        case 1: node_->parameters["delimiter"] = ";"; break;
        case 2: node_->parameters["delimiter"] = "\\t"; break;
        case 3: node_->parameters["delimiter"] = custom_delimiter_; break;
    }

    node_->parameters["has_header"] = has_header_ ? "true" : "false";
    node_->parameters["skip_empty_lines"] = skip_empty_lines_ ? "true" : "false";
    node_->parameters["skip_first_n"] = std::to_string(skip_first_n_);
    node_->parameters["limit_rows"] = std::to_string(limit_rows_);

    has_changes_ = false;
    spdlog::info("CSVReaderDialog: Applied settings for file '{}'", file_path_);
}

void CSVReaderDialog::Reset() {
    if (!node_) return;

    node_->parameters = original_params_;

    // Re-initialize UI state
    if (original_params_.count("file_path")) {
        strncpy(file_path_, original_params_["file_path"].c_str(), sizeof(file_path_) - 1);
    }

    preview_loaded_ = false;
    has_changes_ = false;
}

void CSVReaderDialog::RenderContent() {
    // Tab bar
    if (ImGui::BeginTabBar("CSVTabs")) {
        if (ImGui::BeginTabItem("Settings")) {
            RenderFileSettingsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Transformation")) {
            RenderTransformationTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Advanced Settings")) {
            RenderAdvancedSettingsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Limit Rows")) {
            RenderLimitRowsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Preview")) {
            RenderPreviewTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void CSVReaderDialog::RenderFileSettingsTab() {
    ImGui::Spacing();

    // File path selector
    std::string path_str = file_path_;
    if (FileSelector("Input File:", path_str, "CSV Files\0*.csv\0All Files\0*.*\0")) {
        strncpy(file_path_, path_str.c_str(), sizeof(file_path_) - 1);
        preview_loaded_ = false;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Column Delimiter
    ImGui::Text("Column Delimiter:");
    const char* delimiters[] = { "Comma (,)", "Semicolon (;)", "Tab", "Custom" };
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::Combo("##delimiter", &delimiter_idx_, delimiters, 4)) {
        preview_loaded_ = false;
        has_changes_ = true;
    }

    if (delimiter_idx_ == 3) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60.0f);
        if (ImGui::InputText("##custom_delim", custom_delimiter_, sizeof(custom_delimiter_))) {
            preview_loaded_ = false;
            has_changes_ = true;
        }
    }

    ImGui::Spacing();

    // Has header row
    if (ImGui::Checkbox("Has column header in first row", &has_header_)) {
        preview_loaded_ = false;
        has_changes_ = true;
    }

    // Skip empty lines
    if (ImGui::Checkbox("Skip empty lines", &skip_empty_lines_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // File info
    if (file_path_[0] != '\0') {
        std::ifstream test_file(file_path_);
        if (test_file.good()) {
            test_file.seekg(0, std::ios::end);
            auto size = test_file.tellg();
            test_file.close();

            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "File found");
            ImGui::Text("Size: %.2f KB", static_cast<float>(size) / 1024.0f);
        } else {
            ValidationMessage("File not found or cannot be opened");
        }
    }
}

void CSVReaderDialog::RenderTransformationTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Column transformation options will be available here.");
    ImGui::Spacing();

    ImGui::Text("Available transformations:");
    ImGui::BulletText("Column type inference");
    ImGui::BulletText("Column renaming");
    ImGui::BulletText("Column filtering");
    ImGui::BulletText("Missing value handling");

    ImGui::Spacing();
    ImGui::TextDisabled("(Configure after loading file in Preview tab)");
}

void CSVReaderDialog::RenderAdvancedSettingsTab() {
    ImGui::Spacing();

    // Skip first N rows
    ImGui::Text("Skip first N rows (after header):");
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputInt("##skip_rows", &skip_first_n_)) {
        if (skip_first_n_ < 0) skip_first_n_ = 0;
        preview_loaded_ = false;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Encoding options
    ImGui::Text("Character Encoding:");
    const char* encodings[] = { "UTF-8", "ASCII", "ISO-8859-1", "Windows-1252" };
    static int encoding_idx = 0;
    ImGui::SetNextItemWidth(200.0f);
    ImGui::Combo("##encoding", &encoding_idx, encodings, 4);

    ImGui::Spacing();

    // Quote character
    ImGui::Text("Quote Character:");
    static char quote_char[4] = "\"";
    ImGui::SetNextItemWidth(60.0f);
    ImGui::InputText("##quote", quote_char, sizeof(quote_char));

    ImGui::Spacing();

    // Comment character
    ImGui::Text("Comment Character:");
    static char comment_char[4] = "#";
    ImGui::SetNextItemWidth(60.0f);
    ImGui::InputText("##comment", comment_char, sizeof(comment_char));
    ImGui::SameLine();
    ImGui::TextDisabled("(Lines starting with this are skipped)");
}

void CSVReaderDialog::RenderLimitRowsTab() {
    ImGui::Spacing();

    ImGui::Text("Limit number of rows to read:");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##limit_rows", &limit_rows_)) {
        if (limit_rows_ < 1) limit_rows_ = 1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(0 = no limit)");

    ImGui::Spacing();

    static bool limit_enabled = true;
    ImGui::Checkbox("Enable row limit", &limit_enabled);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
        "Note: Row limits are useful for:\n"
        "- Testing pipelines with large files\n"
        "- Memory management\n"
        "- Quick previews");
}

void CSVReaderDialog::RenderPreviewTab() {
    ImGui::Spacing();

    // Load preview button
    if (ImGui::Button("Load Preview") || (!preview_loaded_ && file_path_[0] != '\0')) {
        LoadPreview();
    }

    if (!preview_error_.empty()) {
        ImGui::Spacing();
        ValidationMessage(preview_error_);
        return;
    }

    if (!preview_loaded_) {
        ImGui::Spacing();
        ImGui::TextDisabled("Select a file and click 'Load Preview'");
        return;
    }

    ImGui::Spacing();
    ImGui::Text("Showing first %d rows", static_cast<int>(preview_rows_.size()));
    ImGui::Separator();

    // Preview table
    if (!preview_headers_.empty()) {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                                ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_Resizable |
                                ImGuiTableFlags_ScrollX |
                                ImGuiTableFlags_ScrollY;

        int num_cols = static_cast<int>(preview_headers_.size());
        if (num_cols > 20) num_cols = 20;  // Limit columns for display

        if (ImGui::BeginTable("##preview_table", num_cols, flags, ImVec2(0, 300))) {
            // Header row
            for (int col = 0; col < num_cols; col++) {
                ImGui::TableSetupColumn(preview_headers_[col].c_str());
            }
            ImGui::TableHeadersRow();

            // Data rows
            for (const auto& row : preview_rows_) {
                ImGui::TableNextRow();
                for (int col = 0; col < num_cols && col < static_cast<int>(row.size()); col++) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(row[col].c_str());
                }
            }

            ImGui::EndTable();
        }
    }
}

void CSVReaderDialog::LoadPreview() {
    preview_headers_.clear();
    preview_rows_.clear();
    preview_error_.clear();
    preview_loaded_ = false;

    if (file_path_[0] == '\0') {
        preview_error_ = "No file selected";
        return;
    }

    std::ifstream file(file_path_);
    if (!file.good()) {
        preview_error_ = "Cannot open file: " + std::string(file_path_);
        return;
    }

    // Determine delimiter
    char delim = ',';
    switch (delimiter_idx_) {
        case 0: delim = ','; break;
        case 1: delim = ';'; break;
        case 2: delim = '\t'; break;
        case 3: delim = custom_delimiter_[0]; break;
    }

    std::string line;
    int line_num = 0;
    int preview_limit = 100;  // Preview first 100 rows

    while (std::getline(file, line) && line_num < preview_limit + skip_first_n_ + (has_header_ ? 1 : 0)) {
        // Skip empty lines if configured
        if (skip_empty_lines_ && line.empty()) continue;

        // Skip first N rows
        if (line_num < skip_first_n_) {
            line_num++;
            continue;
        }

        // Parse line
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delim)) {
            // Trim whitespace
            size_t start = cell.find_first_not_of(" \t\r\n");
            size_t end = cell.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            } else {
                cell.clear();
            }
            cells.push_back(cell);
        }

        // First row after skip is header (if has_header)
        if (has_header_ && line_num == skip_first_n_) {
            preview_headers_ = cells;
        } else {
            preview_rows_.push_back(cells);
        }

        line_num++;
    }

    // Generate default headers if no header row
    if (!has_header_ && !preview_rows_.empty()) {
        for (size_t i = 0; i < preview_rows_[0].size(); i++) {
            preview_headers_.push_back("Column " + std::to_string(i + 1));
        }
    }

    preview_loaded_ = true;
    spdlog::info("CSVReaderDialog: Loaded preview with {} columns, {} rows",
                 preview_headers_.size(), preview_rows_.size());
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
        case 1:
        default:
            return "word";
    }
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
    max_length_ = IsPaddingNode() ? 512 : 256;
    max_vocab_size_ = IsVocabularyNode() ? -1 : 10000;
    min_word_freq_ = IsVocabularyNode() ? 1 : 2;
    pad_value_ = 0;
    lowercase_ = true;
    padding_ = true;
    truncation_ = true;
    std::strncpy(text_col_, "text", sizeof(text_col_) - 1);
    text_col_[sizeof(text_col_) - 1] = '\0';
    label_col_[0] = '\0';
    vocab_file_[0] = '\0';

    auto method = node_->parameters.find("method");
    if (method != node_->parameters.end()) {
        if (method->second == "whitespace") tokenizer_type_ = 0;
        else if (method->second == "word") tokenizer_type_ = 1;
        else if (method->second == "character") tokenizer_type_ = 2;
    }

    if (node_->parameters.count("text_col")) {
        CopyParam(node_, "text_col", text_col_, sizeof(text_col_));
    } else {
        CopyParam(node_, "text_column", text_col_, sizeof(text_col_));
    }
    CopyParam(node_, "label_col", label_col_, sizeof(label_col_));
    CopyParam(node_, "vocab_file", vocab_file_, sizeof(vocab_file_));

    ReadIntParam(node_, "tokenizer_type", tokenizer_type_);
    ReadIntParam(node_, "max_length", max_length_);
    ReadIntParam(node_, "max_vocab_size", max_vocab_size_);
    ReadIntParam(node_, "min_word_freq", min_word_freq_);
    ReadIntParam(node_, "min_freq", min_word_freq_);
    ReadIntParam(node_, "min_frequency", min_word_freq_);
    ReadIntParam(node_, "pad_value", pad_value_);
    ReadBoolParam(node_, "lowercase", lowercase_);
    ReadBoolParam(node_, "padding", padding_);
    ReadBoolParam(node_, "truncation", truncation_);

    if (tokenizer_type_ < 0 || tokenizer_type_ > 2) tokenizer_type_ = 1;
    if (max_length_ < 1) max_length_ = 1;
    if (min_word_freq_ < 1) min_word_freq_ = 1;
}

void TokenizerDialog::Apply() {
    if (!node_) return;

    if (IsTokenizerNode()) {
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
    }

    if (IsVocabularyNode()) {
        node_->parameters["min_freq"] = std::to_string(min_word_freq_);
        node_->parameters["min_word_freq"] = std::to_string(min_word_freq_);
        node_->parameters["max_vocab_size"] = std::to_string(max_vocab_size_);
        node_->parameters["vocab_file"] = vocab_file_;
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

void TokenizerDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    preview_tokens_.clear();
    has_changes_ = false;
}

void TokenizerDialog::RenderContent() {
    if (ImGui::BeginTabBar("TextPreprocessingTabs")) {
        if (IsTokenizerNode() && ImGui::BeginTabItem("Tokenizer")) {
            RenderTokenizerTab();
            ImGui::EndTabItem();
        }
        if ((IsTokenizerNode() || IsVocabularyNode()) &&
            ImGui::BeginTabItem("Vocabulary")) {
            RenderVocabularyTab();
            ImGui::EndTabItem();
        }
        if ((IsTokenizerNode() || IsPaddingNode()) &&
            ImGui::BeginTabItem("Padding")) {
            RenderPaddingTab();
            ImGui::EndTabItem();
        }
        if (IsTokenizerNode() && ImGui::BeginTabItem("Preview")) {
            RenderPreviewTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void TokenizerDialog::RenderTokenizerTab() {
    ImGui::Spacing();

    ImGui::Text("Text column:");
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("##text_col", text_col_, sizeof(text_col_))) {
        has_changes_ = true;
    }

    ImGui::Text("Label column:");
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("##label_col", label_col_, sizeof(label_col_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("optional");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Tokenizer:");
    const char* methods[] = { "Whitespace", "Word", "Character" };
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::Combo("##tokenizer_type", &tokenizer_type_, methods, 3)) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    if (ImGui::Checkbox("Convert to lowercase", &lowercase_)) {
        has_changes_ = true;
    }
}

void TokenizerDialog::RenderVocabularyTab() {
    ImGui::Spacing();

    ImGui::Text("Minimum token frequency:");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##min_word_freq", &min_word_freq_)) {
        if (min_word_freq_ < 1) min_word_freq_ = 1;
        has_changes_ = true;
    }

    ImGui::Spacing();

    ImGui::Text("Maximum vocabulary size:");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_vocab", &max_vocab_size_)) {
        if (max_vocab_size_ < -1) max_vocab_size_ = -1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("-1 = unlimited");

    if (IsVocabularyNode()) {
        ImGui::Spacing();
        ImGui::Text("Vocabulary file:");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##vocab_file", vocab_file_, sizeof(vocab_file_))) {
            has_changes_ = true;
        }
    }
}

void TokenizerDialog::RenderPaddingTab() {
    ImGui::Spacing();

    ImGui::Text("Maximum sequence length:");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_length", &max_length_)) {
        if (max_length_ < 1) max_length_ = 1;
        has_changes_ = true;
    }

    if (IsPaddingNode()) {
        ImGui::Spacing();
        ImGui::Text("Pad value:");
        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::InputInt("##pad_value", &pad_value_)) {
            has_changes_ = true;
        }
    }

    ImGui::Spacing();

    if (IsTokenizerNode()) {
        if (ImGui::Checkbox("Pad sequences", &padding_)) {
            has_changes_ = true;
        }
    }
    if (ImGui::Checkbox("Truncate long sequences", &truncation_)) {
        has_changes_ = true;
    }
}

void TokenizerDialog::RenderPreviewTab() {
    ImGui::Spacing();

    ImGui::Text("Sample Text:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##sample", sample_text_, sizeof(sample_text_), ImVec2(0, 100))) {
        preview_tokens_.clear();
    }

    ImGui::Spacing();

    if (ImGui::Button("Tokenize Preview")) {
        preview_tokens_.clear();

        std::string text = sample_text_;
        if (lowercase_) {
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
        }

        if (tokenizer_type_ == 2) {
            for (char c : text) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    preview_tokens_.push_back(std::string(1, c));
                }
            }
        } else if (tokenizer_type_ == 1) {
            std::string token;
            for (char c : text) {
                unsigned char uc = static_cast<unsigned char>(c);
                if (std::isalnum(uc) || c == '_') {
                    token.push_back(c);
                } else if (!token.empty()) {
                    preview_tokens_.push_back(token);
                    token.clear();
                }
            }
            if (!token.empty()) {
                preview_tokens_.push_back(token);
            }
        } else {
            std::stringstream ss(text);
            std::string token;
            while (ss >> token) {
                if (!token.empty()) {
                    preview_tokens_.push_back(token);
                }
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Show tokens
    if (!preview_tokens_.empty()) {
        ImGui::Text("Tokens (%d):", static_cast<int>(preview_tokens_.size()));

        // Display tokens as colored chips
        float wrap_x = ImGui::GetContentRegionAvail().x;
        float current_x = 0.0f;

        for (size_t i = 0; i < preview_tokens_.size(); i++) {
            const auto& token = preview_tokens_[i];

            // Calculate token width
            ImVec2 text_size = ImGui::CalcTextSize(token.c_str());
            float chip_width = text_size.x + 16.0f;

            // Wrap to next line if needed
            if (current_x + chip_width > wrap_x && current_x > 0) {
                ImGui::NewLine();
                current_x = 0.0f;
            }

            // Draw token chip
            ImGui::PushID(static_cast<int>(i));
            ImVec4 bg_color = ImVec4(0.2f, 0.4f, 0.6f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, bg_color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
            ImGui::SmallButton(token.c_str());
            ImGui::PopStyleColor(2);
            ImGui::PopID();

            ImGui::SameLine();
            current_x += chip_width + ImGui::GetStyle().ItemSpacing.x;
        }
        ImGui::NewLine();
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
    RegisterDialog(NT::CSVFile, [](MLNode* node) {
        return std::make_unique<CSVReaderDialog>(node);
    });

    RegisterDialog(NT::TextTokenizer, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });
    RegisterDialog(NT::TextVocabulary, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });
    RegisterDialog(NT::TextPadding, [](MLNode* node) {
        return std::make_unique<TokenizerDialog>(node);
    });

    RegisterDialog(NT::FilterRows, [](MLNode* node) {
        return std::make_unique<FilterDialog>(node);
    });

    // Register more dialogs for complex nodes
    RegisterDialog(NT::ExcelFile, [](MLNode* node) {
        return std::make_unique<CSVReaderDialog>(node);  // Reuse CSV dialog for now
    });

    // Smart I/O Nodes - Universal dialogs
    RegisterDialog(NT::DataInput, [](MLNode* node) {
        return std::make_unique<DataInputDialog>(node);
    });

    RegisterDialog(NT::DataOutput, [](MLNode* node) {
        return std::make_unique<DataOutputDialog>(node);
    });

    // Data pipeline nodes - single source of truth for batching and splitting
    RegisterDialog(NT::DataLoader, [](MLNode* node) {
        return std::make_unique<DataLoaderDialog>(node);
    });

    RegisterDialog(NT::DataSplit, [](MLNode* node) {
        return std::make_unique<DataSplitDialog>(node);
    });

    // Visualization framework — chart nodes live in their own TU
    // under src/gui/visualization/. BarChart is the first; Histogram /
    // LinePlot / ScatterPlot / PieChart follow the same shape.
    RegisterDialog(NT::BarChart, [](MLNode* node) {
        return std::make_unique<visualization::BarChartDialog>(node);
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
