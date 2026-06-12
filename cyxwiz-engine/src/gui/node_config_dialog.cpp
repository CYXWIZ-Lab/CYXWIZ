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
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <set>
#include <unordered_map>

namespace gui {

// Alias to avoid any potential macro conflicts
using NT = NodeType;

namespace {

void HelpTooltip(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace

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

bool WriteEmbeddingMatrix(const std::filesystem::path& path,
                          int rows,
                          int cols,
                          int padding_idx,
                          int init_mode,
                          std::string& error) {
    if (rows <= 0 || cols <= 0) {
        error = "Embedding rows and dimensions must be positive.";
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        error = "Could not open output file: " + path.string();
        return false;
    }

    std::mt19937 rng(42);
    const float limit = std::sqrt(6.0f / static_cast<float>(rows + cols));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-limit, limit);

    out << "# cyxwiz_embedding rows=" << rows << " dim=" << cols << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float value = 0.0f;
            if (r == padding_idx) {
                value = 0.0f;
            } else if (init_mode == 2) {
                value = (r == c) ? 1.0f : 0.0f;
            } else if (init_mode == 1) {
                value = uniform(rng);
            } else {
                value = normal(rng);
            }
            if (c > 0) out << ' ';
            out << value;
        }
        out << '\n';
    }
    return true;
}

bool InspectEmbeddingMatrix(const std::filesystem::path& path,
                            int& rows,
                            int& cols,
                            std::string& error) {
    rows = 0;
    cols = -1;
    std::ifstream in(path);
    if (!in.is_open()) {
        error = "Could not open weights file: " + path.string();
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        float value = 0.0f;
        int count = 0;
        while (iss >> value) ++count;
        if (count <= 0) continue;
        if (cols < 0) cols = count;
        if (count != cols) {
            error = "Inconsistent embedding row width in " + path.string();
            return false;
        }
        ++rows;
    }
    if (rows <= 0 || cols <= 0) {
        error = "No numeric embedding rows found in " + path.string();
        return false;
    }
    return true;
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
    source_csv_[0] = '\0';
    status_message_.clear();
    status_is_error_ = false;

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
    CopyParam(node_, "source_csv", source_csv_, sizeof(source_csv_));

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
        node_->parameters["vocab_file"] = vocab_file_;
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

    std::unordered_map<std::string, int> counts;
    std::string line;
    size_t rows = 0;
    while (ReadCsvRecord(in, line)) {
        const auto fields = SplitCsvLineSimple(line);
        if (text_idx >= static_cast<int>(fields.size())) continue;
        const auto tokens = TokenizePreviewText(fields[static_cast<size_t>(text_idx)],
                                                tokenizer_type_, lowercase_);
        if (tokens.empty()) continue;
        ++rows;
        for (const auto& token : tokens) {
            ++counts[token];
        }
    }

    std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;
              });

    const std::vector<std::string> specials = {"[PAD]", "[UNK]", "[BOS]", "[EOS]"};
    std::set<std::string> emitted(specials.begin(), specials.end());
    std::vector<std::string> vocab = specials;
    const int cap = max_vocab_size_ > 0 ? max_vocab_size_ : std::numeric_limits<int>::max();
    for (const auto& [token, count] : sorted) {
        if (count < min_word_freq_) continue;
        if (emitted.count(token) > 0) continue;
        if (static_cast<int>(vocab.size()) >= cap) break;
        emitted.insert(token);
        vocab.push_back(token);
    }

    if (out_path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
    }
    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open()) {
        status_message_ = "Could not write vocab file: " + out_path.string();
        status_is_error_ = true;
        return false;
    }
    for (const auto& token : vocab) {
        out << token << '\n';
    }

    max_vocab_size_ = static_cast<int>(vocab.size());
    if (node_) {
        node_->parameters["max_vocab_size"] = std::to_string(max_vocab_size_);
        node_->parameters["vocab_file"] = vocab_file_;
        node_->parameters["source_csv"] = source_csv_;
    }
    status_message_ = "Built " + std::to_string(vocab.size()) +
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
    std::ifstream in(vocab_file_);
    if (!in.is_open()) {
        status_message_ = "Could not open vocab file: " + std::string(vocab_file_);
        status_is_error_ = true;
        return false;
    }
    size_t count = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) ++count;
    }
    status_message_ = "Vocab file contains " + std::to_string(count) + " entries.";
    status_is_error_ = false;
    return true;
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
    HelpTooltip("CSV/Arrow column that contains the raw sentences or documents. For the sentiment graph this is usually 'statement'.");
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::InputText("##text_col", text_col_, sizeof(text_col_))) {
        has_changes_ = true;
    }

    ImGui::Text("Label column:");
    HelpTooltip("Optional target column used for supervised training. Leave blank for unlabeled text; for sentiment this is usually 'status'.");
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
    HelpTooltip("Controls how raw text is split before vocabulary lookup: whitespace splits on spaces, word keeps words and punctuation, character emits one token per character.");
    const char* methods[] = { "Whitespace", "Word", "Character" };
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::Combo("##tokenizer_type", &tokenizer_type_, methods, 3)) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    if (ImGui::Checkbox("Convert to lowercase", &lowercase_)) {
        has_changes_ = true;
    }
    HelpTooltip("Lowercasing reduces duplicate vocabulary entries like 'Happy' and 'happy'. Disable it when capitalization carries meaning.");
}

void TokenizerDialog::RenderVocabularyTab() {
    ImGui::Spacing();

    if (IsVocabularyNode()) {
        ImGui::Text("Source CSV:");
        HelpTooltip("Dataset file scanned when building a vocabulary from this node. Quoted multi-line CSV text is supported.");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##source_csv", source_csv_, sizeof(source_csv_))) {
            has_changes_ = true;
        }
        ImGui::Text("Text column:");
        HelpTooltip("Column in the source CSV used to count tokens for the vocabulary file.");
        ImGui::SetNextItemWidth(220.0f);
        if (ImGui::InputText("##vocab_text_col", text_col_, sizeof(text_col_))) {
            has_changes_ = true;
        }
        ImGui::Spacing();
    }

    ImGui::Text("Minimum token frequency:");
    HelpTooltip("Tokens seen fewer than this many times are excluded. Higher values shrink the vocabulary and map rare words to [UNK].");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##min_word_freq", &min_word_freq_)) {
        if (min_word_freq_ < 1) min_word_freq_ = 1;
        has_changes_ = true;
    }

    ImGui::Spacing();

    ImGui::Text("Maximum vocabulary size:");
    HelpTooltip("Upper bound for vocabulary entries, including special tokens like [PAD] and [UNK]. Use -1 for no cap.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_vocab", &max_vocab_size_)) {
        if (max_vocab_size_ < -1) max_vocab_size_ = -1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("-1 = unlimited");

    if (IsTokenizerNode() || IsVocabularyNode()) {
        ImGui::Spacing();
        ImGui::Text("Vocabulary file:");
        HelpTooltip("One-token-per-line vocabulary file. Tokenizer can load it directly; TextVocabulary can build and save it from the source CSV.");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##vocab_file", vocab_file_, sizeof(vocab_file_))) {
            has_changes_ = true;
        }

        ImGui::Spacing();
        if (IsVocabularyNode() && ImGui::Button("Build and Save Vocabulary")) {
            BuildVocabularyFile();
            has_changes_ = true;
        }
        if (IsVocabularyNode()) {
            HelpTooltip("Scans the source CSV text column, counts tokens, applies frequency and size limits, then writes the vocabulary file.");
        }
        if (IsVocabularyNode()) {
            ImGui::SameLine();
        }
        if (ImGui::Button("Inspect File")) {
            InspectVocabularyFile();
        }
        HelpTooltip("Reads the vocabulary file and reports how many entries it contains.");
        if (!status_message_.empty()) {
            ValidationMessage(status_message_, status_is_error_);
        }
    }
}

void TokenizerDialog::RenderPaddingTab() {
    ImGui::Spacing();

    ImGui::Text("Maximum sequence length:");
    HelpTooltip("Fixed token count emitted for each sample. Short sequences are padded; long sequences can be truncated.");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::InputInt("##max_length", &max_length_)) {
        if (max_length_ < 1) max_length_ = 1;
        has_changes_ = true;
    }

    if (IsPaddingNode()) {
        ImGui::Spacing();
        ImGui::Text("Pad value:");
        HelpTooltip("Token id used to fill short sequences. Use 0 when the vocabulary starts with [PAD] and the Embedding padding index is 0.");
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
        HelpTooltip("Pads shorter token sequences to the maximum sequence length so batches have a stable tensor shape.");
    }
    if (ImGui::Checkbox("Truncate long sequences", &truncation_)) {
        has_changes_ = true;
    }
    HelpTooltip("Cuts sequences longer than the maximum length. Disable only if downstream nodes can handle variable or longer sequences.");
}

void TokenizerDialog::RenderPreviewTab() {
    ImGui::Spacing();

    ImGui::Text("Sample Text:");
    HelpTooltip("Local preview text for checking tokenizer behavior. It does not modify the dataset.");
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
    HelpTooltip("Runs the selected tokenizer settings against the sample text and shows the resulting tokens.");

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

// ==================== EmbeddingDialog ====================

EmbeddingDialog::EmbeddingDialog(MLNode* node)
    : NodeConfigDialog("Embedding Configuration", node) {
    LoadFromNode();
}

void EmbeddingDialog::LoadFromNode() {
    if (!node_) return;

    num_embeddings_ = 10000;
    embedding_dim_ = 256;
    padding_idx_ = -1;
    max_norm_ = 0.0f;
    freeze_ = false;
    init_mode_ = 0;
    weights_file_[0] = '\0';
    output_file_[0] = '\0';
    status_message_.clear();
    status_is_error_ = false;

    ReadIntParam(node_, "num_embeddings", num_embeddings_);
    ReadIntParam(node_, "embedding_dim", embedding_dim_);
    ReadIntParam(node_, "padding_idx", padding_idx_);
    ReadBoolParam(node_, "freeze", freeze_);
    CopyParam(node_, "weights_file", weights_file_, sizeof(weights_file_));
    CopyParam(node_, "embedding_weights_file", weights_file_, sizeof(weights_file_));
    CopyParam(node_, "output_weights_file", output_file_, sizeof(output_file_));

    auto max_norm_it = node_->parameters.find("max_norm");
    if (max_norm_it != node_->parameters.end() && !max_norm_it->second.empty()) {
        try { max_norm_ = std::stof(max_norm_it->second); } catch (...) {}
    }
    auto init_it = node_->parameters.find("init_mode");
    if (init_it != node_->parameters.end()) {
        if (init_it->second == "uniform") init_mode_ = 1;
        else if (init_it->second == "one_hot") init_mode_ = 2;
        else init_mode_ = 0;
    }

    if (num_embeddings_ < 2) num_embeddings_ = 2;
    if (embedding_dim_ < 1) embedding_dim_ = 1;
}

void EmbeddingDialog::Apply() {
    if (!node_) return;

    if (num_embeddings_ < 2) num_embeddings_ = 2;
    if (embedding_dim_ < 1) embedding_dim_ = 1;
    if (padding_idx_ >= num_embeddings_) padding_idx_ = num_embeddings_ - 1;
    if (max_norm_ < 0.0f) max_norm_ = 0.0f;

    node_->parameters["num_embeddings"] = std::to_string(num_embeddings_);
    node_->parameters["embedding_dim"] = std::to_string(embedding_dim_);
    node_->parameters["padding_idx"] = std::to_string(padding_idx_);
    node_->parameters["max_norm"] = std::to_string(max_norm_);
    node_->parameters["freeze"] = freeze_ ? "true" : "false";
    node_->parameters["weights_file"] = weights_file_;
    node_->parameters["embedding_weights_file"] = weights_file_;
    node_->parameters["output_weights_file"] = output_file_;
    switch (init_mode_) {
        case 1: node_->parameters["init_mode"] = "uniform"; break;
        case 2: node_->parameters["init_mode"] = "one_hot"; break;
        default: node_->parameters["init_mode"] = "normal"; break;
    }

    has_changes_ = false;
    spdlog::info("EmbeddingDialog: Applied embedding settings");
}

void EmbeddingDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    has_changes_ = false;
}

bool EmbeddingDialog::BuildAndSaveWeights() {
    if (output_file_[0] == '\0') {
        status_message_ = "Choose an output weights file before building.";
        status_is_error_ = true;
        return false;
    }

    std::string error;
    if (!WriteEmbeddingMatrix(output_file_, num_embeddings_, embedding_dim_,
                              padding_idx_, init_mode_, error)) {
        status_message_ = error;
        status_is_error_ = true;
        return false;
    }

    std::strncpy(weights_file_, output_file_, sizeof(weights_file_) - 1);
    weights_file_[sizeof(weights_file_) - 1] = '\0';
    status_message_ = "Wrote embedding matrix " + std::to_string(num_embeddings_) +
                      " x " + std::to_string(embedding_dim_) + ".";
    status_is_error_ = false;
    has_changes_ = true;
    return true;
}

bool EmbeddingDialog::InspectWeightFile() {
    if (weights_file_[0] == '\0') {
        status_message_ = "Choose a weights file to inspect.";
        status_is_error_ = true;
        return false;
    }
    int rows = 0;
    int cols = 0;
    std::string error;
    if (!InspectEmbeddingMatrix(weights_file_, rows, cols, error)) {
        status_message_ = error;
        status_is_error_ = true;
        return false;
    }

    status_message_ = "Weights file shape: " + std::to_string(rows) +
                      " x " + std::to_string(cols) + ".";
    status_is_error_ = false;
    if (rows != num_embeddings_ || cols != embedding_dim_) {
        status_message_ += " This does not match the current node shape.";
        status_is_error_ = true;
    }
    return !status_is_error_;
}

void EmbeddingDialog::RenderContent() {
    if (ImGui::BeginTabBar("EmbeddingTabs")) {
        if (ImGui::BeginTabItem("Shape")) {
            RenderShapeTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Weights")) {
            RenderWeightsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Advanced")) {
            RenderAdvancedTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void EmbeddingDialog::RenderShapeTab() {
    ImGui::Spacing();
    ImGui::Text("Vocabulary size:");
    HelpTooltip("Number of token ids the embedding table can look up. This must match the tokenizer/vocabulary size.");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##num_embeddings", &num_embeddings_)) {
        if (num_embeddings_ < 2) num_embeddings_ = 2;
        has_changes_ = true;
    }

    ImGui::Text("Embedding dimension:");
    HelpTooltip("Number of float features learned for each token id. Larger values can model richer text patterns but cost more memory and compute.");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##embedding_dim", &embedding_dim_)) {
        if (embedding_dim_ < 1) embedding_dim_ = 1;
        has_changes_ = true;
    }

    ImGui::Text("Padding index:");
    HelpTooltip("Token id treated as padding. Its vector is forced to zero so padded positions do not carry text meaning.");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputInt("##padding_idx", &padding_idx_)) {
        if (padding_idx_ < -1) padding_idx_ = -1;
        has_changes_ = true;
    }
    ImGui::TextDisabled("-1 disables padding-zero behavior; 0 is typical for [PAD].");
}

void EmbeddingDialog::RenderWeightsTab() {
    ImGui::Spacing();
    ImGui::Text("Load pretrained weights:");
    HelpTooltip("Optional text matrix file with one embedding row per token id. Its shape must be vocabulary_size x embedding_dimension.");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##weights_file", weights_file_, sizeof(weights_file_))) {
        has_changes_ = true;
    }
    if (ImGui::Button("Inspect Weights")) {
        InspectWeightFile();
    }
    HelpTooltip("Checks that the selected weights file has a consistent matrix shape and matches this node.");
    ImGui::SameLine();
    if (ImGui::Checkbox("Freeze loaded weights", &freeze_)) {
        has_changes_ = true;
    }
    HelpTooltip("When enabled, pretrained vectors are used as fixed features and are not updated during training.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Build starter weights:");
    HelpTooltip("Creates an initial embedding matrix from this node's shape. This is useful when you want a saved matrix file before training.");
    const char* modes[] = {"Random normal", "Random uniform", "One-hot / truncated"};
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::Combo("##init_mode", &init_mode_, modes, 3)) {
        has_changes_ = true;
    }
    ImGui::Text("Output weights file:");
    HelpTooltip("Destination path for the generated embedding matrix. The file can be loaded back through 'Load pretrained weights'.");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##output_weights_file", output_file_, sizeof(output_file_))) {
        has_changes_ = true;
    }
    if (ImGui::Button("Build, Save, and Use")) {
        BuildAndSaveWeights();
    }
    HelpTooltip("Writes the starter matrix and immediately selects it as this node's weights file.");

    if (!status_message_.empty()) {
        ValidationMessage(status_message_, status_is_error_);
    }
}

void EmbeddingDialog::RenderAdvancedTab() {
    ImGui::Spacing();
    ImGui::Text("Max norm:");
    HelpTooltip("Optional length cap for each token vector during lookup. Use 0 to disable clipping.");
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::InputFloat("##max_norm", &max_norm_, 0.1f, 1.0f, "%.4f")) {
        if (max_norm_ < 0.0f) max_norm_ = 0.0f;
        has_changes_ = true;
    }
    ImGui::TextDisabled("0 disables norm clipping. Positive values clip each vector during lookup.");
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
    RegisterDialog(NT::Embedding, [](MLNode* node) {
        return std::make_unique<EmbeddingDialog>(node);
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
