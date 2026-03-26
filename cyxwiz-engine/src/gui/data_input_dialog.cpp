// Data Input/Output Dialog Implementations
// Part of CyxWiz Studio smart data loading system
// KNIME-style tabbed interface with live preview

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

#ifdef CreateDialog
#undef CreateDialog
#endif

#include "node_config_dialog.h"
#include "node_editor.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace gui {

// ==================== DataInputDialog ====================

DataInputDialog::DataInputDialog(MLNode* node)
    : NodeConfigDialog("Data Input", node)
{
    if (node_) {
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
            DetectFileType();
        }
        if (node_->parameters.count("memory_policy")) {
            memory_policy_ = (node_->parameters["memory_policy"] == "disc")
                ? MemoryPolicy::WriteToDisc : MemoryPolicy::CacheInMemory;
        }
    }
    available_presets_ = {"Default", "Large Files", "Low Memory", "Fast Load"};
}

void DataInputDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;
    const char* types[] = {"auto", "csv", "tsv", "json", "parquet", "excel", "hdf5"};
    node_->parameters["type"] = types[detected_type_];
    node_->parameters["has_header"] = has_header_ ? "true" : "false";
    node_->parameters["skip_rows"] = std::to_string(skip_rows_);
    node_->parameters["max_rows"] = std::to_string(max_rows_);
    node_->parameters["where_clause"] = where_clause_;
    node_->parameters["chunk_size_kb"] = auto_chunk_size_ ? "auto" : std::to_string(chunk_size_kb_);
    node_->parameters["lru_chunks"] = std::to_string(lru_chunks_);
    node_->parameters["prefetch"] = prefetch_enabled_ ? "true" : "false";
    node_->parameters["memory_policy"] = (memory_policy_ == MemoryPolicy::WriteToDisc) ? "disc" : "memory";
    node_->parameters["configured"] = "true";

    has_changes_ = false;
    spdlog::info("DataInputDialog: Applied settings for '{}'", file_path_);
}

void DataInputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    preview_loaded_ = false;
    has_changes_ = false;
}

void DataInputDialog::RenderContent() {
    // KNIME-style tab bar at top
    if (ImGui::BeginTabBar("DataInputTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Settings")) {
            RenderSettingsTab();
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
        if (ImGui::BeginTabItem("Encoding")) {
            RenderEncodingTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Memory Policy")) {
            RenderMemoryPolicyTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataInputDialog::RenderSettingsTab() {
    // ===== FILE SELECTION SECTION =====
    // Prominent file input with visual feedback
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.14f, 0.18f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    if (ImGui::BeginChild("FileSelection", ImVec2(0, 130), true)) {
        // Section header with icon-like indicator
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
        ImGui::Text("[FILE]  Input File");
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        // File path - PROMINENT display
        ImGui::Text("Path:");
        ImGui::SameLine(60);

        // Style the input field
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.10f, 0.14f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.12f, 0.15f, 0.20f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);

        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        std::string path_str = file_path_;
        if (ImGui::InputText("##filepath", file_path_, sizeof(file_path_))) {
            DetectFileType();
            preview_loaded_ = false;
            has_changes_ = true;
        }
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);

        ImGui::SameLine();

        // Styled Browse button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.88f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.62f, 0.98f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        if (ImGui::Button("Browse...", ImVec2(80, 0))) {
            if (FileSelector("Select Data File:", path_str, "All Files\0*.*\0CSV\0*.csv\0Excel\0*.xlsx;*.xls\0JSON\0*.json;*.jsonl\0Parquet\0*.parquet;*.pq\0HDF5\0*.h5;*.hdf5\0")) {
                strncpy(file_path_, path_str.c_str(), sizeof(file_path_) - 1);
                DetectFileType();
                preview_loaded_ = false;
                has_changes_ = true;
            }
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);

        // File info display
        ImGui::Spacing();
        if (strlen(file_path_) > 0) {
            // Show detected type with color coding
            ImVec4 type_color = ImVec4(0.5f, 0.8f, 0.5f, 1.0f);  // Green for detected
            ImGui::TextColored(type_color, "Type: %s", GetFileTypeName());
            ImGui::SameLine(150);

            // Show file size
            if (file_size_ > 0) {
                if (file_size_ >= 1024 * 1024) {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), "Size: %.1f MB", file_size_ / (1024.0f * 1024.0f));
                } else if (file_size_ >= 1024) {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), "Size: %.1f KB", file_size_ / 1024.0f);
                } else {
                    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.8f, 1.0f), "Size: %zu bytes", file_size_);
                }
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.5f, 0.3f, 1.0f), "No file selected - click Browse to select a data file");
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    ImGui::Spacing();

    // ===== READER OPTIONS SECTION =====
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.14f, 0.18f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    if (ImGui::BeginChild("ReaderOptions", ImVec2(0, 200), true)) {
        // Section header
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
        ImGui::Text("[OPTIONS]  Reader Options");
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Spacing();

        // Format detection row
        ImGui::Text("Format:");
        ImGui::SameLine(80);

        // Styled autodetect button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.4f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        if (ImGui::Button("Autodetect", ImVec2(90, 0))) {
            DetectFileType();
            preview_loaded_ = false;
        }
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);

        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "-> %s", GetFileTypeName());

        // Delimiter options (for CSV/TSV)
        if (detected_type_ <= 2) {
            ImGui::Spacing();

            // Column delimiter
            ImGui::Text("Column delimiter");
            ImGui::SameLine(120);
            ImGui::SetNextItemWidth(50);
            static char delim[4] = ",";
            if (ImGui::InputText("##coldelim", delim, sizeof(delim))) {
                custom_delimiter_[0] = delim[0];
                has_changes_ = true;
                preview_loaded_ = false;
            }

            // Row delimiter
            ImGui::SameLine(200);
            ImGui::Text("Row delimiter");
            ImGui::SameLine(320);
            static int row_delim = 0;
            ImGui::RadioButton("Line break", &row_delim, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Custom", &row_delim, 1);

            // Quote chars
            ImGui::Text("Quote char");
            ImGui::SameLine(120);
            ImGui::SetNextItemWidth(30);
            ImGui::InputText("##quote", quote_char_, sizeof(quote_char_));
            ImGui::SameLine(200);
            ImGui::Text("Quote escape char");
            ImGui::SameLine(340);
            ImGui::SetNextItemWidth(30);
            static char escape_char[4] = "\"";
            ImGui::InputText("##escape", escape_char, sizeof(escape_char));

            // Comment char
            ImGui::Text("Comment char");
            ImGui::SameLine(120);
            ImGui::SetNextItemWidth(30);
            static char comment_char[4] = "#";
            ImGui::InputText("##comment", comment_char, sizeof(comment_char));

            // Checkboxes
            ImGui::Spacing();
            if (ImGui::Checkbox("Has column header", &has_header_)) {
                has_changes_ = true;
                preview_loaded_ = false;
            }
            ImGui::SameLine(200);
            static bool has_row_id = false;
            ImGui::Checkbox("Has RowID", &has_row_id);

            static bool short_data_rows = false;
            ImGui::Checkbox("Support short data rows", &short_data_rows);
        } else if (detected_type_ == 5) {  // Excel
            ImGui::Text("Sheet:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(100);
            ImGui::InputInt("##sheet", &sheet_idx_);
            ImGui::Text("Range:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(100);
            ImGui::InputText("##range", sheet_range_, sizeof(sheet_range_));
            ImGui::SameLine();
            ImGui::TextDisabled("(e.g., A1:D100)");
        } else if (detected_type_ == 6) {  // HDF5
            ImGui::Text("Dataset key:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(200);
            ImGui::InputText("##hdf5key", hdf5_dataset_, sizeof(hdf5_dataset_));
        } else if (detected_type_ == 3) {  // JSON
            if (ImGui::Checkbox("JSON Lines format", &json_lines_)) {
                has_changes_ = true;
            }
            ImGui::Text("JSON Path:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(200);
            ImGui::InputText("##jsonpath", json_path_, sizeof(json_path_));
            ImGui::SameLine();
            ImGui::TextDisabled("(e.g., $.data.records)");
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    ImGui::Spacing();

    // Preview section
    RenderPreviewTable();
}

void DataInputDialog::RenderTransformationTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Column transformations");
    ImGui::Separator();
    ImGui::Spacing();

    // Column selection
    if (ImGui::Checkbox("Include all columns", &select_all_columns_)) {
        has_changes_ = true;
        for (size_t i = 0; i < selected_columns_.size(); i++) {
            selected_columns_[i] = select_all_columns_;
        }
    }

    if (!select_all_columns_ && !available_columns_.empty()) {
        ImGui::Spacing();
        ImGui::Text("Select columns to include:");
        ImGui::BeginChild("ColumnList", ImVec2(0, 200), true);
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); i++) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
                has_changes_ = true;
            }
        }
        ImGui::EndChild();
    }

    if (available_columns_.empty() && strlen(file_path_) > 0) {
        ImGui::TextDisabled("Click 'Autodetect format' to load column list");
    }
}

void DataInputDialog::RenderAdvancedSettingsTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Advanced parsing options");
    ImGui::Separator();
    ImGui::Spacing();

    // Type inference
    static int type_limit = 10000;
    ImGui::Text("Type inference rows:");
    ImGui::SameLine(200);
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("##typelimit", &type_limit);
    ImGui::SameLine();
    ImGui::TextDisabled("(rows to scan for type detection)");

    ImGui::Spacing();

    // Missing values
    ImGui::Text("Missing value patterns:");
    ImGui::SameLine(200);
    static char missing_patterns[256] = "NA, N/A, null, NaN, -";
    ImGui::SetNextItemWidth(300);
    ImGui::InputText("##missing", missing_patterns, sizeof(missing_patterns));

    ImGui::Spacing();

    // Date/time format
    ImGui::Text("Date format:");
    ImGui::SameLine(200);
    static char date_format[64] = "auto";
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##dateformat", date_format, sizeof(date_format));
    ImGui::SameLine();
    ImGui::TextDisabled("(e.g., YYYY-MM-DD)");
}

void DataInputDialog::RenderLimitRowsTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Row filtering options");
    ImGui::Separator();
    ImGui::Spacing();

    // Skip rows
    ImGui::Text("Skip first N rows:");
    ImGui::SameLine(200);
    ImGui::SetNextItemWidth(100);
    if (ImGui::InputInt("##skip", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(excluding header)");

    ImGui::Spacing();

    // Limit rows
    static bool limit_enabled = false;
    ImGui::Checkbox("Limit number of rows", &limit_enabled);
    if (limit_enabled) {
        ImGui::SameLine(200);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##maxrows", &max_rows_)) {
            if (max_rows_ < 1) max_rows_ = 1;
            has_changes_ = true;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // WHERE clause filter
    ImGui::Text("Filter rows (SQL WHERE clause):");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##where", where_clause_, sizeof(where_clause_), ImVec2(0, 60))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("Example: age > 18 AND status = 'active'");
}

void DataInputDialog::RenderEncodingTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Character encoding");
    ImGui::Separator();
    ImGui::Spacing();

    const char* encodings[] = {"Auto-detect", "UTF-8", "UTF-16", "ASCII", "ISO-8859-1", "Windows-1252", "UTF-8 with BOM"};
    ImGui::Text("File encoding:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##encoding", &encoding_idx_, encodings, 7)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Auto-detect will scan the first bytes of the file to determine encoding.");
}

void DataInputDialog::RenderMemoryPolicyTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Memory management");
    ImGui::Separator();
    ImGui::Spacing();

    // Memory policy
    int policy = (memory_policy_ == MemoryPolicy::CacheInMemory) ? 0 : 1;
    ImGui::Text("Memory policy:");
    ImGui::Spacing();
    if (ImGui::RadioButton("Cache in memory (fast access)", &policy, 0)) {
        memory_policy_ = MemoryPolicy::CacheInMemory;
        has_changes_ = true;
    }
    ImGui::TextDisabled("   Load entire dataset into RAM for fast random access");
    ImGui::Spacing();
    if (ImGui::RadioButton("Stream from disk (low memory)", &policy, 1)) {
        memory_policy_ = MemoryPolicy::WriteToDisc;
        has_changes_ = true;
    }
    ImGui::TextDisabled("   Process in chunks, suitable for datasets larger than RAM");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Streaming options
    ImGui::Text("Streaming options (when using disk mode):");
    ImGui::Spacing();

    if (ImGui::Checkbox("Auto chunk size", &auto_chunk_size_)) {
        has_changes_ = true;
    }

    if (!auto_chunk_size_) {
        ImGui::Text("Chunk size:");
        ImGui::SameLine(150);
        ImGui::SetNextItemWidth(100);
        if (ImGui::InputInt("##chunk", &chunk_size_kb_)) {
            if (chunk_size_kb_ < 64) chunk_size_kb_ = 64;
            if (chunk_size_kb_ > 65536) chunk_size_kb_ = 65536;
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("KB");
    }

    ImGui::Text("LRU cache chunks:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("##lru", &lru_chunks_)) {
        if (lru_chunks_ < 1) lru_chunks_ = 1;
        if (lru_chunks_ > 64) lru_chunks_ = 64;
        has_changes_ = true;
    }

    if (ImGui::Checkbox("Enable prefetch", &prefetch_enabled_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(preload next chunk while processing current)");

    // RAM estimate
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    UpdateRAMEstimate();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Estimated RAM usage: %.1f MB", estimated_ram_mb_);
}

void DataInputDialog::RenderPreviewTable() {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.11f, 0.13f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    if (ImGui::BeginChild("Preview", ImVec2(0, 250), true)) {
        // Section header
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
        ImGui::Text("[PREVIEW]  Data Preview");
        ImGui::PopStyleColor();

        // Info text
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 350);
        ImGui::TextDisabled("Column types based on first 10000 rows");
        ImGui::Separator();
        ImGui::Spacing();

        if (strlen(file_path_) == 0) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.6f, 1.0f), "  Select a file to see data preview");
        } else if (!preview_loaded_) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.88f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.62f, 0.98f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
            if (ImGui::Button("  Load Preview  ", ImVec2(120, 30))) {
                LoadColumnList();
                preview_loaded_ = true;
            }
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            ImGui::TextDisabled("Click to load data preview from file");
        } else if (preview_data_.empty()) {
            ImGui::TextColored(ImVec4(0.8f, 0.5f, 0.3f, 1.0f), "  No data found in file");
        } else {
            // Render table with scrolling
            if (ImGui::BeginTable("PreviewTable", static_cast<int>(preview_columns_.size()) + 1,
                ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit)) {

                // Header row with RowID and column names
                ImGui::TableSetupColumn("Row ID", ImGuiTableColumnFlags_WidthFixed, 60.0f);
                for (const auto& col : preview_columns_) {
                    ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 100.0f);
                }
                ImGui::TableHeadersRow();

                // Data rows
                int row_idx = 0;
                for (const auto& row : preview_data_) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextDisabled("Row%d", row_idx);

                    int col_idx = 1;
                    for (const auto& cell : row) {
                        if (col_idx <= static_cast<int>(preview_columns_.size())) {
                            ImGui::TableSetColumnIndex(col_idx);
                            ImGui::TextUnformatted(cell.c_str());
                        }
                        col_idx++;
                    }
                    row_idx++;
                    if (row_idx >= 20) break;  // Limit preview rows
                }
                ImGui::EndTable();
            }
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

const char* DataInputDialog::GetFileTypeName() const {
    const char* names[] = {"Auto", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5"};
    return names[detected_type_];
}

// Keep legacy method names for compatibility
void DataInputDialog::RenderSourceTab() { /* Merged into RenderSettingsTab */ }
void DataInputDialog::RenderFormatOptionsTab() { /* Merged into RenderSettingsTab */ }
void DataInputDialog::RenderColumnSelectionTab() { RenderTransformationTab(); }
void DataInputDialog::RenderRowFilterTab() { RenderLimitRowsTab(); }
void DataInputDialog::RenderStreamingTab() { /* Merged into RenderMemoryPolicyTab */ }
void DataInputDialog::RenderPresetTab() { /* Removed - presets not in KNIME style */ }
void DataInputDialog::RenderPreviewSection() { RenderPreviewTable(); }

void DataInputDialog::DetectFileType() {
    std::string path(file_path_);
    if (path.empty()) {
        detected_type_ = 0;  // Auto
        return;
    }

    // Get file size
    try {
        file_size_ = static_cast<int64_t>(std::filesystem::file_size(path));
    } catch (...) {
        file_size_ = 0;
    }

    // Detect by extension
    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    if (ext == "csv") detected_type_ = 1;
    else if (ext == "tsv" || ext == "tab") detected_type_ = 2;
    else if (ext == "json" || ext == "jsonl") detected_type_ = 3;
    else if (ext == "parquet" || ext == "pq") detected_type_ = 4;
    else if (ext == "xlsx" || ext == "xls") detected_type_ = 5;
    else if (ext == "h5" || ext == "hdf5" || ext == "hdf") detected_type_ = 6;
    else detected_type_ = 0;  // Auto

    spdlog::debug("DataInputDialog: Detected type {} for file '{}'", detected_type_, path);
}

void DataInputDialog::LoadColumnList() {
    std::string path(file_path_);
    if (path.empty()) return;

    available_columns_.clear();
    preview_data_.clear();
    preview_columns_.clear();

    // Try to read first few lines for preview
    std::ifstream file(path);
    if (!file.is_open()) {
        spdlog::warn("DataInputDialog: Could not open file for preview: {}", path);
        return;
    }

    // Determine delimiter
    char delimiter = ',';
    if (detected_type_ == 2) delimiter = '\t';  // TSV

    std::string line;
    int line_count = 0;

    while (std::getline(file, line) && line_count < 25) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delimiter)) {
            // Trim whitespace
            size_t start = cell.find_first_not_of(" \t\r\n\"");
            size_t end = cell.find_last_not_of(" \t\r\n\"");
            if (start != std::string::npos && end != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            cells.push_back(cell);
        }

        if (line_count == 0 && has_header_) {
            // First line is header
            preview_columns_ = cells;
            available_columns_ = cells;
            selected_columns_.resize(cells.size(), true);
        } else {
            preview_data_.push_back(cells);

            // If no header, generate column names
            if (!has_header_ && line_count == 0) {
                for (size_t i = 0; i < cells.size(); i++) {
                    preview_columns_.push_back("Column" + std::to_string(i + 1));
                    available_columns_.push_back("Column" + std::to_string(i + 1));
                }
                selected_columns_.resize(cells.size(), true);
            }
        }
        line_count++;
    }

    spdlog::info("DataInputDialog: Loaded {} columns and {} preview rows from '{}'",
                 available_columns_.size(), preview_data_.size(), path);
}

void DataInputDialog::UpdateRAMEstimate() {
    if (file_size_ <= 0) {
        estimated_ram_mb_ = 0;
        return;
    }

    // Rough estimate: file size * 2 for in-memory representation
    double multiplier = (memory_policy_ == MemoryPolicy::CacheInMemory) ? 2.0 : 0.1;
    estimated_ram_mb_ = (file_size_ / (1024.0 * 1024.0)) * multiplier;
}

void DataInputDialog::LoadPreset(const std::string& name) {
    if (name == "Large Files") {
        memory_policy_ = MemoryPolicy::WriteToDisc;
        auto_chunk_size_ = true;
        prefetch_enabled_ = true;
        lru_chunks_ = 8;
    } else if (name == "Low Memory") {
        memory_policy_ = MemoryPolicy::WriteToDisc;
        chunk_size_kb_ = 1024;
        auto_chunk_size_ = false;
        lru_chunks_ = 2;
    } else if (name == "Fast Load") {
        memory_policy_ = MemoryPolicy::CacheInMemory;
        prefetch_enabled_ = true;
    } else {  // Default
        memory_policy_ = MemoryPolicy::CacheInMemory;
        auto_chunk_size_ = true;
        prefetch_enabled_ = false;
        lru_chunks_ = 4;
    }
    has_changes_ = true;
}

void DataInputDialog::SavePreset(const std::string& name) {
    // TODO: Persist to config file
    spdlog::info("DataInputDialog: Saved preset '{}'", name);
}

// ==================== DataOutputDialog ====================

DataOutputDialog::DataOutputDialog(MLNode* node)
    : NodeConfigDialog("Data Output", node)
{
    if (node_) {
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
        }
        if (node_->parameters.count("file_type")) {
            std::string type = node_->parameters["file_type"];
            if (type == "csv") output_type_ = 0;
            else if (type == "tsv") output_type_ = 1;
            else if (type == "json") output_type_ = 2;
            else if (type == "parquet") output_type_ = 3;
            else if (type == "excel") output_type_ = 4;
            else if (type == "hdf5") output_type_ = 5;
        }
    }
}

void DataOutputDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;
    const char* types[] = {"csv", "tsv", "json", "parquet", "excel", "hdf5"};
    node_->parameters["file_type"] = types[output_type_];
    node_->parameters["overwrite"] = overwrite_ ? "true" : "false";
    node_->parameters["include_header"] = include_header_ ? "true" : "false";
    const char* compressions[] = {"none", "gzip", "snappy", "zstd"};
    node_->parameters["compression"] = compressions[compression_];
    node_->parameters["configured"] = "true";

    has_changes_ = false;
    spdlog::info("DataOutputDialog: Applied settings for '{}'", file_path_);
}

void DataOutputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    has_changes_ = false;
}

void DataOutputDialog::RenderContent() {
    // KNIME-style tab bar
    if (ImGui::BeginTabBar("DataOutputTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Settings")) {
            RenderSettingsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Advanced")) {
            RenderAdvancedTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataOutputDialog::RenderSettingsTab() {
    ImGui::Spacing();

    // Output location
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.17f, 1.0f));
    if (ImGui::BeginChild("OutputLocation", ImVec2(0, 80), true)) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Output location");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("File");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
        std::string path_str = file_path_;
        if (ImGui::InputText("##outpath", file_path_, sizeof(file_path_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...")) {
            // TODO: Save file dialog
            has_changes_ = true;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::Spacing();

    // Writer options
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.17f, 1.0f));
    if (ImGui::BeginChild("WriterOptions", ImVec2(0, 150), true)) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Writer options");
        ImGui::Separator();
        ImGui::Spacing();

        // Format
        const char* formats[] = {"CSV", "TSV", "JSON", "Parquet", "Excel (.xlsx)", "HDF5"};
        ImGui::Text("Format:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(150);
        if (ImGui::Combo("##format", &output_type_, formats, 6)) {
            has_changes_ = true;
        }

        // CSV/TSV options
        if (output_type_ <= 1) {
            if (ImGui::Checkbox("Include header row", &include_header_)) {
                has_changes_ = true;
            }
        }

        // Overwrite
        ImGui::Spacing();
        if (ImGui::Checkbox("Overwrite existing file", &overwrite_)) {
            has_changes_ = true;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void DataOutputDialog::RenderAdvancedTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Compression");
    ImGui::Separator();
    ImGui::Spacing();

    const char* compressions[] = {"None", "gzip", "Snappy", "zstd"};
    ImGui::Text("Compression:");
    ImGui::SameLine(120);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##compression", &compression_, compressions, 4)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    if (compression_ > 0) {
        ImGui::TextDisabled("Compression is applied during write. Parquet uses Snappy by default.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Encoding
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Encoding");
    ImGui::Spacing();

    const char* encodings[] = {"UTF-8", "UTF-8 with BOM", "ASCII", "ISO-8859-1"};
    static int out_encoding = 0;
    ImGui::Text("Output encoding:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(150);
    ImGui::Combo("##outencoding", &out_encoding, encodings, 4);
}

} // namespace gui
