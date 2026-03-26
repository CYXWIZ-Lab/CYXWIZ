// Data Input/Output Dialog Implementations
// Part of CyxWiz Studio smart data loading system

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
    : NodeConfigDialog("Data Input Configuration", node)
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
    // Left side: Collapsible sections
    ImGui::BeginChild("Settings", ImVec2(ImGui::GetContentRegionAvail().x * 0.6f, 0), true);

    if (ImGui::CollapsingHeader("Source", ImGuiTreeNodeFlags_DefaultOpen)) RenderSourceTab();
    if (ImGui::CollapsingHeader("Format Options", ImGuiTreeNodeFlags_DefaultOpen)) RenderFormatOptionsTab();
    if (ImGui::CollapsingHeader("Column Selection")) RenderColumnSelectionTab();
    if (ImGui::CollapsingHeader("Row Filter")) RenderRowFilterTab();
    if (ImGui::CollapsingHeader("Streaming", ImGuiTreeNodeFlags_DefaultOpen)) RenderStreamingTab();
    if (ImGui::CollapsingHeader("Memory Policy", ImGuiTreeNodeFlags_DefaultOpen)) RenderMemoryPolicyTab();
    if (ImGui::CollapsingHeader("Presets")) RenderPresetTab();

    ImGui::EndChild();
    ImGui::SameLine();

    // Right side: Preview
    ImGui::BeginChild("Preview", ImVec2(0, 0), true);
    RenderPreviewSection();
    ImGui::EndChild();
}

void DataInputDialog::RenderSourceTab() {
    ImGui::Spacing();

    std::string path_str = file_path_;
    if (FileSelector("Location:", path_str, "All Files\0*.*\0CSV\0*.csv\0JSON\0*.json\0")) {
        strncpy(file_path_, path_str.c_str(), sizeof(file_path_) - 1);
        DetectFileType();
        preview_loaded_ = false;
        has_changes_ = true;
    }

    ImGui::Spacing();

    const char* types[] = {"Auto-Detect", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5"};
    ImGui::Text("Type:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::Combo("##type", &detected_type_, types, 7)) {
        preview_loaded_ = false;
        has_changes_ = true;
    }

    const char* encodings[] = {"Auto", "UTF-8", "ASCII", "ISO-8859-1", "Windows-1252"};
    ImGui::Text("Encoding:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::Combo("##encoding", &encoding_idx_, encodings, 5)) {
        has_changes_ = true;
    }

    if (file_size_ > 0) {
        ImGui::Spacing();
        if (file_size_ < 1024 * 1024)
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Size: %.1f KB", file_size_ / 1024.0f);
        else
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "Size: %.1f MB", file_size_ / (1024.0f * 1024.0f));
    }
}

void DataInputDialog::RenderFormatOptionsTab() {
    ImGui::Spacing();

    if (detected_type_ <= 2) {  // CSV/TSV
        const char* delimiters[] = {"Comma (,)", "Semicolon (;)", "Tab", "Custom"};
        ImGui::Text("Delimiter:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::Combo("##delimiter", &delimiter_idx_, delimiters, 4)) {
            preview_loaded_ = false;
            has_changes_ = true;
        }

        if (delimiter_idx_ == 3) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(40.0f);
            if (ImGui::InputText("##custom", custom_delimiter_, sizeof(custom_delimiter_))) {
                preview_loaded_ = false;
                has_changes_ = true;
            }
        }

        if (ImGui::Checkbox("Has header row", &has_header_)) {
            preview_loaded_ = false;
            has_changes_ = true;
        }

        ImGui::Text("Quote char:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(40.0f);
        ImGui::InputText("##quote", quote_char_, sizeof(quote_char_));

    } else if (detected_type_ == 3) {  // JSON
        if (ImGui::Checkbox("JSON Lines format", &json_lines_)) {
            has_changes_ = true;
        }
        ImGui::Text("JSON Path:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        ImGui::InputText("##jsonpath", json_path_, sizeof(json_path_));
        ImGui::SameLine();
        ImGui::TextDisabled("(e.g., $.data.records)");

    } else if (detected_type_ == 5) {  // Excel
        ImGui::Text("Sheet:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputInt("##sheet", &sheet_idx_);
        ImGui::Text("Range:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        ImGui::InputText("##range", sheet_range_, sizeof(sheet_range_));
        ImGui::SameLine();
        ImGui::TextDisabled("(e.g., A1:D100)");

    } else if (detected_type_ == 6) {  // HDF5
        ImGui::Text("Dataset:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200.0f);
        ImGui::InputText("##hdf5ds", hdf5_dataset_, sizeof(hdf5_dataset_));
    }
}

void DataInputDialog::RenderColumnSelectionTab() {
    ImGui::Spacing();

    if (ImGui::Checkbox("Select all columns", &select_all_columns_)) {
        has_changes_ = true;
        for (size_t i = 0; i < selected_columns_.size(); i++) {
            selected_columns_[i] = select_all_columns_;
        }
    }

    if (!select_all_columns_ && !available_columns_.empty()) {
        ImGui::Spacing();
        ImGui::BeginChild("ColumnList", ImVec2(0, 150), true);
        for (size_t i = 0; i < available_columns_.size() && i < selected_columns_.size(); i++) {
            bool selected = selected_columns_[i];
            if (ImGui::Checkbox(available_columns_[i].c_str(), &selected)) {
                selected_columns_[i] = selected;
                has_changes_ = true;
            }
        }
        ImGui::EndChild();
    }

    if (available_columns_.empty()) {
        ImGui::TextDisabled("Load preview to see available columns");
    }
}

void DataInputDialog::RenderRowFilterTab() {
    ImGui::Spacing();

    ImGui::Text("Skip first:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::InputInt("##skip", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("rows");

    ImGui::Text("Max rows:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::InputInt("##max", &max_rows_)) {
        if (max_rows_ < 0) max_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(0 = unlimited)");

    ImGui::Spacing();
    ImGui::Text("WHERE:");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##where", where_clause_, sizeof(where_clause_))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("SQL-like: column > 10 AND status = 'active'");
}

void DataInputDialog::RenderStreamingTab() {
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Always applied for large files");
    ImGui::Spacing();

    if (ImGui::Checkbox("Auto chunk size", &auto_chunk_size_)) {
        has_changes_ = true;
    }

    if (!auto_chunk_size_) {
        ImGui::Text("Chunk size:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.0f);
        if (ImGui::InputInt("##chunk", &chunk_size_kb_)) {
            if (chunk_size_kb_ < 64) chunk_size_kb_ = 64;
            if (chunk_size_kb_ > 65536) chunk_size_kb_ = 65536;
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("KB");
    }

    ImGui::Text("LRU chunks:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    if (ImGui::InputInt("##lru", &lru_chunks_)) {
        if (lru_chunks_ < 2) lru_chunks_ = 2;
        if (lru_chunks_ > 100) lru_chunks_ = 100;
        has_changes_ = true;
    }

    if (ImGui::Checkbox("Enable prefetch", &prefetch_enabled_)) {
        has_changes_ = true;
    }
}

void DataInputDialog::RenderMemoryPolicyTab() {
    ImGui::Spacing();

    bool cache_mem = (memory_policy_ == MemoryPolicy::CacheInMemory);
    if (ImGui::RadioButton("Cache in Memory", cache_mem)) {
        memory_policy_ = MemoryPolicy::CacheInMemory;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(Fast access, uses RAM)");

    bool write_disc = (memory_policy_ == MemoryPolicy::WriteToDisc);
    if (ImGui::RadioButton("Write to Disc", write_disc)) {
        memory_policy_ = MemoryPolicy::WriteToDisc;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Slow - Zero RAM");

    if (memory_policy_ == MemoryPolicy::WriteToDisc) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f),
            "Warning: Write to Disc mode is slower but uses minimal RAM.\n"
            "Recommended for files > 2GB or low-memory systems.");
    }
}

void DataInputDialog::RenderPresetTab() {
    ImGui::Spacing();

    ImGui::Text("Load preset:");
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::BeginCombo("##preset", preset_name_)) {
        for (const auto& p : available_presets_) {
            if (ImGui::Selectable(p.c_str(), p == preset_name_)) {
                LoadPreset(p);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Save as:");
    ImGui::SetNextItemWidth(120.0f);
    static char new_preset[64] = {};
    ImGui::InputText("##newpreset", new_preset, sizeof(new_preset));
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        if (new_preset[0] != '\0') {
            SavePreset(new_preset);
        }
    }
}

void DataInputDialog::RenderPreviewSection() {
    ImGui::Text("Preview");
    ImGui::Separator();

    if (ImGui::Button("Refresh Preview")) {
        preview_loaded_ = false;
        LoadColumnList();
    }

    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Est. RAM: %.1f MB", estimated_ram_mb_);

    ImGui::Spacing();

    if (!preview_error_.empty()) {
        ValidationMessage(preview_error_);
    } else if (preview_loaded_ && !preview_headers_.empty()) {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;

        int cols = std::min(static_cast<int>(preview_headers_.size()), 10);
        if (ImGui::BeginTable("##preview", cols, flags, ImVec2(0, 200))) {
            for (int i = 0; i < cols; i++) {
                ImGui::TableSetupColumn(preview_headers_[i].c_str());
            }
            ImGui::TableHeadersRow();

            for (const auto& row : preview_rows_) {
                ImGui::TableNextRow();
                for (int i = 0; i < cols && i < static_cast<int>(row.size()); i++) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(row[i].c_str());
                }
            }
            ImGui::EndTable();
        }
    } else {
        ImGui::TextDisabled("Click 'Refresh Preview' to load data");
    }
}

void DataInputDialog::DetectFileType() {
    if (file_path_[0] == '\0') return;

    std::string ext = std::filesystem::path(file_path_).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".csv") detected_type_ = 1;
    else if (ext == ".tsv") detected_type_ = 2;
    else if (ext == ".json") detected_type_ = 3;
    else if (ext == ".parquet") detected_type_ = 4;
    else if (ext == ".xlsx" || ext == ".xls") detected_type_ = 5;
    else if (ext == ".h5" || ext == ".hdf5") detected_type_ = 6;
    else detected_type_ = 0;

    if (std::filesystem::exists(file_path_)) {
        file_size_ = std::filesystem::file_size(file_path_);
        UpdateRAMEstimate();
    }
}

void DataInputDialog::LoadColumnList() {
    preview_headers_.clear();
    preview_rows_.clear();
    available_columns_.clear();
    selected_columns_.clear();
    preview_error_.clear();
    preview_loaded_ = false;

    if (file_path_[0] == '\0') {
        preview_error_ = "No file selected";
        return;
    }

    std::ifstream file(file_path_);
    if (!file.good()) {
        preview_error_ = "Cannot open file";
        return;
    }

    char delim = ',';
    if (delimiter_idx_ == 1) delim = ';';
    else if (delimiter_idx_ == 2) delim = '\t';
    else if (delimiter_idx_ == 3 && custom_delimiter_[0]) delim = custom_delimiter_[0];

    std::string line;
    int row_count = 0;
    while (std::getline(file, line) && row_count < 20) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, delim)) {
            cells.push_back(cell);
        }

        if (has_header_ && row_count == 0) {
            preview_headers_ = cells;
            available_columns_ = cells;
            selected_columns_.resize(cells.size(), true);
        } else {
            preview_rows_.push_back(cells);
        }
        row_count++;
    }

    if (!has_header_ && !preview_rows_.empty()) {
        for (size_t i = 0; i < preview_rows_[0].size(); i++) {
            preview_headers_.push_back("Column " + std::to_string(i + 1));
            available_columns_.push_back("Column " + std::to_string(i + 1));
        }
        selected_columns_.resize(preview_headers_.size(), true);
    }

    preview_loaded_ = true;
}

void DataInputDialog::UpdateRAMEstimate() {
    float ratio = 1.0f;
    if (!select_all_columns_ && !selected_columns_.empty()) {
        int selected = 0;
        for (bool s : selected_columns_) if (s) selected++;
        ratio = static_cast<float>(selected) / selected_columns_.size();
    }

    if (memory_policy_ == MemoryPolicy::WriteToDisc) {
        estimated_ram_mb_ = (chunk_size_kb_ * lru_chunks_) / 1024.0f;
    } else {
        estimated_ram_mb_ = (file_size_ / (1024.0f * 1024.0f)) * ratio * 1.5f;
    }
}

void DataInputDialog::LoadPreset(const std::string& name) {
    strncpy(preset_name_, name.c_str(), sizeof(preset_name_) - 1);

    if (name == "Large Files") {
        auto_chunk_size_ = false;
        chunk_size_kb_ = 4096;
        lru_chunks_ = 5;
        memory_policy_ = MemoryPolicy::WriteToDisc;
    } else if (name == "Low Memory") {
        auto_chunk_size_ = false;
        chunk_size_kb_ = 512;
        lru_chunks_ = 3;
        memory_policy_ = MemoryPolicy::WriteToDisc;
    } else if (name == "Fast Load") {
        auto_chunk_size_ = true;
        lru_chunks_ = 20;
        prefetch_enabled_ = true;
        memory_policy_ = MemoryPolicy::CacheInMemory;
    } else {  // Default
        auto_chunk_size_ = true;
        chunk_size_kb_ = 1024;
        lru_chunks_ = 10;
        prefetch_enabled_ = true;
        memory_policy_ = MemoryPolicy::CacheInMemory;
    }
    has_changes_ = true;
}

void DataInputDialog::SavePreset(const std::string& name) {
    available_presets_.push_back(name);
    strncpy(preset_name_, name.c_str(), sizeof(preset_name_) - 1);
    spdlog::info("DataInputDialog: Saved preset '{}'", name);
}

// ==================== DataOutputDialog ====================

DataOutputDialog::DataOutputDialog(MLNode* node)
    : NodeConfigDialog("Data Output Configuration", node)
{
    if (node_) {
        if (node_->parameters.count("output_path")) {
            strncpy(output_path_, node_->parameters["output_path"].c_str(), sizeof(output_path_) - 1);
        }
    }
}

void DataOutputDialog::Apply() {
    if (!node_) return;
    node_->parameters["output_path"] = output_path_;
    const char* formats[] = {"csv", "tsv", "json", "parquet", "excel", "hdf5"};
    node_->parameters["format"] = formats[output_format_];
    node_->parameters["overwrite"] = overwrite_existing_ ? "true" : "false";
    node_->parameters["append"] = append_mode_ ? "true" : "false";
    node_->parameters["include_header"] = include_header_ ? "true" : "false";
    has_changes_ = false;
}

void DataOutputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    has_changes_ = false;
}

void DataOutputDialog::RenderContent() {
    if (ImGui::BeginTabBar("OutputTabs")) {
        if (ImGui::BeginTabItem("Output")) {
            RenderOutputTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Format Options")) {
            RenderFormatOptionsTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataOutputDialog::RenderOutputTab() {
    ImGui::Spacing();

    std::string path_str = output_path_;
    if (FileSelector("Output Path:", path_str, "All Files\0*.*\0")) {
        strncpy(output_path_, path_str.c_str(), sizeof(output_path_) - 1);
        has_changes_ = true;
    }

    ImGui::Spacing();

    const char* formats[] = {"CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5"};
    ImGui::Text("Format:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::Combo("##format", &output_format_, formats, 6)) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    if (ImGui::Checkbox("Overwrite existing file", &overwrite_existing_)) {
        has_changes_ = true;
    }

    if (ImGui::Checkbox("Append mode", &append_mode_)) {
        has_changes_ = true;
    }
}

void DataOutputDialog::RenderFormatOptionsTab() {
    ImGui::Spacing();

    if (ImGui::Checkbox("Include header row", &include_header_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    const char* compressions[] = {"None", "gzip", "snappy", "zstd"};
    ImGui::Text("Compression:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100.0f);
    if (ImGui::Combo("##compression", &compression_, compressions, 4)) {
        has_changes_ = true;
    }
}

} // namespace gui
