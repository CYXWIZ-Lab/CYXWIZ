// Data output, loader, and split node dialog implementations.

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

#include "node_config_dialog.h"
#include "../core/worker_defaults.h"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>

namespace gui {

namespace {

void CopyToBuffer(char* destination,
                  std::size_t destination_size,
                  const std::string& value) {
    if (destination_size == 0) return;
    std::strncpy(destination, value.c_str(), destination_size - 1);
    destination[destination_size - 1] = '\0';
}

std::string LowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

bool ReadBoolParamValue(const std::map<std::string, std::string>& params,
                        const char* key,
                        bool fallback) {
    auto it = params.find(key);
    if (it == params.end()) return fallback;
    return it->second == "true" || it->second == "1" || it->second == "yes";
}

int ReadIntParamValue(const std::map<std::string, std::string>& params,
                      const char* key,
                      int fallback) {
    auto it = params.find(key);
    if (it == params.end() || it->second.empty()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

std::string ReadStringParamValue(
    const std::map<std::string, std::string>& params,
    const char* key,
    const std::string& fallback = "") {
    auto it = params.find(key);
    return it == params.end() ? fallback : it->second;
}

int CompressionIndexFromName(const std::string& value) {
    const std::string normalized = LowerAscii(value);
    if (normalized == "none" || normalized == "uncompressed") return 0;
    if (normalized == "gzip") return 2;
    if (normalized == "zstd") return 3;
    if (normalized == "brotli") return 4;
    return 1;
}

const char* CompressionNameFromIndex(int index) {
    static const char* kCompressions[] = {
        "none", "snappy", "gzip", "zstd", "brotli"
    };
    if (index < 0 || index >= 5) return "snappy";
    return kCompressions[index];
}

int DataFormatIndexFromName(const std::string& value) {
    const std::string normalized = LowerAscii(value);
    if (normalized == "csv") return 1;
    if (normalized == "tsv") return 2;
    if (normalized == "json" || normalized == "jsonl" || normalized == "ndjson") return 3;
    if (normalized == "txt" || normalized == "text") return 4;
    if (normalized == "arff") return 5;
    if (normalized == "npy") return 6;
    if (normalized == "h5" || normalized == "hdf5" || normalized == "hdf") return 7;
    if (normalized == "parquet" || normalized == "pq") return 8;
    if (normalized == "feather" || normalized == "fea") return 9;
    if (normalized == "arrow") return 10;
    if (normalized == "ipc") return 11;
    return 0;
}

const char* DataFormatNameFromIndex(int index) {
    static const char* kFormats[] = {
        "auto", "csv", "tsv", "jsonl", "txt", "arff", "npy", "hdf5", "parquet", "feather", "arrow", "ipc"
    };
    if (index < 0 || index >= 12) return "auto";
    return kFormats[index];
}

const char* DataFormatExtensionFromIndex(int index) {
    static const char* kExtensions[] = {
        ".parquet", ".csv", ".tsv", ".jsonl", ".txt", ".arff", ".npy", ".h5", ".parquet", ".feather", ".arrow", ".ipc"
    };
    if (index < 0 || index >= 12) return ".parquet";
    return kExtensions[index];
}

bool DataFormatExtensionMatchesIndex(const std::filesystem::path& path,
                                     int format_index) {
    if (format_index == 0 || path.empty()) return true;
    std::string extension = LowerAscii(path.extension().string());
    switch (format_index) {
        case 1:
            return extension == ".csv";
        case 2:
            return extension == ".tsv";
        case 3:
            return extension == ".json" || extension == ".jsonl" ||
                   extension == ".ndjson";
        case 4:
            return extension == ".txt" || extension == ".text";
        case 5:
            return extension == ".arff";
        case 6:
            return extension == ".npy";
        case 7:
            return extension == ".h5" || extension == ".hdf5" ||
                   extension == ".hdf";
        case 8:
            return extension == ".parquet" || extension == ".pq";
        case 9:
            return extension == ".feather" || extension == ".fea";
        case 10:
            return extension == ".arrow";
        case 11:
            return extension == ".ipc";
        default:
            return true;
    }
}

std::string DelimiterLabel(char delimiter) {
    switch (delimiter) {
        case ',': return "comma (,)";
        case '\t': return "tab";
        case ';': return "semicolon (;)";
        case '|': return "pipe (|)";
        default: return std::string("'") + delimiter + "'";
    }
}

bool BrowseDataInput(char* destination, std::size_t destination_size) {
#ifdef _WIN32
    OPENFILENAMEA ofn = {};
    char file[512] = {};
    CopyToBuffer(file, sizeof(file), destination);
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter =
        "Supported Data Files\0*.csv;*.tsv;*.json;*.jsonl;*.ndjson;*.txt;*.text;*.arff;*.npy;*.h5;*.hdf5;*.hdf;*.parquet;*.pq;*.feather;*.fea;*.arrow;*.ipc\0"
        "CSV/TSV Files\0*.csv;*.tsv\0"
        "JSON Lines Files\0*.json;*.jsonl;*.ndjson\0"
        "Text Files\0*.txt;*.text\0"
        "ARFF Files\0*.arff\0"
        "NumPy Files\0*.npy\0"
        "HDF5 Files\0*.h5;*.hdf5;*.hdf\0"
        "Parquet Files\0*.parquet;*.pq\0"
        "Arrow IPC/Feather Files\0*.feather;*.fea;*.arrow;*.ipc\0"
        "All Files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof(file);
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn)) {
        CopyToBuffer(destination, destination_size, file);
        return true;
    }
#else
    (void)destination;
    (void)destination_size;
#endif
    return false;
}

bool BrowseDataOutput(char* destination,
                      std::size_t destination_size,
                      int output_format_index) {
#ifdef _WIN32
    OPENFILENAMEA ofn = {};
    char file[512] = {};
    CopyToBuffer(file, sizeof(file), destination);
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter =
        "Supported Data Files\0*.csv;*.tsv;*.json;*.jsonl;*.ndjson;*.txt;*.text;*.arff;*.npy;*.h5;*.hdf5;*.hdf;*.parquet;*.pq;*.feather;*.fea;*.arrow;*.ipc\0"
        "CSV Files\0*.csv\0"
        "TSV Files\0*.tsv\0"
        "JSON Lines Files\0*.json;*.jsonl;*.ndjson\0"
        "Text Files\0*.txt;*.text\0"
        "ARFF Files\0*.arff\0"
        "NumPy Files\0*.npy\0"
        "HDF5 Files\0*.h5;*.hdf5;*.hdf\0"
        "Parquet Files\0*.parquet;*.pq\0"
        "Arrow IPC/Feather Files\0*.feather;*.fea;*.arrow;*.ipc\0"
        "All Files\0*.*\0";
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof(file);
    const char* extension = DataFormatExtensionFromIndex(output_format_index);
    ofn.lpstrDefExt = extension[0] == '.' ? extension + 1 : extension;
    ofn.Flags = OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;
    if (GetSaveFileNameA(&ofn)) {
        CopyToBuffer(destination, destination_size, file);
        return true;
    }
#else
    (void)destination;
    (void)destination_size;
    (void)output_format_index;
#endif
    return false;
}

} // namespace

// ==================== DataOutputDialog ====================

DataOutputDialog::DataOutputDialog(MLNode* node)
    : NodeConfigDialog("Data Output", node)
{
    if (node_) {
        auto file_path_it = node_->parameters.find("file_path");
        if (file_path_it == node_->parameters.end() ||
            file_path_it->second.empty()) {
            file_path_it = node_->parameters.find("path");
        }
        if (file_path_it != node_->parameters.end()) {
            CopyToBuffer(file_path_, sizeof(file_path_), file_path_it->second);
        }

        auto file_type_it = node_->parameters.find("file_type");
        if (file_type_it == node_->parameters.end() ||
            file_type_it->second.empty()) {
            file_type_it = node_->parameters.find("format");
        }
        if (file_type_it != node_->parameters.end()) {
            std::string type = LowerAscii(file_type_it->second);
            if (type == "csv") output_type_ = 0;
            else if (type == "parquet") output_type_ = 1;
        }
    }
}

void DataOutputDialog::Apply() {
    if (!node_) return;

    node_->parameters["file_path"] = file_path_;
    const char* types[] = {"csv", "parquet"};
    node_->parameters["file_type"] = types[output_type_];
    node_->parameters["overwrite"] = overwrite_ ? "true" : "false";
    node_->parameters["include_header"] = include_header_ ? "true" : "false";
    const char* compressions[] = {"none", "gzip", "snappy", "zstd"};
    node_->parameters["compression"] = compressions[compression_];
    node_->parameters["configured"] = "true";

    if (strlen(file_path_) > 0) {
        std::filesystem::path p(file_path_);
        node_->description = "Writing to " + p.filename().string();
    }

    has_changes_ = false;
}

void DataOutputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    has_changes_ = false;
}

void DataOutputDialog::RenderContent() {
    if (ImGui::BeginTabBar("DataOutputTabs")) {
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
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Output Location");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("File:");
    ImGui::SameLine(60);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 80);
    if (ImGui::InputText("##outpath", file_path_, sizeof(file_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse", ImVec2(70, 0))) {
        // TODO: Save file dialog
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Format");
    ImGui::Separator();
    ImGui::Spacing();

    const char* formats[] = {"CSV", "Parquet"};
    ImGui::Text("Format:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##format", &output_type_, formats, 2)) {
        has_changes_ = true;
    }

    if (output_type_ == 0) {
        if (ImGui::Checkbox("Include header", &include_header_)) {
            has_changes_ = true;
        }
    }

    if (ImGui::Checkbox("Overwrite existing", &overwrite_)) {
        has_changes_ = true;
    }
}

void DataOutputDialog::RenderAdvancedTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Compression");
    ImGui::Separator();
    ImGui::Spacing();

    const char* compressions[] = {"None", "gzip", "Snappy", "zstd"};
    ImGui::Text("Compression:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##compression", &compression_, compressions, 4)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Encoding");
    ImGui::Separator();
    ImGui::Spacing();

    const char* encodings[] = {"UTF-8", "UTF-8 with BOM", "ASCII", "ISO-8859-1"};
    static int out_encoding = 0;
    ImGui::Text("Encoding:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("##outencoding", &out_encoding, encodings, 4);
}

// ==================== DataConvertDialog ====================

DataConvertDialog::DataConvertDialog(MLNode* node)
    : NodeConfigDialog("Data Convert", node)
{
    LoadFromNode();
}

void DataConvertDialog::LoadFromNode() {
    if (!node_) return;
    CopyToBuffer(input_path_, sizeof(input_path_),
                 ReadStringParamValue(node_->parameters, "input_path"));
    CopyToBuffer(output_path_, sizeof(output_path_),
                 ReadStringParamValue(node_->parameters, "output_path"));
    input_format_ = DataFormatIndexFromName(
        ReadStringParamValue(node_->parameters, "input_format", "auto"));
    output_format_ = DataFormatIndexFromName(
        ReadStringParamValue(node_->parameters, "output_format", "auto"));
    const std::string delimiter =
        ReadStringParamValue(node_->parameters, "delimiter", "auto");
    auto_detect_delimiter_ = LowerAscii(delimiter) == "auto";
    CopyToBuffer(delimiter_, sizeof(delimiter_),
                 auto_detect_delimiter_ ? "," : delimiter);
    has_header_ = ReadBoolParamValue(node_->parameters, "header", true);
    allow_newlines_in_values_ =
        ReadBoolParamValue(node_->parameters, "allow_newlines_in_values", true);
    skip_rows_ = ReadIntParamValue(node_->parameters, "skip_rows", 0);
    compression_ = CompressionIndexFromName(
        ReadStringParamValue(node_->parameters, "compression", "snappy"));
    row_group_size_ = ReadIntParamValue(
        node_->parameters, "row_group_size", 1048576);
    overwrite_ = ReadBoolParamValue(node_->parameters, "overwrite", false);
    create_parent_dirs_ = ReadBoolParamValue(
        node_->parameters, "create_parent_dirs", true);
    write_manifest_ = ReadBoolParamValue(
        node_->parameters, "write_manifest", true);
    status_message_ = ReadStringParamValue(
        node_->parameters, "status", "Not run");
    status_is_error_ = false;
    log_lines_.clear();
    if (!status_message_.empty() && status_message_ != "Not run") {
        log_lines_.push_back(status_message_);
    }
}

void DataConvertDialog::Apply() {
    if (!node_) return;

    node_->parameters["input_path"] = input_path_;
    node_->parameters["input_format"] = DataFormatNameFromIndex(input_format_);
    node_->parameters["output_path"] = output_path_;
    node_->parameters["output_format"] = DataFormatNameFromIndex(output_format_);
    node_->parameters["delimiter"] =
        auto_detect_delimiter_ ? "auto" : std::string(delimiter_);
    node_->parameters["header"] = has_header_ ? "true" : "false";
    node_->parameters["allow_newlines_in_values"] =
        allow_newlines_in_values_ ? "true" : "false";
    node_->parameters["skip_rows"] = std::to_string(skip_rows_);
    node_->parameters["compression"] = CompressionNameFromIndex(compression_);
    node_->parameters["row_group_size"] = std::to_string(row_group_size_);
    node_->parameters["overwrite"] = overwrite_ ? "true" : "false";
    node_->parameters["create_parent_dirs"] =
        create_parent_dirs_ ? "true" : "false";
    node_->parameters["write_manifest"] = write_manifest_ ? "true" : "false";
    node_->parameters["configured"] =
        (output_path_[0] != '\0') ? "true" : "false";
    node_->parameters["status"] = status_message_;
    if (last_result_.ok) {
        node_->parameters["rows_written"] =
            std::to_string(last_result_.rows_written);
        node_->parameters["converted_output_path"] = last_result_.output_path;
        node_->parameters["parquet_output_path"] = last_result_.output_path;
        node_->parameters["manifest_path"] = last_result_.manifest_path;
        node_->description = "Converted " +
            std::to_string(last_result_.rows_written) + " rows";
    } else if (output_path_[0] != '\0') {
        node_->description = std::filesystem::path(output_path_).filename().string();
    }

    has_changes_ = false;
}

void DataConvertDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    has_changes_ = false;
}

void DataConvertDialog::RenderContent() {
    ImGui::TextWrapped(
        "Convert source data between supported table file formats and write "
        "an optional sidecar manifest.");
    ImGui::Spacing();

    if (!status_message_.empty()) {
        const ImVec4 color = status_is_error_
            ? ImVec4(0.95f, 0.35f, 0.30f, 1.0f)
            : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::TextColored(color, "%s", status_message_.c_str());
        ImGui::Separator();
    }

    if (ImGui::BeginTabBar("DataConvertTabs")) {
        if (ImGui::BeginTabItem("Source")) {
            RenderSourceTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Output")) {
            RenderOutputTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Options")) {
            RenderOptionsTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Preview")) {
            RenderPreviewTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Run")) {
            RenderRunTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Logs")) {
            RenderLogsTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
}

void DataConvertDialog::RenderSourceTab() {
    ImGui::Spacing();
    ImGui::Text("Input data file");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90.0f);
    if (ImGui::InputText("##input_path", input_path_, sizeof(input_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse", ImVec2(80.0f, 0.0f)) &&
        BrowseDataInput(input_path_, sizeof(input_path_))) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    const char* formats[] = {
        "Auto", "CSV", "TSV", "JSONL", "Text", "ARFF", "NumPy", "HDF5", "Parquet", "Feather", "Arrow", "IPC"
    };
    ImGui::Text("Input format");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("##input_format", &input_format_, formats, 12)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Auto detects from extension.");
}

void DataConvertDialog::RenderOptionsTab() {
    ImGui::Spacing();
    ImGui::Text("Delimited input parsing");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Auto-detect delimiter", &auto_detect_delimiter_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Checks comma, tab, semicolon, and pipe.");

    ImGui::Text("Delimiter");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(90.0f);
    ImGui::BeginDisabled(auto_detect_delimiter_);
    if (ImGui::InputText("##delimiter", delimiter_, sizeof(delimiter_))) {
        has_changes_ = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled(auto_detect_delimiter_
        ? "Detected when preview or conversion runs."
        : "Use comma for CSV, tab for TSV.");

    if (ImGui::Checkbox("First row contains headers", &has_header_)) {
        has_changes_ = true;
    }
    if (ImGui::Checkbox("Allow quoted multiline values",
                        &allow_newlines_in_values_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Needed for text columns that contain line breaks.");

    ImGui::Text("Skip rows");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputInt("##skip_rows", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Text("Parquet writer");
    ImGui::Separator();
    ImGui::Spacing();

    const char* compressions[] = {
        "None", "Snappy", "Gzip", "Zstd", "Brotli"
    };
    ImGui::Text("Compression");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(140.0f);
    if (ImGui::Combo("##compression", &compression_, compressions, 5)) {
        has_changes_ = true;
    }

    ImGui::Text("Row group size");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(140.0f);
    if (ImGui::InputInt("##row_group_size", &row_group_size_)) {
        if (row_group_size_ < 1) row_group_size_ = 1;
        has_changes_ = true;
    }
}

void DataConvertDialog::RenderOutputTab() {
    ImGui::Spacing();
    ImGui::Text("Output data file");
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90.0f);
    if (ImGui::InputText("##output_path", output_path_, sizeof(output_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Save As", ImVec2(80.0f, 0.0f)) &&
        BrowseDataOutput(output_path_, sizeof(output_path_), output_format_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    const char* formats[] = {
        "Auto", "CSV", "TSV", "JSONL", "Text", "ARFF", "NumPy", "HDF5", "Parquet", "Feather", "Arrow", "IPC"
    };
    ImGui::Text("Output format");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("##output_format", &output_format_, formats, 12)) {
        if (output_format_ != 0 && output_path_[0] != '\0') {
            std::filesystem::path out(output_path_);
            out.replace_extension(DataFormatExtensionFromIndex(output_format_));
            CopyToBuffer(output_path_, sizeof(output_path_), out.string());
        }
        has_changes_ = true;
    }

    if (ImGui::Button("Use input name + selected extension")) {
        std::filesystem::path in(input_path_);
        if (!in.empty()) {
            in.replace_extension(DataFormatExtensionFromIndex(output_format_));
            CopyToBuffer(output_path_, sizeof(output_path_), in.string());
            has_changes_ = true;
        }
    }

    if (output_format_ != 0 && output_path_[0] != '\0' &&
        !DataFormatExtensionMatchesIndex(std::filesystem::path(output_path_),
                                         output_format_)) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.20f, 1.0f),
                           "Output extension does not match the selected format. Expected %s.",
                           DataFormatExtensionFromIndex(output_format_));
    }

    ImGui::Spacing();
    if (ImGui::Checkbox("Overwrite existing file", &overwrite_)) {
        has_changes_ = true;
    }
    if (ImGui::Checkbox("Create parent folders", &create_parent_dirs_)) {
        has_changes_ = true;
    }
    if (ImGui::Checkbox("Write conversion manifest", &write_manifest_)) {
        has_changes_ = true;
    }
}

void DataConvertDialog::RenderPreviewTab() {
    ImGui::Spacing();
    if (ImGui::Button("Preview schema")) {
        PreviewInput();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Reads the input and Arrow-inferred types.");

    ImGui::Spacing();
    if (!preview_.ok) {
        if (!preview_.error.empty()) {
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.30f, 1.0f),
                               "%s", preview_.error.c_str());
        }
        return;
    }

    ImGui::Text("Rows: %lld   Columns: %lld",
                static_cast<long long>(preview_.rows),
                static_cast<long long>(preview_.columns));
    if (ImGui::BeginTable("DataConvertSchema", 3,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("Column");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Nullable");
        ImGui::TableHeadersRow();
        for (const auto& column : preview_.schema) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(column.name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(column.type.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(column.nullable ? "yes" : "no");
        }
        ImGui::EndTable();
    }

    if (!preview_.sample_rows.empty()) {
        ImGui::Spacing();
        ImGui::Text("Sample rows");
        if (ImGui::BeginTable("DataConvertSampleRows",
                              static_cast<int>(preview_.schema.size()),
                              ImGuiTableFlags_Borders |
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_ScrollX)) {
            for (const auto& column : preview_.schema) {
                ImGui::TableSetupColumn(column.name.c_str());
            }
            ImGui::TableHeadersRow();
            for (const auto& row : preview_.sample_rows) {
                ImGui::TableNextRow();
                for (int column = 0;
                     column < static_cast<int>(preview_.schema.size());
                     ++column) {
                    ImGui::TableSetColumnIndex(column);
                    const char* value =
                        column < static_cast<int>(row.size())
                            ? row[static_cast<size_t>(column)].c_str()
                            : "";
                    ImGui::TextUnformatted(value);
                }
            }
            ImGui::EndTable();
        }
    }
}

void DataConvertDialog::RenderRunTab() {
    ImGui::Spacing();
    ImGui::TextWrapped(
        "Run conversion when the source and output settings are correct. "
        "Downstream nodes can consume the generated output directly.");
    ImGui::Spacing();

    if (ImGui::Button("Run conversion", ImVec2(160.0f, 0.0f))) {
        RunConversion();
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply settings")) {
        Apply();
    }

    ImGui::Spacing();
    if (last_result_.ok) {
        const std::string output_path = last_result_.output_path.empty()
            ? std::string(output_path_)
            : last_result_.output_path;
        ImGui::TextWrapped("Output: %s", output_path.c_str());
        ImGui::Text("Rows written: %lld",
                    static_cast<long long>(last_result_.rows_written));
        ImGui::Text("Columns: %lld",
                    static_cast<long long>(last_result_.columns));
        ImGui::Text("Bytes written: %lld",
                    static_cast<long long>(last_result_.bytes_written));
        if (!last_result_.manifest_path.empty()) {
            ImGui::TextWrapped("Sidecar manifest: %s",
                               last_result_.manifest_path.c_str());
        }
    }
}

void DataConvertDialog::RenderLogsTab() {
    ImGui::Spacing();
    if (ImGui::Button("Clear logs")) {
        log_lines_.clear();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Shows messages from this dialog session.");

    ImGui::Spacing();
    if (ImGui::BeginChild("DataConvertLogs",
                          ImVec2(0.0f, 0.0f),
                          true,
                          ImGuiWindowFlags_HorizontalScrollbar)) {
        if (log_lines_.empty()) {
            ImGui::TextDisabled("No conversion messages yet.");
        } else {
            for (const auto& line : log_lines_) {
                ImGui::TextWrapped("%s", line.c_str());
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                ImGui::SetScrollHereY(1.0f);
            }
        }
    }
    ImGui::EndChild();
}

cyxwiz::DataConvertOptions DataConvertDialog::BuildOptions() const {
    cyxwiz::DataConvertOptions options;
    options.input_path = input_path_;
    options.output_path = output_path_;
    options.input_format = DataFormatNameFromIndex(input_format_);
    options.output_format = DataFormatNameFromIndex(output_format_);
    options.delimiter = delimiter_[0] == '\0' ? ',' : delimiter_[0];
    options.auto_detect_delimiter = auto_detect_delimiter_;
    options.has_header = has_header_;
    options.allow_newlines_in_values = allow_newlines_in_values_;
    options.skip_rows = skip_rows_;
    options.parquet_compression = CompressionNameFromIndex(compression_);
    options.row_group_size = row_group_size_;
    options.overwrite = overwrite_;
    options.create_parent_dirs = create_parent_dirs_;
    options.write_manifest = write_manifest_;
    return options;
}

void DataConvertDialog::PreviewInput() {
    preview_ = cyxwiz::DataConvertService::Preview(BuildOptions());
    if (preview_.ok) {
        std::ostringstream msg;
        msg << "Preview loaded: " << preview_.rows << " rows, "
            << preview_.columns << " columns.";
        if (auto_detect_delimiter_) {
            msg << " Detected delimiter: "
                << DelimiterLabel(preview_.detected_delimiter) << ".";
        }
        SetStatus(msg.str(), false);
    } else {
        SetStatus(preview_.error, true);
    }
}

void DataConvertDialog::RunConversion() {
    last_result_ = cyxwiz::DataConvertService::Convert(BuildOptions());
    if (last_result_.ok) {
        std::ostringstream msg;
        if (last_result_.skipped_fresh_output) {
            msg << "Conversion skipped: existing output is fresh at "
                << last_result_.output_path << ".";
        } else {
            msg << "Conversion complete: " << last_result_.rows_written
                << " rows written to " << last_result_.output_path << ".";
        }
        if (auto_detect_delimiter_) {
            msg << " Detected delimiter: "
                << DelimiterLabel(last_result_.detected_delimiter) << ".";
        }
        SetStatus(msg.str(), false);
        Apply();
    } else {
        SetStatus(last_result_.error, true);
    }
}

void DataConvertDialog::SetStatus(std::string message, bool is_error) {
    status_message_ = std::move(message);
    status_is_error_ = is_error;
    AddLogLine((is_error ? "Error: " : "Info: ") + status_message_);
    has_changes_ = true;
}

void DataConvertDialog::AddLogLine(const std::string& message) {
    if (message.empty()) return;
    log_lines_.push_back(message);
    constexpr size_t kMaxLogLines = 200;
    if (log_lines_.size() > kMaxLogLines) {
        log_lines_.erase(log_lines_.begin(),
                         log_lines_.begin() +
                             static_cast<std::ptrdiff_t>(log_lines_.size() -
                                                         kMaxLogLines));
    }
}

// ==================== DataLoaderDialog ====================

namespace {
void ReadIntParam(const std::map<std::string, std::string>& params,
                  const char* key, int& out) {
    auto it = params.find(key);
    if (it != params.end() && !it->second.empty()) {
        try { out = std::stoi(it->second); } catch (...) {}
    }
}

void ReadFloatParam(const std::map<std::string, std::string>& params,
                    const char* key, float& out) {
    auto it = params.find(key);
    if (it != params.end() && !it->second.empty()) {
        try { out = std::stof(it->second); } catch (...) {}
    }
}

void ReadBoolParam(const std::map<std::string, std::string>& params,
                   const char* key, bool& out) {
    auto it = params.find(key);
    if (it != params.end()) out = (it->second == "true");
}

int BalanceModeIndexFromName(const std::string& value) {
    std::string normalized = LowerAscii(value);
    if (normalized == "weighted sampler") normalized = "weighted_sampler";
    if (normalized == "oversample") return 1;
    if (normalized == "undersample") return 2;
    if (normalized == "weighted_sampler") return 3;
    return 0;
}

const char* BalanceModeName(int index) {
    switch (index) {
        case 1: return "oversample";
        case 2: return "undersample";
        case 3: return "weighted_sampler";
        default: return "none";
    }
}

bool IsPositiveIntegerText(const char* text) {
    if (!text || text[0] == '\0') return false;
    for (const char* c = text; *c != '\0'; ++c) {
        if (!std::isdigit(static_cast<unsigned char>(*c))) return false;
    }
    return true;
}
}

DataLoaderDialog::DataLoaderDialog(MLNode* node)
    : NodeConfigDialog("Data Loader", node)
{
    if (node_) {
        ReadIntParam(node_->parameters, "epochs", epochs_);
        ReadIntParam(node_->parameters, "batch_size", batch_size_);
        ReadBoolParam(node_->parameters, "shuffle", shuffle_);
        ReadBoolParam(node_->parameters, "drop_last", drop_last_);
        num_workers_ = cyxwiz::GetDefaultNumWorkers();
        ReadIntParam(node_->parameters, "num_workers", num_workers_);
        ReadIntParam(node_->parameters, "prefetch_factor", prefetch_factor_);
        if (prefetch_factor_ < 0) prefetch_factor_ = 0;
        ReadIntParam(node_->parameters, "log_interval", log_interval_);
        if (log_interval_ < 0) log_interval_ = 0;
        ReadIntParam(node_->parameters, "validation_freq", validation_freq_);
        if (validation_freq_ < 1) validation_freq_ = 1;
        ReadIntParam(node_->parameters, "seed", seed_);
        if (seed_ < 0) seed_ = 0;
        ReadIntParam(node_->parameters, "grad_accum_steps", grad_accum_steps_);
        if (grad_accum_steps_ < 1) grad_accum_steps_ = 1;
        ReadBoolParam(node_->parameters, "balance_classes", balance_classes_);
        balance_mode_index_ = BalanceModeIndexFromName(
            ReadStringParamValue(node_->parameters, "balance_mode", "none"));
        if (balance_classes_ && balance_mode_index_ == 0) {
            balance_mode_index_ = 1;
        }
        CopyToBuffer(balance_target_, sizeof(balance_target_),
                     ReadStringParamValue(node_->parameters, "balance_target", "max"));
        if (balance_target_[0] == '\0') {
            CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
        }
        ReadIntParam(node_->parameters, "balance_seed", balance_seed_);
        if (balance_seed_ < 0) balance_seed_ = 0;
        ReadBoolParam(node_->parameters, "pin_memory", pin_memory_requested_);
        ReadBoolParam(node_->parameters, "save_best_checkpoint", save_best_checkpoint_);
        ReadIntParam(node_->parameters, "early_stopping_patience", early_stopping_patience_);
        CopyToBuffer(checkpoint_dir_, sizeof(checkpoint_dir_),
                     ReadStringParamValue(node_->parameters, "checkpoint_dir"));
    }
}

void DataLoaderDialog::Apply() {
    if (!node_) return;
    node_->parameters["epochs"] = std::to_string(epochs_);
    node_->parameters["batch_size"] = std::to_string(batch_size_);
    node_->parameters["shuffle"] = shuffle_ ? "true" : "false";
    node_->parameters["drop_last"] = drop_last_ ? "true" : "false";
    node_->parameters["num_workers"] = std::to_string(num_workers_);
    node_->parameters["prefetch_factor"] = std::to_string(prefetch_factor_);
    node_->parameters["log_interval"] = std::to_string(log_interval_);
    node_->parameters["validation_freq"] = std::to_string(validation_freq_);
    node_->parameters["seed"] = std::to_string(seed_);
    node_->parameters["grad_accum_steps"] = std::to_string(grad_accum_steps_);
    node_->parameters["balance_classes"] = balance_classes_ ? "true" : "false";
    node_->parameters["balance_mode"] = BalanceModeName(
        balance_classes_ ? balance_mode_index_ : 0);
    node_->parameters["balance_target"] = balance_target_;
    node_->parameters["balance_seed"] = std::to_string(balance_seed_);
    node_->parameters["save_best_checkpoint"] = save_best_checkpoint_ ? "true" : "false";
    node_->parameters["early_stopping_patience"] = std::to_string(early_stopping_patience_);
    node_->parameters["checkpoint_dir"] = checkpoint_dir_;
    node_->description = "epochs=" + std::to_string(epochs_) +
                          ", batch=" + std::to_string(batch_size_) +
                          (shuffle_ ? ", shuffled" : ", ordered") +
                          (balance_classes_
                               ? ", balanced=" + std::string(BalanceModeName(balance_mode_index_))
                               : "");
    has_changes_ = false;
}

void DataLoaderDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    // Re-initialize local UI state from the restored params so sliders match.
    epochs_ = 10;
    batch_size_ = 32;
    shuffle_ = true;
    drop_last_ = false;
    num_workers_ = cyxwiz::GetDefaultNumWorkers();
    prefetch_factor_ = 2;
    log_interval_ = 10;
    validation_freq_ = 1;
    seed_ = 42;
    grad_accum_steps_ = 1;
    balance_classes_ = false;
    balance_mode_index_ = 0;
    CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
    balance_seed_ = 42;
    pin_memory_requested_ = false;
    save_best_checkpoint_ = true;
    early_stopping_patience_ = 5;
    checkpoint_dir_[0] = '\0';
    ReadIntParam(original_params_, "epochs", epochs_);
    ReadIntParam(original_params_, "batch_size", batch_size_);
    ReadBoolParam(original_params_, "shuffle", shuffle_);
    ReadBoolParam(original_params_, "drop_last", drop_last_);
    ReadIntParam(original_params_, "num_workers", num_workers_);
    ReadIntParam(original_params_, "prefetch_factor", prefetch_factor_);
    if (prefetch_factor_ < 0) prefetch_factor_ = 0;
    ReadIntParam(original_params_, "log_interval", log_interval_);
    if (log_interval_ < 0) log_interval_ = 0;
    ReadIntParam(original_params_, "validation_freq", validation_freq_);
    if (validation_freq_ < 1) validation_freq_ = 1;
    ReadIntParam(original_params_, "seed", seed_);
    if (seed_ < 0) seed_ = 0;
    ReadIntParam(original_params_, "grad_accum_steps", grad_accum_steps_);
    if (grad_accum_steps_ < 1) grad_accum_steps_ = 1;
    ReadBoolParam(original_params_, "balance_classes", balance_classes_);
    balance_mode_index_ = BalanceModeIndexFromName(
        ReadStringParamValue(original_params_, "balance_mode", "none"));
    if (balance_classes_ && balance_mode_index_ == 0) {
        balance_mode_index_ = 1;
    }
    CopyToBuffer(balance_target_, sizeof(balance_target_),
                 ReadStringParamValue(original_params_, "balance_target", "max"));
    if (balance_target_[0] == '\0') {
        CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
    }
    ReadIntParam(original_params_, "balance_seed", balance_seed_);
    if (balance_seed_ < 0) balance_seed_ = 0;
    ReadBoolParam(original_params_, "pin_memory", pin_memory_requested_);
    ReadBoolParam(original_params_, "save_best_checkpoint", save_best_checkpoint_);
    ReadIntParam(original_params_, "early_stopping_patience", early_stopping_patience_);
    CopyToBuffer(checkpoint_dir_, sizeof(checkpoint_dir_),
                 ReadStringParamValue(original_params_, "checkpoint_dir"));
    has_changes_ = false;
}

void DataLoaderDialog::RenderContent() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Training Loop");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Epochs:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##epochs", &epochs_)) {
        if (epochs_ < 1) epochs_ = 1;
        if (epochs_ > 100000) epochs_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(full passes over training split)");

    ImGui::Spacing();
    ImGui::TextColored(accent, "Batching");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Batch size:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##batch_size", &batch_size_)) {
        if (batch_size_ < 1) batch_size_ = 1;
        if (batch_size_ > 100000) batch_size_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(samples per gradient step)");

    ImGui::Spacing();
    if (ImGui::Checkbox("Shuffle each epoch", &shuffle_)) has_changes_ = true;
    ImGui::TextDisabled("  Reshuffles training samples at the start of every epoch.");

    ImGui::Spacing();
    if (ImGui::Checkbox("Drop last incomplete batch", &drop_last_)) has_changes_ = true;
    ImGui::TextDisabled("  Discard the final batch if it has fewer than batch_size samples.");

    ImGui::Spacing();
    ImGui::Text("Grad accum:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##grad_accum_steps", &grad_accum_steps_)) {
        if (grad_accum_steps_ < 1) grad_accum_steps_ = 1;
        if (grad_accum_steps_ > 100000) grad_accum_steps_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(optimizer step every N batches)");

    ImGui::Spacing();
    ImGui::Text("Validate every:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##validation_freq", &validation_freq_)) {
        if (validation_freq_ < 1) validation_freq_ = 1;
        if (validation_freq_ > 100000) validation_freq_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(epochs; final epoch always validates)");

    ImGui::Spacing();
    ImGui::Text("Seed:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##dataloader_seed", &seed_)) {
        if (seed_ < 0) seed_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(split/shuffle order)");

    ImGui::Spacing();
    ImGui::TextColored(accent, "Class Imbalance");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Balance training classes", &balance_classes_)) {
        if (balance_classes_ && balance_mode_index_ == 0) {
            balance_mode_index_ = 1;
        }
        if (!balance_classes_) {
            balance_mode_index_ = 0;
        }
        has_changes_ = true;
    }
    ImGui::TextDisabled("  Applies only to training batches. Validation and test distributions stay unchanged.");

    const char* balance_modes[] = {
        "None", "Oversample minority", "Undersample majority", "Weighted sampler"
    };
    ImGui::BeginDisabled(!balance_classes_);
    ImGui::Text("Mode:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(190);
    if (ImGui::Combo("##balance_mode", &balance_mode_index_,
                     balance_modes, IM_ARRAYSIZE(balance_modes))) {
        if (balance_mode_index_ == 0) {
            balance_classes_ = false;
        }
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(train split only)");

    const char* balance_targets[] = {"max", "median", "min", "custom number"};
    int target_index = 0;
    if (std::strcmp(balance_target_, "median") == 0) {
        target_index = 1;
    } else if (std::strcmp(balance_target_, "min") == 0) {
        target_index = 2;
    } else if (IsPositiveIntegerText(balance_target_)) {
        target_index = 3;
    }
    ImGui::Text("Target:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(190);
    if (ImGui::Combo("##balance_target_mode", &target_index,
                     balance_targets, IM_ARRAYSIZE(balance_targets))) {
        if (target_index == 0) {
            CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
        } else if (target_index == 1) {
            CopyToBuffer(balance_target_, sizeof(balance_target_), "median");
        } else if (target_index == 2) {
            CopyToBuffer(balance_target_, sizeof(balance_target_), "min");
        } else if (!IsPositiveIntegerText(balance_target_)) {
            CopyToBuffer(balance_target_, sizeof(balance_target_), "1");
        }
        has_changes_ = true;
    }
    if (target_index == 3) {
        int target_count = 1;
        try {
            target_count = std::max(1, std::stoi(balance_target_));
        } catch (...) {
            target_count = 1;
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(110);
        if (ImGui::InputInt("##balance_target_count", &target_count)) {
            if (target_count < 1) target_count = 1;
            CopyToBuffer(balance_target_, sizeof(balance_target_),
                         std::to_string(target_count));
            has_changes_ = true;
        }
    }

    ImGui::Text("Balance seed:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##balance_seed", &balance_seed_)) {
        if (balance_seed_ < 0) balance_seed_ = 0;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(deterministic sampler)");
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::TextColored(accent, "Checkpointing");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Save best validation checkpoint", &save_best_checkpoint_)) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("  Runtime-supported: saves the best validation epoch when validation loss improves.");

    ImGui::Text("Early stop patience:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##early_stopping_patience", &early_stopping_patience_)) {
        if (early_stopping_patience_ < 0) early_stopping_patience_ = 0;
        if (early_stopping_patience_ > 100000) early_stopping_patience_ = 100000;
        has_changes_ = true;
    }
    ImGui::TextDisabled("  Runtime-supported: 0 disables early stopping.");

    ImGui::Text("Checkpoint dir:");
    ImGui::SameLine(150);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    if (ImGui::InputText("##checkpoint_dir", checkpoint_dir_, sizeof(checkpoint_dir_))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("  Empty uses the default run-local .cyxwiz/checkpoints path.");

    ImGui::Spacing();
    ImGui::TextColored(accent, "Performance");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Worker threads:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##num_workers", &num_workers_)) {
        if (num_workers_ < 0) num_workers_ = 0;
        if (num_workers_ > 64) num_workers_ = 64;
        has_changes_ = true;
    }
    if (num_workers_ > 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                           "  Used as synchronous per-batch workers where supported; Prefetch controls async queuing.");
    } else {
        ImGui::TextDisabled("  0 = load batches on the training thread (current behavior).");
    }

    ImGui::Spacing();
    ImGui::Text("Prefetch:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##prefetch_factor", &prefetch_factor_)) {
        if (prefetch_factor_ < 0) prefetch_factor_ = 0;
        if (prefetch_factor_ > 64) prefetch_factor_ = 64;
        has_changes_ = true;
    }
    if (prefetch_factor_ > 0) {
        ImGui::TextDisabled("  Bounded async queue depth for Arrow/Parquet batchers.");
    } else {
        ImGui::TextDisabled("  0 = disable async prefetch.");
    }

    ImGui::Spacing();
    ImGui::Text("Pin memory:");
    ImGui::SameLine(130);
    ImGui::TextDisabled(pin_memory_requested_ ? "unsupported (saved true)" : "unsupported");
    if (pin_memory_requested_) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f),
                           "  This loaded graph requests pin_memory=true, but current batchers ignore it.");
    } else {
        ImGui::TextDisabled("  Serialized for compatibility only; no pinned host-memory transfer backend exists yet.");
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Logging");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Log every:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##log_interval", &log_interval_)) {
        if (log_interval_ < 0) log_interval_ = 0;
        if (log_interval_ > 100000) log_interval_ = 100000;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(batches; 0 logs first batch only)");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(
        "This node is the single source of truth for batching. If batch_size is\n"
        "also set on the optimizer node, the DataLoader value wins and a warning\n"
        "is logged at training start.");
}

// ==================== DataSplitDialog ====================

DataSplitDialog::DataSplitDialog(MLNode* node)
    : NodeConfigDialog("Data Split", node)
{
    if (node_) {
        ReadFloatParam(node_->parameters, "train_ratio", train_ratio_);
        ReadFloatParam(node_->parameters, "val_ratio", val_ratio_);
        ReadFloatParam(node_->parameters, "test_ratio", test_ratio_);
        ReadIntParam(node_->parameters, "seed", seed_);
        ReadBoolParam(node_->parameters, "stratified", stratified_);
    }
}

void DataSplitDialog::Apply() {
    if (!node_) return;
    // Guard against all-zero sliders, then normalize to 1.0 if the user drifted.
    float sum = train_ratio_ + val_ratio_ + test_ratio_;
    if (sum <= 0.001f) {
        train_ratio_ = 0.8f;
        val_ratio_ = 0.1f;
        test_ratio_ = 0.1f;
    } else if (std::abs(sum - 1.0f) > 0.01f) {
        train_ratio_ /= sum;
        val_ratio_ /= sum;
        test_ratio_ /= sum;
    }
    node_->parameters["train_ratio"] = std::to_string(train_ratio_);
    node_->parameters["val_ratio"] = std::to_string(val_ratio_);
    node_->parameters["test_ratio"] = std::to_string(test_ratio_);
    node_->parameters["seed"] = std::to_string(seed_);
    node_->parameters["stratified"] = stratified_ ? "true" : "false";
    char buf[128];
    snprintf(buf, sizeof(buf), "%.0f/%.0f/%.0f",
             train_ratio_ * 100.0f, val_ratio_ * 100.0f, test_ratio_ * 100.0f);
    node_->description = buf;
    has_changes_ = false;
}

void DataSplitDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    // Re-initialize local UI state from the restored params so sliders match.
    train_ratio_ = 0.8f;
    val_ratio_ = 0.1f;
    test_ratio_ = 0.1f;
    seed_ = 42;
    stratified_ = true;
    ReadFloatParam(original_params_, "train_ratio", train_ratio_);
    ReadFloatParam(original_params_, "val_ratio", val_ratio_);
    ReadFloatParam(original_params_, "test_ratio", test_ratio_);
    ReadIntParam(original_params_, "seed", seed_);
    ReadBoolParam(original_params_, "stratified", stratified_);
    has_changes_ = false;
}

void DataSplitDialog::RenderContent() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Split Ratios");
    ImGui::Separator();
    ImGui::Spacing();

    bool changed = false;
    ImGui::Text("Train:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##train", &train_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    ImGui::Text("Validation:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##val", &val_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    ImGui::Text("Test:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::SliderFloat("##test", &test_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;

    if (changed) has_changes_ = true;

    float sum = train_ratio_ + val_ratio_ + test_ratio_;
    ImGui::Spacing();
    if (std::abs(sum - 1.0f) > 0.01f) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                           "Sum = %.2f (will be normalized to 1.00 on Apply)", sum);
    } else {
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1.0f), "Sum = %.2f", sum);
    }

    ImGui::Spacing();
    ImGui::TextColored(accent, "Options");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Seed:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(180);
    if (ImGui::InputInt("##seed", &seed_)) has_changes_ = true;
    ImGui::TextDisabled("  Controls the shuffled order used when partitioning.");

    ImGui::Spacing();
    if (ImGui::Checkbox("Stratified split", &stratified_)) has_changes_ = true;
    ImGui::TextDisabled("  Preserve class distribution across splits (classification only).");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled(
        "If no DataSplit node is in the graph, training uses defaults (80/10/10).\n"
        "This node resolves partition policy for training. Older saved graphs\n"
        "may show Train/Val/Test tensor output pins as legacy compatibility\n"
        "pins; runtime validation/test paths are created from resolved dataset\n"
        "partitions, not from separate canvas branches.");
}

} // namespace gui
