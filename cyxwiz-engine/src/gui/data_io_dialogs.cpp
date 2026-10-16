// Data output, loader, and split node dialog implementations.

#include "node_config_dialog.h"
#include "data_convert_format_controls.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/async_task_manager.h"
#include "../core/data_convert_formats.h"
#include "../core/file_dialogs.h"
#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/graph_compiler.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/training_parameter_contract.h"
#include "../core/worker_defaults.h"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <spdlog/spdlog.h>
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

namespace dc = cyxwiz::data_convert;

bool IsDelimitedDataInput(const std::string& format, const char* path) {
    const auto effective = dc::Resolve(format, std::filesystem::path(path).extension().string());
    return effective == dc::Format::Csv || effective == dc::Format::Tsv;
}
const char* DataFormatExtension(const std::string& name) {
    const auto* info = dc::Find(name);
    return info ? info->default_extension : ".parquet";
}

bool DataFormatExtensionMatches(const std::filesystem::path& path, const std::string& name) {
    if (dc::Normalize(name) == "auto" || name.empty() || path.empty()) return true;
    return dc::ExtensionMatches(path.extension().string(), dc::FromName(name));
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
    const auto filters = cyxwiz::FileDialogs::FilterList{
        {"Supported Data Files", dc::ExtensionFilter(dc::Direction::Input, dc::kBuildFeatures)},
        {"All Files", "*"}};
    if (auto selected = cyxwiz::FileDialogs::OpenFile(
            "Select Data File", filters, destination[0] == '\0' ? nullptr : destination)) {
        CopyToBuffer(destination, destination_size, *selected);
        return true;
    }
    return false;
}

bool BrowseDataOutput(char* destination,
                      std::size_t destination_size,
                      const std::string& output_format) {
    const auto filters = cyxwiz::FileDialogs::FilterList{
        {"Supported Data Files", dc::ExtensionFilter(dc::Direction::Output, dc::kBuildFeatures)},
        {"All Files", "*"}};
    const std::filesystem::path existing_path(destination);
    const std::string default_path = existing_path.has_parent_path()
        ? existing_path.parent_path().string()
        : std::string{};
    const std::string default_name = existing_path.filename().empty()
        ? "output" + std::string(DataFormatExtension(output_format))
        : existing_path.filename().string();
    if (auto selected = cyxwiz::FileDialogs::SaveFile(
            "Save Data File", filters, default_path.empty() ? nullptr : default_path.c_str(), default_name.c_str())) {
        CopyToBuffer(destination, destination_size, *selected);
        return true;
    }
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
        const std::string dialog_format = output_type_ == 1 ? "parquet" : "csv";
        if (BrowseDataOutput(file_path_, sizeof(file_path_), dialog_format)) {
            has_changes_ = true;
        }
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

}

// ==================== DataConvertDialog ====================

DataConvertDialog::DataConvertDialog(MLNode* node)
    : NodeConfigDialog("Data Convert", node)
{
    LoadFromNode();
}

DataConvertDialog::~DataConvertDialog() {
    CancelPreview();
}

void DataConvertDialog::LoadFromNode() {
    if (!node_) return;
    CopyToBuffer(input_path_, sizeof(input_path_),
                 ReadStringParamValue(node_->parameters, "input_path"));
    CopyToBuffer(output_path_, sizeof(output_path_),
                 ReadStringParamValue(node_->parameters, "output_path"));
    input_format_ = ReadStringParamValue(node_->parameters, "input_format", "auto");
    output_format_ = ReadStringParamValue(node_->parameters, "output_format", "auto");
    CopyToBuffer(excel_sheet_, sizeof(excel_sheet_),
                 ReadStringParamValue(node_->parameters, "excel_sheet"));
    CopyToBuffer(excel_start_column_, sizeof(excel_start_column_),
                 ReadStringParamValue(node_->parameters, "excel_start_column", "A"));
    const std::string delimiter =
        ReadStringParamValue(node_->parameters, "delimiter", "auto");
    auto_detect_delimiter_ = LowerAscii(delimiter) == "auto";
    CopyToBuffer(delimiter_, sizeof(delimiter_),
                 auto_detect_delimiter_ ? "," : delimiter);
    const std::string decimal_point =
        ReadStringParamValue(node_->parameters, "decimal_point", ".");
    decimal_point_ = !decimal_point.empty() && decimal_point.front() == ','
        ? ',' : '.';
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
    if (!node_ || IsBusy()) return;

    node_->parameters["input_path"] = input_path_;
    node_->parameters["input_format"] = input_format_;
    node_->parameters["excel_sheet"] = excel_sheet_;
    node_->parameters["excel_start_column"] = excel_start_column_;
    node_->parameters["output_path"] = output_path_;
    node_->parameters["output_format"] = output_format_;
    node_->parameters["delimiter"] =
        auto_detect_delimiter_ ? "auto" : std::string(delimiter_);
    node_->parameters["decimal_point"] = std::string(1, decimal_point_);
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
    CancelPreview();
    // The task manager owns any active file write. Closing/resetting only
    // detaches its UI result; it must not falsely report that output was cancelled.
    conversion_state_.reset();
    last_result_ = {};
    node_->parameters = original_params_;
    LoadFromNode();
    has_changes_ = false;
}

void DataConvertDialog::RenderContent() {
    PollPreviewResult();
    PollConversionResult();

    ImGui::TextWrapped(
        "Convert source data between supported table file formats and write "
        "an optional sidecar manifest.");
    ImGui::Spacing();

    if (conversion_state_) {
        ImGui::TextWrapped("Conversion is running in the background. Closing or cancelling "
                           "this dialog does not cancel the file write. See Tasks for completion.");
    }

    if (!status_message_.empty()) {
        const ImVec4 color = status_is_error_
            ? cyxwiz::ui::CurrentTokens().error
            : ImGui::GetStyleColorVec4(ImGuiCol_Text);
        ImGui::TextColored(color, "%s", status_message_.c_str());
        ImGui::Separator();
    }

    if (ImGui::BeginTabBar("DataConvertTabs")) {
        if (ImGui::BeginTabItem("Source")) {
            ImGui::BeginDisabled(IsBusy());
            RenderSourceTab();
            ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Output")) {
            ImGui::BeginDisabled(IsBusy());
            RenderOutputTab();
            ImGui::EndDisabled();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Options")) {
            ImGui::BeginDisabled(IsBusy());
            RenderOptionsTab();
            ImGui::EndDisabled();
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
    ImGui::Text("Input format");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(160.0f);
    if (RenderDataConvertFormatCombo("##input_format", input_format_, dc::Direction::Input, dc::kBuildFeatures)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Auto detects from extension.");
    const auto effective_format = dc::Resolve(input_format_,
        std::filesystem::path(input_path_).extension().string());
    if (effective_format == dc::Format::Excel) {
        ImGui::Spacing();
        if (!dc::Available(effective_format, dc::Direction::Input, dc::kBuildFeatures))
            ImGui::TextWrapped("XLSX is unavailable in this build (OpenXLSX is required).");
        ImGui::TextWrapped("%s", dc::kXlsxRestrictions);
        ImGui::BeginDisabled(!dc::kBuildFeatures.xlsx);
        if (ImGui::InputText("Worksheet", excel_sheet_, sizeof(excel_sheet_))) has_changes_ = true;
        ImGui::TextDisabled("Blank selects the first worksheet.");
        if (ImGui::InputText("Start column", excel_start_column_, sizeof(excel_start_column_))) has_changes_ = true;
        ImGui::TextWrapped("Excel column label A to XFD (for example B or AA). Read from this column to the worksheet's last column. Set Skip rows under Options; the first remaining row is the header when enabled.");
        ImGui::EndDisabled();
    }


    if (effective_format == dc::Format::Text) {
        ImGui::Spacing();
        ImGui::TextColored(
            cyxwiz::ui::CurrentTokens().warning,
            "Plain Text produces one column named 'text'.");
        ImGui::TextWrapped(
            "For a semicolon-delimited .txt table, select CSV here and set the delimiter to semicolon. The file extension does not need to change.");
    }
}

void DataConvertDialog::RenderOptionsTab() {
    const auto input = dc::Resolve(input_format_, std::filesystem::path(input_path_).extension().string());
    const bool delimited_input = input == dc::Format::Csv || input == dc::Format::Tsv;
    const bool excel_input = input == dc::Format::Excel;
    ImGui::Spacing();
    ImGui::Text("Input parsing");
    ImGui::Separator();
    ImGui::Spacing();

    if (!delimited_input) {
        ImGui::TextColored(
            cyxwiz::ui::CurrentTokens().warning,
            "Delimiter settings apply only to CSV/TSV. Headers also apply to XLSX.");
        ImGui::TextWrapped(
            "Select CSV for comma-, semicolon-, or pipe-delimited data, including delimited files whose extension is .txt.");
        ImGui::Spacing();
    }

    ImGui::BeginDisabled(!delimited_input);
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

    ImGui::Text("Decimal separator");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(120.0f);
    const char* decimal_separators[] = {"Dot (.)", "Comma (,)"};
    int decimal_index = decimal_point_ == ',' ? 1 : 0;
    if (ImGui::Combo("##data_convert_decimal", &decimal_index,
                     decimal_separators, 2)) {
        decimal_point_ = decimal_index == 1 ? ',' : '.';
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Used while parsing numeric input values.");

    ImGui::EndDisabled();
    ImGui::BeginDisabled(!delimited_input && !excel_input);
    if (ImGui::Checkbox("First selected row contains headers", &has_header_)) {
        has_changes_ = true;
    }
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!delimited_input);
    if (ImGui::Checkbox("Allow quoted multiline values",
                        &allow_newlines_in_values_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Needed for text columns that contain line breaks.");
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!delimited_input && !excel_input && input != dc::Format::Text);
    ImGui::Text("Skip rows");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::InputInt("##skip_rows", &skip_rows_)) {
        if (skip_rows_ < 0) skip_rows_ = 0;
        has_changes_ = true;
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Text("Parquet writer");
    ImGui::Separator();
    ImGui::Spacing();

    const auto output = dc::Resolve(output_format_, std::filesystem::path(output_path_).extension().string());
    ImGui::BeginDisabled(output != dc::Format::Parquet);
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
    ImGui::EndDisabled();
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
    ImGui::Text("Output format");
    ImGui::SameLine(130.0f);
    ImGui::SetNextItemWidth(160.0f);
    if (RenderDataConvertFormatCombo("##output_format", output_format_, dc::Direction::Output, dc::kBuildFeatures)) {
        if (!output_format_.empty() && output_format_ != "auto" && output_path_[0] != '\0') {
            std::filesystem::path out(output_path_);
            out.replace_extension(DataFormatExtension(output_format_));
            CopyToBuffer(output_path_, sizeof(output_path_), out.string());
        }
        has_changes_ = true;
    }

    if (ImGui::Button("Use input name + selected extension")) {
        std::filesystem::path in(input_path_);
        if (!in.empty()) {
            in.replace_extension(DataFormatExtension(output_format_));
            CopyToBuffer(output_path_, sizeof(output_path_), in.string());
            has_changes_ = true;
        }
    }

    if (!output_format_.empty() && output_format_ != "auto" && output_path_[0] != '\0' &&
        !DataFormatExtensionMatches(std::filesystem::path(output_path_),
                                         output_format_)) {
        ImGui::Spacing();
        ImGui::TextColored(cyxwiz::ui::CurrentTokens().warning,
                           "Output extension does not match the selected format. Expected %s.",
                           DataFormatExtension(output_format_));
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
    ImGui::BeginDisabled(IsBusy());
    if (ImGui::Button("Preview schema")) {
        PreviewInput();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (preview_loading_) {
        ImGui::TextDisabled("Preview is running in the background (Task %llu)...",
                            static_cast<unsigned long long>(preview_task_id_));
    } else {
        ImGui::TextDisabled("Reads the input and Arrow-inferred types.");
    }

    ImGui::Spacing();
    if (preview_loading_) {
        float progress = 0.0f;
        if (const auto task =
                cyxwiz::AsyncTaskManager::Instance().GetTask(preview_task_id_)) {
            progress = task->GetProgress();
        }
        ImGui::ProgressBar(progress,
                           ImVec2(-1.0f, 0.0f),
                           "Reading source without blocking the engine");
    }
    if (!preview_.ok) {
        if (!preview_.error.empty()) {
            ImGui::TextColored(cyxwiz::ui::CurrentTokens().error,
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
        "Downstream nodes can consume the generated output directly. "
        "Once queued, the file write runs to completion even if this dialog closes.");
    ImGui::Spacing();

    ImGui::BeginDisabled(IsBusy());
    if (ImGui::Button("Run conversion", ImVec2(160.0f, 0.0f))) {
        RunConversion();
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply settings")) {
        Apply();
    }
    ImGui::EndDisabled();

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
    options.input_format = input_format_;
    options.excel_sheet = excel_sheet_;
    options.excel_start_column = excel_start_column_;
    options.output_format = output_format_;
    options.delimiter = delimiter_[0] == '\0' ? ',' : delimiter_[0];
    options.decimal_point = decimal_point_;
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
    if (IsBusy()) return;

    auto state = std::make_shared<PreviewLoadState>();
    const auto options = BuildOptions();
    preview_load_state_ = state;
    preview_loading_ = true;
    preview_ = {};
    SetStatus("Preview queued in the background.", false);

    try {
        preview_task_id_ = cyxwiz::AsyncTaskManager::Instance().Submit(
            cyxwiz::MakeDataConvertPreviewTask(options, state));
    } catch (const std::exception& error) {
        preview_load_state_.reset();
        preview_loading_ = false;
        preview_task_id_ = 0;
        SetStatus(std::string("Could not queue preview: ") + error.what(), true);
        return;
    }
    spdlog::info("DataConvertDialog: queued async preview task {} for '{}'",
                 preview_task_id_, options.input_path);
}

void DataConvertDialog::PollPreviewResult() {
    auto state = preview_load_state_;
    if (!state || !state->done.load()) return;

    preview_ = std::move(state->result);
    preview_load_state_.reset();
    preview_task_id_ = 0;
    preview_loading_ = false;

    if (preview_.ok) {
        std::ostringstream msg;
        msg << "Preview loaded: " << preview_.rows << " rows, "
            << preview_.columns << " columns.";
        if (auto_detect_delimiter_ && IsDelimitedDataInput(input_format_, input_path_)) {
            msg << " Detected delimiter: "
                << DelimiterLabel(preview_.detected_delimiter) << ".";
        }
        SetStatus(msg.str(), false);
    } else {
        SetStatus(preview_.error.empty()
                      ? "Data Convert preview failed."
                      : preview_.error,
                  true);
    }
}

void DataConvertDialog::CancelPreview() {
    if (preview_task_id_ != 0) {
        cyxwiz::AsyncTaskManager::Instance().Cancel(preview_task_id_);
    }
    preview_load_state_.reset();
    preview_task_id_ = 0;
    preview_loading_ = false;
}

void DataConvertDialog::RunConversion() {
    if (IsBusy()) return;
    const auto options = BuildOptions();
    auto state = std::make_shared<cyxwiz::DataConvertTaskResult>();
    auto task = cyxwiz::MakeDataConvertTask(options, state);
    try {
        cyxwiz::AsyncTaskManager::Instance().Submit(task);
        last_result_ = {};
        conversion_state_ = state;
        SetStatus("Conversion queued in the background; file writes cannot be cancelled.", false);
    } catch (const std::exception& error) {
        SetStatus(std::string("Could not queue conversion: ") + error.what(), true);
    }
}

void DataConvertDialog::PollConversionResult() {
    auto state = conversion_state_;
    if (!state || !state->done.load()) return;
    last_result_ = std::move(state->result);
    conversion_state_.reset();
    if (last_result_.ok) {
        std::ostringstream msg;
        if (last_result_.skipped_fresh_output) {
            msg << "Conversion skipped: existing output is fresh at "
                << last_result_.output_path << ".";
        } else {
            msg << "Conversion complete: " << last_result_.rows_written
                << " rows written to " << last_result_.output_path << ".";
        }
        if (auto_detect_delimiter_ && IsDelimitedDataInput(input_format_, input_path_)) {
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

namespace {

constexpr float kFieldLabel = 170.0f;

void Field(const char* label) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(cyxwiz::ui::CurrentTokens().text_dim, "%s", label);
    ImGui::SameLine(kFieldLabel);
}

std::string GroupDigits(int64_t value) {
    std::string digits = std::to_string(value < 0 ? -value : value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return value < 0 ? "-" + digits : digits;
}

// Upstream of `node_id`: the loaded row count of the Data Input and the
// Data Split's train ratio (each -1 when not found).
struct UpstreamData {
    int64_t rows = -1;
    float train_ratio = -1.0f;
    std::string input_name;
};

UpstreamData FindUpstreamData(NodeEditor* editor, int node_id) {
    UpstreamData found;
    if (!editor) return found;
    const auto& nodes = editor->GetNodes();
    const auto& links = editor->GetLinks();
    std::set<int> seen{node_id};
    std::vector<int> frontier{node_id};
    for (size_t i = 0; i < frontier.size(); ++i) {
        for (const auto& link : links) {
            if (link.to_node != frontier[i] || !seen.insert(link.from_node).second) continue;
            const auto it = std::find_if(nodes.begin(), nodes.end(),
                                         [&](const MLNode& n) { return n.id == link.from_node; });
            if (it == nodes.end()) continue;
            const auto param = [&](const char* key) {
                const auto p = it->parameters.find(key);
                return p == it->parameters.end() ? std::string() : p->second;
            };
            if (it->type == NodeType::DataSplit && found.train_ratio < 0.0f) {
                try { found.train_ratio = std::stof(param("train_ratio")); } catch (...) {}
            }
            if (it->type == NodeType::DataInput && found.rows < 0) {
                // The rows of the dataset it loaded, from the registry.
                auto& registry = cyxwiz::DataRegistry::Instance();
                const std::string name = param("dataset_name");
                if (const auto* text = registry.GetTextDatasetEntry(name))
                    found.rows = static_cast<int64_t>(text->num_samples);
                else if (auto arrow = registry.GetArrowDataset(name)) found.rows = arrow->GetNumRows();
                else if (auto parquet = registry.GetParquetBackedDataset(name)) found.rows = parquet->GetNumRows();
                else if (const auto* image = registry.GetImageDatasetEntry(name))
                    found.rows = static_cast<int64_t>(image->num_images);
                else if (const auto* audio = registry.GetAudioDatasetEntry(name))
                    found.rows = static_cast<int64_t>(audio->num_samples);
                found.input_name = it->name;
            }
            frontier.push_back(it->id);
        }
    }
    return found;
}

}  // namespace

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
        ReadIntParam(node_->parameters, "model_seed", model_seed_);
        ReadIntParam(node_->parameters,
                     cyxwiz::training_contract::kGradientAccumulationStepsKey,
                     grad_accum_steps_);
        grad_accum_steps_ =
            cyxwiz::training_contract::ClampGradientAccumulationSteps(
                grad_accum_steps_);
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
        CopyToBuffer(initialize_from_checkpoint_, sizeof(initialize_from_checkpoint_),
                     ReadStringParamValue(node_->parameters, "initialize_from_checkpoint"));
    }
    if (node_) {
    generation_preview_enabled_ = false;
    generation_preview_every_epochs_ = 20;
    generation_preview_max_new_tokens_ = 32;
    ReadBoolParam(node_->parameters, "generation_preview_enabled", generation_preview_enabled_);
    ReadIntParam(node_->parameters, "generation_preview_every_epochs", generation_preview_every_epochs_);
    ReadIntParam(node_->parameters, "generation_preview_max_new_tokens", generation_preview_max_new_tokens_);
    CopyToBuffer(generation_preview_prompts_, sizeof(generation_preview_prompts_),
                 ReadStringParamValue(node_->parameters, "generation_preview_prompts"));
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
    node_->parameters["model_seed"] = std::to_string(model_seed_);
    node_->parameters[cyxwiz::training_contract::kGradientAccumulationStepsKey] =
        std::to_string(
            cyxwiz::training_contract::ClampGradientAccumulationSteps(
                grad_accum_steps_));
    node_->parameters["balance_classes"] = balance_classes_ ? "true" : "false";
    node_->parameters["balance_mode"] = BalanceModeName(
        balance_classes_ ? balance_mode_index_ : 0);
    node_->parameters["balance_target"] = balance_target_;
    node_->parameters["balance_seed"] = std::to_string(balance_seed_);
    node_->parameters["save_best_checkpoint"] = save_best_checkpoint_ ? "true" : "false";
    node_->parameters["early_stopping_patience"] = std::to_string(early_stopping_patience_);
    node_->parameters["checkpoint_dir"] = checkpoint_dir_;
    if (initialize_from_checkpoint_[0] != '\0')
        node_->parameters["initialize_from_checkpoint"] = initialize_from_checkpoint_;
    else
        node_->parameters.erase("initialize_from_checkpoint");
    node_->parameters["generation_preview_enabled"] = generation_preview_enabled_ ? "true" : "false";
    node_->parameters["generation_preview_every_epochs"] = std::to_string(generation_preview_every_epochs_);
    node_->parameters["generation_preview_max_new_tokens"] = std::to_string(generation_preview_max_new_tokens_);
    node_->parameters["generation_preview_prompts"] = generation_preview_prompts_;
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
    model_seed_ = -1;
    grad_accum_steps_ =
        cyxwiz::training_contract::kGradientAccumulationStepsDefault;
    balance_classes_ = false;
    balance_mode_index_ = 0;
    CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
    balance_seed_ = 42;
    pin_memory_requested_ = false;
    save_best_checkpoint_ = true;
    early_stopping_patience_ = 5;
    checkpoint_dir_[0] = '\0';
    initialize_from_checkpoint_[0] = '\0';
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
    ReadIntParam(original_params_, "model_seed", model_seed_);
    ReadIntParam(original_params_,
                 cyxwiz::training_contract::kGradientAccumulationStepsKey,
                 grad_accum_steps_);
    grad_accum_steps_ =
        cyxwiz::training_contract::ClampGradientAccumulationSteps(
            grad_accum_steps_);
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
    CopyToBuffer(initialize_from_checkpoint_, sizeof(initialize_from_checkpoint_),
                 ReadStringParamValue(original_params_, "initialize_from_checkpoint"));
    has_changes_ = false;
    generation_preview_enabled_ = false;
    generation_preview_every_epochs_ = 20;
    generation_preview_max_new_tokens_ = 32;
    ReadBoolParam(original_params_, "generation_preview_enabled", generation_preview_enabled_);
    ReadIntParam(original_params_, "generation_preview_every_epochs", generation_preview_every_epochs_);
    ReadIntParam(original_params_, "generation_preview_max_new_tokens", generation_preview_max_new_tokens_);
    CopyToBuffer(generation_preview_prompts_, sizeof(generation_preview_prompts_),
                 ReadStringParamValue(original_params_, "generation_preview_prompts"));

}

void DataLoaderDialog::RenderContent() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    // Six headed sections in one scroll became five tabs (board 5); every
    // field stays.
    if (!ImGui::BeginTabBar("##data_loader_tabs")) return;
    if (ImGui::BeginTabItem("Training")) {
        ImGui::Spacing();
        Field("Epochs");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##epochs", &epochs_)) {
            epochs_ = std::clamp(epochs_, 1, 100000);
            has_changes_ = true;
        }
        Field("Validate every");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##validation_freq", &validation_freq_)) {
            validation_freq_ = std::clamp(validation_freq_, 1, 100000);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("epoch (the last one always validates)");
        Field("Batch size");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##batch_size", &batch_size_)) {
            batch_size_ = std::clamp(batch_size_, 1, 100000);
            has_changes_ = true;
        }
        // Batches per epoch from the graph's data: the Data Input's rows times
        // the Data Split's train share.
        const auto upstream = node_ ? FindUpstreamData(node_editor_, node_->id) : UpstreamData{};
        if (upstream.rows > 0) {
            const double train = static_cast<double>(upstream.rows) *
                                 (upstream.train_ratio > 0.0f ? upstream.train_ratio : 0.8);
            const double batches = drop_last_ ? std::floor(train / batch_size_) : std::ceil(train / batch_size_);
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "about %s batches per epoch",
                               GroupDigits(static_cast<int64_t>(batches)).c_str());
        }
        Field("Seed");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##dataloader_seed", &seed_)) {
            if (seed_ < 0) seed_ = 0;
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("split and shuffle order");
        Field("Model seed");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##model_rng_seed", &model_seed_)) has_changes_ = true;
        ImGui::SameLine();
        ImGui::TextDisabled("weights and dropout of a fresh run; -1 unset");
        if (model_seed_ < -1) ImGui::TextColored(t.error, "The model seed must be -1 or nonnegative.");
        if (ImGui::Checkbox("Shuffle every epoch", &shuffle_)) has_changes_ = true;
        if (ImGui::Checkbox("Drop the last short batch", &drop_last_)) has_changes_ = true;
        ImGui::Spacing();
        ImGui::TextColored(t.text_faint, "Generation previews, logging, workers and pin memory are under Advanced.");
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Batches")) {
        ImGui::Spacing();
        Field("Accumulate gradients");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##grad_accum_steps", &grad_accum_steps_)) {
            grad_accum_steps_ = cyxwiz::training_contract::ClampGradientAccumulationSteps(grad_accum_steps_);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("batches per optimizer step");
        Field("Prefetch");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##prefetch_factor", &prefetch_factor_)) {
            prefetch_factor_ = std::clamp(prefetch_factor_, 0, 64);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled(prefetch_factor_ > 0 ? "batches queued ahead (Arrow / Parquet batchers)"
                                                 : "0 = no queue ahead");
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Balance")) {
        ImGui::Spacing();
        if (ImGui::Checkbox("Balance training classes", &balance_classes_)) {
            if (balance_classes_ && balance_mode_index_ == 0) balance_mode_index_ = 1;
            if (!balance_classes_) balance_mode_index_ = 0;
            has_changes_ = true;
        }
        ImGui::TextDisabled("Training batches only; validation and test keep their distribution.");
        const char* balance_modes[] = {"None", "Oversample minority", "Undersample majority", "Weighted sampler"};
        ImGui::BeginDisabled(!balance_classes_);
        Field("Mode");
        ImGui::SetNextItemWidth(200);
        if (ImGui::Combo("##balance_mode", &balance_mode_index_, balance_modes, IM_ARRAYSIZE(balance_modes))) {
            if (balance_mode_index_ == 0) balance_classes_ = false;
            has_changes_ = true;
        }
        const char* balance_targets[] = {"max", "median", "min", "custom number"};
        int target_index = 0;
        if (std::strcmp(balance_target_, "median") == 0) target_index = 1;
        else if (std::strcmp(balance_target_, "min") == 0) target_index = 2;
        else if (IsPositiveIntegerText(balance_target_)) target_index = 3;
        Field("Target per class");
        ImGui::SetNextItemWidth(200);
        if (ImGui::Combo("##balance_target_mode", &target_index, balance_targets, IM_ARRAYSIZE(balance_targets))) {
            if (target_index == 0) CopyToBuffer(balance_target_, sizeof(balance_target_), "max");
            else if (target_index == 1) CopyToBuffer(balance_target_, sizeof(balance_target_), "median");
            else if (target_index == 2) CopyToBuffer(balance_target_, sizeof(balance_target_), "min");
            else if (!IsPositiveIntegerText(balance_target_))
                CopyToBuffer(balance_target_, sizeof(balance_target_), "1");
            has_changes_ = true;
        }
        if (target_index == 3) {
            int target_count = 1;
            try { target_count = std::max(1, std::stoi(balance_target_)); } catch (...) { target_count = 1; }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(110);
            if (ImGui::InputInt("##balance_target_count", &target_count)) {
                if (target_count < 1) target_count = 1;
                CopyToBuffer(balance_target_, sizeof(balance_target_), std::to_string(target_count));
                has_changes_ = true;
            }
        }
        Field("Balance seed");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##balance_seed", &balance_seed_)) {
            if (balance_seed_ < 0) balance_seed_ = 0;
            has_changes_ = true;
        }
        ImGui::EndDisabled();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Checkpoints")) {
        ImGui::Spacing();
        if (ImGui::Checkbox("Save the best validation checkpoint", &save_best_checkpoint_)) has_changes_ = true;
        ImGui::TextDisabled("Saved whenever the validation loss improves.");
        Field("Early stop patience");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##early_stopping_patience", &early_stopping_patience_)) {
            early_stopping_patience_ = std::clamp(early_stopping_patience_, 0, 100000);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("epochs without improvement; 0 = off");
        Field("Checkpoint folder");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##checkpoint_dir", checkpoint_dir_, sizeof(checkpoint_dir_))) has_changes_ = true;
        ImGui::TextDisabled("Empty uses the project's checkpoints folder (.cyxwiz/checkpoints without a project).");
        Field("Start from");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##initialize_from_checkpoint", initialize_from_checkpoint_,
                             sizeof(initialize_from_checkpoint_)))
            has_changes_ = true;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("Optional: a checkpoint folder from an earlier run (e.g. .../train-<id>/best). Its weights "
                            "are this model's starting point and every layer keeps training on this data (full "
                            "fine-tuning; nothing is frozen). Optimizer, schedule, epochs and seed start fresh. Layer "
                            "names, shapes and the tokenizer must match. Empty = random weights.");
        ImGui::PopTextWrapPos();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Advanced")) {
        ImGui::Spacing();
        Field("Worker threads");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##num_workers", &num_workers_)) {
            num_workers_ = std::clamp(num_workers_, 0, 64);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled(num_workers_ > 0 ? "per-batch workers where supported"
                                             : "0 = batches load on the training thread");
        Field("Pin memory");
        ImGui::TextColored(pin_memory_requested_ ? t.warning : t.text_dim, "%s",
                           pin_memory_requested_ ? "not supported on this device (this graph asks for it; ignored)"
                                                 : "not supported on this device");
        Field("Report every");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##log_interval", &log_interval_)) {
            log_interval_ = std::clamp(log_interval_, 0, 100000);
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("batches; 0 = only the first and the last");
        ImGui::Spacing();
        if (ImGui::Checkbox("Generate samples during training", &generation_preview_enabled_)) has_changes_ = true;
        ImGui::BeginDisabled(!generation_preview_enabled_);
        Field("Every");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##gen_every", &generation_preview_every_epochs_)) has_changes_ = true;
        ImGui::SameLine();
        ImGui::TextDisabled("completed epochs");
        Field("New tokens");
        ImGui::SetNextItemWidth(120);
        if (ImGui::InputInt("##gen_tokens", &generation_preview_max_new_tokens_)) has_changes_ = true;
        ImGui::SameLine();
        ImGui::TextDisabled("1 to 128");
        Field("Prompts");
        if (ImGui::InputTextMultiline("##gen_prompts", generation_preview_prompts_, sizeof(generation_preview_prompts_),
                                      ImVec2(-1, 100)))
            has_changes_ = true;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("One prompt per line (1 to 8); for multi-line chat prompts put a line of --- between "
                            "prompts (an empty line before --- keeps a trailing newline). Greedy previews use the "
                            "causal token-window vocabulary; training pauses between epochs; results go to the "
                            "Console and the run's artifacts.");
        ImGui::PopTextWrapPos();
        ImGui::EndDisabled();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "This node reads the Data Split's partitions and is the one place batching is "
                                     "set: a batch size on the optimizer is ignored (a warning says so at training "
                                     "start).");
    ImGui::PopTextWrapPos();
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
    resolution_attempted_ = false;
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
    resolution_attempted_ = false;
    has_changes_ = false;
}

void DataSplitDialog::RefreshResolvedManifest() {
    resolution_attempted_ = true;
    resolved_partitions_ = cyxwiz::ResolvedDatasetPartitions{};
    resolution_status_.clear();
    if (!node_editor_) {
        resolution_status_ =
            "Graph context is unavailable. Reopen this dialog from the node Properties panel.";
        return;
    }

    cyxwiz::GraphCompiler compiler;
    auto config = compiler.Compile(
        node_editor_->GetNodes(), node_editor_->GetLinks(), true);
    resolved_partitions_ = std::move(config.dataset_roles);
    if (!resolved_partitions_.train.IsSupplied()) {
        resolution_status_ =
            "No Training Dataset is resolved on the active training path.";
        return;
    }
    resolution_status_ = config.is_valid
        ? "Resolved from the active graph topology."
        : "Partition facts are shown, but the complete graph still has validation errors.";
}

void DataSplitDialog::RenderContent() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    const bool legacy_contract = node_ && node_->parameters.count("data_boundary_pin_contract") > 0 &&
                                 node_->parameters.at("data_boundary_pin_contract") == "legacy.v1";
    if (legacy_contract) {
        // Only an old graph shows this.
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.caution, "This graph uses the old Data Boundary v1: its two tensor inputs and six "
                                      "Train / Val / Test outputs stay connected as saved.");
        ImGui::TextDisabled("Migrating turns Data Input -> Data Split -> Data Loader into Dataset v2: duplicate "
                            "label links go and label targets read Data Loader.Labels. It is refused when Val / Test "
                            "canvas branches or an ambiguous boundary would lose meaning.");
        ImGui::PopTextWrapPos();
        if (cyxwiz::ui::SecondaryButton("Migrate graph to Dataset v2", node_editor_ != nullptr,
                                        "Open the dialog from the canvas", cyxwiz::ui::ButtonSize::Small)) {
            const auto result = node_editor_->MigrateLegacyDataBoundary();
            migration_status_ = result.message;
            if (result.success) resolution_attempted_ = false;
        }
        if (!migration_status_.empty()) ImGui::TextWrapped("%s", migration_status_.c_str());
        ImGui::Spacing();
    }

    // The three parts as one bar, with their rows from the graph's data.
    const auto upstream = node_ ? FindUpstreamData(node_editor_, node_->id) : UpstreamData{};
    const float sum = train_ratio_ + val_ratio_ + test_ratio_;
    const float norm = sum > 0.001f ? sum : 1.0f;
    const float parts[3] = {train_ratio_ / norm, val_ratio_ / norm, test_ratio_ / norm};
    const char* names[3] = {"train", "val", "test"};
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = ImGui::GetFrameHeight();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    float x = at.x;
    auto* draw = ImGui::GetWindowDrawList();
    std::string unfitted;  // parts too narrow for their label: listed under the bar
    for (int k = 0; k < 3; ++k) {
        const float w = width * parts[k];
        if (w <= 0.5f) continue;
        draw->AddRectFilled(ImVec2(x, at.y), ImVec2(x + w - 2.0f, at.y + height),
                            ImGui::GetColorU32(cyxwiz::ui::WithAlpha(t.series[k], 0.55f)), 4.0f);
        std::string label = names[k];
        if (k == 0) label += " " + std::to_string(static_cast<int>(std::lround(parts[k] * 100.0f))) + "%";
        if (upstream.rows > 0)
            label += (k == 0 ? " \xC2\xB7 about " : " ") +
                     GroupDigits(static_cast<int64_t>(std::llround(upstream.rows * parts[k])));
        const ImVec2 size = ImGui::CalcTextSize(label.c_str());
        if (size.x + 12.0f < w)
            draw->AddText(ImVec2(x + 6.0f, at.y + (height - size.y) * 0.5f), ImGui::GetColorU32(t.text_bright),
                          label.c_str());
        else
            unfitted += (unfitted.empty() ? "" : "   ") + label;
        x += w;
    }
    ImGui::Dummy(ImVec2(width, height));
    if (!unfitted.empty()) ImGui::TextColored(t.text_dim, "%s", unfitted.c_str());
    ImGui::Spacing();

    bool changed = false;
    Field("Train");
    ImGui::SetNextItemWidth(220);
    if (ImGui::SliderFloat("##train", &train_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;
    Field("Validation");
    ImGui::SetNextItemWidth(220);
    if (ImGui::SliderFloat("##val", &val_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;
    Field("Test");
    ImGui::SetNextItemWidth(220);
    if (ImGui::SliderFloat("##test", &test_ratio_, 0.0f, 1.0f, "%.2f")) changed = true;
    if (changed) has_changes_ = true;
    if (std::abs(sum - 1.0f) > 0.01f)
        ImGui::TextColored(t.warning, "Sum %.2f (Apply scales the three to 1.00)", sum);
    else
        ImGui::TextColored(t.text_dim, "Sum %.2f", sum);

    Field("Seed");
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##seed", &seed_)) has_changes_ = true;
    ImGui::SameLine();
    ImGui::TextDisabled("the shuffled order the parts are cut from");
    if (ImGui::Checkbox("Stratified: keep each class's share in every part", &stratified_)) has_changes_ = true;

    ImGui::Spacing();
    if (!resolution_attempted_) RefreshResolvedManifest();
    const auto& partitions = resolved_partitions_;
    if (partitions.train.IsSupplied()) {
        ImGui::TextColored(t.text_dim, "From the graph:");
        ImGui::SameLine();
        std::string from = partitions.train.dataset_name;
        if (upstream.rows > 0) from += ", " + GroupDigits(upstream.rows) + " samples";
        ImGui::TextUnformatted(from.c_str());
    } else {
        ImGui::TextColored(t.warning, "%s", resolution_status_.c_str());
    }
    ImGui::SameLine();
    if (cyxwiz::ui::LinkButton("Refresh from graph")) RefreshResolvedManifest();

    if (partitions.train.IsSupplied() && ImGui::TreeNodeEx("Details (the resolved manifest)")) {
        const auto& manifest = partitions.manifest;
        const auto rows_text = [](int64_t rows) {
            return rows >= 0 ? GroupDigits(rows) : std::string("resolved at batching");
        };
        const auto short_id = [](const std::string& value) {
            return value.empty() ? std::string("unavailable") : value.substr(0, std::min<size_t>(12, value.size()));
        };
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", resolution_status_.c_str());
        ImGui::Text("Train: %s, external, %s rows", partitions.train.dataset_name.c_str(),
                    rows_text(manifest.train_rows).c_str());
        ImGui::Text("Validation: %s, %s, %s rows, schema %s",
                    (partitions.dev.IsSupplied() ? partitions.dev.dataset_name
                                                   : partitions.train.dataset_name).c_str(),
                    cyxwiz::PartitionOriginName(manifest.dev_origin), rows_text(manifest.dev_rows).c_str(),
                    cyxwiz::PartitionCompatibilityName(manifest.dev_compatibility));
        ImGui::Text("Test: %s, %s, %s rows, schema %s",
                    (partitions.test.IsSupplied() ? partitions.test.dataset_name
                                                   : partitions.train.dataset_name).c_str(),
                    cyxwiz::PartitionOriginName(manifest.test_origin), rows_text(manifest.test_rows).c_str(),
                    cyxwiz::PartitionCompatibilityName(manifest.test_compatibility));
        ImGui::TextDisabled("Validation: %s", manifest.dev_resolution_reason.c_str());
        ImGui::TextDisabled("Test: %s", manifest.test_resolution_reason.c_str());
        ImGui::Text("Policy: %s, seed %d, stratified %s, manifest %s",
                    cyxwiz::PartitionSplitMethodName(manifest.split_method), manifest.seed,
                    manifest.stratified ? "yes" : "no", short_id(manifest.manifest_fingerprint).c_str());
        ImGui::Text("Source ids: train %s, validation %s, test %s, feature schema %s",
                    short_id(manifest.training_source_fingerprint).c_str(),
                    short_id(manifest.validation_source_fingerprint).c_str(),
                    short_id(manifest.test_source_fingerprint).c_str(),
                    short_id(manifest.feature_schema_fingerprint).c_str());
        ImGui::TextDisabled("Leakage: validation %s, test %s. Overlap with external roles is checked again before "
                            "training.",
                            cyxwiz::PartitionLeakageStatusName(manifest.dev_leakage),
                            cyxwiz::PartitionLeakageStatusName(manifest.test_leakage));
        ImGui::PopTextWrapPos();
        ImGui::TreePop();
    }

    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Without a Data Split node training uses 80/10/10. This node sets how the data "
                                     "is partitioned; validation and test come from these partitions, not from canvas "
                                     "branches (old graphs may still show Train / Val / Test output pins).");
    ImGui::PopTextWrapPos();
}

} // namespace gui
