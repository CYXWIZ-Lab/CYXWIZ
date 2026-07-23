// DataInputDialog source, preview-loading, and browse helpers.

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#ifdef CreateDialog
#undef CreateDialog
#endif

#include "node_config_dialog.h"
#include "data_input_preview.h"
#include "../core/data_preview_service.h"
#include "../core/data_registry.h"

#include <cstring>
#include <string>
#include <utility>

#include <spdlog/spdlog.h>

namespace gui {

const char* DataInputDialog::GetFileTypeName() const {
    return data_input::FileTypeName(detected_type_);
}

void DataInputDialog::DetectFileType() {
    detected_type_ = data_input::DetectFileTypeForPath(file_path_, &file_size_);
}

void DataInputDialog::DetectFileCategory() {
    file_category_ = data_input::DetectFileCategoryForPath(file_path_, file_category_);
}

void DataInputDialog::LoadPreview() {
    preview_error_.clear();
    preview_columns_.clear();
    preview_data_.clear();
    preview_total_rows_ = 0;
    preview_next_offset_ = 0;
    preview_has_next_ = false;
    preview_backend_.clear();
    label_distribution_.clear();
    label_distribution_column_.clear();
    label_distribution_total_ = 0;

    if (!IsPreviewSupported()) {
        preview_error_ = PreviewUnavailableMessage();
        preview_loaded_ = true;
        UpdateRAMEstimate();
        return;
    }

    const bool can_preview_registered_tabular =
        source_type_ == SourceType::File &&
        (file_category_ == FileCategory::Tabular ||
         file_category_ == FileCategory::TimeSeries) &&
        !loaded_dataset_name_.empty() &&
        data_load_state_ == DataLoadState::InMemory;
    if (can_preview_registered_tabular) {
        cyxwiz::DataPreviewRequest request;
        request.dataset_name = loaded_dataset_name_;
        request.offset = 0;
        request.row_limit = 20;
        auto page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
            cyxwiz::DataRegistry::Instance(), request);
        if (!page.ok) {
            preview_error_ = page.reason;
            preview_loaded_ = true;
            UpdateRAMEstimate();
            return;
        }

        preview_backend_ = page.backend;
        preview_total_rows_ = page.total_rows;
        preview_next_offset_ = page.next_offset;
        preview_has_next_ = page.has_next;
        preview_columns_.reserve(page.schema.size());
        for (const auto& column : page.schema) {
            preview_columns_.push_back(column.name);
        }
        available_columns_ = preview_columns_;
        preview_data_ = std::move(page.rows);
        selected_columns_.assign(preview_columns_.size(), true);
        preview_loaded_ = true;
        UpdateRAMEstimate();
        return;
    }

    if (source_type_ == SourceType::File &&
        (file_category_ == FileCategory::Tabular ||
         file_category_ == FileCategory::Text ||
         file_category_ == FileCategory::TimeSeries)) {
        LoadColumnList();
    }
    preview_loaded_ = true;
    UpdateRAMEstimate();
}

void DataInputDialog::LoadColumnList() {
    if (strlen(file_path_) == 0) {
        return;
    }

    const auto table = data_input::LoadDelimitedPreview(
        file_path_,
        has_header_,
        custom_delimiter_[0],
        detected_type_,
        skip_rows_);
    if (!table.error.empty()) {
        preview_error_ = table.error;
        return;
    }

    preview_columns_ = table.columns;
    available_columns_ = table.columns;
    preview_data_ = table.rows;
    preview_total_rows_ = static_cast<int64_t>(table.rows.size());
    preview_next_offset_ = preview_total_rows_;
    preview_has_next_ = false;
    preview_backend_ = "Source";
    selected_columns_.assign(table.columns.size(), true);

    UpdateTextLabelDistribution();
}

void DataInputDialog::UpdateRAMEstimate() {
    if (file_size_ <= 0) {
        estimated_ram_mb_ = 0;
        return;
    }
    estimated_ram_mb_ = static_cast<float>(file_size_ / (1024.0 * 1024.0));
}

void DataInputDialog::BrowseFile() {
#ifdef _WIN32
    const char* filter = nullptr;

    switch (file_category_) {
        case FileCategory::Tabular:
            filter = "Supported Tabular Data\0*.csv;*.tsv;*.parquet;*.feather;*.fea;*.arrow;*.ipc\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0Parquet\0*.parquet\0"
                     "Feather\0*.feather;*.fea\0Arrow / IPC\0*.arrow;*.ipc\0All Files\0*.*\0";
            break;
        case FileCategory::Image:
            filter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff;*.webp\0All Files\0*.*\0";
            break;
        case FileCategory::Audio:
            filter = "Audio Files\0*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.aac\0All Files\0*.*\0";
            break;
        case FileCategory::Video:
            filter = "Video Files\0*.mp4;*.avi;*.mov;*.mkv;*.webm;*.wmv\0All Files\0*.*\0";
            break;
        case FileCategory::Text:
            filter = "Text Data\0*.csv;*.tsv;*.json;*.jsonl;*.txt\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0"
                     "JSON\0*.json;*.jsonl\0Plain Text\0*.txt\0"
                     "All Files\0*.*\0";
            break;
        case FileCategory::TimeSeries:
            filter = "Time Series Data\0*.csv;*.tsv;*.parquet;*.feather;*.fea;*.arrow;*.ipc\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0Parquet\0*.parquet\0"
                     "Feather\0*.feather;*.fea\0Arrow / IPC\0*.arrow;*.ipc\0"
                     "All Files\0*.*\0";
            break;
    }

    OPENFILENAMEA ofn = {};
    char file[512] = {};
    strncpy(file, file_path_, sizeof(file) - 1);

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = file;
    ofn.nMaxFile = sizeof(file);
    ofn.lpstrTitle = "Select Data File";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        strncpy(file_path_, file, sizeof(file_path_) - 1);
        DetectFileType();
        DetectFileCategory();
        LoadColumnList();
        label_column_idx_ = -1;
        has_changes_ = true;
        preview_loaded_ = false;
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

void DataInputDialog::BrowseFolder() {
#ifdef _WIN32
    IFileDialog* pfd = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr,
                                  CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pfd));
    if (SUCCEEDED(hr)) {
        DWORD options = 0;
        pfd->GetOptions(&options);
        pfd->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        pfd->SetTitle(L"Select Image Folder");

        hr = pfd->Show(nullptr);
        if (SUCCEEDED(hr)) {
            IShellItem* psi = nullptr;
            hr = pfd->GetResult(&psi);
            if (SUCCEEDED(hr)) {
                PWSTR wide_path = nullptr;
                hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &wide_path);
                if (SUCCEEDED(hr) && wide_path) {
                    char narrow[512] = {};
                    WideCharToMultiByte(CP_ACP, 0, wide_path, -1,
                                        narrow, sizeof(narrow) - 1, nullptr, nullptr);
                    strncpy(folder_path_, narrow, sizeof(folder_path_) - 1);
                    has_changes_ = true;
                    preview_loaded_ = false;
                    CoTaskMemFree(wide_path);
                }
                psi->Release();
            }
        }
        pfd->Release();
    }
#else
    spdlog::warn("Folder browser not implemented for this platform");
#endif
}

std::string DataInputDialog::CurrentSourcePath() const {
    if (source_type_ == SourceType::File) {
        const bool uses_folder = data_input::UsesFolderPath(file_category_, text_layout_);
        return uses_folder ? std::string(folder_path_) : std::string(file_path_);
    }
    if (source_type_ == SourceType::MLDataset) {
        return std::string(dataset_name_);
    }
    if (source_type_ == SourceType::Database) {
        if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
            return std::string(db_file_);
        }
        return std::string(db_host_) + "/" + std::string(db_name_);
    }
    if (source_type_ == SourceType::Cloud) {
        std::string source = cloud_bucket_;
        if (strlen(cloud_path_) > 0) {
            source += "/" + std::string(cloud_path_);
        }
        return source;
    }
    return {};
}

std::string DataInputDialog::CurrentSourceLabel() const {
    return data_input::CurrentSourceLabel(source_type_, file_category_, text_layout_);
}

std::string DataInputDialog::CurrentApplySummary() const {
    return data_input::CurrentApplySummary(
        source_type_,
        file_category_,
        text_layout_,
        force_disk_backed_,
        max_rows_,
        skip_rows_);
}

const char* DataInputDialog::BackendSummary() const {
    return data_input::BackendSummary(loaded_backend_);
}

bool DataInputDialog::IsApplySupported() const {
    return data_input::IsApplySupported(source_type_, file_category_);
}

bool DataInputDialog::IsPreviewSupported() const {
    return data_input::IsPreviewSupported(source_type_, file_category_);
}

const char* DataInputDialog::UnsupportedApplyMessage() const {
    return data_input::UnsupportedApplyMessage(source_type_, file_category_);
}

const char* DataInputDialog::PreviewUnavailableMessage() const {
    return data_input::PreviewUnavailableMessage(source_type_, file_category_);
}

void DataInputDialog::MarkApplyUnsupported(const char* message) {
    apply_status_message_ = message ? message : "This data source is not available yet.";
    apply_success_ = false;
    apply_status_timer_ = 8.0f;
    if (node_) {
        node_->parameters["data_loaded"] = "false";
    }
    apply_in_progress_ = false;
    has_changes_ = false;
    spdlog::warn("DataInputDialog: unsupported Apply path - {}", apply_status_message_);
}

std::string DataInputDialog::FormatBytes(size_t bytes) {
    return data_input::FormatBytes(bytes);
}

std::string DataInputDialog::GenerateDatasetName() const {
    return data_input::GenerateDatasetName(
        source_type_,
        file_path_,
        folder_path_,
        dataset_name_,
        db_name_,
        cloud_bucket_);
}

} // namespace gui
