// DataInputDialog source, preview-loading, and browse helpers.

#ifdef CreateDialog
#undef CreateDialog
#endif

#include "node_config_dialog.h"
#include "data_input_preview.h"
#include "../core/async_task_manager.h"
#include "../core/data_preview_service.h"
#include "../core/data_registry.h"
#include "../core/file_dialogs.h"

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
    ResetPreviewPaging();
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

    if (CanPageRegisteredPreview()) {
        preview_is_paged_ = true;
        preview_loaded_ = true;
        RequestPreviewPage(0);
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

void DataInputDialog::RefreshColumnList() {
    ResetPreviewPaging();
    std::string selected_label;
    if (label_column_idx_ >= 0 &&
        label_column_idx_ < static_cast<int>(available_columns_.size())) {
        selected_label = available_columns_[label_column_idx_];
    }

    preview_error_.clear();
    preview_columns_.clear();
    preview_data_.clear();
    available_columns_.clear();
    selected_columns_.clear();
    label_column_idx_ = -1;

    const bool is_delimited = detected_type_ >= 0 && detected_type_ <= 2;
    const bool is_tabular_source =
        source_type_ == SourceType::File &&
        (file_category_ == FileCategory::Tabular ||
         file_category_ == FileCategory::Text ||
         file_category_ == FileCategory::TimeSeries);
    if (strlen(file_path_) > 0 && is_delimited && is_tabular_source) {
        LoadColumnList();
    }

    if (!selected_label.empty()) {
        for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
            if (available_columns_[i] == selected_label) {
                label_column_idx_ = i;
                break;
            }
        }
    }
    preview_loaded_ = false;
}

bool DataInputDialog::CanPageRegisteredPreview() const {
    if (source_type_ != SourceType::File ||
        (file_category_ != FileCategory::Tabular &&
         file_category_ != FileCategory::TimeSeries) ||
        loaded_dataset_name_.empty() ||
        data_load_state_ != DataLoadState::InMemory ||
        (loaded_backend_ != 1 && loaded_backend_ != 2) ||
        !node_) {
        return false;
    }

    const auto parameter_matches = [this](const char* key,
                                           const std::string& value) {
        const auto it = node_->parameters.find(key);
        return it != node_->parameters.end() && it->second == value;
    };
    if (!parameter_matches("file_path", file_path_) ||
        !parameter_matches("type", data_input::FileTypeParam(detected_type_)) ||
        !parameter_matches("has_header", has_header_ ? "true" : "false") ||
        !parameter_matches("delimiter", custom_delimiter_) ||
        !parameter_matches("decimal_point", std::string(1, decimal_point_)) ||
        !parameter_matches("missing_value_tokens", missing_value_tokens_) ||
        !parameter_matches("skip_rows", std::to_string(skip_rows_)) ||
        !parameter_matches("max_rows", std::to_string(max_rows_))) {
        return false;
    }

    const auto source_dataset =
        cyxwiz::DataRegistry::Instance().FindTabularDatasetBySourcePath(file_path_);
    return source_dataset && *source_dataset == loaded_dataset_name_;
}

void DataInputDialog::ResetPreviewPaging() {
    if (preview_page_task_id_ != 0) {
        cyxwiz::AsyncTaskManager::Instance().Cancel(preview_page_task_id_);
    }
    ++preview_generation_;
    preview_page_cache_.Clear();
    preview_page_state_.reset();
    preview_page_task_id_ = 0;
    preview_loading_offset_ = -1;
    preview_failed_offset_ = -1;
    preview_is_paged_ = false;
    preview_page_loading_ = false;
    preview_page_error_.clear();
}

void DataInputDialog::RequestPreviewPage(int64_t row_index) {
    if (!preview_is_paged_ || preview_page_loading_ ||
        loaded_dataset_name_.empty()) {
        return;
    }

    const int64_t offset = preview_page_cache_.AlignOffset(row_index);
    if (offset >= preview_total_rows_ && preview_total_rows_ > 0) return;
    if (preview_page_cache_.ContainsPage(offset)) return;
    if (preview_failed_offset_ == offset) return;

    cyxwiz::DataPreviewRequest request;
    request.dataset_name = loaded_dataset_name_;
    request.offset = offset;
    request.row_limit = preview_page_cache_.PageSize();
    // The dialog renders the complete registered schema. Leave the column
    // selection empty so each page is resolved by stable column position.
    // Round-tripping names here can change the result for legal schemas that
    // contain duplicate or empty field names.

    auto state = std::make_shared<PreviewPageLoadState>();
    state->generation = preview_generation_;
    state->offset = offset;
    preview_page_state_ = state;
    preview_page_loading_ = true;
    preview_loading_offset_ = offset;
    preview_page_error_.clear();

    preview_page_task_id_ = cyxwiz::AsyncTaskManager::Instance().RunAsync(
        "Loading preview page",
        [state, request](cyxwiz::LambdaTask& task) {
            try {
                auto cancellable_request = request;
                cancellable_request.cancel_requested = [&task]() {
                    return task.ShouldStop();
                };
                state->page = cyxwiz::DataPreviewService::PreviewRegisteredTabular(
                    cyxwiz::DataRegistry::Instance(), cancellable_request);
            } catch (const std::exception& e) {
                state->page.dataset_name = request.dataset_name;
                state->page.reason = std::string("Preview page failed: ") + e.what();
            } catch (...) {
                state->page.dataset_name = request.dataset_name;
                state->page.reason = "Preview page failed with an unknown error";
            }
            state->done.store(true);
        });
}

void DataInputDialog::PollPreviewPageResult() {
    auto state = preview_page_state_;
    if (!state || !state->done.load()) return;

    preview_page_state_.reset();
    preview_page_task_id_ = 0;
    preview_page_loading_ = false;
    preview_loading_offset_ = -1;

    if (!preview_is_paged_ || state->generation != preview_generation_) return;

    auto& page = state->page;
    if (!page.ok) {
        preview_failed_offset_ = state->offset;
        if (preview_page_cache_.PageCount() == 0) {
            preview_error_ = page.reason.empty()
                ? "Preview page could not be loaded"
                : page.reason;
        } else {
            preview_page_error_ = page.reason.empty()
                ? "Additional preview rows could not be loaded"
                : page.reason;
        }
        return;
    }

    std::vector<std::string> page_columns;
    page_columns.reserve(page.schema.size());
    for (const auto& column : page.schema) {
        page_columns.push_back(column.name);
    }

    if (preview_columns_.empty()) {
        preview_columns_ = page_columns;
        available_columns_ = preview_columns_;
        selected_columns_.assign(preview_columns_.size(), true);
    } else if (page_columns != preview_columns_) {
        // The registered dataset may have been refreshed while a page was in
        // flight. Treat the result as a new preview generation instead of
        // surfacing a terminal paging error. The current page remains useful;
        // missing pages are fetched lazily against the refreshed schema.
        spdlog::info(
            "DataInputDialog: preview schema refreshed for '{}' while paging "
            "(old columns={}, new columns={})",
            loaded_dataset_name_,
            preview_columns_.size(),
            page_columns.size());
        ++preview_generation_;
        preview_page_cache_.Clear();
        preview_failed_offset_ = -1;
        preview_page_error_.clear();
        preview_error_.clear();
        preview_columns_ = std::move(page_columns);
        available_columns_ = preview_columns_;
        selected_columns_.assign(preview_columns_.size(), true);
        label_column_idx_ = -1;
    }

    preview_backend_ = page.backend;
    preview_total_rows_ = page.total_rows;
    preview_next_offset_ = page.next_offset;
    preview_has_next_ = page.has_next;
    preview_page_cache_.PutPage(page.offset, std::move(page.rows));
    if (preview_failed_offset_ == page.offset) preview_failed_offset_ = -1;
    preview_page_error_.clear();
}

void DataInputDialog::UpdateRAMEstimate() {
    if (file_size_ <= 0) {
        estimated_ram_mb_ = 0;
        return;
    }
    estimated_ram_mb_ = static_cast<float>(file_size_ / (1024.0 * 1024.0));
}

void DataInputDialog::BrowseFile() {
    cyxwiz::FileDialogs::FilterList filters;

    switch (file_category_) {
        case FileCategory::Tabular:
            filters = {{"Supported Tabular Data", "csv,tsv,parquet,feather,fea,arrow,ipc"},
                       {"CSV", "csv"}, {"TSV", "tsv"}, {"Parquet", "parquet"},
                       {"Feather", "feather,fea"}, {"Arrow / IPC", "arrow,ipc"}, {"All Files", "*"}};
            break;
        case FileCategory::Image:
            filters = {{"Image Files", "jpg,jpeg,png,bmp,gif,tiff,webp"}, {"All Files", "*"}};
            break;
        case FileCategory::Audio:
            filters = {{"Audio Files", "wav,mp3,flac,ogg,m4a,aac"}, {"All Files", "*"}};
            break;
        case FileCategory::Video:
            filters = {{"Video Files", "mp4,avi,mov,mkv,webm,wmv"}, {"All Files", "*"}};
            break;
        case FileCategory::Text:
            filters = {{"Text Data", "csv,tsv,json,jsonl,txt"}, {"CSV", "csv"},
                       {"TSV", "tsv"}, {"JSON", "json,jsonl"}, {"Plain Text", "txt"}, {"All Files", "*"}};
            break;
        case FileCategory::TimeSeries:
            filters = {{"Time Series Data", "csv,tsv,parquet,feather,fea,arrow,ipc"},
                       {"CSV", "csv"}, {"TSV", "tsv"}, {"Parquet", "parquet"},
                       {"Feather", "feather,fea"}, {"Arrow / IPC", "arrow,ipc"}, {"All Files", "*"}};
            break;
    }

    if (auto selected = cyxwiz::FileDialogs::OpenFile(
            "Select Data File", filters, file_path_[0] == '\0' ? nullptr : file_path_)) {
        strncpy(file_path_, selected->c_str(), sizeof(file_path_) - 1);
        file_path_[sizeof(file_path_) - 1] = '\0';
        DetectFileType();
        DetectFileCategory();
        RefreshColumnList();
        has_changes_ = true;
    }
}

void DataInputDialog::BrowseFolder() {
    if (auto selected = cyxwiz::FileDialogs::SelectFolder(
            "Select Image Folder", folder_path_[0] == '\0' ? nullptr : folder_path_)) {
        strncpy(folder_path_, selected->c_str(), sizeof(folder_path_) - 1);
        folder_path_[sizeof(folder_path_) - 1] = '\0';
        has_changes_ = true;
        preview_loaded_ = false;
    }
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
