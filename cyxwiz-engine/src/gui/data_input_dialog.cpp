// Data Input/Output Dialog Implementations
// Part of CyxWiz Studio smart data loading system
// Comprehensive support for files, ML datasets, databases, and cloud storage

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

#ifdef CreateDialog
#undef CreateDialog
#endif

#include "node_config_dialog.h"
#include "node_editor.h"
#include "../core/data_registry.h"
#include "../core/formats/audio_dataset.h"
#include "../core/formats/text_dataset.h"
#include "../core/arrow_dataset.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/async_task_manager.h"
#include <spdlog/spdlog.h>
#include <implot.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <limits>
#include <arrow/api.h>
#include <arrow/table.h>

namespace fs = std::filesystem;

namespace gui {

// ==================== DataInputDialog ====================

DataInputDialog::DataInputDialog(MLNode* node)
    : NodeConfigDialog("Data Input", node)
{
    if (node_) {
        // Restore from parameters
        if (node_->parameters.count("source_type")) {
            std::string st = node_->parameters["source_type"];
            if (st == "file") source_type_ = SourceType::File;
            else if (st == "ml_dataset") source_type_ = SourceType::MLDataset;
            else if (st == "database") source_type_ = SourceType::Database;
            else if (st == "cloud") source_type_ = SourceType::Cloud;
        }
        if (node_->parameters.count("file_path")) {
            strncpy(file_path_, node_->parameters["file_path"].c_str(), sizeof(file_path_) - 1);
            DetectFileType();
            DetectFileCategory();
            LoadColumnList();  // Load columns to enable label selection
        }
        if (node_->parameters.count("folder_path")) {
            strncpy(folder_path_, node_->parameters["folder_path"].c_str(), sizeof(folder_path_) - 1);
        }
        if (node_->parameters.count("dataset_name")) {
            strncpy(dataset_name_, node_->parameters["dataset_name"].c_str(), sizeof(dataset_name_) - 1);
        }
        // Restore label column selection
        if (node_->parameters.count("label_column") && !node_->parameters["label_column"].empty()) {
            std::string label_col = node_->parameters["label_column"];
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                if (available_columns_[i] == label_col) {
                    label_column_idx_ = i;
                    break;
                }
            }
        }

        // Restore the Force disk-backed cache checkbox from node params.
        // Without this, the toggle state is lost every time the dialog is
        // closed, and the user silently falls back to the default auto-
        // detect path on the next Apply even though they intended to keep
        // using disk-backed mode. Persistence via node_->parameters so it
        // travels with the project file too.
        if (node_->parameters.count("force_disk_backed") &&
            node_->parameters["force_disk_backed"] == "true") {
            force_disk_backed_ = true;
        }

        // Restore the image layout strategy from node params.
        if (node_->parameters.count("image_layout")) {
            try {
                int raw = std::stoi(node_->parameters["image_layout"]);
                if (raw == static_cast<int>(ImageLayout::FlatWithCSV)) {
                    image_layout_ = ImageLayout::FlatWithCSV;
                } else {
                    image_layout_ = ImageLayout::ClassSubdirs;
                }
            } catch (...) {
                image_layout_ = ImageLayout::ClassSubdirs;
            }
        }

        // Restore the file category. Without this, the Apply path dispatches
        // by whatever file_category_ was initialised to (Tabular) or by
        // DetectFileCategory() which only runs if file_path_ is set — neither
        // matches the actual persisted category for audio / image folders.
        // So picking Audio, loading a folder, closing the dialog, and
        // reopening would silently fall through to the image branch on the
        // next Apply. Reading it back from params fixes the regression.
        if (node_->parameters.count("file_category")) {
            const std::string& cat = node_->parameters["file_category"];
            if (cat == "tabular") file_category_ = FileCategory::Tabular;
            else if (cat == "image") file_category_ = FileCategory::Image;
            else if (cat == "audio") file_category_ = FileCategory::Audio;
            else if (cat == "video") file_category_ = FileCategory::Video;
            else if (cat == "text") file_category_ = FileCategory::Text;
            else if (cat == "timeseries") file_category_ = FileCategory::TimeSeries;
        }

        // Restore text dialog state. Without this, picking Text, filling
        // in text_column / tokenizer settings, closing the dialog, and
        // reopening would reset the column names to the default "text"
        // and lose the user's choice. Mirrors the audio restore block.
        if (node_->parameters.count("text_layout")) {
            try {
                int raw = std::stoi(node_->parameters["text_layout"]);
                if (raw == static_cast<int>(TextLayout::CorpusSubdirs)) {
                    text_layout_ = TextLayout::CorpusSubdirs;
                } else {
                    text_layout_ = TextLayout::SingleFile;
                }
            } catch (...) {
                text_layout_ = TextLayout::SingleFile;
            }
        }
        if (node_->parameters.count("text_column")) {
            strncpy(text_column_,
                    node_->parameters["text_column"].c_str(),
                    sizeof(text_column_) - 1);
            text_column_[sizeof(text_column_) - 1] = '\0';
        }
        if (node_->parameters.count("text_label_column")) {
            strncpy(text_label_column_,
                    node_->parameters["text_label_column"].c_str(),
                    sizeof(text_label_column_) - 1);
            text_label_column_[sizeof(text_label_column_) - 1] = '\0';
        }
        if (node_->parameters.count("text_tokenizer_type")) {
            try { text_tokenizer_type_ = std::stoi(node_->parameters["text_tokenizer_type"]); }
            catch (...) { text_tokenizer_type_ = 1; }
        }
        if (node_->parameters.count("text_max_length")) {
            try { text_max_length_ = std::stoi(node_->parameters["text_max_length"]); }
            catch (...) { text_max_length_ = 512; }
        }
        if (node_->parameters.count("text_lowercase")) {
            text_lowercase_ = (node_->parameters["text_lowercase"] == "true");
        }
        if (node_->parameters.count("text_min_freq")) {
            try { text_min_freq_ = std::stoi(node_->parameters["text_min_freq"]); }
            catch (...) { text_min_freq_ = 1; }
        }
        if (node_->parameters.count("text_max_vocab_size")) {
            try { text_max_vocab_size_ = std::stoi(node_->parameters["text_max_vocab_size"]); }
            catch (...) { text_max_vocab_size_ = -1; }
        }

        // Restore the audio layout strategy from node params (ClassSubdirs
        // vs FlatWithCSV + label CSV + column overrides). Mirrors image_layout_.
        if (node_->parameters.count("audio_layout")) {
            try {
                int raw = std::stoi(node_->parameters["audio_layout"]);
                if (raw == static_cast<int>(AudioLayout::FlatWithCSV)) {
                    audio_layout_ = AudioLayout::FlatWithCSV;
                } else {
                    audio_layout_ = AudioLayout::ClassSubdirs;
                }
            } catch (...) {
                audio_layout_ = AudioLayout::ClassSubdirs;
            }
        }
        if (node_->parameters.count("audio_labels_csv")) {
            strncpy(audio_labels_csv_,
                    node_->parameters["audio_labels_csv"].c_str(),
                    sizeof(audio_labels_csv_) - 1);
            audio_labels_csv_[sizeof(audio_labels_csv_) - 1] = '\0';
        }
        if (node_->parameters.count("audio_filename_col")) {
            strncpy(audio_filename_col_,
                    node_->parameters["audio_filename_col"].c_str(),
                    sizeof(audio_filename_col_) - 1);
            audio_filename_col_[sizeof(audio_filename_col_) - 1] = '\0';
        }
        if (node_->parameters.count("audio_label_col")) {
            strncpy(audio_label_col_,
                    node_->parameters["audio_label_col"].c_str(),
                    sizeof(audio_label_col_) - 1);
            audio_label_col_[sizeof(audio_label_col_) - 1] = '\0';
        }

        // Restore data load state by probing the registry directly. The
        // registry is the source of truth — the parameters["data_loaded"]
        // hint goes stale when an async Apply finishes after the dialog
        // has been closed (PollAsyncLoadResult only runs while the dialog
        // is open, so the provisional "false" never gets flipped back).
        // Trusting the registry sidesteps that race entirely and also
        // re-syncs the param so the compile gate sees consistent state.
        if (node_->parameters.count("dataset_name") && !node_->parameters["dataset_name"].empty()) {
            loaded_dataset_name_ = node_->parameters["dataset_name"];
            auto& registry = cyxwiz::DataRegistry::Instance();

            // Reset the estimate flag up front. Tabular Arrow/Parquet branches
            // use actual memory; lazy-loaded image/audio branches will flip
            // it back to true.
            loaded_memory_is_estimate_ = false;

            auto arrow_ds = registry.GetArrowDataset(loaded_dataset_name_);
            auto parquet_ds = registry.GetParquetBackedDataset(loaded_dataset_name_);

            auto img_entry = registry.GetImageDatasetEntry(loaded_dataset_name_);
            auto audio_entry_ptr = registry.GetAudioDatasetEntry(loaded_dataset_name_);
            auto text_entry_ptr = registry.GetTextDatasetEntry(loaded_dataset_name_);

            if (arrow_ds) {
                loaded_rows_ = arrow_ds->GetNumRows();
                loaded_cols_ = arrow_ds->GetNumColumns();
                loaded_memory_bytes_ = arrow_ds->GetMemoryUsage();
                data_load_state_ = DataLoadState::InMemory;
                loaded_backend_ = 1;
                apply_success_ = true;
                apply_status_message_ = "Loaded " + loaded_dataset_name_ + " (" +
                    std::to_string(loaded_rows_) + " rows, " +
                    std::to_string(loaded_cols_) + " cols, " +
                    FormatBytes(loaded_memory_bytes_) + ")";
                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                node_->parameters["memory_bytes"] = std::to_string(loaded_memory_bytes_);
                spdlog::debug("DataInputDialog: restored in-memory Arrow state for '{}'",
                              loaded_dataset_name_);
            } else if (parquet_ds) {
                loaded_rows_ = parquet_ds->GetNumRows();
                loaded_cols_ = parquet_ds->GetNumColumns();
                loaded_memory_bytes_ = parquet_ds->GetMemoryUsage();
                data_load_state_ = DataLoadState::InMemory;
                loaded_backend_ = 2;
                apply_success_ = true;
                apply_status_message_ = "Loaded " + loaded_dataset_name_ +
                    " via Parquet cache (" +
                    std::to_string(loaded_rows_) + " rows, " +
                    std::to_string(loaded_cols_) + " cols, " +
                    FormatBytes(loaded_memory_bytes_) + " on disk)";
                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                node_->parameters["memory_bytes"] = std::to_string(loaded_memory_bytes_);
                spdlog::debug("DataInputDialog: restored Parquet-backed state for '{}'",
                              loaded_dataset_name_);
            } else if (img_entry) {
                loaded_rows_ = static_cast<int64_t>(img_entry->num_images);
                loaded_cols_ = 1;
                // Match the Apply path estimate: 224×224×3 float per image.
                // Actual target size comes from the Resize node and isn't
                // known here, so this is a conservative default estimate.
                size_t per_image = 224ull * 224 * 3 * sizeof(float);
                loaded_memory_bytes_ = img_entry->num_images * per_image;
                loaded_memory_is_estimate_ = true;
                data_load_state_ = DataLoadState::InMemory;
                loaded_backend_ = 3;  // 3 = image
                apply_success_ = true;
                apply_status_message_ = "Loaded " + loaded_dataset_name_ +
                    " (" + std::to_string(img_entry->num_images) + " images, " +
                    std::to_string(img_entry->num_classes) + " classes)";
                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                spdlog::debug("DataInputDialog: restored image dataset state for '{}'",
                              loaded_dataset_name_);
            } else if (audio_entry_ptr) {
                // Audio restore: mirror image branch. Uses the probed
                // feature_rows/feature_cols persisted on the entry so
                // we don't have to re-probe on every dialog reopen.
                loaded_rows_ = static_cast<int64_t>(audio_entry_ptr->num_samples);
                loaded_cols_ = 1;
                size_t per_sample = (audio_entry_ptr->feature_rows > 0 &&
                                     audio_entry_ptr->feature_cols > 0)
                    ? static_cast<size_t>(audio_entry_ptr->feature_rows) *
                      static_cast<size_t>(audio_entry_ptr->feature_cols) * sizeof(float)
                    : static_cast<size_t>(audio_entry_ptr->n_mels) * 313 * sizeof(float);
                loaded_memory_bytes_ = audio_entry_ptr->num_samples * per_sample;
                loaded_memory_is_estimate_ = true;
                data_load_state_ = DataLoadState::InMemory;
                loaded_backend_ = 4;  // 4 = audio
                apply_success_ = true;
                apply_status_message_ = "Loaded " + loaded_dataset_name_ +
                    " (" + std::to_string(audio_entry_ptr->num_samples) + " audio files, " +
                    std::to_string(audio_entry_ptr->num_classes) + " classes)";
                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                spdlog::debug("DataInputDialog: restored audio dataset state for '{}' "
                              "({} samples, feature shape [{}x{}])",
                              loaded_dataset_name_, audio_entry_ptr->num_samples,
                              audio_entry_ptr->feature_rows, audio_entry_ptr->feature_cols);
            } else if (text_entry_ptr) {
                // Text restore: mirror audio branch. Uses the vocab_size
                // and num_samples persisted on the entry at Apply time.
                loaded_rows_ = static_cast<int64_t>(text_entry_ptr->num_samples);
                loaded_cols_ = 1;
                // Estimate: max_length * sizeof(float) per sample. Actual
                // memory during training is dominated by the tokenized
                // in-RAM corpus, not the training tensor batches.
                size_t per_sample = static_cast<size_t>(text_entry_ptr->max_length) * sizeof(float);
                loaded_memory_bytes_ = text_entry_ptr->num_samples * per_sample;
                loaded_memory_is_estimate_ = true;
                data_load_state_ = DataLoadState::InMemory;
                loaded_backend_ = 5;  // 5 = text
                apply_success_ = true;
                apply_status_message_ = "Loaded " + loaded_dataset_name_ +
                    " (" + std::to_string(text_entry_ptr->num_samples) + " text samples, " +
                    std::to_string(text_entry_ptr->num_classes) + " classes, vocab " +
                    std::to_string(text_entry_ptr->vocab_size) + ")";
                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                spdlog::debug("DataInputDialog: restored text dataset state for '{}' "
                              "({} samples, vocab {})",
                              loaded_dataset_name_, text_entry_ptr->num_samples,
                              text_entry_ptr->vocab_size);
            } else {
                // Registry doesn't have it. Either the user never applied,
                // or project close / graph clear unregistered. Reflect
                // reality in the param hint and show "Not Loaded".
                node_->parameters["data_loaded"] = "false";
                spdlog::debug("DataInputDialog: dataset '{}' not in registry - "
                              "showing as Not Loaded",
                              loaded_dataset_name_);
            }
        }
    }
}

void DataInputDialog::Apply() {
    if (!node_) return;

    // Source type
    const char* source_types[] = {"file", "ml_dataset", "database", "cloud"};
    node_->parameters["source_type"] = source_types[static_cast<int>(source_type_)];

    // File category
    const char* categories[] = {"tabular", "image", "audio", "video", "text", "timeseries"};
    node_->parameters["file_category"] = categories[static_cast<int>(file_category_)];

    // Common parameters
    node_->parameters["file_path"] = file_path_;
    node_->parameters["folder_path"] = folder_path_;
    node_->parameters["configured"] = "true";
    // Persist the Force disk-backed toggle so reopening the dialog (or
    // reopening the project) keeps the user's choice. Without this, the
    // checkbox silently resets to false and the next Apply falls back to
    // auto-detect, surprising users who explicitly asked for disk-backed.
    node_->parameters["force_disk_backed"] = force_disk_backed_ ? "true" : "false";

    // Persist audio layout + FlatWithCSV fields so reopen restores them.
    // Without this, picking FlatWithCSV and a labels CSV, closing the dialog,
    // and reopening would silently reset to ClassSubdirs and lose the CSV
    // path, causing the next Apply to fail with "no class subdirectories".
    node_->parameters["audio_layout"] =
        std::to_string(static_cast<int>(audio_layout_));
    node_->parameters["audio_labels_csv"] = audio_labels_csv_;
    node_->parameters["audio_filename_col"] = audio_filename_col_;
    node_->parameters["audio_label_col"] = audio_label_col_;

    // Source-specific parameters
    if (source_type_ == SourceType::File) {
        const char* types[] = {"auto", "csv", "tsv", "json", "parquet", "excel", "hdf5", "feather", "arrow", "txt", "arff"};
        node_->parameters["type"] = types[detected_type_];
        node_->parameters["has_header"] = has_header_ ? "true" : "false";
        node_->parameters["delimiter"] = custom_delimiter_;
        node_->parameters["skip_rows"] = std::to_string(skip_rows_);
        node_->parameters["max_rows"] = std::to_string(max_rows_);
        node_->parameters["encoding"] = std::to_string(encoding_idx_);

        if (file_category_ == FileCategory::Image) {
            node_->parameters["target_width"] = std::to_string(target_width_);
            node_->parameters["target_height"] = std::to_string(target_height_);
            node_->parameters["normalize"] = normalize_images_ ? "true" : "false";
            node_->parameters["rgb"] = rgb_mode_ ? "true" : "false";
            node_->parameters["labels_csv"] = labels_csv_;
            node_->parameters["image_layout"] = std::to_string(static_cast<int>(image_layout_));
        } else if (file_category_ == FileCategory::Audio) {
            node_->parameters["sample_rate"] = std::to_string(sample_rate_);
            node_->parameters["mono"] = mono_ ? "true" : "false";
        } else if (file_category_ == FileCategory::Text) {
            // Persist text dialog state so reopening the dialog restores
            // the layout choice, column mapping, and tokenizer defaults.
            node_->parameters["text_layout"] =
                std::to_string(static_cast<int>(text_layout_));
            node_->parameters["text_column"] = text_column_;
            node_->parameters["text_label_column"] = text_label_column_;
            node_->parameters["text_tokenizer_type"] = std::to_string(text_tokenizer_type_);
            node_->parameters["text_max_length"] = std::to_string(text_max_length_);
            node_->parameters["text_lowercase"] = text_lowercase_ ? "true" : "false";
            node_->parameters["text_min_freq"] = std::to_string(text_min_freq_);
            node_->parameters["text_max_vocab_size"] = std::to_string(text_max_vocab_size_);
        }

        // Excel specific
        if (detected_type_ == 5) {
            node_->parameters["sheet_idx"] = std::to_string(sheet_idx_);
            node_->parameters["sheet_range"] = sheet_range_;
        }
        // HDF5 specific
        if (detected_type_ == 6) {
            node_->parameters["hdf5_dataset"] = hdf5_dataset_;
        }
        // JSON specific
        if (detected_type_ == 3) {
            node_->parameters["json_lines"] = json_lines_ ? "true" : "false";
            node_->parameters["json_path"] = json_path_;
        }

        // Label column for ML training
        if (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(available_columns_.size())) {
            node_->parameters["label_column"] = available_columns_[label_column_idx_];
        } else {
            node_->parameters["label_column"] = "";
        }

        // Auto-generate description
        if (strlen(file_path_) > 0) {
            fs::path p(file_path_);
            std::string desc = "Reading " + p.filename().string();
            if (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(available_columns_.size())) {
                desc += " [label: " + available_columns_[label_column_idx_] + "]";
            }
            node_->description = desc;
        } else if (strlen(folder_path_) > 0) {
            fs::path p(folder_path_);
            node_->description = "Loading from " + p.filename().string();
        }
    }
    else if (source_type_ == SourceType::MLDataset) {
        const char* ml_types[] = {"mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet", "image_folder", "huggingface", "kaggle", "custom"};
        node_->parameters["ml_dataset_type"] = ml_types[static_cast<int>(ml_dataset_type_)];
        node_->parameters["dataset_name"] = dataset_name_;
        node_->parameters["dataset_subset"] = dataset_subset_;
        node_->parameters["cache_dir"] = cache_dir_;

        if (ml_dataset_type_ == MLDatasetType::HuggingFace) {
            node_->parameters["hf_token"] = hf_token_;
        } else if (ml_dataset_type_ == MLDatasetType::Kaggle) {
            node_->parameters["kaggle_slug"] = kaggle_slug_;
        }

        node_->description = std::string("Loading ") + dataset_name_;
    }
    else if (source_type_ == SourceType::Database) {
        const char* db_types[] = {"sqlite", "postgresql", "mysql", "duckdb"};
        node_->parameters["database_type"] = db_types[static_cast<int>(database_type_)];
        node_->parameters["db_host"] = db_host_;
        node_->parameters["db_port"] = std::to_string(db_port_);
        node_->parameters["db_name"] = db_name_;
        node_->parameters["db_user"] = db_user_;
        node_->parameters["db_file"] = db_file_;
        node_->parameters["sql_query"] = sql_query_;

        node_->description = "Query from " + std::string(db_name_);
    }
    else if (source_type_ == SourceType::Cloud) {
        node_->parameters["cloud_bucket"] = cloud_bucket_;
        node_->parameters["cloud_path"] = cloud_path_;
        node_->parameters["cloud_credentials"] = cloud_credentials_;

        node_->description = "Loading from " + std::string(cloud_bucket_);
    }

    // === NEW: Actually load data and provide feedback ===
    apply_in_progress_ = true;
    apply_success_ = false;
    apply_status_message_.clear();

    // Phase 0 gate: video is still not wired (audio landed in Phase 2).
    // Picking a video file would previously fall through to LoadArrowTable
    // which tries to parse the MP4 as Arrow IPC. Fail loudly until
    // Phase 5 (video) wires up proper loaders. See
    // docs/Data Studio/multi_format_data_pipeline_design.md.
    if (source_type_ == SourceType::File &&
        file_category_ == FileCategory::Video) {
        apply_status_message_ = "Video data is not yet supported. "
            "Coming in a future release - use tabular, image, or audio for now.";
        apply_success_ = false;
        apply_status_timer_ = 8.0f;
        node_->parameters["data_loaded"] = "false";
        apply_in_progress_ = false;
        has_changes_ = false;
        spdlog::warn("DataInputDialog: Video category selected - not yet supported");
        return;
    }

    // Guard: Image category with a single file (not a folder) is not a
    // valid training use case. The user probably used the top-level file
    // picker instead of the Image Loading Options folder picker.
    if (source_type_ == SourceType::File &&
        file_category_ == FileCategory::Image &&
        strlen(file_path_) > 0 && strlen(folder_path_) == 0) {
        apply_status_message_ = "Image training requires a folder of images, "
            "not a single file. Set the Folder path in the Image Loading "
            "Options section above.";
        apply_success_ = false;
        apply_status_timer_ = 8.0f;
        node_->parameters["data_loaded"] = "false";
        apply_in_progress_ = false;
        has_changes_ = false;
        spdlog::warn("DataInputDialog: Image category with single file - need folder instead");
        return;
    }

    // Text loading (Phase 3).
    //
    // Takes priority over the tabular CSV branch since both can trigger
    // on a non-empty file_path_. We dispatch explicitly by file_category_
    // to route the user's intent: Text now goes through the SAME async
    // AsyncTaskManager worker pattern that Arrow/Parquet uses, because
    // the original assumption ("text files are small, a sync probe is
    // fine") was wrong — 52k-row CSVs with word tokenization take 1.5-2
    // seconds of UI-thread freeze during TextDataset construction, and
    // the user sees no feedback.
    //
    // Two sub-paths, picked by text_layout_:
    //   - SingleFile:    uses file_path_, handles CSV/JSON/TXT
    //   - CorpusSubdirs: uses folder_path_, handles folder/<class>/*.txt
    //
    // TextDataset's constructor auto-detects file vs directory from the
    // path itself, so the only thing differing between the two paths is
    // which field we pull the source from. The worker body runs the
    // tokenizer + vocab build, then calls RegisterTextDataset under its
    // own lock. PollAsyncLoadResult drains the result onto the UI thread
    // on the next frame.
    const bool text_single_ready =
        file_category_ == FileCategory::Text &&
        text_layout_ == TextLayout::SingleFile &&
        strlen(file_path_) > 0;
    const bool text_corpus_ready =
        file_category_ == FileCategory::Text &&
        text_layout_ == TextLayout::CorpusSubdirs &&
        strlen(folder_path_) > 0;
    if (source_type_ == SourceType::File &&
        (text_single_ready || text_corpus_ready)) {
        // Already loading — defensive against a double-click. Apply is
        // also greyed out via IsBusy() while is_loading_async_ is true.
        if (is_loading_async_) {
            spdlog::warn("DataInputDialog: Text Apply ignored - load already in progress");
            apply_in_progress_ = false;
            return;
        }

        // Pick the source path — file for SingleFile mode, folder for
        // corpus mode. TextDataset dispatches internally on is_directory.
        const std::string text_source_path =
            text_corpus_ready ? std::string(folder_path_)
                              : std::string(file_path_);
        auto& registry = cyxwiz::DataRegistry::Instance();

        // Capture OLD dataset_name for cleanup on re-Apply with new file.
        // We do this on the UI thread before launching the worker so the
        // registry doesn't temporarily hold two entries under different
        // names. UnregisterTextDataset/Tabular are fast (map erase), no
        // reason to offload them.
        std::string previous_dataset_name;
        {
            auto pit = node_->parameters.find("dataset_name");
            if (pit != node_->parameters.end()) {
                previous_dataset_name = pit->second;
            }
        }

        loaded_dataset_name_ = GenerateDatasetName();
        if (!previous_dataset_name.empty() &&
            previous_dataset_name != loaded_dataset_name_) {
            registry.UnregisterTextDataset(previous_dataset_name);
        }
        // Cross-category cleanup: if the user toggled from Tabular to
        // Text on the same file, clear the stale Tabular entry so
        // IsTabularDataset(name) doesn't mislead downstream code.
        // This is safe on the UI thread because it's touching a
        // DIFFERENT registry map than the one the async worker will
        // register into.
        registry.UnregisterTabularDataset(loaded_dataset_name_);
        // NOTE: We intentionally do NOT pre-clear the same-category
        // text entry here. `RegisterTextDataset` uses map[name]=entry
        // which atomically replaces any existing entry under the
        // same name, so a premature unregister just creates a window
        // where the registry is empty for the duration of the async
        // load — and if the user clicks Compile during that window,
        // the compile gate correctly reports "Data is not loaded"
        // (it's literally true at that moment). Let the worker do
        // the replacement.

        // Snapshot every dialog field the worker needs, by value. Never
        // capture references into the dialog — the dialog can be closed
        // while the worker is still running.
        std::string captured_source     = text_source_path;
        std::string captured_name       = loaded_dataset_name_;
        std::string captured_text_col   = text_column_;
        std::string captured_label_col  = text_label_column_;
        bool captured_has_labels        = text_corpus_ready ||
                                          (strlen(text_label_column_) > 0);
        int  captured_tok_type          = text_tokenizer_type_;
        int  captured_max_length        = text_max_length_;
        bool captured_lowercase         = text_lowercase_;
        int  captured_min_freq          = text_min_freq_;
        int  captured_max_vocab         = text_max_vocab_size_;

        auto state = std::make_shared<AsyncLoadState>();
        state->dataset_name = captured_name;
        state->source_path  = captured_source;
        async_load_state_   = state;
        is_loading_async_   = true;

        // Provisional node params: mark NOT loaded until the worker
        // finishes. Same reasoning as the Arrow path — a Train click
        // during load should be caught by the compile gate, not try to
        // train on a missing dataset.
        node_->parameters["dataset_name"] = captured_name;
        node_->parameters["data_loaded"] = "false";

        apply_status_message_ = std::string("Tokenizing ") +
            fs::path(captured_source).filename().string() + "...";
        apply_status_timer_ = 0.0f;  // hide stale status during load

        auto& mgr = cyxwiz::AsyncTaskManager::Instance();
        loading_task_id_ = mgr.RunAsync(
            "Loading text " + captured_name,
            [captured_source, captured_name, captured_text_col, captured_label_col,
             captured_has_labels, captured_tok_type, captured_max_length,
             captured_lowercase, captured_min_freq, captured_max_vocab, state]
            (cyxwiz::LambdaTask& task) {
                try {
                    task.ReportProgress(0.1f, "Building tokenizer");

                    cyxwiz::TextDatasetConfig probe_cfg;
                    probe_cfg.text_column  = captured_text_col;
                    probe_cfg.label_column = captured_label_col;
                    probe_cfg.has_labels   = captured_has_labels;
                    switch (captured_tok_type) {
                        case 0: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Whitespace; break;
                        case 2: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Character; break;
                        default: probe_cfg.tokenizer_type = cyxwiz::TokenizerType::Word; break;
                    }
                    probe_cfg.max_length     = captured_max_length;
                    probe_cfg.lowercase      = captured_lowercase;
                    probe_cfg.do_padding     = true;
                    probe_cfg.do_truncation  = true;
                    probe_cfg.min_word_freq  = captured_min_freq;
                    probe_cfg.max_vocab_size = captured_max_vocab;

                    cyxwiz::TextDataset probe(captured_source, probe_cfg);
                    auto info = probe.GetInfo();

                    task.ReportProgress(0.9f, "Registering dataset");

                    auto& reg = cyxwiz::DataRegistry::Instance();
                    cyxwiz::DataRegistry::TextDatasetEntry text_entry;
                    text_entry.source_path    = captured_source;
                    text_entry.text_column    = captured_text_col;
                    text_entry.label_column   = captured_label_col;
                    text_entry.has_labels     = captured_has_labels;
                    text_entry.tokenizer_type = captured_tok_type;
                    text_entry.max_length     = captured_max_length;
                    text_entry.lowercase      = captured_lowercase;
                    text_entry.do_padding     = true;
                    text_entry.do_truncation  = true;
                    text_entry.min_word_freq  = captured_min_freq;
                    text_entry.max_vocab_size = captured_max_vocab;
                    text_entry.num_samples    = info.num_samples;
                    text_entry.num_classes    = info.num_classes;
                    text_entry.class_names    = info.class_names;
                    text_entry.vocab_size     = probe.GetVocabSize();

                    reg.RegisterTextDataset(captured_name, text_entry);

                    state->success      = true;
                    state->backend      = 5;  // 5 = text in-memory
                    state->rows         = static_cast<int64_t>(info.num_samples);
                    state->cols         = 1;
                    size_t per_sample   = static_cast<size_t>(captured_max_length) * sizeof(float);
                    state->bytes        = info.num_samples * per_sample;
                    state->num_classes  = info.num_classes;
                    state->vocab_size   = probe.GetVocabSize();
                    state->message      = "Loaded " + std::to_string(info.num_samples) +
                                          " text samples (" + std::to_string(info.num_classes) +
                                          " classes, vocab " + std::to_string(probe.GetVocabSize()) + ")";
                } catch (const std::exception& e) {
                    state->success = false;
                    state->message = std::string("Error loading text: ") + e.what();
                    spdlog::error("DataInputDialog async text load: {}", state->message);
                }
                // Atomic done flag is the publish barrier — must be set
                // last so PollAsyncLoadResult only reads fully initialized
                // state.
                state->done.store(true);
            });

        spdlog::info("DataInputDialog: queued async text load (task {}, name '{}')",
                     loading_task_id_, captured_name);

        apply_in_progress_ = false;
        has_changes_ = false;
        return;
    } else if (source_type_ == SourceType::File && strlen(file_path_) > 0) {
        auto& registry = cyxwiz::DataRegistry::Instance();

        // Capture the OLD dataset_name (from the previous Apply, if any)
        // before we overwrite it. If the user re-Applies with a different
        // file the new auto-generated name will differ (it's based on the
        // file stem) and the old entry would otherwise leak in the
        // registry. LoadTabularCSV unregisters the *new* name on entry
        // but doesn't know about the old one.
        std::string previous_dataset_name;
        {
            auto pit = node_->parameters.find("dataset_name");
            if (pit != node_->parameters.end()) {
                previous_dataset_name = pit->second;
            }
        }

        loaded_dataset_name_ = GenerateDatasetName();
        if (!previous_dataset_name.empty() &&
            previous_dataset_name != loaded_dataset_name_) {
            registry.UnregisterTabularDataset(previous_dataset_name);
        }

        try {
            std::shared_ptr<cyxwiz::ArrowDataset> dataset;

            // Load based on detected file type
            const char* types[] = {"auto", "csv", "tsv", "json", "parquet", "excel", "hdf5", "feather", "arrow", "txt", "arff"};
            int type_idx = (detected_type_ >= 0 && detected_type_ < 11) ? detected_type_ : 0;
            std::string file_type = types[type_idx];

            if (file_type == "csv" || file_type == "tsv" || file_type == "txt" || file_type == "arff") {
                // === ASYNC CSV LOAD ===
                // The CSV path can take 10+ seconds when force_disk_backed
                // triggers a CSV->Parquet conversion, so it runs on a
                // worker thread. Everything below is the launcher; the
                // actual result is picked up later by PollAsyncLoadResult
                // running on the UI thread.
                if (is_loading_async_) {
                    // Already loading — ignore the second click. The Apply
                    // button is also greyed out via IsBusy(), so this is
                    // mostly defensive.
                    spdlog::warn("DataInputDialog: Apply ignored - load already in progress");
                    apply_in_progress_ = false;
                    return;
                }

                char delim = (file_type == "tsv") ? '\t' : custom_delimiter_[0];
                // max_rows_ from the Limit Rows tab: 0 means "all rows".
                int64_t cap = (max_rows_ > 0) ? static_cast<int64_t>(max_rows_) : 0;

                // Snapshot all inputs by value so the worker isn't racing
                // the UI thread on dialog state.
                std::string captured_path = file_path_;
                std::string captured_name = loaded_dataset_name_;
                bool captured_header = has_header_;
                int captured_skip = skip_rows_;
                int64_t captured_cap = cap;
                bool captured_force = force_disk_backed_;

                auto state = std::make_shared<AsyncLoadState>();
                state->dataset_name = captured_name;
                state->source_path = captured_path;
                async_load_state_ = state;
                is_loading_async_ = true;

                // Provisional node params: mark NOT loaded until the worker
                // finishes. This is important — if the user clicks Train
                // before the load completes, the compile gate sees
                // data_loaded=false and refuses, instead of trying to
                // train on a dataset that isn't there yet.
                node_->parameters["dataset_name"] = captured_name;
                node_->parameters["data_loaded"] = "false";

                apply_status_message_ = std::string("Loading ") +
                    fs::path(captured_path).filename().string() + "...";
                apply_status_timer_ = 0.0f;  // hide stale status during load

                auto& mgr = cyxwiz::AsyncTaskManager::Instance();
                loading_task_id_ = mgr.RunAsync(
                    "Loading " + captured_name,
                    [captured_path, captured_name, captured_header, delim,
                     captured_skip, captured_cap, captured_force, state]
                    (cyxwiz::LambdaTask& task) {
                        try {
                            task.ReportProgress(0.1f, "Reading CSV");

                            auto& reg = cyxwiz::DataRegistry::Instance();
                            auto backend = reg.LoadTabularCSV(
                                captured_path, captured_name,
                                captured_header, delim,
                                captured_skip, captured_cap,
                                captured_force);

                            task.ReportProgress(0.9f, "Finalizing");

                            if (backend == cyxwiz::DataRegistry::TabularLoadBackend::InMemory) {
                                auto ds = reg.GetArrowDataset(captured_name);
                                if (ds) {
                                    state->success = true;
                                    state->backend = 1;
                                    state->rows = ds->GetNumRows();
                                    state->cols = ds->GetNumColumns();
                                    state->bytes = ds->GetMemoryUsage();
                                    state->message = "Loaded in memory";
                                } else {
                                    state->success = false;
                                    state->message = "Load completed but dataset missing from registry";
                                }
                            } else if (backend == cyxwiz::DataRegistry::TabularLoadBackend::DiskBacked) {
                                auto pq = reg.GetParquetBackedDataset(captured_name);
                                if (pq) {
                                    state->success = true;
                                    state->backend = 2;
                                    state->rows = pq->GetNumRows();
                                    state->cols = pq->GetNumColumns();
                                    state->bytes = pq->GetMemoryUsage();
                                    state->message = "Loaded via Parquet cache";
                                } else {
                                    state->success = false;
                                    state->message = "Disk-backed load completed but dataset missing";
                                }
                            } else {
                                state->success = false;
                                state->message = "Failed to load CSV - check file format";
                            }
                        } catch (const std::exception& e) {
                            state->success = false;
                            state->message = std::string("Error: ") + e.what();
                            spdlog::error("DataInputDialog async load: {}", state->message);
                        }
                        // Atomic done flag is the publish barrier — must be
                        // set last so PollAsyncLoadResult only reads fully
                        // initialized state.
                        state->done.store(true);
                    });

                spdlog::info("DataInputDialog: queued async CSV load (task {}, name '{}')",
                             loading_task_id_, captured_name);

                // Reset the sync apply flag — we're done with the
                // synchronous metadata-write portion of Apply. The async
                // result will arrive via PollAsyncLoadResult.
                apply_in_progress_ = false;
                has_changes_ = false;
                return;
            } else if (file_type == "parquet") {
                dataset = registry.LoadParquetToArrow(file_path_, loaded_dataset_name_);
            } else if (file_type == "json") {
                dataset = registry.LoadJSONToArrow(file_path_, loaded_dataset_name_, json_lines_);
            } else if (file_type == "excel") {
                dataset = registry.LoadExcelToArrow(file_path_, loaded_dataset_name_, sheet_idx_);
            } else {
                // Auto-detect
                dataset = registry.LoadArrowTable(file_path_, loaded_dataset_name_);
            }

            if (dataset) {
                loaded_rows_ = dataset->GetNumRows();
                loaded_cols_ = dataset->GetNumColumns();
                loaded_memory_bytes_ = dataset->GetMemoryUsage();
                data_load_state_ = DataLoadState::InMemory;

                // Store in node parameters
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                node_->parameters["memory_bytes"] = std::to_string(loaded_memory_bytes_);
                node_->parameters["dataset_name"] = loaded_dataset_name_;
                node_->parameters["data_loaded"] = "true";

                // Update node description with stats
                fs::path p(file_path_);
                node_->description = p.filename().string() + "\n" +
                    std::to_string(loaded_rows_) + " rows, " +
                    std::to_string(loaded_cols_) + " cols";

                // Format status message
                apply_status_message_ = "Loaded " + p.filename().string() +
                    " (" + std::to_string(loaded_rows_) + " rows, " +
                    std::to_string(loaded_cols_) + " cols, " +
                    FormatBytes(loaded_memory_bytes_) + ")";

                apply_success_ = true;
                spdlog::info("DataInputDialog: {}", apply_status_message_);
            } else if (!apply_success_) {
                // Only treat a null Arrow `dataset` as a failure if no prior
                // success path set apply_success_ already. The disk-backed
                // Parquet branch returns without populating `dataset` (that
                // variable is Arrow-typed), and previously that fall-through
                // clobbered data_loaded back to "false", defeating the state
                // restore on dialog reopen. This guard keeps the Parquet
                // success state intact.
                apply_status_message_ = "Failed to load data - check file format";
                node_->parameters["data_loaded"] = "false";
            }
        } catch (const std::exception& e) {
            apply_status_message_ = std::string("Error: ") + e.what();
            node_->parameters["data_loaded"] = "false";
            spdlog::error("DataInputDialog: {}", apply_status_message_);
        }
    } else if (source_type_ == SourceType::File && strlen(folder_path_) > 0 &&
               file_category_ == FileCategory::Audio) {
        // Audio folder loading (Phase 2).
        //
        // Mirrors the image folder path: scan the folder via AudioDataset,
        // populate an AudioDatasetEntry with feature config from the dialog,
        // register it in DataRegistry. Training dispatch in
        // StartTrainingFromGraph routes IsAudioDataset to StartTrainingAudio
        // which builds an AudioDatasetBatcher around the entry.
        //
        // Two layouts are supported (mirroring images):
        //   - ClassSubdirs: folder/<class>/*.wav (drone dataset)
        //   - FlatWithCSV:  folder/*.flac + labels.csv (call center dataset)
        // Layout is picked by the user via the audio_layout_ combo.
        //
        // Feature extraction happens lazily inside AudioDataset::GetItem at
        // training time via libsndfile + FFTW3, NOT here. We just record
        // metadata so the registry has a handle for the dispatch to find.
        auto& registry = cyxwiz::DataRegistry::Instance();
        loaded_dataset_name_ = GenerateDatasetName();
        registry.UnregisterTabularDataset(loaded_dataset_name_);
        registry.UnregisterAudioDataset(loaded_dataset_name_);

        try {
            // FlatWithCSV requires an actual CSV path. Fail fast with a
            // clear error rather than constructing AudioDataset and getting
            // a confusing "no audio files" message.
            if (audio_layout_ == AudioLayout::FlatWithCSV && strlen(audio_labels_csv_) == 0) {
                apply_status_message_ = "Flat folder layout requires a Labels CSV - "
                    "pick one or switch to 'Class subdirectories'";
                node_->parameters["data_loaded"] = "false";
                spdlog::error("DataInputDialog: audio FlatWithCSV selected but labels_csv is empty");
                apply_in_progress_ = false;
                apply_status_timer_ = 6.0f;
                has_changes_ = false;
                return;
            }

            // Build the entry from the dialog state. Defaults match what the
            // RenderAudioOptions tab exposes; advanced fields (n_fft etc.)
            // stay at sensible defaults until users ask for them.
            cyxwiz::DataRegistry::AudioDatasetEntry audio_entry;
            audio_entry.folder_path = folder_path_;
            audio_entry.labeled_subdirs = (audio_layout_ == AudioLayout::ClassSubdirs);
            if (audio_layout_ == AudioLayout::FlatWithCSV) {
                audio_entry.csv_path = audio_labels_csv_;
                audio_entry.filename_col = audio_filename_col_;  // "" = auto
                audio_entry.label_col = audio_label_col_;        // "" = auto
            }
            audio_entry.feature_type = 1;          // MelSpectrogram default
            audio_entry.target_sr = sample_rate_;
            audio_entry.max_duration = (duration_sec_ > 0.0f) ? duration_sec_ : 5.0f;
            audio_entry.n_fft = 512;
            audio_entry.hop_length = 256;
            audio_entry.n_mels = 128;
            audio_entry.n_mfcc = 13;

            // Probe the folder by constructing an AudioDataset once.
            // This validates the folder structure and surfaces a clear
            // error before training.
            cyxwiz::AudioDatasetConfig probe_cfg;
            probe_cfg.feature_type = cyxwiz::AudioDatasetConfig::FeatureType::MelSpectrogram;
            probe_cfg.target_sr = audio_entry.target_sr;
            probe_cfg.max_duration = audio_entry.max_duration;
            probe_cfg.labeled_subdirs = audio_entry.labeled_subdirs;
            probe_cfg.n_fft = audio_entry.n_fft;
            probe_cfg.hop_length = audio_entry.hop_length;
            probe_cfg.n_mels = audio_entry.n_mels;
            probe_cfg.csv_path = audio_entry.csv_path;
            probe_cfg.filename_col = audio_entry.filename_col;
            probe_cfg.label_col = audio_entry.label_col;

            cyxwiz::AudioDataset probe(folder_path_, probe_cfg);
            auto info = probe.GetInfo();
            audio_entry.num_samples = info.num_samples;
            audio_entry.num_classes = info.num_classes;
            audio_entry.class_names = probe.GetClassNames();
            // Persist the actual probed feature shape so dialog reopen
            // and Memory tab display don't have to re-probe.
            if (info.shape.size() >= 2) {
                audio_entry.feature_rows = static_cast<int>(info.shape[0]);
                audio_entry.feature_cols = static_cast<int>(info.shape[1]);
            }

            registry.RegisterAudioDataset(loaded_dataset_name_, audio_entry);

            loaded_rows_ = static_cast<int64_t>(info.num_samples);
            loaded_cols_ = 1;
            // Estimate the if-fully-cached size so the Memory tab shows
            // a real number instead of 0 B. Audio is lazy-loaded per
            // sample at training time, so this is upper-bound, not actual.
            // Per sample: feature_rows × feature_cols × float (4 B).
            size_t per_sample = (audio_entry.feature_rows > 0 && audio_entry.feature_cols > 0)
                ? static_cast<size_t>(audio_entry.feature_rows) *
                  static_cast<size_t>(audio_entry.feature_cols) * sizeof(float)
                : static_cast<size_t>(audio_entry.n_mels) * 313 * sizeof(float);
            loaded_memory_bytes_ = info.num_samples * per_sample;
            loaded_memory_is_estimate_ = true;
            loaded_backend_ = 4;  // 4 = audio
            data_load_state_ = DataLoadState::InMemory;

            node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
            node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
            node_->parameters["dataset_name"] = loaded_dataset_name_;
            node_->parameters["data_loaded"] = "true";

            fs::path p(folder_path_);
            node_->description = p.filename().string() + "\n" +
                std::to_string(loaded_rows_) + " audio files, " +
                std::to_string(info.num_classes) + " classes";

            apply_status_message_ = "Loaded " + std::to_string(loaded_rows_) +
                " audio files (" + std::to_string(info.num_classes) +
                " classes) from " + p.filename().string();
            apply_success_ = true;
            spdlog::info("DataInputDialog: {}", apply_status_message_);
        } catch (const std::exception& e) {
            apply_status_message_ = std::string("Error loading audio folder: ") + e.what();
            node_->parameters["data_loaded"] = "false";
            apply_success_ = false;
            spdlog::error("DataInputDialog: {}", apply_status_message_);
        }
    } else if (source_type_ == SourceType::File && strlen(folder_path_) > 0) {
        // Image folder loading.
        //
        // Phase 0: routes through the LEGACY DatasetHandle path
        // (LoadImageFolder) which uses ImageFolderDataset and loads real
        // pixel data via an LRU cache. Training dispatch then falls through
        // to StartTraining (legacy) for this dataset, since it's in
        // datasets_ not arrow_datasets_ / parquet_backed_datasets_.
        //
        // We deliberately do NOT call LoadImageFolderToArrow here. That
        // loader only stores metadata (file_path as string + label_id as
        // int32), which caused a crash at training time because the Arrow
        // batcher skipped string columns and the model got a tensor with 1
        // feature instead of H*W*C. LoadImageFolderToArrow is still in the
        // registry for Data Studio pipeline-executor use, which needs a
        // tabular view of the folder metadata — that use case is fine.
        //
        // Phase 1 replaces this legacy route with a proper ImageDatasetBatcher
        // + image_datasets_ registry map and a node-based preprocessing
        // pipeline. See multi_format_data_pipeline_design.md.
        auto& registry = cyxwiz::DataRegistry::Instance();
        loaded_dataset_name_ = GenerateDatasetName();

        // Clear any stale Arrow / Parquet entry under the same name so the
        // training dispatch (which checks Arrow first) correctly falls
        // through to the legacy path for this dataset.
        registry.UnregisterTabularDataset(loaded_dataset_name_);

        try {
            // Dispatch to the loader matching the user's explicit layout
            // choice from the ImageLayout combo in RenderImageOptions.
            // No auto-detect — what the user picks is what we load.
            // Adding a new layout is an entry in kImageLayoutOptions + a
            // case in this switch.
            //
            // Note: LoadImageCSV bakes in a 224x224 target size for now.
            // Phase 1 moves the target size to a dedicated Resize node
            // so the user can pick any size declaratively.
            cyxwiz::DatasetHandle handle;
            switch (image_layout_) {
                case ImageLayout::FlatWithCSV:
                    if (strlen(labels_csv_) == 0) {
                        apply_status_message_ = "Flat folder layout requires a Labels CSV - "
                            "pick one or switch to 'Class subdirectories'";
                        node_->parameters["data_loaded"] = "false";
                        spdlog::error("DataInputDialog: FlatWithCSV selected but labels_csv is empty");
                        apply_in_progress_ = false;
                        apply_status_timer_ = 6.0f;
                        has_changes_ = false;
                        return;
                    }
                    handle = registry.LoadImageCSV(folder_path_, labels_csv_, loaded_dataset_name_);
                    spdlog::info("DataInputDialog: layout=FlatWithCSV, folder='{}', csv='{}'",
                                 folder_path_, labels_csv_);
                    break;

                case ImageLayout::ClassSubdirs:
                default:
                    handle = registry.LoadImageFolder(folder_path_, loaded_dataset_name_);
                    spdlog::info("DataInputDialog: layout=ClassSubdirs, folder='{}'",
                                 folder_path_);
                    break;
            }

            if (handle.IsValid()) {
                auto info = handle.GetInfo();
                loaded_dataset_name_ = info.name;  // may have been uniquified
                loaded_rows_ = static_cast<int64_t>(info.num_samples);
                loaded_cols_ = 1;
                // Estimate if-fully-cached size so the Memory tab shows
                // a useful number for this lazy-loaded image dataset.
                // Default to 224x224x3 RGB float since the actual target
                // size comes from the graph's Resize node and isn't known
                // here. Marks as estimate so the UI labels it correctly.
                size_t per_image = 224ull * 224 * 3 * sizeof(float);
                loaded_memory_bytes_ = info.num_samples * per_image;
                loaded_memory_is_estimate_ = true;
                data_load_state_ = DataLoadState::InMemory;

                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                node_->parameters["dataset_name"] = loaded_dataset_name_;
                node_->parameters["data_loaded"] = "true";

                // Register in the new image_datasets_ map so the Phase 1
                // training dispatch (IsImageDataset) finds it and routes
                // to ImageDatasetBatcher instead of the legacy path.
                cyxwiz::DataRegistry::ImageDatasetEntry img_entry;
                img_entry.folder_path = folder_path_;
                img_entry.labels_csv = labels_csv_;
                img_entry.layout = static_cast<int>(image_layout_);
                img_entry.num_images = info.num_samples;
                img_entry.num_classes = info.num_classes;
                img_entry.class_names = info.class_names;
                registry.RegisterImageDataset(loaded_dataset_name_, img_entry);

                fs::path p(folder_path_);
                node_->description = p.filename().string() + "\n" +
                    std::to_string(loaded_rows_) + " images, " +
                    std::to_string(info.num_classes) + " classes";

                apply_status_message_ = "Loaded " + std::to_string(loaded_rows_) +
                    " images (" + std::to_string(info.num_classes) + " classes) from " +
                    p.filename().string();
                apply_success_ = true;
                spdlog::info("DataInputDialog: {}", apply_status_message_);
            } else {
                apply_status_message_ = "Failed to load image folder - "
                    "expected either class subdirectories (cats/, dogs/, ...) "
                    "OR a flat folder with a Labels CSV selected above";
                node_->parameters["data_loaded"] = "false";
                spdlog::error("DataInputDialog: image folder load returned invalid handle for '{}' "
                              "(labels_csv='{}')",
                              folder_path_, labels_csv_);
            }
        } catch (const std::exception& e) {
            apply_status_message_ = std::string("Error: ") + e.what();
            node_->parameters["data_loaded"] = "false";
            spdlog::error("DataInputDialog: {}", apply_status_message_);
        }
    }

    apply_in_progress_ = false;
    apply_status_timer_ = 5.0f;  // Show for 5 seconds

    has_changes_ = false;
    spdlog::info("DataInputDialog: Applied settings");
}

void DataInputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    preview_loaded_ = false;
    has_changes_ = false;
}

void DataInputDialog::PollAsyncLoadResult() {
    // Cheap fast path: nothing pending.
    if (!is_loading_async_ || !async_load_state_) return;

    // Worker hasn't published yet — keep showing the loading UI.
    if (!async_load_state_->done.load()) return;

    // === Worker finished. Drain the result onto the UI thread. ===
    auto state = async_load_state_;
    async_load_state_.reset();
    is_loading_async_ = false;
    loading_task_id_ = 0;

    if (!node_) return;

    if (state->success) {
        loaded_rows_ = state->rows;
        loaded_cols_ = state->cols;
        loaded_memory_bytes_ = state->bytes;
        loaded_backend_ = state->backend;
        loaded_dataset_name_ = state->dataset_name;
        data_load_state_ = DataLoadState::InMemory;
        apply_success_ = true;

        node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
        node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
        node_->parameters["memory_bytes"] = std::to_string(loaded_memory_bytes_);
        node_->parameters["dataset_name"] = loaded_dataset_name_;
        node_->parameters["data_loaded"] = "true";

        fs::path p(state->source_path);
        if (state->backend == 5) {
            // Text load: rows = num_samples, cols = 1, plus class /
            // vocab metadata. Description format differs from tabular
            // because the interesting knobs are different.
            loaded_memory_is_estimate_ = true;
            node_->description = p.filename().string() + "\n" +
                std::to_string(loaded_rows_) + " samples, " +
                std::to_string(state->num_classes) + " classes, vocab " +
                std::to_string(state->vocab_size);
            apply_status_message_ = state->message.empty()
                ? std::string("Loaded text from ") + p.filename().string()
                : state->message;
        } else {
            std::string backing_suffix = (loaded_backend_ == 2) ? " (disk-backed)" : "";
            node_->description = p.filename().string() + "\n" +
                std::to_string(loaded_rows_) + " rows, " +
                std::to_string(loaded_cols_) + " cols" + backing_suffix;

            std::string size_label = (loaded_backend_ == 2) ? " on disk" : "";
            apply_status_message_ = "Loaded " + p.filename().string() +
                (loaded_backend_ == 2 ? " via Parquet cache (" : " (") +
                std::to_string(loaded_rows_) + " rows, " +
                std::to_string(loaded_cols_) + " cols, " +
                FormatBytes(loaded_memory_bytes_) + size_label + ")";
        }

        spdlog::info("DataInputDialog: async load complete - {}", apply_status_message_);
    } else {
        apply_success_ = false;
        loaded_backend_ = 0;
        node_->parameters["data_loaded"] = "false";
        apply_status_message_ = state->message.empty()
            ? std::string("Failed to load data")
            : state->message;
        spdlog::error("DataInputDialog: async load failed - {}", apply_status_message_);
    }

    apply_status_timer_ = 5.0f;  // 5 second fade-out
    apply_in_progress_ = false;
}

void DataInputDialog::RenderContent() {
    // Pick up the async load result, if the worker has finished, and apply
    // it to the dialog/node state. Cheap when no load is in flight.
    PollAsyncLoadResult();

    // KNIME-style tab bar at TOP based on source type
    if (source_type_ == SourceType::File) {
        if (ImGui::BeginTabBar("DataInputTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Settings")) {
                RenderFileSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Data Profiling")) {
                RenderDataProfilingTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Transformation")) {
                RenderTransformationTab();
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
            if (ImGui::BeginTabItem("Memory")) {
                RenderMemoryTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::MLDataset) {
        if (ImGui::BeginTabBar("MLDatasetTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Dataset")) {
                RenderMLDatasetSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Options")) {
                RenderMLDatasetOptions();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::Database) {
        if (ImGui::BeginTabBar("DatabaseTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Connection")) {
                RenderDatabaseSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Query")) {
                RenderSQLQuery();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    else if (source_type_ == SourceType::Cloud) {
        RenderCloudSource();
    }

    // === Status bar (Apply feedback) ===
    // Three states:
    //   1. is_loading_async_ -> indeterminate progress bar + ticking dots
    //   2. apply_status_timer_ > 0 -> fade-out success/error from a finished
    //      apply (sync OR async)
    //   3. otherwise -> nothing
    if (is_loading_async_) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImVec4 loading_color(0.3f, 0.6f, 1.0f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, loading_color);

        // Animated dots: 1-3 dots, ~2 cycles per second.
        loading_anim_phase_ += ImGui::GetIO().DeltaTime * 2.0f;
        int dots = 1 + (static_cast<int>(loading_anim_phase_) % 3);
        std::string anim_dots(dots, '.');
        std::string label = apply_status_message_.empty()
            ? "Loading"
            : apply_status_message_;
        // Trim a trailing "..." from the message so we don't end up
        // with five dots.
        while (!label.empty() && label.back() == '.') label.pop_back();
        ImGui::Text("%s%s", label.c_str(), anim_dots.c_str());

        ImGui::PopStyleColor();

        // Indeterminate-ish progress bar driven by the same phase.
        float bar_phase = std::fmod(loading_anim_phase_ * 0.5f, 1.0f);
        ImGui::ProgressBar(bar_phase, ImVec2(-1, 6.0f), "");

        ImGui::TextDisabled("This dialog is non-blocking — you can close it; "
                            "the load continues in the background.");
    } else if (apply_status_timer_ > 0.0f) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Fade out effect
        float alpha = std::min(1.0f, apply_status_timer_);
        ImVec4 color = apply_success_
            ? ImVec4(0.2f, 0.8f, 0.2f, alpha)  // Green for success
            : ImVec4(0.9f, 0.3f, 0.3f, alpha); // Red for error

        ImGui::PushStyleColor(ImGuiCol_Text, color);

        // Icon
        if (apply_success_) {
            ImGui::TextUnformatted("\xE2\x9C\x93");  // Checkmark (UTF-8)
        } else {
            ImGui::TextUnformatted("\xE2\x9C\x97");  // X mark (UTF-8)
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(apply_status_message_.c_str());

        ImGui::PopStyleColor();

        // Decrease timer (assuming ~60fps, subtract per frame)
        apply_status_timer_ -= ImGui::GetIO().DeltaTime;
    }

    // Preview section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderPreviewPanel();

    // Source type selector at BOTTOM
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderSourceSelector();
}

void DataInputDialog::RenderSourceSelector() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATA SOURCE");
    ImGui::Spacing();

    // Source type radio buttons in a row
    int source_idx = static_cast<int>(source_type_);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(20, 4));
    if (ImGui::RadioButton("File", &source_idx, 0)) {
        source_type_ = SourceType::File;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("ML Dataset", &source_idx, 1)) {
        source_type_ = SourceType::MLDataset;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Database", &source_idx, 2)) {
        source_type_ = SourceType::Database;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Cloud", &source_idx, 3)) {
        source_type_ = SourceType::Cloud;
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::PopStyleVar();
}

void DataInputDialog::RenderFileSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    ImVec4 info_color = style.Colors[ImGuiCol_TextDisabled];

    // File category selector
    ImGui::TextColored(accent, "File Type");
    ImGui::Spacing();

    int cat_idx = static_cast<int>(file_category_);
    if (ImGui::RadioButton("Tabular", &cat_idx, 0)) {
        file_category_ = FileCategory::Tabular;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Image", &cat_idx, 1)) {
        file_category_ = FileCategory::Image;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Audio", &cat_idx, 2)) {
        file_category_ = FileCategory::Audio;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Video", &cat_idx, 3)) {
        file_category_ = FileCategory::Video;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Text", &cat_idx, 4)) {
        file_category_ = FileCategory::Text;
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Time Series", &cat_idx, 5)) {
        file_category_ = FileCategory::TimeSeries;
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Single-file path section — shown for Tabular and for Text in
    // SingleFile layout mode. In CorpusSubdirs mode Text uses a folder
    // picker inside RenderTextOptions instead (mirroring image/audio).
    // Image uses its own folder/CSV inputs in RenderImageOptions.
    // Audio/Video are not yet supported and show their own messages.
    // Hiding this for other categories prevents the user from picking
    // a .jpg in a file dialog that's meant for CSVs.
    const bool show_file_picker =
        (file_category_ == FileCategory::Tabular) ||
        (file_category_ == FileCategory::TimeSeries) ||
        (file_category_ == FileCategory::Text &&
         text_layout_ == TextLayout::SingleFile);
    if (show_file_picker) {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.FrameRounding);
        if (ImGui::BeginChild("FileSelection", ImVec2(0, 100), true)) {
            ImGui::TextColored(accent, "Input File");
            ImGui::Separator();
            ImGui::Spacing();

            // File path input
            ImGui::Text("Path:");
            ImGui::SameLine(60);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 6));
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
            if (ImGui::InputText("##filepath", file_path_, sizeof(file_path_))) {
                DetectFileType();
                DetectFileCategory();
                LoadColumnList();
                has_changes_ = true;
                preview_loaded_ = false;
            }
            ImGui::PopStyleVar();
            ImGui::SameLine();
            if (ImGui::Button("Browse...", ImVec2(80, 0))) {
                BrowseFile();
            }

            // File info
            if (strlen(file_path_) > 0) {
                ImGui::TextColored(info_color, "Type: %s", GetFileTypeName());
                ImGui::SameLine(150);
                if (file_size_ > 0) {
                    if (file_size_ >= 1024 * 1024)
                        ImGui::TextColored(info_color, "Size: %.1f MB", file_size_ / (1024.0f * 1024.0f));
                    else if (file_size_ >= 1024)
                        ImGui::TextColored(info_color, "Size: %.1f KB", file_size_ / 1024.0f);
                    else
                        ImGui::TextColored(info_color, "Size: %zu B", file_size_);
                }
            } else {
                ImGui::TextDisabled("No file selected - click Browse to select a data file");
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    ImGui::Spacing();

    // Category-specific options
    switch (file_category_) {
        case FileCategory::Tabular:    RenderTabularOptions(); break;
        case FileCategory::Image:      RenderImageOptions(); break;
        case FileCategory::Audio:      RenderAudioOptions(); break;
        case FileCategory::Video:      RenderVideoOptions(); break;
        case FileCategory::Text:       RenderTextOptions(); break;
        // Phase 4: Time Series shares the Tabular loader entirely — the
        // only semantic difference is the file_category="timeseries" stamp
        // on the DataInput node, which the downstream TimeSeriesWindow
        // operator uses as a sanity hint. All format/delimiter/column
        // config is identical to Tabular.
        case FileCategory::TimeSeries: RenderTabularOptions(); break;
    }
}

void DataInputDialog::RenderTabularOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    if (ImGui::CollapsingHeader("Format Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Format auto-detect or manual selection
        ImGui::Text("Format:");
        ImGui::SameLine(100);
        const char* formats[] = {"Auto", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5", "Feather", "Arrow", "TXT", "ARFF"};
        ImGui::SetNextItemWidth(120);
        if (ImGui::Combo("##format", &detected_type_, formats, 11)) {
            has_changes_ = true;
            preview_loaded_ = false;
        }

        // CSV/TSV/TXT/ARFF specific options (delimiter-based)
        if (detected_type_ <= 2 || detected_type_ == 9 || detected_type_ == 10) {
            ImGui::Spacing();

            ImGui::Text("Delimiter:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(50);
            if (ImGui::InputText("##delim", custom_delimiter_, sizeof(custom_delimiter_))) {
                has_changes_ = true;
                preview_loaded_ = false;
            }

            ImGui::SameLine(180);
            if (ImGui::Checkbox("Has header", &has_header_)) {
                has_changes_ = true;
                preview_loaded_ = false;
            }

            ImGui::Text("Quote char:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(30);
            ImGui::InputText("##quote", quote_char_, sizeof(quote_char_));
        }
        // Excel specific
        else if (detected_type_ == 5) {
            ImGui::Text("Sheet index:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(60);
            ImGui::InputInt("##sheet", &sheet_idx_);

            ImGui::Text("Cell range:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(100);
            ImGui::InputText("##range", sheet_range_, sizeof(sheet_range_));
            ImGui::SameLine();
            ImGui::TextDisabled("(e.g., A1:D100)");
        }
        // HDF5 specific
        else if (detected_type_ == 6) {
            ImGui::Text("Dataset key:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(150);
            ImGui::InputText("##hdf5key", hdf5_dataset_, sizeof(hdf5_dataset_));
        }
        // JSON specific
        else if (detected_type_ == 3) {
            if (ImGui::Checkbox("JSON Lines format", &json_lines_)) {
                has_changes_ = true;
            }
            ImGui::Text("JSON Path:");
            ImGui::SameLine(100);
            ImGui::SetNextItemWidth(150);
            ImGui::InputText("##jsonpath", json_path_, sizeof(json_path_));
            ImGui::SameLine();
            ImGui::TextDisabled("($.data.rows)");
        }
    }

    // Column Mapping - Label/Target column selection
    if (ImGui::CollapsingHeader("Column Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
            "For ML training, specify which column contains labels/targets");
        ImGui::Spacing();

        // Label column selector
        ImGui::Text("Label Column:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(200);

        // Build combo items from available columns
        const char* preview = (label_column_idx_ >= 0 && label_column_idx_ < static_cast<int>(available_columns_.size()))
            ? available_columns_[label_column_idx_].c_str()
            : "(None - select label column)";

        if (ImGui::BeginCombo("##labelcol", preview)) {
            // Option for no label
            if (ImGui::Selectable("(None)", label_column_idx_ == -1)) {
                label_column_idx_ = -1;
                has_changes_ = true;
            }

            // List available columns
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                bool is_selected = (label_column_idx_ == i);
                if (ImGui::Selectable(available_columns_[i].c_str(), is_selected)) {
                    label_column_idx_ = i;
                    has_changes_ = true;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        // Hint for common label column names
        if (label_column_idx_ < 0 && !available_columns_.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "!");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Select the target column for training.\nCommon names: 'label', 'class', 'target', 'y'");
            }
        }

        // Auto-detect common label column names
        if (label_column_idx_ < 0 && !available_columns_.empty()) {
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                std::string lower = available_columns_[i];
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower == "label" || lower == "labels" || lower == "class" || lower == "target" || lower == "y") {
                    // Found a common label column - suggest it
                    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f),
                        "Suggestion: '%s' looks like a label column", available_columns_[i].c_str());
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Use")) {
                        label_column_idx_ = i;
                        has_changes_ = true;
                    }
                    break;
                }
            }
        }

        ImGui::Spacing();
        ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
            "Features: All other columns will be used as input features");
    }

}

void DataInputDialog::RenderImageOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Image Loading Options");
    ImGui::Spacing();

    // Image dataset layout options. Each entry describes a supported
    // on-disk layout, the combo-box display name, a one-line description,
    // and whether the user needs to pick a Labels CSV. Adding a new
    // layout for Phase 1 means adding one entry here and one case in
    // Apply's dispatch — nothing else in the dialog changes. Kept as a
    // function-local static so we don't need to expose the private
    // ImageLayout enum at file scope.
    struct ImageLayoutOption {
        ImageLayout value;
        const char* display;
        const char* description;
        bool needs_labels_csv;
    };
    static const ImageLayoutOption kImageLayoutOptions[] = {
        {ImageLayout::ClassSubdirs,
         "Class subdirectories (ImageNet-style)",
         "One subfolder per class; folder name becomes the label. "
         "E.g. root/cat/*.jpg, root/dog/*.jpg",
         false},
        {ImageLayout::FlatWithCSV,
         "Flat folder + CSV labels (Kaggle-style)",
         "All images in one folder; a CSV file maps each filename to its label. "
         "E.g. root/img1.jpg + labels.csv",
         true},
        // Phase 1 extends here: Unlabeled, COCO, YOLO, JSONLabels, HDF5Labels.
    };
    constexpr int kImageLayoutOptionsCount =
        static_cast<int>(sizeof(kImageLayoutOptions) / sizeof(kImageLayoutOptions[0]));

    // === Layout strategy combo ===
    // The user explicitly picks how their image dataset is laid out on
    // disk. No auto-detect magic — what they pick is what Apply dispatches
    // to. Adding new layouts is a one-line entry in kImageLayoutOptions.
    int current_idx = 0;
    for (int i = 0; i < kImageLayoutOptionsCount; ++i) {
        if (kImageLayoutOptions[i].value == image_layout_) {
            current_idx = i;
            break;
        }
    }

    ImGui::Text("Dataset layout:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10);
    if (ImGui::BeginCombo("##imglayout", kImageLayoutOptions[current_idx].display)) {
        for (int i = 0; i < kImageLayoutOptionsCount; ++i) {
            bool selected = (i == current_idx);
            if (ImGui::Selectable(kImageLayoutOptions[i].display, selected)) {
                image_layout_ = kImageLayoutOptions[i].value;
                has_changes_ = true;
                preview_loaded_ = false;
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    // Description line below the combo explains what this layout means.
    ImGui::TextDisabled("   %s", kImageLayoutOptions[current_idx].description);

    ImGui::Spacing();

    // === Folder input (required for all layouts) ===
    ImGui::Text("Folder:");
    ImGui::SameLine(130);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    if (ImGui::InputText("##imgfolder", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##imgbrowse", ImVec2(80, 0))) {
        BrowseFolder();
    }

    // === Labels CSV input (only shown for layouts that need it) ===
    const bool needs_csv = kImageLayoutOptions[current_idx].needs_labels_csv;
    if (needs_csv) {
        ImGui::Text("Labels CSV:");
        ImGui::SameLine(130);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##labelscsv", labels_csv_, sizeof(labels_csv_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse##csvbrowse", ImVec2(80, 0))) {
#ifdef _WIN32
            OPENFILENAMEA ofn = {};
            char file[512] = {};
            strncpy(file, labels_csv_, sizeof(file) - 1);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFilter = "CSV Files\0*.csv\0All Files\0*.*\0";
            ofn.lpstrFile = file;
            ofn.nMaxFile = sizeof(file);
            ofn.lpstrTitle = "Select Labels CSV";
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
            if (GetOpenFileNameA(&ofn)) {
                strncpy(labels_csv_, file, sizeof(labels_csv_) - 1);
                has_changes_ = true;
            }
#else
            spdlog::warn("Labels CSV file browser not implemented for this platform");
#endif
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Resize options
    if (ImGui::CollapsingHeader("Resize", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Target size:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##width", &target_width_);
        ImGui::SameLine();
        ImGui::Text("x");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##height", &target_height_);

        // Preset sizes
        ImGui::Text("Presets:");
        ImGui::SameLine(100);
        if (ImGui::Button("224x224")) { target_width_ = target_height_ = 224; has_changes_ = true; }
        ImGui::SameLine();
        if (ImGui::Button("256x256")) { target_width_ = target_height_ = 256; has_changes_ = true; }
        ImGui::SameLine();
        if (ImGui::Button("512x512")) { target_width_ = target_height_ = 512; has_changes_ = true; }
    }

    if (ImGui::CollapsingHeader("Normalization")) {
        if (ImGui::Checkbox("Normalize to [0, 1]", &normalize_images_)) {
            has_changes_ = true;
        }
        if (ImGui::Checkbox("Convert to RGB", &rgb_mode_)) {
            has_changes_ = true;
        }
    }
}

void DataInputDialog::RenderAudioOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Audio Loading Options");
    ImGui::Spacing();

    // Layout selector — ClassSubdirs vs FlatWithCSV. Mirrors the image dialog.
    static const char* kAudioLayoutNames[] = {
        "Class subdirectories (folder/<class>/*.wav)",
        "Flat folder + Labels CSV"
    };
    int layout_idx = static_cast<int>(audio_layout_);
    ImGui::Text("Layout:");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##audiolayout", &layout_idx, kAudioLayoutNames, IM_ARRAYSIZE(kAudioLayoutNames))) {
        audio_layout_ = static_cast<AudioLayout>(layout_idx);
        has_changes_ = true;
    }

    ImGui::Spacing();

    // Folder picker — same field as image folder. The Apply path
    // dispatches by file_category so wires don't cross.
    ImGui::Text("Folder:");
    ImGui::SetNextItemWidth(-80.0f);
    if (ImGui::InputText("##audiofolder", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##audio")) {
        BrowseFolder();
    }

    // FlatWithCSV: show the CSV picker + optional column overrides
    if (audio_layout_ == AudioLayout::FlatWithCSV) {
        ImGui::TextDisabled("Flat folder layout: all audio files in one directory, labels in CSV.");

        ImGui::Spacing();
        ImGui::Text("Labels CSV:");
        ImGui::SetNextItemWidth(-80.0f);
        if (ImGui::InputText("##audiocsv", audio_labels_csv_, sizeof(audio_labels_csv_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse##audiocsv")) {
            // Reuse the same file browser as everything else
            BrowseFile();
            // BrowseFile writes to file_path_; copy it to the CSV field
            if (strlen(file_path_) > 0) {
                strncpy(audio_labels_csv_, file_path_,
                        sizeof(audio_labels_csv_) - 1);
                audio_labels_csv_[sizeof(audio_labels_csv_) - 1] = '\0';
            }
        }

        ImGui::Spacing();
        ImGui::TextDisabled("Column auto-detect: leave blank to use header names like "
                            "filename / file / id / call_id and label / class / category. "
                            "Override below if your CSV uses unusual column names.");

        ImGui::Text("Filename col:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiofnamecol", audio_filename_col_,
                             sizeof(audio_filename_col_))) {
            has_changes_ = true;
        }

        ImGui::Text("Label col:");
        ImGui::SameLine(120);
        ImGui::SetNextItemWidth(180);
        if (ImGui::InputText("##audiolabelcol", audio_label_col_,
                             sizeof(audio_label_col_))) {
            has_changes_ = true;
        }
    } else {
        ImGui::TextDisabled("Class subdirectories layout: folder/class_a/*.wav, folder/class_b/*.wav");
    }

    // Quick scan preview if folder exists
    if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
        if (audio_layout_ == AudioLayout::ClassSubdirs) {
            int subdir_count = 0;
            int audio_file_count = 0;
            try {
                for (const auto& entry : fs::directory_iterator(folder_path_)) {
                    if (entry.is_directory()) {
                        subdir_count++;
                        for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                            if (sub.is_regular_file()) {
                                std::string ext = sub.path().extension().string();
                                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                                if (ext == ".wav" || ext == ".flac" || ext == ".ogg" ||
                                    ext == ".aiff" || ext == ".aif" || ext == ".mp3") {
                                    audio_file_count++;
                                }
                            }
                        }
                    }
                }
            } catch (...) { /* permission error etc., ignore */ }
            if (subdir_count > 0) {
                ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                                   "Detected: %d classes, %d audio files",
                                   subdir_count, audio_file_count);
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f),
                                   "No class subdirectories found");
            }
        } else {
            // Flat folder count
            int flat_count = 0;
            try {
                for (const auto& entry : fs::recursive_directory_iterator(folder_path_)) {
                    if (entry.is_regular_file()) {
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".wav" || ext == ".flac" || ext == ".ogg" ||
                            ext == ".aiff" || ext == ".aif" || ext == ".mp3") {
                            flat_count++;
                        }
                    }
                }
            } catch (...) { }
            ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                               "Detected: %d audio files in folder tree", flat_count);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Sample rate:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##samplerate", &sample_rate_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Hz");

    ImGui::Text("Max duration:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputFloat("##duration", &duration_sec_, 0.5f, 1.0f, "%.1f")) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("seconds (pad/truncate)");

    if (ImGui::Checkbox("Convert to mono", &mono_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Feature extraction defaults to MelSpectrogram (n_mels=128, n_fft=512).");
    ImGui::TextDisabled("Spectrogram / MFCC nodes will override this in a later phase.");
    ImGui::TextDisabled("Supported: WAV, FLAC, OGG, AIFF");
}

void DataInputDialog::RenderVideoOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Video Loading Options");
    ImGui::Spacing();

    static int frame_extraction = 0;
    ImGui::RadioButton("Extract all frames", &frame_extraction, 0);
    ImGui::RadioButton("Extract N frames", &frame_extraction, 1);
    ImGui::RadioButton("Sample at FPS", &frame_extraction, 2);

    if (frame_extraction == 1) {
        static int n_frames = 16;
        ImGui::Text("Frames:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##nframes", &n_frames);
    } else if (frame_extraction == 2) {
        static float fps = 1.0f;
        ImGui::Text("FPS:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(80);
        ImGui::InputFloat("##fps", &fps, 0.1f, 1.0f, "%.1f");
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: MP4, AVI, MOV, WebM");
}

void DataInputDialog::RenderTextOptions() {
    // Phase 3 text options. Two layouts:
    //   1. SingleFile    — CSV / JSON / TXT picked via shared file picker
    //   2. CorpusSubdirs — folder/<class>/*.txt picked via folder picker
    //                      shown inline here. Analogous to image
    //                      ClassSubdirs / audio ClassSubdirs.
    //
    // Column mapping controls only make sense in SingleFile mode — a
    // folder corpus gets its labels from subdirectory names, not CSV
    // columns. We hide the column inputs when CorpusSubdirs is picked
    // so the user can't fill in fields that get ignored at load time.
    //
    // Graph preprocessing nodes (TextTokenizer / TextVocabulary /
    // TextPadding) override the tokenizer defaults at compile time if
    // present, so these are just sensible fallbacks for the common
    // case of "drop a DataInput, point at a CSV, run".
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "Text Loading Options");
    ImGui::Spacing();

    // Layout selector — mirrors the image / audio combo at the top of
    // their options panels. Switching between modes preserves whichever
    // path (file vs folder) was previously filled in, so round-tripping
    // doesn't destroy state.
    static const char* kTextLayoutNames[] = {
        "Single file (CSV / JSON / TXT)",
        "Corpus subdirectories (folder/<class>/*.txt)"
    };
    int layout_idx = static_cast<int>(text_layout_);
    ImGui::Text("Layout:");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##textlayout", &layout_idx, kTextLayoutNames, IM_ARRAYSIZE(kTextLayoutNames))) {
        text_layout_ = static_cast<TextLayout>(layout_idx);
        has_changes_ = true;
    }
    ImGui::Spacing();

    if (text_layout_ == TextLayout::SingleFile) {
        ImGui::TextDisabled("Loads from CSV/TSV (text column + label column), "
                            "JSON / JSONL (text field + label field), or plain "
                            "TXT (one unlabeled sample per line).");
    } else {
        ImGui::TextDisabled("Loads text from folder/<class>/*.txt — each "
                            "subdirectory is a class, each .txt file is one "
                            "sample. Mirrors ImageFolder / audio ClassSubdirs.");
    }
    ImGui::Spacing();

    // CorpusSubdirs: folder picker lives inline here (file picker at the
    // top is hidden in this mode). Shows a quick scan of subdirs when
    // the path exists — mirrors the image/audio folder feedback.
    if (text_layout_ == TextLayout::CorpusSubdirs) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(accent, "Folder");
        ImGui::Spacing();

        ImGui::Text("Path:");
        ImGui::SameLine(60);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
        if (ImGui::InputText("##textfolder", folder_path_, sizeof(folder_path_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Browse...##textfolder", ImVec2(80, 0))) {
            BrowseFolder();
        }

        if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
            int subdir_count = 0;
            int file_count = 0;
            try {
                for (const auto& entry : fs::directory_iterator(folder_path_)) {
                    if (!entry.is_directory()) continue;
                    subdir_count++;
                    for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                        if (!sub.is_regular_file()) continue;
                        std::string ext = sub.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (ext == ".txt" || ext == ".text" || ext == ".md") {
                            file_count++;
                        }
                    }
                }
            } catch (...) { /* permission errors etc., ignore */ }
            if (subdir_count > 0) {
                ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.6f, 1.0f),
                                   "Detected: %d classes, %d text files",
                                   subdir_count, file_count);
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.4f, 1.0f),
                                   "No class subdirectories found — corpus "
                                   "layout expects folder/<class>/*.txt");
            }
        }
        ImGui::Spacing();
    }

    // Column mapping is only meaningful for SingleFile CSV/JSON mode.
    // In corpus mode the label comes from the subdirectory name, so we
    // hide these fields to avoid confusion.
    if (text_layout_ == TextLayout::SingleFile) {
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(accent, "Column Mapping");
        ImGui::Spacing();

        ImGui::Text("Text col:");
        ImGui::SameLine(140);
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##textcol", text_column_, sizeof(text_column_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(CSV column / JSON field holding sentences)");

        ImGui::Text("Label col:");
        ImGui::SameLine(140);
        ImGui::SetNextItemWidth(220);
        if (ImGui::InputText("##textlabelcol", text_label_column_, sizeof(text_label_column_))) {
            has_changes_ = true;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(blank = unlabeled corpus / plain TXT)");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Tokenizer Defaults");
    ImGui::Spacing();
    ImGui::TextDisabled("Overridden by a TextTokenizer node in the graph, if present.");
    ImGui::Spacing();

    static const char* kTokenizerNames[] = {
        "Whitespace (split on spaces)",
        "Word (splits on punctuation too)",
        "Character (one token per char)"
    };
    ImGui::Text("Tokenizer:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(260);
    if (ImGui::Combo("##texttok", &text_tokenizer_type_, kTokenizerNames, IM_ARRAYSIZE(kTokenizerNames))) {
        has_changes_ = true;
    }

    ImGui::Text("Max length:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textmaxlen", &text_max_length_)) {
        if (text_max_length_ < 1) text_max_length_ = 1;
        if (text_max_length_ > 8192) text_max_length_ = 8192;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("tokens (pad/truncate)");

    if (ImGui::Checkbox("Lowercase text", &text_lowercase_)) {
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Vocabulary");
    ImGui::Spacing();

    ImGui::Text("Min freq:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textminfreq", &text_min_freq_)) {
        if (text_min_freq_ < 1) text_min_freq_ = 1;
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(drop words occurring fewer times)");

    ImGui::Text("Max vocab:");
    ImGui::SameLine(140);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("##textmaxvocab", &text_max_vocab_size_)) {
        has_changes_ = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(-1 = unlimited)");

    ImGui::Spacing();
    ImGui::TextDisabled("Supported: CSV, TSV, JSON, TXT (one line per sample), folder of text files");
}

void DataInputDialog::RenderTransformationTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Column Transformations");
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
        ImGui::TextDisabled("Load preview to see column list");
    }
}

void DataInputDialog::RenderLimitRowsTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Row Filtering Options");
    ImGui::Separator();
    ImGui::Spacing();

    // Skip rows
    ImGui::Text("Skip first N rows:");
    ImGui::SameLine(180);
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
        ImGui::SameLine(180);
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
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Character Encoding");
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

void DataInputDialog::RenderMemoryTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Memory Management");
    ImGui::Separator();
    ImGui::Spacing();

    // === Current data status ===
    ImGui::Text("Current Status:");
    ImGui::SameLine(120);

    switch (data_load_state_) {
        case DataLoadState::InMemory: {
            // Green = in-memory Arrow, blue = disk-backed Parquet cache.
            // Both are "loaded and ready to train" from the user's POV;
            // the color signals where the data actually lives.
            // When the dataset is image/audio (lazy-loaded), the byte
            // count is an if-fully-cached estimate, not actual RAM use —
            // label it accordingly so users aren't misled.
            if (loaded_backend_ == 2) {
                ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "Loaded via Parquet cache");
                ImGui::SameLine();
                ImGui::TextDisabled("- %s on disk", FormatBytes(loaded_memory_bytes_).c_str());
            } else if (loaded_memory_is_estimate_) {
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 0.4f, 1.0f), "Lazy-loaded");
                ImGui::SameLine();
                ImGui::TextDisabled("- ~%s if fully cached (estimated)",
                                    FormatBytes(loaded_memory_bytes_).c_str());
            } else {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "In Memory");
                ImGui::SameLine();
                ImGui::TextDisabled("- %s", FormatBytes(loaded_memory_bytes_).c_str());
            }

            ImGui::Text("Rows:");
            ImGui::SameLine(120);
            ImGui::Text("%lld", static_cast<long long>(loaded_rows_));
            ImGui::SameLine(200);
            ImGui::Text("Columns:");
            ImGui::SameLine(280);
            ImGui::Text("%lld", static_cast<long long>(loaded_cols_));

            // Explain the backing for disk-backed loads so users understand
            // they're not consuming RAM proportional to the file size.
            if (loaded_backend_ == 2) {
                ImGui::TextDisabled("  Training reads pages lazily via memory-mapped I/O. "
                                    "RAM use bounded by the OS page cache, not the file size.");
            }

            ImGui::Spacing();

            // Unload button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.4f, 0.4f, 1.0f));
            if (ImGui::Button("Unload from Memory", ImVec2(180, 0))) {
                UnloadDataset();
            }
            ImGui::PopStyleColor(2);
            ImGui::SameLine();
            ImGui::TextDisabled("Free RAM by removing cached data");
            break;
        }
        case DataLoadState::OnDisk:
            ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.0f, 1.0f), "On Disk (streaming)");
            break;
        case DataLoadState::NotLoaded:
        default:
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Not Loaded");
            ImGui::TextDisabled("Click Apply to load data");
            break;
    }

    // Pre-load size estimate (honest: shows file size on disk, not a made-up
    // multiplier). Actual in-memory footprint after load is shown by the
    // Current Status section above once the data is applied, and is typically
    // smaller than file size thanks to CompactIntegerColumns downcasting
    // int64 columns to uint8/16/32 where the data fits.
    if (file_size_ > 0 && data_load_state_ != DataLoadState::InMemory) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        UpdateRAMEstimate();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f),
                           "File size on disk: %.1f MB", estimated_ram_mb_);
        ImGui::TextDisabled("   Actual RAM usage after load is usually smaller due to integer column compaction.");
    }

    // Advanced: force disk-backed cache (escape hatch for testing the
    // Parquet path on datasets that would otherwise take the in-memory
    // route). Default off — LoadTabularCSV picks in-memory vs disk-backed
    // automatically based on file_size vs available RAM (75% threshold).
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(accent, "Advanced");
    ImGui::Spacing();
    if (ImGui::Checkbox("Force disk-backed cache", &force_disk_backed_)) {
        has_changes_ = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Normally the engine loads small CSVs directly into RAM and spills\n"
            "larger-than-RAM CSVs to a Parquet cache on disk. This checkbox\n"
            "forces the disk-backed cache path even for small files — useful\n"
            "for testing or benchmarking the lazy load code path. Takes effect\n"
            "on the next Apply.");
    }
    ImGui::TextDisabled("   When on, the next Apply writes a Parquet cache in the system temp dir.");
}

void DataInputDialog::RenderMLDatasetOptions() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "Dataset Options");
    ImGui::Separator();
    ImGui::Spacing();

    // Split options
    ImGui::Text("Data Split:");
    ImGui::Spacing();

    static int split_mode = 0;
    ImGui::RadioButton("Use default split", &split_mode, 0);
    ImGui::RadioButton("Custom split", &split_mode, 1);

    if (split_mode == 1) {
        static float train_ratio = 0.8f;
        static float val_ratio = 0.1f;
        ImGui::Text("Train:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##train", &train_ratio, 0.1f, 0.9f, "%.0f%%");
        ImGui::Text("Val:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("##val", &val_ratio, 0.0f, 0.5f, "%.0f%%");
        ImGui::Text("Test:");
        ImGui::SameLine(80);
        float test_ratio = 1.0f - train_ratio - val_ratio;
        ImGui::Text("%.0f%%", test_ratio * 100);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Preprocessing
    ImGui::TextColored(accent, "Preprocessing");
    ImGui::Spacing();

    if (ImGui::Checkbox("Normalize to [0, 1]", &normalize_images_)) {
        has_changes_ = true;
    }

    static bool shuffle = true;
    if (ImGui::Checkbox("Shuffle on load", &shuffle)) {
        has_changes_ = true;
    }
}

void DataInputDialog::RenderMLDatasetSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "ML DATASET");
    ImGui::Spacing();

    // Dataset source selector
    int ml_idx = static_cast<int>(ml_dataset_type_);
    const char* ml_items[] = {
        "MNIST", "CIFAR-10", "CIFAR-100", "Fashion-MNIST",
        "ImageNet", "Image Folder", "HuggingFace", "Kaggle", "Custom"
    };

    ImGui::Text("Dataset:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(150);
    if (ImGui::Combo("##mldataset", &ml_idx, ml_items, 9)) {
        ml_dataset_type_ = static_cast<MLDatasetType>(ml_idx);
        has_changes_ = true;
        preview_loaded_ = false;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Dataset-specific configuration
    switch (ml_dataset_type_) {
        case MLDatasetType::MNIST:
        case MLDatasetType::CIFAR10:
        case MLDatasetType::CIFAR100:
        case MLDatasetType::FashionMNIST:
            RenderBuiltinDatasets();
            break;
        case MLDatasetType::ImageNet:
        case MLDatasetType::ImageFolder:
            RenderImageFolderPicker();
            break;
        case MLDatasetType::HuggingFace:
            RenderHuggingFaceConfig();
            break;
        case MLDatasetType::Kaggle:
            RenderKaggleConfig();
            break;
        case MLDatasetType::Custom:
            // Custom uses file source
            ImGui::TextDisabled("Use File source for custom datasets");
            break;
    }
}

void DataInputDialog::RenderBuiltinDatasets() {
    const char* dataset_names[] = {"mnist", "cifar10", "cifar100", "fashion_mnist"};
    const char* descriptions[] = {
        "Handwritten digits (28x28, 10 classes, 60K train)",
        "Color images (32x32x3, 10 classes, 50K train)",
        "Color images (32x32x3, 100 classes, 50K train)",
        "Fashion items (28x28, 10 classes, 60K train)"
    };

    int idx = static_cast<int>(ml_dataset_type_);
    if (idx < 4) {
        strncpy(dataset_name_, dataset_names[idx], sizeof(dataset_name_) - 1);
        ImGui::TextWrapped("%s", descriptions[idx]);
    }

    ImGui::Spacing();

    // Cache directory
    ImGui::Text("Cache dir:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 90);
    ImGui::InputText("##cachedir", cache_dir_, sizeof(cache_dir_));
    ImGui::SameLine();
    if (ImGui::Button("...##cachebrowse", ImVec2(80, 0))) {
        BrowseFolder();
        strncpy(cache_dir_, folder_path_, sizeof(cache_dir_) - 1);
    }

    if (strlen(cache_dir_) == 0) {
        ImGui::TextDisabled("Default: ~/.cyxwiz/datasets/");
    }

    ImGui::Spacing();

    // Download button
    if (is_downloading_) {
        ImGui::ProgressBar(download_progress_, ImVec2(-1, 0), "Downloading...");
    } else {
        if (ImGui::Button("Download Dataset", ImVec2(-1, 30))) {
            DownloadMLDataset();
        }
    }

    if (!status_message_.empty()) {
        ImVec4 color = status_message_.find("Error") != std::string::npos
            ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
            : ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
        ImGui::TextColored(color, "%s", status_message_.c_str());
    }
}

void DataInputDialog::RenderImageFolderPicker() {
    ImGui::Text("Root folder:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##imgroot", folder_path_, sizeof(folder_path_))) {
        has_changes_ = true;
        preview_loaded_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("...##rootbrowse", ImVec2(90, 0))) {
        BrowseFolder();
    }

    ImGui::Spacing();
    ImGui::TextDisabled("Structure: root/class_name/image.jpg");

    // Show detected classes
    if (strlen(folder_path_) > 0 && fs::exists(folder_path_)) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        std::vector<std::string> classes;
        for (const auto& entry : fs::directory_iterator(folder_path_)) {
            if (entry.is_directory()) {
                classes.push_back(entry.path().filename().string());
            }
        }

        if (!classes.empty()) {
            ImGui::Text("Detected %zu classes:", classes.size());
            ImGui::BeginChild("ClassList", ImVec2(0, 100), true);
            for (const auto& c : classes) {
                ImGui::BulletText("%s", c.c_str());
            }
            ImGui::EndChild();
        }
    }
}

void DataInputDialog::RenderHuggingFaceConfig() {
    ImGui::Text("Dataset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    if (ImGui::InputText("##hfname", dataset_name_, sizeof(dataset_name_))) {
        has_changes_ = true;
    }
    ImGui::TextDisabled("e.g., mnist, imdb, squad, coco");

    ImGui::Spacing();

    ImGui::Text("Subset:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##hfsubset", dataset_subset_, sizeof(dataset_subset_));

    ImGui::Text("Split:");
    ImGui::SameLine(90);
    static int split_idx = 0;
    const char* splits[] = {"train", "validation", "test"};
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("##hfsplit", &split_idx, splits, 3);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Auth token:");
    ImGui::SameLine(90);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 100);
    ImGui::InputText("##hftoken", hf_token_, sizeof(hf_token_), ImGuiInputTextFlags_Password);
    ImGui::TextDisabled("Required for private/gated datasets");

    ImGui::Spacing();

    if (ImGui::Button("Load from HuggingFace", ImVec2(-1, 30))) {
        DownloadMLDataset();
    }
}

void DataInputDialog::RenderKaggleConfig() {
    ImGui::Text("Dataset slug:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##kaggleslug", kaggle_slug_, sizeof(kaggle_slug_));
    ImGui::TextDisabled("e.g., zalando-research/fashionmnist");

    ImGui::Spacing();

    ImGui::Text("or Competition:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    static char competition[128] = "";
    ImGui::InputText("##kagglecomp", competition, sizeof(competition));
    ImGui::TextDisabled("e.g., titanic, digit-recognizer");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextDisabled("API credentials from ~/.kaggle/kaggle.json");

    if (ImGui::Button("Download from Kaggle", ImVec2(-1, 30))) {
        DownloadMLDataset();
    }
}

void DataInputDialog::RenderDatabaseSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "DATABASE CONNECTION");
    ImGui::Spacing();

    // Database type
    int db_idx = static_cast<int>(database_type_);
    const char* db_types[] = {"SQLite", "PostgreSQL", "MySQL", "DuckDB"};

    ImGui::Text("Type:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##dbtype", &db_idx, db_types, 4)) {
        database_type_ = static_cast<DatabaseType>(db_idx);
        has_changes_ = true;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderDatabaseConnection();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderSQLQuery();
}

void DataInputDialog::RenderDatabaseConnection() {
    if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
        // File-based database
        ImGui::Text("Database file:");
        ImGui::SameLine(100);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
        ImGui::InputText("##dbfile", db_file_, sizeof(db_file_));
        ImGui::SameLine();
        if (ImGui::Button("...##dbbrowse", ImVec2(100, 0))) {
            std::string path;
            const char* filter = (database_type_ == DatabaseType::SQLite)
                ? "SQLite\0*.db;*.sqlite;*.sqlite3\0All Files\0*.*\0"
                : "DuckDB\0*.duckdb;*.db\0All Files\0*.*\0";
            if (FileSelector("Select Database:", path, filter)) {
                strncpy(db_file_, path.c_str(), sizeof(db_file_) - 1);
                has_changes_ = true;
            }
        }
    } else {
        // Network database
        ImGui::Text("Host:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbhost", db_host_, sizeof(db_host_));
        ImGui::SameLine();
        ImGui::Text("Port:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(60);
        ImGui::InputInt("##dbport", &db_port_);

        ImGui::Text("Database:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(150);
        ImGui::InputText("##dbname", db_name_, sizeof(db_name_));

        ImGui::Text("User:");
        ImGui::SameLine(80);
        ImGui::SetNextItemWidth(120);
        ImGui::InputText("##dbuser", db_user_, sizeof(db_user_));
        ImGui::SameLine();
        ImGui::Text("Password:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputText("##dbpass", db_password_, sizeof(db_password_), ImGuiInputTextFlags_Password);
    }

    ImGui::Spacing();

    // Connect button
    ImVec4 btn_color = db_connected_
        ? ImVec4(0.2f, 0.7f, 0.2f, 1.0f)
        : ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, btn_color);
    if (ImGui::Button(db_connected_ ? "Connected" : "Test Connection", ImVec2(150, 0))) {
        TestDatabaseConnection();
    }
    ImGui::PopStyleColor();
}

void DataInputDialog::RenderSQLQuery() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "SQL Query");
    ImGui::Spacing();

    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextMultiline("##sqlquery", sql_query_, sizeof(sql_query_), ImVec2(0, 100))) {
        has_changes_ = true;
    }

    ImGui::Spacing();

    // Quick templates
    if (ImGui::Button("SELECT *")) {
        strncpy(sql_query_, "SELECT * FROM table_name LIMIT 1000", sizeof(sql_query_) - 1);
    }
    ImGui::SameLine();
    if (ImGui::Button("SHOW TABLES")) {
        if (database_type_ == DatabaseType::SQLite || database_type_ == DatabaseType::DuckDB) {
            strncpy(sql_query_, "SELECT name FROM sqlite_master WHERE type='table'", sizeof(sql_query_) - 1);
        } else if (database_type_ == DatabaseType::PostgreSQL) {
            strncpy(sql_query_, "SELECT tablename FROM pg_tables WHERE schemaname='public'", sizeof(sql_query_) - 1);
        } else {
            strncpy(sql_query_, "SHOW TABLES", sizeof(sql_query_) - 1);
        }
    }

    ImGui::Spacing();

    if (ImGui::Button("Execute Query", ImVec2(-1, 25))) {
        LoadPreview();
    }
}

void DataInputDialog::RenderCloudSource() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "CLOUD STORAGE");
    ImGui::Spacing();

    // Cloud provider
    static int cloud_provider = 0;
    ImGui::RadioButton("AWS S3", &cloud_provider, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Google Cloud", &cloud_provider, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Azure Blob", &cloud_provider, 2);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Bucket:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("##bucket", cloud_bucket_, sizeof(cloud_bucket_));

    ImGui::Text("Path:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudpath", cloud_path_, sizeof(cloud_path_));

    ImGui::Spacing();

    ImGui::Text("Credentials:");
    ImGui::SameLine(100);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 110);
    ImGui::InputText("##cloudcreds", cloud_credentials_, sizeof(cloud_credentials_));
    ImGui::SameLine();
    if (ImGui::Button("...##credbrowse", ImVec2(100, 0))) {
        std::string path;
        if (FileSelector("Select Credentials:", path, "JSON\0*.json\0All Files\0*.*\0")) {
            strncpy(cloud_credentials_, path.c_str(), sizeof(cloud_credentials_) - 1);
        }
    }

    ImGui::Spacing();
    ImGui::TextDisabled("AWS: ~/.aws/credentials or JSON key");
    ImGui::TextDisabled("GCS: service-account.json");
    ImGui::TextDisabled("Azure: connection string or SAS token");

    ImGui::Spacing();

    if (ImGui::Button("Connect & List Files", ImVec2(-1, 30))) {
        LoadPreview();
    }
}

void DataInputDialog::RenderPreviewPanel() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::TextColored(accent, "PREVIEW");
    ImGui::Separator();
    ImGui::Spacing();

    if (!preview_loaded_) {
        if (ImGui::Button("Load Preview", ImVec2(-1, 30))) {
            LoadPreview();
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Configure source and click\nLoad Preview to see data");
    } else if (!preview_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Error: %s", preview_error_.c_str());
    } else {
        // Render based on category
        if (source_type_ == SourceType::File) {
            switch (file_category_) {
                case FileCategory::Tabular: RenderTabularPreview(); break;
                case FileCategory::Image:   RenderImagePreview(); break;
                case FileCategory::Audio:   RenderAudioPreview(); break;
                case FileCategory::Text:    RenderTextPreview(); break;
                default: RenderTabularPreview(); break;
            }
        } else {
            // Default to tabular preview for other sources
            RenderTabularPreview();
        }
    }

    // RAM estimate
    if (preview_loaded_ && estimated_ram_mb_ > 0) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Est. RAM: %.1f MB", estimated_ram_mb_);
    }
}

void DataInputDialog::RenderTabularPreview() {
    if (preview_columns_.empty()) {
        ImGui::TextDisabled("No data to preview");
        return;
    }

    // Column info
    ImGui::Text("%zu columns, %zu rows", preview_columns_.size(), preview_data_.size());
    ImGui::Spacing();

    // Table
    if (ImGui::BeginTable("Preview", static_cast<int>(preview_columns_.size()),
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        // Headers
        for (const auto& col : preview_columns_) {
            ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 80.0f);
        }
        ImGui::TableHeadersRow();

        // Data rows (limit to 20)
        int row_idx = 0;
        for (const auto& row : preview_data_) {
            if (row_idx >= 20) break;
            ImGui::TableNextRow();
            int col_idx = 0;
            for (const auto& cell : row) {
                if (col_idx < static_cast<int>(preview_columns_.size())) {
                    ImGui::TableSetColumnIndex(col_idx);
                    ImGui::TextUnformatted(cell.c_str());
                }
                col_idx++;
            }
            row_idx++;
        }
        ImGui::EndTable();
    }
}

void DataInputDialog::RenderImagePreview() {
    if (preview_image_textures_.empty()) {
        ImGui::TextDisabled("No images to preview");
        ImGui::TextDisabled("(Image preview coming soon)");
        return;
    }

    ImGui::Text("%zu images", preview_image_textures_.size());
    ImGui::Spacing();

    // Grid of thumbnails
    float thumb_size = 64.0f;
    float avail_width = ImGui::GetContentRegionAvail().x;
    int cols = std::max(1, static_cast<int>(avail_width / (thumb_size + 8)));

    int idx = 0;
    for (ImTextureID tex_id : preview_image_textures_) {
        if (idx > 0 && (idx % cols) != 0) {
            ImGui::SameLine();
        }

        ImGui::BeginGroup();
        ImGui::Image(tex_id, ImVec2(thumb_size, thumb_size));
        if (idx < static_cast<int>(preview_image_labels_.size())) {
            ImGui::TextDisabled("%s", preview_image_labels_[idx].c_str());
        }
        ImGui::EndGroup();

        idx++;
        if (idx >= 12) break;  // Limit preview images
    }
}

void DataInputDialog::RenderAudioPreview() {
    ImGui::TextDisabled("Audio preview not yet implemented");
    ImGui::TextDisabled("Waveform visualization coming soon");
}

// ==================== Helper Functions ====================

const char* DataInputDialog::GetFileTypeName() const {
    const char* names[] = {"Auto", "CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5", "Feather", "Arrow", "TXT", "ARFF"};
    if (detected_type_ >= 0 && detected_type_ < 11) {
        return names[detected_type_];
    }
    return "Unknown";
}

void DataInputDialog::DetectFileType() {
    std::string path(file_path_);
    if (path.empty()) {
        detected_type_ = 0;
        return;
    }

    // Get file size
    try {
        file_size_ = static_cast<size_t>(fs::file_size(path));
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
    else if (ext == "feather" || ext == "fea") detected_type_ = 7;
    else if (ext == "arrow" || ext == "ipc") detected_type_ = 8;
    else if (ext == "txt") detected_type_ = 9;
    else if (ext == "arff") detected_type_ = 10;
    else detected_type_ = 0;
}

void DataInputDialog::DetectFileCategory() {
    std::string path(file_path_);
    if (path.empty()) return;

    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    // Image extensions
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "gif" || ext == "tiff" || ext == "webp") {
        file_category_ = FileCategory::Image;
    }
    // Audio extensions
    else if (ext == "wav" || ext == "mp3" || ext == "flac" || ext == "ogg" ||
             ext == "m4a" || ext == "aac") {
        file_category_ = FileCategory::Audio;
    }
    // Video extensions
    else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
             ext == "webm" || ext == "wmv") {
        file_category_ = FileCategory::Video;
    }
    // Default to tabular — but preserve Text and TimeSeries selections made
    // by the user before picking the file. Both are built on top of the
    // tabular loader; clobbering them back to Tabular after the user clicks
    // Browse would silently lose an explicit category choice.
    else {
        if (file_category_ != FileCategory::Text &&
            file_category_ != FileCategory::TimeSeries) {
            file_category_ = FileCategory::Tabular;
        }
    }
}

void DataInputDialog::RenderTextPreview() {
    // Text preview reuses the CSV head read by LoadColumnList. We render
    // a simple column table (identical to RenderTabularPreview) plus a
    // small header showing which column the dialog has mapped to
    // text/label — so the user can verify their mapping before Apply
    // without reading the full tokenize+vocab round-trip error at train
    // time. Corpus mode (folder/<class>/*.txt) has no CSV to preview;
    // show the folder scan result instead.
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    if (text_layout_ == TextLayout::CorpusSubdirs) {
        if (strlen(folder_path_) == 0) {
            ImGui::TextDisabled("No folder selected");
            return;
        }
        if (!fs::exists(folder_path_)) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "Folder does not exist");
            return;
        }
        int subdir_count = 0, file_count = 0;
        try {
            for (const auto& entry : fs::directory_iterator(folder_path_)) {
                if (!entry.is_directory()) continue;
                subdir_count++;
                for (const auto& sub : fs::recursive_directory_iterator(entry.path())) {
                    if (!sub.is_regular_file()) continue;
                    std::string ext = sub.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".txt" || ext == ".text" || ext == ".md") file_count++;
                }
            }
        } catch (...) {}
        ImGui::TextColored(accent, "Corpus scan");
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Folder:  %s", folder_path_);
        ImGui::Text("Classes: %d subdirectories", subdir_count);
        ImGui::Text("Samples: %d text files", file_count);
        ImGui::Spacing();
        ImGui::TextDisabled("Each subdirectory name becomes a class label. "
                            "Apply to tokenize + build vocab (runs on a worker thread).");
        return;
    }

    // SingleFile mode: show the CSV head via the existing preview_columns_
    // / preview_data_ state populated by LoadColumnList.
    if (preview_columns_.empty()) {
        ImGui::TextDisabled("No data to preview — click Load Preview above.");
        return;
    }

    // Header strip: which column is the text? which is the label? Use the
    // dialog's mapped field names and verify they're actually in the CSV
    // header row. Wrong/missing columns show in red so the user catches
    // the typo before Apply.
    auto col_exists = [&](const std::string& name) {
        if (name.empty()) return false;
        for (const auto& c : preview_columns_) if (c == name) return true;
        return false;
    };

    ImGui::TextColored(accent, "Column mapping");
    ImGui::Separator();
    ImGui::Spacing();

    std::string text_col_str = text_column_;
    std::string label_col_str = text_label_column_;
    ImVec4 ok_color  = ImVec4(0.6f, 0.9f, 0.6f, 1.0f);
    ImVec4 err_color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);

    ImGui::Text("Text:  ");
    ImGui::SameLine();
    if (text_col_str.empty()) {
        ImGui::TextColored(err_color, "(not set)");
    } else if (col_exists(text_col_str)) {
        ImGui::TextColored(ok_color, "%s", text_col_str.c_str());
    } else {
        ImGui::TextColored(err_color, "%s (not in file)", text_col_str.c_str());
    }

    ImGui::Text("Label: ");
    ImGui::SameLine();
    if (label_col_str.empty()) {
        ImGui::TextDisabled("(unlabeled)");
    } else if (col_exists(label_col_str)) {
        ImGui::TextColored(ok_color, "%s", label_col_str.c_str());
    } else {
        ImGui::TextColored(err_color, "%s (not in file)", label_col_str.c_str());
    }

    ImGui::Spacing();
    ImGui::Text("%zu columns, %zu rows shown", preview_columns_.size(), preview_data_.size());
    ImGui::Spacing();

    // Class distribution bar chart. Skipped in unlabeled mode or when
    // the label column isn't in the file — nothing to count. Computes
    // once per (file_path, label_column) pair via a cache key.
    if (!label_col_str.empty() && col_exists(label_col_str)) {
        ComputeTextLabelDistribution();

        if (!preview_label_counts_.empty()) {
            ImGui::TextColored(accent, "Class distribution");
            ImGui::Separator();
            ImGui::Text("%d rows scanned, %zu unique classes%s",
                        preview_label_counts_total_rows_,
                        preview_label_counts_.size(),
                        preview_label_counts_truncated_ ? " (truncated)" : "");

            // Flag imbalance: if max/min count ratio > 10x, say so. Users
            // catch class-imbalance bugs before hitting a stuck loss.
            if (preview_label_counts_.size() >= 2) {
                int max_c = preview_label_counts_.front().second;
                int min_c = preview_label_counts_.back().second;
                if (min_c > 0 && max_c / min_c >= 10) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                                       "  imbalance %.1fx",
                                       static_cast<float>(max_c) /
                                       static_cast<float>(min_c));
                }
            }

            // Horizontal bar chart via ImPlot. Height scales with class
            // count (~18 px per class) so 2-class stays compact and 20-
            // class stays readable. Capped at 360 px so it doesn't
            // dominate the preview.
            int n = static_cast<int>(preview_label_counts_.size());
            std::vector<double> values;
            std::vector<const char*> labels;
            values.reserve(n);
            labels.reserve(n);
            for (const auto& kv : preview_label_counts_) {
                values.push_back(static_cast<double>(kv.second));
                labels.push_back(kv.first.c_str());
            }
            float plot_h = std::min(360.0f, 18.0f * n + 40.0f);
            if (ImPlot::BeginPlot("##ClassDist", ImVec2(-1, plot_h),
                ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText |
                ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("count", nullptr,
                                  ImPlotAxisFlags_AutoFit,
                                  ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisTicks(ImAxis_Y1, 0, n - 1, n,
                                       labels.data(), false);
                ImPlot::PlotBars("##counts", values.data(), n, 0.7,
                                 0.0, ImPlotBarsFlags_Horizontal);
                ImPlot::EndPlot();
            }
            ImGui::Spacing();
        }
    }

    // Column table (same structure as RenderTabularPreview).
    if (ImGui::BeginTable("TextPreview", static_cast<int>(preview_columns_.size()),
        ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        for (const auto& col : preview_columns_) {
            ImGui::TableSetupColumn(col.c_str(), ImGuiTableColumnFlags_WidthFixed, 200.0f);
        }
        ImGui::TableHeadersRow();

        int row_idx = 0;
        for (const auto& row : preview_data_) {
            if (row_idx >= 15) break;
            ImGui::TableNextRow();
            int col_idx = 0;
            for (const auto& cell : row) {
                if (col_idx < static_cast<int>(preview_columns_.size())) {
                    ImGui::TableSetColumnIndex(col_idx);
                    ImGui::TextUnformatted(cell.c_str());
                }
                col_idx++;
            }
            row_idx++;
        }
        ImGui::EndTable();
    }
}

void DataInputDialog::LoadPreview() {
    preview_error_.clear();
    preview_columns_.clear();
    preview_data_.clear();

    // Tabular and Text both read the first N rows of a CSV/TSV/JSON file
    // through LoadColumnList. Text category uses the same column table for
    // now; a dedicated renderer highlights the mapped text + label columns.
    if (source_type_ == SourceType::File &&
        (file_category_ == FileCategory::Tabular ||
         file_category_ == FileCategory::Text ||
         file_category_ == FileCategory::TimeSeries)) {
        LoadColumnList();
    }
    // TODO: Add image, audio, database preview loading

    preview_loaded_ = true;
    UpdateRAMEstimate();
}

void DataInputDialog::ComputeTextLabelDistribution() {
    // Cache key: if neither the file nor the label column has changed,
    // return without touching the disk. Makes this safe to call every
    // frame from RenderTextPreview.
    std::string cache_key =
        std::string(file_path_) + "|" + std::string(text_label_column_);
    if (cache_key == preview_label_counts_cache_key_) {
        return;
    }

    preview_label_counts_.clear();
    preview_label_counts_total_rows_ = 0;
    preview_label_counts_truncated_ = false;
    preview_label_counts_cache_key_ = cache_key;

    if (strlen(file_path_) == 0 || strlen(text_label_column_) == 0) {
        return;
    }

    std::ifstream file(file_path_);
    if (!file.is_open()) return;

    char delim = custom_delimiter_[0];
    if (delim == '\0') delim = ',';
    if (detected_type_ == 2) delim = '\t';

    // Find the label column index from the header row.
    std::string line;
    if (!std::getline(file, line)) return;
    int label_idx = -1;
    {
        std::stringstream ss(line);
        std::string cell;
        int i = 0;
        while (std::getline(ss, cell, delim)) {
            size_t start = cell.find_first_not_of(" \t\r\n\"");
            size_t end   = cell.find_last_not_of(" \t\r\n\"");
            if (start != std::string::npos && end != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            if (cell == text_label_column_) { label_idx = i; break; }
            i++;
        }
    }
    if (label_idx < 0) return;

    // Stream the full file, counting labels. Caps: stop after 1M rows or
    // 100 unique labels so a free-text column doesn't blow up RAM.
    constexpr int kMaxRows   = 1'000'000;
    constexpr int kMaxUnique = 100;
    std::map<std::string, int> counts;
    int row = 0;
    while (std::getline(file, line) && row < kMaxRows) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        std::string label_val;
        while (std::getline(ss, cell, delim)) {
            if (col == label_idx) {
                size_t start = cell.find_first_not_of(" \t\r\n\"");
                size_t end   = cell.find_last_not_of(" \t\r\n\"");
                if (start != std::string::npos && end != std::string::npos) {
                    label_val = cell.substr(start, end - start + 1);
                } else {
                    label_val = cell;
                }
                break;
            }
            col++;
        }
        if (label_val.empty()) { row++; continue; }
        if (counts.size() >= kMaxUnique && counts.find(label_val) == counts.end()) {
            preview_label_counts_truncated_ = true;
            row++;
            continue;
        }
        counts[label_val]++;
        row++;
    }
    if (row >= kMaxRows) preview_label_counts_truncated_ = true;
    preview_label_counts_total_rows_ = row;

    // Flatten + sort by count descending so the bar chart reads top-down
    // by frequency, and imbalance pops out at a glance.
    preview_label_counts_.reserve(counts.size());
    for (auto& kv : counts) {
        preview_label_counts_.emplace_back(kv.first, kv.second);
    }
    std::sort(preview_label_counts_.begin(), preview_label_counts_.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
}

void DataInputDialog::LoadColumnList() {
    std::string path(file_path_);
    if (path.empty()) return;

    std::ifstream file(path);
    if (!file.is_open()) {
        preview_error_ = "Cannot open file";
        return;
    }

    char delim = custom_delimiter_[0];
    if (delim == '\0') delim = ',';
    if (detected_type_ == 2) delim = '\t';

    std::string line;
    int line_count = 0;

    while (std::getline(file, line) && line_count < 25) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delim)) {
            // Trim whitespace and quotes
            size_t start = cell.find_first_not_of(" \t\r\n\"");
            size_t end = cell.find_last_not_of(" \t\r\n\"");
            if (start != std::string::npos && end != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            cells.push_back(cell);
        }

        if (line_count == 0 && has_header_) {
            preview_columns_ = cells;
            available_columns_ = cells;
            selected_columns_.resize(cells.size(), true);
        } else {
            preview_data_.push_back(cells);
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
}

void DataInputDialog::UpdateRAMEstimate() {
    // Report the plain file size on disk in MB. The actual in-memory footprint
    // after Arrow loads and CompactIntegerColumns runs is typically smaller,
    // and is shown in the Current Status section of the Memory tab after load.
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
            filter = "All Data Files\0*.csv;*.tsv;*.xlsx;*.xls;*.json;*.parquet;*.h5;*.hdf5;*.feather;*.arrow;*.txt;*.arff\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0Excel\0*.xlsx;*.xls\0JSON\0*.json\0Parquet\0*.parquet\0"
                     "HDF5\0*.h5;*.hdf5\0Feather\0*.feather\0Arrow\0*.arrow;*.ipc\0Text\0*.txt\0ARFF\0*.arff\0All Files\0*.*\0";
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
            filter = "Time Series Data\0*.csv;*.tsv;*.parquet;*.feather;*.arrow\0"
                     "CSV\0*.csv\0TSV\0*.tsv\0Parquet\0*.parquet\0"
                     "Feather\0*.feather\0Arrow\0*.arrow;*.ipc\0"
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
        LoadColumnList();  // Load column names for label selection
        label_column_idx_ = -1;  // Reset label selection when file changes
        has_changes_ = true;
        preview_loaded_ = false;
    }
#else
    spdlog::warn("File browser not implemented for this platform");
#endif
}

void DataInputDialog::BrowseFolder() {
#ifdef _WIN32
    // Use the modern IFileDialog in folder-pick mode (same look as the
    // file picker the Labels CSV Browse button opens). The old
    // SHBrowseForFolder shows a Win95-style tree that looks out of place
    // in a modern app.
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

void DataInputDialog::TestDatabaseConnection() {
    // TODO: Implement actual database connection test
    db_connected_ = true;
    status_message_ = "Connected successfully";
    spdlog::info("Database connection test: simulated success");
}

void DataInputDialog::DownloadMLDataset() {
    is_downloading_ = true;
    download_progress_ = 0.0f;

    // TODO: Implement actual download
    // For now, simulate progress
    is_downloading_ = false;
    download_progress_ = 1.0f;
    status_message_ = "Dataset ready (simulated)";

    spdlog::info("ML dataset download: {}", dataset_name_);
}

// ==================== DataInputDialog Helper Methods ====================

std::string DataInputDialog::FormatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    char buffer[32];
    if (unit_idx == 0) {
        snprintf(buffer, sizeof(buffer), "%zu %s", bytes, units[unit_idx]);
    } else {
        snprintf(buffer, sizeof(buffer), "%.1f %s", size, units[unit_idx]);
    }
    return std::string(buffer);
}

std::string DataInputDialog::GenerateDatasetName() const {
    // Generate a unique dataset name based on source
    std::string name;

    if (source_type_ == SourceType::File) {
        if (strlen(file_path_) > 0) {
            fs::path p(file_path_);
            name = p.stem().string();
        } else if (strlen(folder_path_) > 0) {
            fs::path p(folder_path_);
            name = p.filename().string();
        }
    } else if (source_type_ == SourceType::MLDataset) {
        name = dataset_name_;
    } else if (source_type_ == SourceType::Database) {
        name = std::string("db_") + db_name_;
    } else if (source_type_ == SourceType::Cloud) {
        name = std::string("cloud_") + cloud_bucket_;
    }

    // Sanitize: replace spaces and special chars
    for (char& c : name) {
        if (!isalnum(c) && c != '_' && c != '-') {
            c = '_';
        }
    }

    if (name.empty()) {
        name = "dataset";
    }

    return name;
}

void DataInputDialog::RenderDataProfilingTab() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

    ImGui::Spacing();
    ImGui::TextColored(accent, "DATA QUALITY ANALYSIS");
    ImGui::Separator();
    ImGui::Spacing();

    // Check if data is loaded
    if (data_load_state_ != DataLoadState::InMemory) {
        ImGui::TextDisabled("No data loaded in memory.");
        ImGui::TextDisabled("Click Apply to load data, then return here to analyze.");
        return;
    }

    // Analyze button
    if (profile_in_progress_) {
        ImGui::TextDisabled("Computing statistics...");
    } else {
        if (ImGui::Button("Analyze Data", ImVec2(150, 30))) {
            ComputeDataProfile();
        }
    }

    if (!profile_computed_) {
        ImGui::Spacing();
        ImGui::TextDisabled("Click 'Analyze Data' to compute column statistics");
        return;
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Data Quality Score
    ImGui::TextColored(accent, "Data Quality Score");
    ImGui::Spacing();

    // Calculate color based on score
    ImVec4 score_color;
    if (data_quality_score_ >= 0.8f)
        score_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
    else if (data_quality_score_ >= 0.5f)
        score_color = ImVec4(0.9f, 0.7f, 0.0f, 1.0f);  // Yellow
    else
        score_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);  // Red

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, score_color);
    ImGui::ProgressBar(data_quality_score_, ImVec2(-1, 20));
    ImGui::PopStyleColor();
    ImGui::Text("%.1f%% complete data (%.1f%% missing values)",
        data_quality_score_ * 100.0f, (1.0f - data_quality_score_) * 100.0f);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Column Statistics Table
    ImGui::TextColored(accent, "COLUMN STATISTICS");
    ImGui::Spacing();

    if (column_stats_.empty()) {
        ImGui::TextDisabled("No column statistics available");
        return;
    }

    if (ImGui::BeginTable("ColumnStats", 8,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
        ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
        ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {

        ImGui::TableSetupColumn("Column", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Unique", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        ImGui::TableSetupColumn("Missing", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Mean", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();

        for (const auto& stat : column_stats_) {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(stat.name.c_str());

            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(stat.dtype.c_str());

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%zu", stat.count);

            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%zu", stat.unique_count);

            ImGui::TableSetColumnIndex(4);
            if (stat.null_percentage > 5.0f) {
                ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "%.1f%%", stat.null_percentage);
            } else {
                ImGui::Text("%.1f%%", stat.null_percentage);
            }

            ImGui::TableSetColumnIndex(5);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.min_val);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableSetColumnIndex(6);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.max_val);
            } else {
                ImGui::TextDisabled("-");
            }

            ImGui::TableSetColumnIndex(7);
            if (stat.dtype == "numeric" || stat.dtype == "integer") {
                ImGui::Text("%.2f", stat.mean);
            } else {
                ImGui::TextDisabled("-");
            }
        }

        ImGui::EndTable();
    }
}

void DataInputDialog::ComputeDataProfile() {
    profile_in_progress_ = true;
    column_stats_.clear();

    // Get dataset from registry
    auto& registry = cyxwiz::DataRegistry::Instance();
    auto dataset = registry.GetArrowDataset(loaded_dataset_name_);

    if (!dataset) {
        profile_in_progress_ = false;
        spdlog::warn("Cannot compute profile: dataset '{}' not found", loaded_dataset_name_);
        return;
    }

    // Get column names and compute basic stats for each
    auto column_names = dataset->GetColumnNames();
    size_t total_nulls = 0;
    size_t total_values = 0;

    for (const auto& col_name : column_names) {
        ColumnStats stat;
        stat.name = col_name;

        // Get column type from Arrow schema
        auto arrow_table = dataset->GetArrowTable();
        if (arrow_table) {
            auto schema = arrow_table->schema();
            auto field = schema->GetFieldByName(col_name);
            if (field) {
                auto type = field->type();
                if (type->id() == arrow::Type::INT64 || type->id() == arrow::Type::INT32 ||
                    type->id() == arrow::Type::INT16 || type->id() == arrow::Type::INT8 ||
                    type->id() == arrow::Type::UINT64 || type->id() == arrow::Type::UINT32 ||
                    type->id() == arrow::Type::UINT16 || type->id() == arrow::Type::UINT8) {
                    stat.dtype = "integer";
                } else if (type->id() == arrow::Type::DOUBLE || type->id() == arrow::Type::FLOAT ||
                           type->id() == arrow::Type::HALF_FLOAT) {
                    stat.dtype = "numeric";
                } else if (type->id() == arrow::Type::STRING || type->id() == arrow::Type::LARGE_STRING) {
                    stat.dtype = "string";
                } else if (type->id() == arrow::Type::BOOL) {
                    stat.dtype = "boolean";
                } else {
                    stat.dtype = "other";
                }
            }

            // Get column data for statistics
            auto column = arrow_table->GetColumnByName(col_name);
            if (column) {
                stat.count = static_cast<size_t>(column->length());
                stat.null_count = static_cast<size_t>(column->null_count());
                stat.null_percentage = stat.count > 0
                    ? (static_cast<float>(stat.null_count) / static_cast<float>(stat.count)) * 100.0f
                    : 0.0f;

                total_nulls += stat.null_count;
                total_values += stat.count;

                // Compute numeric statistics
                if (stat.dtype == "numeric" || stat.dtype == "integer") {
                    double sum = 0.0;
                    double min_v = std::numeric_limits<double>::max();
                    double max_v = std::numeric_limits<double>::lowest();
                    size_t valid_count = 0;

                    // Helper lambda to process numeric values
                    auto process_value = [&](double val) {
                        sum += val;
                        min_v = std::min(min_v, val);
                        max_v = std::max(max_v, val);
                        valid_count++;
                    };

                    for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                        auto chunk = column->chunk(chunk_idx);
                        // Handle all numeric types
                        if (auto arr = std::dynamic_pointer_cast<arrow::DoubleArray>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(arr->Value(i));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::FloatArray>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::Int64Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::Int32Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::Int16Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::Int8Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::UInt64Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        } else if (auto arr = std::dynamic_pointer_cast<arrow::UInt32Array>(chunk)) {
                            for (int64_t i = 0; i < arr->length(); i++) {
                                if (!arr->IsNull(i)) process_value(static_cast<double>(arr->Value(i)));
                            }
                        }
                    }

                    if (valid_count > 0) {
                        stat.min_val = min_v;
                        stat.max_val = max_v;
                        stat.mean = sum / static_cast<double>(valid_count);
                    }
                }

                // Estimate unique count (simplified - use hash set for smaller datasets)
                std::unordered_set<std::string> unique_vals;
                if (stat.count <= 10000) {  // Only for smaller datasets
                    for (int chunk_idx = 0; chunk_idx < column->num_chunks(); chunk_idx++) {
                        auto chunk = column->chunk(chunk_idx);
                        if (auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(chunk)) {
                            for (int64_t i = 0; i < str_array->length(); i++) {
                                if (!str_array->IsNull(i)) {
                                    unique_vals.insert(str_array->GetString(i));
                                }
                            }
                        }
                    }
                }
                stat.unique_count = unique_vals.empty() ? 0 : unique_vals.size();
            }
        }

        column_stats_.push_back(stat);
    }

    // Calculate overall data quality score (based on missing values)
    if (total_values > 0) {
        data_quality_score_ = 1.0f - (static_cast<float>(total_nulls) / static_cast<float>(total_values));
    } else {
        data_quality_score_ = 1.0f;
    }

    profile_computed_ = true;
    profile_in_progress_ = false;

    spdlog::info("Data profile computed: {} columns, quality score: {:.1f}%",
        column_stats_.size(), data_quality_score_ * 100.0f);
}

void DataInputDialog::UnloadDataset() {
    if (loaded_dataset_name_.empty()) return;

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnloadDataset(loaded_dataset_name_);

    data_load_state_ = DataLoadState::NotLoaded;
    loaded_rows_ = 0;
    loaded_cols_ = 0;
    loaded_memory_bytes_ = 0;
    profile_computed_ = false;
    column_stats_.clear();

    if (node_) {
        node_->parameters["data_loaded"] = "false";
    }

    spdlog::info("Dataset '{}' unloaded from memory", loaded_dataset_name_);
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

    if (strlen(file_path_) > 0) {
        fs::path p(file_path_);
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

    const char* formats[] = {"CSV", "TSV", "JSON", "Parquet", "Excel", "HDF5"};
    ImGui::Text("Format:");
    ImGui::SameLine(80);
    ImGui::SetNextItemWidth(120);
    if (ImGui::Combo("##format", &output_type_, formats, 6)) {
        has_changes_ = true;
    }

    if (output_type_ <= 1) {
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
}

DataLoaderDialog::DataLoaderDialog(MLNode* node)
    : NodeConfigDialog("Data Loader", node)
{
    if (node_) {
        ReadIntParam(node_->parameters, "batch_size", batch_size_);
        ReadBoolParam(node_->parameters, "shuffle", shuffle_);
        ReadBoolParam(node_->parameters, "drop_last", drop_last_);
        ReadIntParam(node_->parameters, "num_workers", num_workers_);
    }
}

void DataLoaderDialog::Apply() {
    if (!node_) return;
    node_->parameters["batch_size"] = std::to_string(batch_size_);
    node_->parameters["shuffle"] = shuffle_ ? "true" : "false";
    node_->parameters["drop_last"] = drop_last_ ? "true" : "false";
    node_->parameters["num_workers"] = std::to_string(num_workers_);
    node_->description = "batch=" + std::to_string(batch_size_) +
                          (shuffle_ ? ", shuffled" : ", ordered");
    has_changes_ = false;
}

void DataLoaderDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    // Re-initialize local UI state from the restored params so sliders match
    batch_size_ = 32;
    shuffle_ = true;
    drop_last_ = false;
    num_workers_ = 0;
    ReadIntParam(original_params_, "batch_size", batch_size_);
    ReadBoolParam(original_params_, "shuffle", shuffle_);
    ReadBoolParam(original_params_, "drop_last", drop_last_);
    ReadIntParam(original_params_, "num_workers", num_workers_);
    has_changes_ = false;
}

void DataLoaderDialog::RenderContent() {
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];

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
                           "  Not yet implemented - batching runs single-threaded.");
    } else {
        ImGui::TextDisabled("  0 = load batches on the training thread (current behavior).");
    }

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
    // Guard against all-zero sliders, then normalize to 1.0 if the user drifted
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
    // Re-initialize local UI state from the restored params so sliders match
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
        "This node is the single source of truth for dataset partitioning.");
}

} // namespace gui
