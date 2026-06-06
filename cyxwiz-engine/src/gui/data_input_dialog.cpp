// Data Input/Output Dialog Implementations
// Part of CyxWiz Studio smart data loading system
// Smart file-backed data loading with planned dataset, database, and cloud modes.

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
#include "loaders/audio_loader.h"
#include "loaders/data_loader.h"
#include "loaders/image_loader.h"
#include "loaders/tabular_loader.h"
#include "loaders/text_loader.h"
#include "../core/data_registry.h"
#include "../core/formats/audio_dataset.h"
#include "../core/formats/text_dataset.h"
#include "../core/arrow_dataset.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/async_task_manager.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <limits>
#include <chrono>
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
            source_type_ = data_input::SourceTypeFromParam(
                node_->parameters["source_type"],
                source_type_);
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
            file_category_ = data_input::FileCategoryFromParam(
                node_->parameters["file_category"],
                file_category_);
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
        //
        // Dispatch by loader: GetByRegisteredDataset resolves the name
        // to the loader that owns it, then RestoreFromRegistry reads
        // the entry. This collapses the pre-refactor 5-branch inline
        // probe into one dispatch.
        if (node_->parameters.count("dataset_name") && !node_->parameters["dataset_name"].empty()) {
            loaded_dataset_name_ = node_->parameters["dataset_name"];
            loaded_memory_is_estimate_ = false;

            auto* loader = cyxwiz::loaders::GetByRegisteredDataset(loaded_dataset_name_);
            cyxwiz::loaders::RestoreState rs;
            if (loader && loader->RestoreFromRegistry(loaded_dataset_name_, *node_, rs) && rs.found) {
                loaded_rows_ = rs.rows;
                loaded_cols_ = rs.cols;
                loaded_memory_bytes_ = rs.bytes;
                loaded_backend_ = rs.backend;
                loaded_memory_is_estimate_ = rs.memory_is_estimate;
                data_load_state_ = DataLoadState::InMemory;
                apply_success_ = true;
                apply_status_message_ = rs.status_message;

                node_->parameters["data_loaded"] = "true";
                node_->parameters["loaded_rows"] = std::to_string(loaded_rows_);
                node_->parameters["loaded_cols"] = std::to_string(loaded_cols_);
                // memory_bytes is persisted for tabular only (matches
                // pre-refactor behavior — image / audio / text never
                // wrote this param because the bytes are an estimate).
                if (rs.backend == 1 || rs.backend == 2) {
                    node_->parameters["memory_bytes"] = std::to_string(loaded_memory_bytes_);
                }
                spdlog::debug("DataInputDialog: restored {} state for '{}' (backend {})",
                              loader->CategoryName(), loaded_dataset_name_, rs.backend);
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

    node_->parameters["source_type"] = data_input::SourceTypeParam(source_type_);
    node_->parameters["file_category"] = data_input::FileCategoryParam(file_category_);

    // Prune stale per-category params when the user switched category
    // since the last Apply. Without this, switching Tabular → Image
    // would leave `label_column`, `delimiter`, `has_header` etc. in
    // node->parameters forever — they'd show up in saved .cyxgraph
    // files and could mislead the compile gate on the next load.
    //
    // Algorithm: union all keys across all registered loaders (those
    // are the "per-category" keys). Remove any such key that's not in
    // the current loader's schema. Common params (dataset_name,
    // file_path, source_type, etc.) are never in any loader's schema
    // and thus never pruned.
    if (auto* current_loader = cyxwiz::loaders::GetByCategory(file_category_)) {
        std::unordered_set<std::string> all_known;
        for (auto* l : cyxwiz::loaders::All()) {
            for (const auto& p : l->NodeParams()) all_known.insert(p.name);
        }
        std::unordered_set<std::string> current_keys;
        for (const auto& p : current_loader->NodeParams()) current_keys.insert(p.name);

        for (auto it = node_->parameters.begin(); it != node_->parameters.end(); ) {
            if (all_known.count(it->first) && !current_keys.count(it->first)) {
                it = node_->parameters.erase(it);
            } else {
                ++it;
            }
        }
    }

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
        node_->parameters["type"] = data_input::FileTypeParam(detected_type_);
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
        node_->parameters["ml_dataset_type"] = data_input::MLDatasetTypeParam(ml_dataset_type_);
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
        node_->parameters["database_type"] = data_input::DatabaseTypeParam(database_type_);
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
    apply_started_at_ = std::chrono::steady_clock::now();
    last_load_elapsed_ms_ = -1.0f;

    if (!IsApplySupported()) {
        MarkApplyUnsupported(UnsupportedApplyMessage());
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
        // Text load — routes through TextLoader. Branch takes priority
        // over the tabular branch below because a CSV file with Text
        // category selected needs the tokenizer / vocab pipeline, not
        // the plain Arrow path. TextDataset auto-detects file vs
        // directory from the source path, so we just pick the right
        // field (file_path_ for SingleFile, folder_path_ for
        // CorpusSubdirs) and the loader forwards it.
        if (is_loading_async_) {
            spdlog::warn("DataInputDialog: Text Apply ignored - load already in progress");
            apply_in_progress_ = false;
            return;
        }

        auto* loader = cyxwiz::loaders::GetByCategory(
            cyxwiz::loaders::FileCategory::Text);
        if (!loader) {
            apply_status_message_ = "No loader registered for Text";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: TextLoader not registered");
            return;
        }

        const std::string text_source_path =
            text_corpus_ready ? std::string(folder_path_)
                              : std::string(file_path_);

        std::string previous_dataset_name;
        if (auto pit = node_->parameters.find("dataset_name");
            pit != node_->parameters.end()) {
            previous_dataset_name = pit->second;
        }
        loaded_dataset_name_ = GenerateDatasetName();

        cyxwiz::loaders::ApplyContext ctx;
        ctx.dataset_name          = loaded_dataset_name_;
        ctx.source_path           = text_source_path;
        ctx.previous_dataset_name = previous_dataset_name;
        ctx.text_column           = text_column_;
        ctx.text_label_column     = text_label_column_;
        // has_labels: always true for corpus-subdirs layout (label =
        // parent folder name); for single-file layout, true only when
        // the user picked a label column.
        ctx.text_has_labels       = text_corpus_ready ||
                                    (strlen(text_label_column_) > 0);
        ctx.text_tokenizer_type   = text_tokenizer_type_;
        ctx.text_max_length       = text_max_length_;
        ctx.text_lowercase        = text_lowercase_;
        ctx.text_min_freq         = text_min_freq_;
        ctx.text_max_vocab_size   = text_max_vocab_size_;

        std::string err;
        if (!loader->ValidateApplyContext(ctx, err)) {
            apply_status_message_ = err;
            node_->parameters["data_loaded"] = "false";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: text validate - {}", err);
            return;
        }

        auto state = std::make_shared<AsyncLoadState>();
        state->dataset_name = loaded_dataset_name_;
        state->source_path  = text_source_path;
        async_load_state_   = state;
        is_loading_async_   = true;

        node_->parameters["dataset_name"] = loaded_dataset_name_;
        node_->parameters["data_loaded"]  = "false";

        apply_status_message_ = std::string("Tokenizing ") +
            fs::path(text_source_path).filename().string() + "...";
        apply_status_timer_ = 0.0f;

        loading_task_id_ = loader->LaunchAsyncLoad(ctx, state);

        spdlog::info("DataInputDialog: queued async text load (task {}, name '{}')",
                     loading_task_id_, loaded_dataset_name_);

        apply_in_progress_ = false;
        has_changes_ = false;
        return;
    } else if (source_type_ == SourceType::File && strlen(file_path_) > 0 &&
               (file_category_ == FileCategory::Tabular ||
                file_category_ == FileCategory::TimeSeries)) {
        // Tabular / TimeSeries load. All sub-formats (CSV / TSV / TXT /
        // ARFF, Parquet, JSON, Excel, auto-detect) route through
        // TabularLoader, which runs the work on an AsyncTaskManager
        // worker and writes the outcome into AsyncLoadState.
        // PollAsyncLoadResult on the UI thread finalizes the node +
        // status message on the next frame.
        //
        // Behavior change vs pre-refactor: Parquet / JSON / Excel used
        // to run synchronously on the UI thread and would freeze the
        // dialog on large files. They're now async too, matching the
        // CSV path — consistent UX, one code path.
        //
        // The category gate (Tabular || TimeSeries) still lives here
        // because Image / Audio / Text all have stale file_path_
        // values from older sessions and we don't want to mis-dispatch
        // them through the tabular loader.
        if (is_loading_async_) {
            spdlog::warn("DataInputDialog: Apply ignored - load already in progress");
            apply_in_progress_ = false;
            return;
        }

        auto* loader = cyxwiz::loaders::GetByCategory(file_category_);
        if (!loader) {
            // Defensive — TabularLoader is registered at startup and
            // GetByCategory also routes TimeSeries back to Tabular, so
            // this should never fire in practice.
            apply_status_message_ = "No loader registered for this category";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: no loader for tabular category");
            return;
        }

        // Capture the OLD dataset_name before GenerateDatasetName
        // overwrites it. The loader clears the old registry entry on
        // re-Apply so a rename (e.g., user picks a different file)
        // doesn't leak the prior dataset.
        std::string previous_dataset_name;
        if (auto pit = node_->parameters.find("dataset_name");
            pit != node_->parameters.end()) {
            previous_dataset_name = pit->second;
        }
        loaded_dataset_name_ = GenerateDatasetName();

        cyxwiz::loaders::ApplyContext ctx;
        ctx.dataset_name          = loaded_dataset_name_;
        ctx.source_path           = file_path_;
        ctx.previous_dataset_name = previous_dataset_name;

        const char* types[] = {"auto", "csv", "tsv", "json", "parquet",
                               "excel", "hdf5", "feather", "arrow",
                               "txt", "arff"};
        int type_idx = (detected_type_ >= 0 && detected_type_ < 11) ? detected_type_ : 0;
        ctx.detected_file_type = types[type_idx];
        ctx.has_header         = has_header_;
        ctx.delimiter          = custom_delimiter_[0];
        ctx.skip_rows          = skip_rows_;
        ctx.max_rows           = (max_rows_ > 0) ? static_cast<int64_t>(max_rows_) : 0;
        ctx.force_disk_backed  = force_disk_backed_;
        ctx.json_lines         = json_lines_;
        ctx.excel_sheet_idx    = sheet_idx_;
        ctx.label_column       =
            (label_column_idx_ >= 0 &&
             label_column_idx_ < static_cast<int>(available_columns_.size()))
                ? available_columns_[label_column_idx_]
                : std::string();

        std::string err;
        if (!loader->ValidateApplyContext(ctx, err)) {
            apply_status_message_ = err;
            node_->parameters["data_loaded"] = "false";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: tabular validate - {}", err);
            return;
        }

        auto state = std::make_shared<AsyncLoadState>();
        state->dataset_name = loaded_dataset_name_;
        state->source_path  = file_path_;
        async_load_state_   = state;
        is_loading_async_   = true;

        // Provisional node params: mark NOT loaded until the worker
        // finishes. If the user clicks Train mid-load, the compile gate
        // sees data_loaded=false and refuses, instead of trying to
        // train on a dataset that isn't in the registry yet.
        node_->parameters["dataset_name"] = loaded_dataset_name_;
        node_->parameters["data_loaded"]  = "false";

        apply_status_message_ = std::string("Loading ") +
            fs::path(file_path_).filename().string() + "...";
        apply_status_timer_ = 0.0f;

        loading_task_id_ = loader->LaunchAsyncLoad(ctx, state);

        spdlog::info("DataInputDialog: queued async tabular load (task {}, name '{}')",
                     loading_task_id_, loaded_dataset_name_);

        apply_in_progress_ = false;
        has_changes_ = false;
        return;
    } else if (source_type_ == SourceType::File && strlen(folder_path_) > 0 &&
               file_category_ == FileCategory::Audio) {
        // Audio folder load — routes through AudioLoader. Feature
        // extraction still runs lazily inside AudioDataset::GetItem
        // at training time; the loader only scans to surface invalid
        // folders early and stash num_samples / num_classes /
        // feature shape in the registry.
        if (is_loading_async_) {
            spdlog::warn("DataInputDialog: Audio Apply ignored - load already in progress");
            apply_in_progress_ = false;
            return;
        }

        auto* loader = cyxwiz::loaders::GetByCategory(
            cyxwiz::loaders::FileCategory::Audio);
        if (!loader) {
            apply_status_message_ = "No loader registered for Audio";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: AudioLoader not registered");
            return;
        }

        std::string previous_dataset_name;
        if (auto pit = node_->parameters.find("dataset_name");
            pit != node_->parameters.end()) {
            previous_dataset_name = pit->second;
        }
        loaded_dataset_name_ = GenerateDatasetName();

        cyxwiz::loaders::ApplyContext ctx;
        ctx.dataset_name           = loaded_dataset_name_;
        ctx.source_path            = folder_path_;
        ctx.previous_dataset_name  = previous_dataset_name;
        ctx.audio_labels_csv       = audio_labels_csv_;
        ctx.audio_filename_col     = audio_filename_col_;
        ctx.audio_label_col        = audio_label_col_;
        ctx.audio_labeled_subdirs  = (audio_layout_ == AudioLayout::ClassSubdirs);
        ctx.audio_target_sr        = sample_rate_;
        ctx.audio_max_duration     = duration_sec_;
        // Feature-extraction knobs are not yet exposed in the dialog
        // UI; pass the same defaults the pre-refactor code baked in
        // so existing training graphs see identical tensor shapes.
        ctx.audio_n_fft            = 512;
        ctx.audio_hop_length       = 256;
        ctx.audio_n_mels           = 128;
        ctx.audio_n_mfcc           = 13;
        ctx.audio_feature_type     = 1;  // MelSpectrogram

        std::string err;
        if (!loader->ValidateApplyContext(ctx, err)) {
            apply_status_message_ = err;
            node_->parameters["data_loaded"] = "false";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: audio validate - {}", err);
            return;
        }

        auto state = std::make_shared<AsyncLoadState>();
        state->dataset_name = loaded_dataset_name_;
        state->source_path  = folder_path_;
        async_load_state_   = state;
        is_loading_async_   = true;

        node_->parameters["dataset_name"] = loaded_dataset_name_;
        node_->parameters["data_loaded"]  = "false";

        apply_status_message_ = std::string("Scanning ") +
            fs::path(folder_path_).filename().string() + "...";
        apply_status_timer_ = 0.0f;

        loading_task_id_ = loader->LaunchAsyncLoad(ctx, state);

        spdlog::info("DataInputDialog: queued async audio load (task {}, name '{}')",
                     loading_task_id_, loaded_dataset_name_);

        apply_in_progress_ = false;
        has_changes_ = false;
        return;
    } else if (source_type_ == SourceType::File && strlen(folder_path_) > 0 &&
               file_category_ == FileCategory::Image) {
        // Image folder load — routes through ImageLoader. Previously
        // this branch was unguarded (`strlen(folder_path_) > 0`) and
        // relied on process-of-elimination after the earlier
        // audio/text branches; the explicit category gate matches the
        // other refactored branches and makes intent obvious.
        //
        // Pixels are loaded lazily at training time by
        // ImageDatasetBatcher — the loader only scans the folder to
        // populate num_samples / num_classes and surface invalid
        // layouts.
        if (is_loading_async_) {
            spdlog::warn("DataInputDialog: Image Apply ignored - load already in progress");
            apply_in_progress_ = false;
            return;
        }

        auto* loader = cyxwiz::loaders::GetByCategory(
            cyxwiz::loaders::FileCategory::Image);
        if (!loader) {
            apply_status_message_ = "No loader registered for Image";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: ImageLoader not registered");
            return;
        }

        std::string previous_dataset_name;
        if (auto pit = node_->parameters.find("dataset_name");
            pit != node_->parameters.end()) {
            previous_dataset_name = pit->second;
        }
        loaded_dataset_name_ = GenerateDatasetName();

        cyxwiz::loaders::ApplyContext ctx;
        ctx.dataset_name          = loaded_dataset_name_;
        ctx.source_path           = folder_path_;
        ctx.previous_dataset_name = previous_dataset_name;
        ctx.image_labels_csv      = labels_csv_;
        ctx.image_layout          = static_cast<int>(image_layout_);
        ctx.image_width           = target_width_;
        ctx.image_height          = target_height_;
        ctx.image_channels        = rgb_mode_ ? 3 : 1;

        std::string err;
        if (!loader->ValidateApplyContext(ctx, err)) {
            apply_status_message_ = err;
            node_->parameters["data_loaded"] = "false";
            apply_success_ = false;
            apply_status_timer_ = 6.0f;
            apply_in_progress_ = false;
            has_changes_ = false;
            spdlog::error("DataInputDialog: image validate - {}", err);
            return;
        }

        auto state = std::make_shared<AsyncLoadState>();
        state->dataset_name = loaded_dataset_name_;
        state->source_path  = folder_path_;
        async_load_state_   = state;
        is_loading_async_   = true;

        node_->parameters["dataset_name"] = loaded_dataset_name_;
        node_->parameters["data_loaded"]  = "false";

        apply_status_message_ = std::string("Scanning ") +
            fs::path(folder_path_).filename().string() + "...";
        apply_status_timer_ = 0.0f;

        loading_task_id_ = loader->LaunchAsyncLoad(ctx, state);

        spdlog::info("DataInputDialog: queued async image load (task {}, name '{}')",
                     loading_task_id_, loaded_dataset_name_);

        apply_in_progress_ = false;
        has_changes_ = false;
        return;
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

        // Per-backend description format lives in the owning loader
        // via DescribeCompletedLoad. Tabular returns the "N rows, M
        // cols" shape (with a "(disk-backed)" suffix for Parquet);
        // Image / Audio / Text format class + vocab counts instead.
        fs::path p(state->source_path);
        auto* loader = cyxwiz::loaders::GetByBackendTag(state->backend);
        if (loader) {
            auto desc = loader->DescribeCompletedLoad(*state);
            loaded_memory_is_estimate_ = desc.memory_is_estimate;
            node_->description = p.filename().string() + "\n" + desc.node_description_suffix;
            apply_status_message_ = state->message.empty()
                ? desc.default_status_message
                : state->message;
        } else {
            // Defensive — every loader we register exposes a BackendTag
            // that maps back here. If we ever hit this, the backend
            // field in AsyncLoadState was set to a value no loader
            // claims.
            loaded_memory_is_estimate_ = false;
            node_->description = p.filename().string();
            apply_status_message_ = state->message.empty()
                ? std::string("Loaded ") + p.filename().string()
                : state->message;
            spdlog::warn("DataInputDialog: no loader for backend tag {}",
                         state->backend);
        }
        if (!state->audit_message.empty()) {
            apply_status_message_ += " " + state->audit_message;
            node_->parameters["audit_errors"] = std::to_string(state->audit_errors);
            node_->parameters["audit_warnings"] = std::to_string(state->audit_warnings);
            audit_issue_lines_ = state->audit_issue_lines;
            if (state->audit_errors > 0) {
                if (loader) {
                    loader->Unregister(state->dataset_name);
                }
                node_->parameters["data_loaded"] = "false";
                data_load_state_ = DataLoadState::NotLoaded;
                apply_success_ = false;
                apply_status_message_ =
                    "Apply refused by dataset audit. " + state->audit_message;
            }
        } else {
            node_->parameters.erase("audit_errors");
            node_->parameters.erase("audit_warnings");
            audit_issue_lines_.clear();
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

    if (apply_started_at_ != std::chrono::steady_clock::time_point{}) {
        const auto elapsed = std::chrono::steady_clock::now() - apply_started_at_;
        last_load_elapsed_ms_ =
            std::chrono::duration<float, std::milli>(elapsed).count();
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
            if (ImGui::BeginTabItem("Audit")) {
                RenderAuditTab();
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

    // Dataset summary
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    RenderDatasetSummaryPanel();

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

} // namespace gui
