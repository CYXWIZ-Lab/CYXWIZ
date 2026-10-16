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
#include "data_studio/profile_view.h"
#include "node_editor.h"
#include "ui_tokens.h"
#include "loaders/audio_loader.h"
#include "loaders/data_loader.h"
#include "loaders/image_loader.h"
#include "loaders/tabular_loader.h"
#include "loaders/text_loader.h"
#include "../core/data_registry.h"
#include "../core/formats/audio_dataset.h"
#include "../core/arrow_dataset.h"
#include "../core/parquet_backed_dataset.h"
#include "../core/async_task_manager.h"
#include "../core/project_data_path.h"
#include "../core/project_manager.h"
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
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace gui {

// ==================== DataInputDialog ====================

DataInputDialog::DataInputDialog(MLNode* node)
    : NodeConfigDialog("Data Input", node)
{
    if (node_) {
        if (node_->parameters.count("archive_member")) {
            strncpy(archive_member_, node_->parameters["archive_member"].c_str(), sizeof(archive_member_) - 1);
        }
        // Apply stores sources inside the project project-relative; the
        // dialog works on the resolved path, as the loaders do, so Preview
        // and the registry lookup open the same file after a reopen.
        const std::string& project_root = cyxwiz::ProjectManager::Instance().GetProjectRoot();
        if (node_->parameters.count("file_path")) {
            const std::string resolved =
                cyxwiz::ResolveProjectDataPath(node_->parameters["file_path"], project_root);
            strncpy(file_path_, resolved.c_str(), sizeof(file_path_) - 1);
            DetectFileType();
            DetectFileCategory();
        }
        if (node_->parameters.count("file_type") &&
            !node_->parameters["file_type"].empty() &&
            node_->parameters["file_type"] != "auto") {
            detected_type_ = data_input::FileTypeFromParam(
                node_->parameters["file_type"],
                detected_type_);
        } else if (node_->parameters.count("type") &&
                   !node_->parameters["type"].empty()) {
            detected_type_ = data_input::FileTypeFromParam(
                node_->parameters["type"],
                detected_type_);
        }

        // Restore parser state before reading the source schema. Column
        // discovery is parser-dependent: for preambled CSV files the first
        // row after skip_rows is the header. Loading columns before restoring
        // these values leaves the label selector bound to metadata text.
        if (node_->parameters.count("has_header")) {
            has_header_ = node_->parameters["has_header"] == "true";
        } else if (node_->parameters.count("header")) {
            has_header_ = node_->parameters["header"] == "true";
        }
        if (node_->parameters.count("delimiter") &&
            !node_->parameters["delimiter"].empty()) {
            strncpy(custom_delimiter_,
                    node_->parameters["delimiter"].c_str(),
                    sizeof(custom_delimiter_) - 1);
            custom_delimiter_[sizeof(custom_delimiter_) - 1] = '\0';
        }
        if (node_->parameters.count("decimal_point") &&
            !node_->parameters["decimal_point"].empty()) {
            const char decimal_point = node_->parameters["decimal_point"][0];
            decimal_point_ = (decimal_point == ',') ? ',' : '.';
        }
        if (node_->parameters.count("quote_char")) {
            strncpy(quote_char_, node_->parameters["quote_char"].c_str(), sizeof(quote_char_) - 1);
            quote_char_[sizeof(quote_char_) - 1] = '\0';
        }
        if (node_->parameters.count("missing_value_tokens")) {
            strncpy(missing_value_tokens_,
                    node_->parameters["missing_value_tokens"].c_str(),
                    sizeof(missing_value_tokens_) - 1);
            missing_value_tokens_[sizeof(missing_value_tokens_) - 1] = '\0';
        }
        if (node_->parameters.count("skip_rows")) {
            try {
                skip_rows_ = std::max(0, std::stoi(node_->parameters["skip_rows"]));
            } catch (...) {
                skip_rows_ = 0;
            }
        }
        if (node_->parameters.count("max_rows")) {
            try {
                max_rows_ = std::max(0, std::stoi(node_->parameters["max_rows"]));
            } catch (...) {
                max_rows_ = 0;
            }
        }
        if (node_->parameters.count("folder_path")) {
            const std::string resolved =
                cyxwiz::ResolveProjectDataPath(node_->parameters["folder_path"], project_root);
            strncpy(folder_path_, resolved.c_str(), sizeof(folder_path_) - 1);
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

        if (node_->parameters.count("labels_csv")) {
            strncpy(labels_csv_, node_->parameters["labels_csv"].c_str(), sizeof(labels_csv_) - 1);
            labels_csv_[sizeof(labels_csv_) - 1] = '\0';
        }
        // Audio loading: sample rate and length (it was never saved; the
        // examples call it max_duration).
        if (node_->parameters.count("sample_rate")) {
            try { sample_rate_ = (std::max)(1, std::stoi(node_->parameters["sample_rate"])); }
            catch (...) {}
        }
        if (node_->parameters.count("max_duration")) {
            try { duration_sec_ = (std::max)(0.1f, std::stof(node_->parameters["max_duration"])); }
            catch (...) {}
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
        LoadSequenceSettings();
        if (strlen(file_path_) > 0) {
            RefreshColumnList();
        }
        if (IsHdf5Source()) {
            SyncHdf5InspectorSource();
            hdf5_inspector_.RestoreSettings(node_->parameters);
        }
        if (node_->parameters.count("selected_columns") &&
            !available_columns_.empty()) {
            try {
                const auto persisted = nlohmann::json::parse(
                    node_->parameters["selected_columns"]);
                if (persisted.is_array() && !persisted.empty()) {
                    std::unordered_set<std::string> included;
                    for (const auto& value : persisted) {
                        if (value.is_string()) {
                            included.insert(value.get<std::string>());
                        }
                    }
                    selected_columns_.assign(available_columns_.size(), false);
                    for (std::size_t i = 0; i < available_columns_.size(); ++i) {
                        selected_columns_[i] =
                            included.count(available_columns_[i]) > 0;
                    }
                    select_all_columns_ = std::all_of(
                        selected_columns_.begin(), selected_columns_.end(),
                        [](bool selected) { return selected; });
                }
            } catch (const std::exception& e) {
                spdlog::warn(
                    "DataInputDialog: ignored invalid selected_columns: {}",
                    e.what());
            }
        }
        if (node_->parameters.count("label_column") &&
            !node_->parameters["label_column"].empty()) {
            const std::string& label_column = node_->parameters["label_column"];
            for (int i = 0; i < static_cast<int>(available_columns_.size()); ++i) {
                if (available_columns_[i] == label_column) {
                    label_column_idx_ = i;
                    if (i < static_cast<int>(selected_columns_.size())) {
                        selected_columns_[static_cast<std::size_t>(i)] = true;
                    }
                    break;
                }
            }
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
        if (IsHdf5Source()) {
            BeginHdf5LoadedSourceVerification();
        } else if (node_->parameters.count("dataset_name") && !node_->parameters["dataset_name"].empty()) {
            loaded_dataset_name_ = node_->parameters["dataset_name"];
            loaded_memory_is_estimate_ = false;

            auto* loader = cyxwiz::loaders::GetByRegisteredDataset(loaded_dataset_name_);
            cyxwiz::loaders::RestoreState rs;
            if (loader && loader->RestoreFromRegistry(loaded_dataset_name_, *node_, rs) && rs.found) {
                loaded_rows_ = rs.rows;
                loaded_cols_ = rs.cols;
                loaded_classes_ = rs.classes;
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

void DataInputDialog::Reset() {
    hdf5_loaded_source_.Cancel();
    hdf5_inspector_.Reset();
    hdf5_source_path_.clear();
    if (!node_) return;
    node_->parameters = original_params_;
    RestoreOriginalHdf5Source();
    if (IsHdf5Source()) BeginHdf5LoadedSourceVerification();
    ResetPreviewPaging();
    preview_loaded_ = false;
    has_changes_ = false;
}

void DataInputDialog::RenderContent() {
    SyncHdf5InspectorSource();
    hdf5_inspector_.Poll();
    PollHdf5LoadedSourceVerification();
    // Pick up the async load result, if the worker has finished, and apply
    // it to the dialog/node state. Cheap when no load is in flight.
    PollAsyncLoadResult();
    PollPreviewPageResult();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 8.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    // The data strip stays above the tabs, so the load state shows on each.
    RenderDataStrip();
    ImGui::Spacing();

    if (ImGui::BeginTabBar("DataInputTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Source")) {
            RenderFileSource();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Preview")) {
            RenderPreviewPanel();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Profile")) {
            RenderDataProfilingTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Rows and columns")) {
            RenderRowsAndColumnsTab();
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

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void DataInputDialog::UnloadDataset() {
    if (loaded_dataset_name_.empty()) return;

    ResetPreviewPaging();
    preview_loaded_ = false;

    auto& registry = cyxwiz::DataRegistry::Instance();
    registry.UnloadDataset(loaded_dataset_name_);

    data_load_state_ = DataLoadState::NotLoaded;
    loaded_rows_ = 0;
    loaded_cols_ = 0;
    loaded_memory_bytes_ = 0;
    profile_view_.reset();
    profile_dataset_.clear();

    if (node_) {
        node_->parameters["data_loaded"] = "false";
    }

    spdlog::info("Dataset '{}' unloaded from memory", loaded_dataset_name_);
}

} // namespace gui
