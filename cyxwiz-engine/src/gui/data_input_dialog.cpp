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
#include <nlohmann/json.hpp>

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
        }
        if (node_->parameters.count("type") && !node_->parameters["type"].empty()) {
            detected_type_ = data_input::FileTypeFromParam(
                node_->parameters["type"],
                detected_type_);
        } else if (node_->parameters.count("file_type") &&
                   !node_->parameters["file_type"].empty()) {
            detected_type_ = data_input::FileTypeFromParam(
                node_->parameters["file_type"],
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
        if (node_->parameters.count("encoding")) {
            try {
                encoding_idx_ = std::clamp(
                    std::stoi(node_->parameters["encoding"]), 0, 6);
            } catch (...) {
                encoding_idx_ = 0;
            }
        }
        if (node_->parameters.count("folder_path")) {
            strncpy(folder_path_, node_->parameters["folder_path"].c_str(), sizeof(folder_path_) - 1);
        }
        if (node_->parameters.count("dataset_name")) {
            strncpy(dataset_name_, node_->parameters["dataset_name"].c_str(), sizeof(dataset_name_) - 1);
        }
        if (node_->parameters.count("dataset_role")) {
            const std::string& role = node_->parameters["dataset_role"];
            if (role == "dev") {
                dataset_role_idx_ = 1;
            } else if (role == "test") {
                dataset_role_idx_ = 2;
            } else {
                dataset_role_idx_ = 0;
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
        if (strlen(file_path_) > 0) {
            RefreshColumnList();
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

void DataInputDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    ResetPreviewPaging();
    preview_loaded_ = false;
    has_changes_ = false;
}

void DataInputDialog::RenderContent() {
    // Pick up the async load result, if the worker has finished, and apply
    // it to the dialog/node state. Cheap when no load is in flight.
    PollAsyncLoadResult();
    PollPreviewPageResult();

    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec4 accent = style.Colors[ImGuiCol_HeaderActive];
    const ImVec4 muted = style.Colors[ImGuiCol_TextDisabled];

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 8.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));

    ImGui::TextColored(accent, "Connect data");
    ImGui::TextColored(muted,
        "Choose a source, inspect the contract, then apply it to this DataInput node.");
    ImGui::Spacing();
    RenderSourceSelector();
    ImGui::Spacing();

    // KNIME-style tab bar at TOP based on source type
    if (source_type_ == SourceType::File) {
        if (ImGui::BeginTabBar("DataInputTabs", ImGuiTabBarFlags_None)) {
            if (ImGui::BeginTabItem("Settings")) {
                RenderFileSource();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Preview")) {
                RenderPreviewPanel();
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
    ImGui::Spacing();
    RenderDatasetSummaryPanel();
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
    profile_computed_ = false;
    column_stats_.clear();

    if (node_) {
        node_->parameters["data_loaded"] = "false";
    }

    spdlog::info("Dataset '{}' unloaded from memory", loaded_dataset_name_);
}

} // namespace gui
