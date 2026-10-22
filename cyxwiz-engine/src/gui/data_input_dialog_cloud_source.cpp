// The Data Input dialog's CyxCloud source (TOFIX164 slice 4b.1, mockup
// approved 2026-10-22): the source choice above the file field, the dataset
// picker and file list of the signed-in account, the Detected line and what
// the graph remembers. Words and rules come from
// core/cloud_source_presentation; this file only draws and lists.
#include "node_config_dialog.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../auth/auth_client.h"
#include "../core/cloud_connection.h"
#include "../core/cloud_dataset_file_format.h"
#include "data_input_capabilities.h"
#include "../core/engine_config.h"
#include "../network/datastream_client.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace gui {

namespace view = cyxwiz::cloud_source_view;

namespace {

void FieldLabel(const char* label) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "%s", label);
}

ImVec4 ToneColor(view::Tone tone) {
    const auto& t = cyxwiz::ui::CurrentTokens();
    switch (tone) {
        case view::Tone::Accent: return t.accent;
        case view::Tone::Success: return t.success;
        case view::Tone::Warning: return t.warning;
        case view::Tone::Error: return t.error;
        default: return t.text_dim;
    }
}

}  // namespace

// What one listing thread brings back; the dialog polls it each frame.
struct DataInputDialog::CloudListTask {
    std::mutex mutex;
    bool done = false;
    bool ok = false;
    std::string error;
    std::string dataset_id;  // for a file listing
    std::vector<view::DatasetIn> datasets;
    std::vector<view::FileIn> files;
};

bool DataInputDialog::CloudSourceChosen() const { return source_kind_ == SourceKind::CyxCloud; }

void DataInputDialog::RefreshCloudDatasets() {
    auto* client = cyxwiz::cloud::Client();
    if (!client || !client->IsConnected()) return;
    auto task = std::make_shared<CloudListTask>();
    cloud_datasets_task_ = task;
    cloud_listed_ = true;
    std::thread([client, task] {
        std::vector<network::CloudDatasetInfo> datasets;
        const bool ok = client->ListDatasets(datasets, 100, 0, /*include_shared=*/true);
        std::lock_guard<std::mutex> lock(task->mutex);
        if (ok) {
            for (const auto& d : datasets) {
                view::DatasetIn in;
                in.id = d.id;
                in.name = d.name;
                in.trust = static_cast<int>(d.trust_level);
                in.size_bytes = d.total_size_bytes;
                in.file_count = d.file_count;
                task->datasets.push_back(std::move(in));
            }
        } else {
            task->error = client->GetLastError();
        }
        task->ok = ok;
        task->done = true;
    }).detach();
}

void DataInputDialog::RefreshCloudFiles(const std::string& dataset_id) {
    auto* client = cyxwiz::cloud::Client();
    if (!client || !client->IsConnected() || dataset_id.empty()) return;
    auto task = std::make_shared<CloudListTask>();
    task->dataset_id = dataset_id;
    cloud_files_task_ = task;
    cloud_files_.clear();
    cloud_files_of_ = dataset_id;
    std::thread([client, task, dataset_id] {
        network::CloudDatasetInfo info;
        std::vector<network::CloudFileInfo> files;
        const bool ok = client->GetDatasetInfo(dataset_id, info, files);
        std::lock_guard<std::mutex> lock(task->mutex);
        if (ok) {
            for (const auto& f : files) {
                view::FileIn in;
                in.id = f.id;
                in.name = std::filesystem::path(f.path).filename().string();
                in.size_bytes = f.size_bytes;
                task->files.push_back(std::move(in));
            }
        } else {
            task->error = client->GetLastError();
        }
        task->ok = ok;
        task->done = true;
    }).detach();
}

void DataInputDialog::PollCloudLists() {
    if (cloud_datasets_task_) {
        std::lock_guard<std::mutex> lock(cloud_datasets_task_->mutex);
        if (cloud_datasets_task_->done) {
            cloud_datasets_error_ = cloud_datasets_task_->ok ? std::string() : cloud_datasets_task_->error;
            if (cloud_datasets_task_->ok) cloud_datasets_ = std::move(cloud_datasets_task_->datasets);
            cloud_datasets_task_.reset();
            // A reopened node: highlight what the graph remembers and list its files.
            if (cloud_selected_dataset_.empty() && cloud_remembered_.Any()) {
                for (const auto& d : cloud_datasets_) {
                    if (d.id == cloud_remembered_.dataset_id) {
                        cloud_selected_dataset_ = d.id;
                        cloud_selected_file_ = cloud_remembered_.file_id;
                    }
                }
            }
            if (!cloud_selected_dataset_.empty() && cloud_files_of_ != cloud_selected_dataset_) {
                RefreshCloudFiles(cloud_selected_dataset_);
            }
        }
    }
    if (cloud_files_task_) {
        std::lock_guard<std::mutex> lock(cloud_files_task_->mutex);
        if (cloud_files_task_->done) {
            if (cloud_files_task_->ok && cloud_files_task_->dataset_id == cloud_selected_dataset_) {
                cloud_files_ = std::move(cloud_files_task_->files);
            } else if (!cloud_files_task_->ok) {
                cloud_datasets_error_ = cloud_files_task_->error;
            }
            cloud_files_task_.reset();
        }
    }
}

view::Input DataInputDialog::CloudInput() const {
    view::Input in;
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    in.signed_in = auth.IsAuthenticated();
    if (in.signed_in) in.account = auth.GetUserInfo().email;
    auto* client = cyxwiz::cloud::Client();
    in.browser_opened = client != nullptr;
    in.connected = client != nullptr && client->IsConnected();
    in.gateway = cyxwiz::core::EngineConfig::Instance().GetCloudGatewayAddress();
    in.connect_error = cyxwiz::cloud::LastConnectError();
    in.loading_datasets = cloud_datasets_task_ != nullptr;
    in.datasets_error = cloud_datasets_error_;
    in.datasets = cloud_datasets_;
    in.selected_dataset_id = cloud_selected_dataset_;
    in.loading_files = cloud_files_task_ != nullptr;
    in.files = cloud_files_;
    in.selected_file_id = cloud_selected_file_;
    in.remembered = cloud_remembered_;
    return in;
}

void DataInputDialog::RenderSourceChoice() {
    static const char* const kKinds[] = {"This computer", "CyxCloud"};
    int kind = source_kind_ == SourceKind::CyxCloud ? 1 : 0;
    if (cyxwiz::ui::SegmentedControl("##source_kind", kKinds, 2, &kind)) {
        source_kind_ = kind == 1 ? SourceKind::CyxCloud : SourceKind::ThisComputer;
        has_changes_ = true;
    }
}

void DataInputDialog::RenderCloudSource() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    PollCloudLists();
    const view::Input input = CloudInput();
    if (input.connected && !cloud_listed_) RefreshCloudDatasets();
    const view::View v = view::Build(input);
    cloud_can_apply_ = v.can_apply;
    cloud_apply_blocked_reason_ = v.apply_blocked_reason;

    // The connection line, beside the source control.
    ImGui::SameLine();
    ImGui::TextColored(ToneColor(v.connection_tone), "%s", v.connection.c_str());
    if (v.offer_try_again) {
        ImGui::SameLine();
        if (cyxwiz::ui::LinkButton("Try again")) {
            if (auto* client = cyxwiz::cloud::Client()) {
                if (client->Connect(input.gateway)) cyxwiz::cloud::SetLastConnectError({});
                else cyxwiz::cloud::SetLastConnectError(client->GetLastError());
            }
            cloud_listed_ = false;
        }
    }
    if (v.offer_open_browser) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "(uploads: Cloud Browser in the sidebar)");
    }

    if (input.connected) {
        // Dataset picker and the files of the chosen dataset, side by side.
        if (ImGui::BeginTable("##cloud_pick", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            FieldLabel("Dataset");
            const char* preview = "Pick a dataset";
            for (const auto& row : v.datasets) {
                if (row.selected) preview = row.name.c_str();
            }
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (ImGui::BeginCombo("##cloud_dataset", preview)) {
                for (const auto& row : v.datasets) {
                    ImGui::PushID(row.id.c_str());
                    if (ImGui::Selectable(row.name.c_str(), row.selected)) {
                        cloud_selected_dataset_ = row.id;
                        cloud_selected_file_.clear();
                        cloud_files_.clear();
                        RefreshCloudFiles(row.id);
                        has_changes_ = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %s", row.trust.c_str(), row.meta.c_str());
                    ImGui::PopID();
                }
                ImGui::EndCombo();
            }
            if (!v.datasets_hint.empty()) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_faint, "%s", v.datasets_hint.c_str());
                ImGui::PopTextWrapPos();
            }
            if (cyxwiz::ui::LinkButton("Refresh")) RefreshCloudDatasets();

            ImGui::TableSetColumnIndex(1);
            FieldLabel("File in the dataset");
            if (!v.files_hint.empty()) {
                ImGui::TextColored(t.text_faint, "%s", v.files_hint.c_str());
            }
            for (const auto& row : v.files) {
                ImGui::PushID(row.id.c_str());
                if (!row.trainable) ImGui::PushStyleColor(ImGuiCol_Text, t.text_faint);
                if (ImGui::Selectable(row.name.c_str(), row.selected)) {
                    cloud_selected_file_ = row.id;
                    has_changes_ = true;
                }
                if (!row.trainable) ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::TextColored(t.text_dim, "%s", row.meta.c_str());
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }

    if (!v.detected.empty()) {
        FieldLabel("Detected");
        ImGui::TextUnformatted(v.detected.c_str());
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", v.detected_note.c_str());
        ImGui::PopTextWrapPos();
    }

    if (!v.remembered.empty()) {
        FieldLabel("What the graph remembers");
        if (ImGui::BeginTable("##cloud_remembered", 2, ImGuiTableFlags_SizingStretchSame)) {
            for (const auto& pair : v.remembered) {
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", pair.key.c_str());
                ImGui::SameLine();
                ImGui::TextUnformatted(pair.value.c_str());
            }
            ImGui::EndTable();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", v.remembered_note.c_str());
        ImGui::PopTextWrapPos();
    }

    ImGui::Spacing();
    ImGui::TextColored(t.text_faint,
                       "Columns and the row use (table, text, token tagging, time series) come with the first "
                       "sample read; a node trains from the file as it is.");
}

void DataInputDialog::ApplyCloudSource() {
    if (!node_) return;
    const view::View v = view::Build(CloudInput());
    if (!v.can_apply) {
        apply_success_ = false;
        apply_status_message_ = v.apply_blocked_reason;
        return;
    }
    view::Remembered chosen;
    for (const auto& d : cloud_datasets_) {
        if (d.id == cloud_selected_dataset_) {
            chosen.dataset_id = d.id;
            chosen.dataset_name = d.name;
        }
    }
    for (const auto& f : cloud_files_) {
        if (f.id == cloud_selected_file_) {
            chosen.file_id = f.id;
            chosen.file_name = f.name;
            chosen.size_bytes = static_cast<uint64_t>(f.size_bytes);
        }
    }
    // A local dataset this node loaded before the switch is let go.
    if (!loaded_dataset_name_.empty()) UnloadDataset();

    auto& p = node_->parameters;
    p["source_type"] = "cloud";
    p["cloud_dataset_id"] = chosen.dataset_id;
    p["cloud_dataset_name"] = chosen.dataset_name;
    p["cloud_file_id"] = chosen.file_id;
    p["cloud_file_name"] = chosen.file_name;
    p["cloud_size_bytes"] = std::to_string(chosen.size_bytes);
    p["file_type"] = cyxwiz::CloudFileExtensionLower(chosen.file_name) == ".parquet" ? "parquet" : "arrow";
    p["file_category"] = "tabular";
    p["configured"] = "true";
    p["data_loaded"] = "false";
    p.erase("file_path");
    p.erase("folder_path");
    p.erase("type");
    if (p["dataset_name"].empty() || p["dataset_name"] != loaded_dataset_name_) {
        p["dataset_name"] = data_input::GenerateDatasetName(chosen.file_name, "");
    }
    node_->description = chosen.file_name + " (CyxCloud)";

    cloud_remembered_ = chosen;
    loaded_dataset_name_.clear();
    data_load_state_ = DataLoadState::NotLoaded;
    apply_success_ = true;
    apply_in_progress_ = false;
    apply_status_message_ = view::ApplyMessage(chosen);
    has_changes_ = false;
    spdlog::info("Data Input '{}': CyxCloud file {} ({}) in dataset {} ({}), {} bytes", node_->name, chosen.file_name,
                 chosen.file_id, chosen.dataset_name, chosen.dataset_id, chosen.size_bytes);
}

}  // namespace gui
