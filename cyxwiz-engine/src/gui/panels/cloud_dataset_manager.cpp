#include "cloud_dataset_manager.h"
#include "../icons.h"
#include <imgui.h>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace gui {

static std::string FormatExpiry(int64_t timestamp) {
    if (timestamp == 0) return "Never";
    std::time_t time = static_cast<std::time_t>(timestamp);
    std::tm* tm = std::localtime(&time);
    if (!tm) return "Invalid";
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M");
    return oss.str();
}

CloudDatasetManagerPanel::CloudDatasetManagerPanel()
    : Panel("Dataset Manager", false) {
}

CloudDatasetManagerPanel::~CloudDatasetManagerPanel() = default;

void CloudDatasetManagerPanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_SLIDERS " Dataset Manager###DatasetManager", &visible_)) {
        focused_ = ImGui::IsWindowFocused();

        // Check connection
        if (!datastream_client_ || !datastream_client_->IsConnected()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                ICON_FA_TRIANGLE_EXCLAMATION " Not connected to CyxCloud Gateway");
            ImGui::End();
            return;
        }

        // Header toolbar
        if (ImGui::Button(ICON_FA_PLUS " Create Dataset")) {
            show_create_wizard_ = true;
            wizard_step_ = 0;
            std::memset(new_dataset_name_, 0, sizeof(new_dataset_name_));
            std::memset(new_dataset_description_, 0, sizeof(new_dataset_description_));
            std::memset(new_dataset_schema_, 0, sizeof(new_dataset_schema_));
            selected_file_ids_.clear();
        }

        if (has_dataset_) {
            ImGui::SameLine();
            if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh")) {
                // Re-fetch dataset info
                if (datastream_client_->GetDatasetInfo(current_dataset_.id,
                                                        current_dataset_, current_files_)) {
                    // Success
                }
            }
        }

        ImGui::Separator();

        // Main content
        if (!has_dataset_) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "No dataset selected. Use the Cloud Browser to select a dataset.");
        } else {
            // Tabs for different sections
            if (ImGui::BeginTabBar("##DatasetManagerTabs")) {
                if (ImGui::BeginTabItem(ICON_FA_CIRCLE_INFO " Info")) {
                    RenderDatasetInfo();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_CODE " Schema")) {
                    RenderSchemaEditor();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_KEY " Access Tokens")) {
                    RenderAccessTokens();
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem(ICON_FA_PLAY " Streaming")) {
                    RenderStreamingConfig();
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }

        // Error display
        if (!last_error_.empty() && error_time_ > 0) {
            error_time_ -= ImGui::GetIO().DeltaTime;
            ImGui::Separator();
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                ICON_FA_CIRCLE_EXCLAMATION " %s", last_error_.c_str());
        }

        // Create dataset wizard (modal)
        if (show_create_wizard_) {
            RenderCreateDatasetWizard();
        }

        // Token creation dialog
        if (show_token_dialog_) {
            ImGui::OpenPopup("Create Access Token");
        }

        if (ImGui::BeginPopupModal("Create Access Token", nullptr,
                                    ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Create a new access token for streaming");
            ImGui::Separator();

            ImGui::Text("Node ID (optional):");
            ImGui::InputText("##token_node", token_node_id_, sizeof(token_node_id_));
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(Leave empty for any node)");

            ImGui::Text("Expires in (hours):");
            ImGui::SliderInt("##token_ttl", &token_ttl_hours_, 1, 720, "%d hours");

            ImGui::Text("Scopes:");
            ImGui::Checkbox("Read", &token_scope_read_);
            ImGui::SameLine();
            ImGui::Checkbox("Stream", &token_scope_stream_);

            ImGui::Separator();

            if (ImGui::Button("Create", ImVec2(100, 0))) {
                CreateNewToken();
                show_token_dialog_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(100, 0))) {
                show_token_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }
    ImGui::End();
}

void CloudDatasetManagerPanel::SetDataset(const network::CloudDatasetInfo& dataset) {
    current_dataset_ = dataset;
    has_dataset_ = true;

    // Fetch full info
    if (datastream_client_) {
        datastream_client_->GetDatasetInfo(dataset.id, current_dataset_, current_files_);
    }
}

void CloudDatasetManagerPanel::ClearDataset() {
    current_dataset_ = network::CloudDatasetInfo{};
    current_files_.clear();
    tokens_.clear();
    has_dataset_ = false;
}

void CloudDatasetManagerPanel::RenderDatasetInfo() {
    ImGui::Text(ICON_FA_DATABASE " %s", current_dataset_.name.c_str());

    ImGui::Columns(2, nullptr, false);

    ImGui::Text("ID:"); ImGui::NextColumn();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", current_dataset_.id.c_str());
    ImGui::NextColumn();

    ImGui::Text("Owner:"); ImGui::NextColumn();
    ImGui::Text("%s", current_dataset_.owner_id.c_str());
    ImGui::NextColumn();

    ImGui::Text("Size:"); ImGui::NextColumn();
    ImGui::Text("%s", current_dataset_.GetSizeString().c_str());
    ImGui::NextColumn();

    ImGui::Text("Files:"); ImGui::NextColumn();
    ImGui::Text("%d", current_dataset_.file_count);
    ImGui::NextColumn();

    ImGui::Text("Version:"); ImGui::NextColumn();
    ImGui::Text("%d", current_dataset_.version);
    ImGui::NextColumn();

    ImGui::Text("Trust Level:"); ImGui::NextColumn();
    ImGui::Text("%s", current_dataset_.GetTrustLevelString().c_str());
    ImGui::NextColumn();

    ImGui::Text("Created:"); ImGui::NextColumn();
    ImGui::Text("%s", current_dataset_.GetCreatedAtString().c_str());
    ImGui::NextColumn();

    ImGui::Columns(1);

    if (!current_dataset_.description.empty()) {
        ImGui::Separator();
        ImGui::Text("Description:");
        ImGui::TextWrapped("%s", current_dataset_.description.c_str());
    }

    // Files list
    if (!current_files_.empty()) {
        ImGui::Separator();
        ImGui::Text("Files (%zu):", current_files_.size());

        if (ImGui::BeginChild("##files", ImVec2(0, 150), true)) {
            for (const auto& f : current_files_) {
                ImGui::Text(ICON_FA_FILE " %s", f.path.c_str());
                ImGui::SameLine(300);
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                    "(%lld bytes)", static_cast<long long>(f.size_bytes));
            }
        }
        ImGui::EndChild();
    }
}

void CloudDatasetManagerPanel::RenderSchemaEditor() {
    ImGui::Text("Dataset Schema (JSON):");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "Define the schema for parsing dataset items");

    ImGui::Separator();

    // Schema editor
    static char schema_buffer[8192] = "";
    if (has_dataset_ && schema_buffer[0] == '\0' && !current_dataset_.schema_json.empty()) {
        strncpy(schema_buffer, current_dataset_.schema_json.c_str(), sizeof(schema_buffer) - 1);
    }

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
    ImGui::InputTextMultiline("##schema", schema_buffer, sizeof(schema_buffer),
                               ImVec2(-1, 300), flags);

    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save Schema")) {
        // TODO: Update dataset schema via API
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f),
            "Schema update not implemented yet");
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE_LEFT " Reset")) {
        if (!current_dataset_.schema_json.empty()) {
            strncpy(schema_buffer, current_dataset_.schema_json.c_str(), sizeof(schema_buffer) - 1);
        } else {
            schema_buffer[0] = '\0';
        }
    }

    // Schema templates
    ImGui::Separator();
    ImGui::Text("Templates:");
    if (ImGui::Button("Image Classification")) {
        strcpy(schema_buffer, R"({
  "type": "image_classification",
  "image_column": "image",
  "label_column": "label",
  "image_format": "png",
  "normalize": true,
  "resize": [224, 224]
})");
    }
    ImGui::SameLine();
    if (ImGui::Button("Text Classification")) {
        strcpy(schema_buffer, R"({
  "type": "text_classification",
  "text_column": "text",
  "label_column": "label",
  "tokenizer": "bert-base-uncased",
  "max_length": 512
})");
    }
    ImGui::SameLine();
    if (ImGui::Button("CSV Tabular")) {
        strcpy(schema_buffer, R"({
  "type": "tabular",
  "format": "csv",
  "columns": {
    "feature1": "float",
    "feature2": "float",
    "label": "int"
  }
})");
    }
}

void CloudDatasetManagerPanel::RenderAccessTokens() {
    ImGui::Text("Access Tokens");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "Tokens allow server nodes to stream data from this dataset");

    ImGui::Separator();

    if (ImGui::Button(ICON_FA_PLUS " Create Token")) {
        show_token_dialog_ = true;
        std::memset(token_node_id_, 0, sizeof(token_node_id_));
        token_ttl_hours_ = 24;
        token_scope_read_ = true;
        token_scope_stream_ = true;
    }

    ImGui::Separator();

    if (tokens_.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            "No access tokens created");
    } else {
        ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("##tokens", 4, flags)) {
            ImGui::TableSetupColumn("Token ID", 0, 150);
            ImGui::TableSetupColumn("Scopes", 0, 100);
            ImGui::TableSetupColumn("Expires", 0, 120);
            ImGui::TableSetupColumn("Actions", 0, 80);
            ImGui::TableHeadersRow();

            for (size_t i = 0; i < tokens_.size(); i++) {
                const auto& token = tokens_[i];
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s...", token.token_preview.c_str());

                ImGui::TableSetColumnIndex(1);
                std::string scopes_str;
                for (const auto& s : token.scopes) {
                    if (!scopes_str.empty()) scopes_str += ", ";
                    scopes_str += s;
                }
                ImGui::Text("%s", scopes_str.c_str());

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%s", FormatExpiry(token.expires_at).c_str());

                ImGui::TableSetColumnIndex(3);
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::SmallButton(ICON_FA_TRASH)) {
                    RevokeToken(token.token_id);
                }
                ImGui::PopID();
            }

            ImGui::EndTable();
        }
    }
}

void CloudDatasetManagerPanel::RenderStreamingConfig() {
    ImGui::Text("Streaming Configuration");
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
        "Configure how data is streamed to training nodes");

    ImGui::Separator();

    // Configuration options
    ImGui::SliderInt("Batch Size", &stream_batch_size_, 1, 256);
    ImGui::SliderInt("Prefetch Batches", &stream_prefetch_, 1, 32);
    ImGui::Checkbox("Shuffle", &stream_shuffle_);

    if (stream_shuffle_) {
        ImGui::InputScalar("Seed", ImGuiDataType_S64, &stream_seed_);
    }

    const char* trust_items[] = {"Self Only", "Signed", "Verified", "Attested", "Any"};
    ImGui::Combo("Max Trust Level", &stream_max_trust_, trust_items, IM_ARRAYSIZE(trust_items));

    ImGui::Separator();

    // Streaming status
    RenderStreamingStatus();
}

void CloudDatasetManagerPanel::RenderStreamingStatus() {
    ImGui::Text("Streaming Status:");

    if (datastream_client_ && datastream_client_->IsStreaming()) {
        float progress = datastream_client_->GetStreamingProgress();
        uint64_t total = datastream_client_->GetTotalBatches();
        uint64_t current = static_cast<uint64_t>(progress * total);

        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
            ICON_FA_PLAY " Streaming active");

        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        ImGui::Text("Batch %llu / %llu", static_cast<unsigned long long>(current),
                    static_cast<unsigned long long>(total));

        if (ImGui::Button(ICON_FA_STOP " Stop Streaming")) {
            datastream_client_->StopStreaming();
        }
    } else {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
            ICON_FA_CIRCLE_PAUSE " Not streaming");

        if (ImGui::Button(ICON_FA_PLAY " Test Stream")) {
            // Start a test stream
            if (datastream_client_) {
                datastream_client_->StartStreaming(
                    current_dataset_.id,
                    stream_batch_size_,
                    [](const network::StreamingBatch& batch) {
                        // Just receive batches for testing
                    },
                    [this](const std::string& error) {
                        last_error_ = error;
                        error_time_ = 5.0f;
                    },
                    nullptr,
                    0,
                    stream_prefetch_,
                    static_cast<network::TrustLevel>(stream_max_trust_),
                    stream_shuffle_,
                    stream_seed_
                );
            }
        }
    }
}

void CloudDatasetManagerPanel::RenderCreateDatasetWizard() {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Create Dataset", &show_create_wizard_);

    // Progress indicator
    ImGui::Text("Step %d of 3", wizard_step_ + 1);
    float progress = (wizard_step_ + 1) / 3.0f;
    ImGui::ProgressBar(progress, ImVec2(-1, 0));
    ImGui::Separator();

    switch (wizard_step_) {
        case 0:  // Basic info
            ImGui::Text("Dataset Name:");
            ImGui::InputText("##name", new_dataset_name_, sizeof(new_dataset_name_));

            ImGui::Text("Description:");
            ImGui::InputTextMultiline("##desc", new_dataset_description_,
                                       sizeof(new_dataset_description_), ImVec2(-1, 100));
            break;

        case 1:  // File selection
            ImGui::Text("Select files to include:");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "Upload files first using the Upload dialog, then select their IDs here");

            // TODO: Show list of uploaded files to select from
            ImGui::Text("(File selection UI placeholder)");
            break;

        case 2:  // Schema
            ImGui::Text("Schema (optional JSON):");
            ImGui::InputTextMultiline("##wizard_schema", new_dataset_schema_,
                                       sizeof(new_dataset_schema_), ImVec2(-1, 200));
            break;
    }

    ImGui::Separator();

    // Navigation
    if (wizard_step_ > 0) {
        if (ImGui::Button(ICON_FA_ARROW_LEFT " Back")) {
            wizard_step_--;
        }
        ImGui::SameLine();
    }

    if (wizard_step_ < 2) {
        if (ImGui::Button("Next " ICON_FA_ARROW_RIGHT)) {
            wizard_step_++;
        }
    } else {
        if (ImGui::Button(ICON_FA_CHECK " Create")) {
            CreateDataset();
            show_create_wizard_ = false;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
        show_create_wizard_ = false;
    }

    ImGui::End();
}

void CloudDatasetManagerPanel::CreateNewToken() {
    if (!datastream_client_ || !has_dataset_) return;

    network::DataStreamAccessToken token;
    std::string node_id = token_node_id_[0] ? token_node_id_ : "";

    if (datastream_client_->CreateAccessToken(current_dataset_.id, token, node_id,
                                               token_ttl_hours_ * 3600)) {
        ManagedToken managed;
        managed.token_id = token.token_id;
        managed.token_preview = token.token.substr(0, 8);
        managed.expires_at = token.expires_at;
        managed.scopes = token.scopes;
        tokens_.push_back(managed);
    } else {
        last_error_ = datastream_client_->GetLastError();
        error_time_ = 5.0f;
    }
}

void CloudDatasetManagerPanel::RevokeToken(const std::string& token_id) {
    if (!datastream_client_) return;

    if (datastream_client_->RevokeAccessToken(token_id)) {
        // Remove from list
        tokens_.erase(std::remove_if(tokens_.begin(), tokens_.end(),
            [&token_id](const ManagedToken& t) { return t.token_id == token_id; }),
            tokens_.end());
    } else {
        last_error_ = datastream_client_->GetLastError();
        error_time_ = 5.0f;
    }
}

void CloudDatasetManagerPanel::CreateDataset() {
    if (!datastream_client_) return;

    network::CloudDatasetInfo new_dataset;
    if (datastream_client_->CreateDataset(new_dataset_name_, new_dataset_description_,
                                           selected_file_ids_, new_dataset_schema_,
                                           new_dataset)) {
        if (on_dataset_created_) {
            on_dataset_created_(new_dataset);
        }

        // Switch to the new dataset
        SetDataset(new_dataset);
    } else {
        last_error_ = datastream_client_->GetLastError();
        error_time_ = 5.0f;
    }
}

} // namespace gui
