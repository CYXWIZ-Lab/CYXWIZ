#include "connection_dialog.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"
#include "common.pb.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>

namespace cyxwiz {

ConnectionDialog::ConnectionDialog(network::GRPCClient* client, network::JobManager* job_manager)
    : client_(client), job_manager_(job_manager), show_(false), connecting_(false), submitting_job_(false) {
    // Default server address
    std::strncpy(server_address_, "localhost:50051", sizeof(server_address_) - 1);
    server_address_[sizeof(server_address_) - 1] = '\0';

    // Initialize job submission fields
    std::strncpy(model_definition_, "# Python model definition\nimport cyxwiz\n", sizeof(model_definition_) - 1);
    model_definition_[sizeof(model_definition_) - 1] = '\0';

    std::strncpy(dataset_uri_, "ipfs://Qm...", sizeof(dataset_uri_) - 1);
    dataset_uri_[sizeof(dataset_uri_) - 1] = '\0';
}

ConnectionDialog::~ConnectionDialog() = default;

void ConnectionDialog::Render() {
    if (!show_) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Server Connection", &show_, ImGuiWindowFlags_NoCollapse)) {
        RenderConnectionPanel();

        ImGui::Separator();

        if (client_ && client_->IsConnected()) {
            RenderJobSubmitPanel();
            ImGui::Separator();
            RenderActiveJobsPanel();
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "Not connected to server");
        }
    }
    ImGui::End();
}

void ConnectionDialog::RenderConnectionPanel() {
    ImGui::SeparatorText("Connection Settings");

    bool is_connected = client_ && client_->IsConnected();

    if (is_connected) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Connected");
        ImGui::SameLine();
        ImGui::Text("to %s", client_->GetServerAddress().c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Disconnected");
    }

    ImGui::Spacing();

    // Server address input
    ImGui::Text("Server Address:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##server_address", server_address_, sizeof(server_address_));

    ImGui::Spacing();

    // Connect/Disconnect button
    if (is_connected) {
        if (ImGui::Button("Disconnect", ImVec2(120, 0))) {
            spdlog::info("Disconnecting from server...");
            client_->Disconnect();
            connection_error_.clear();

            if (connection_callback_) {
                connection_callback_(false);
            }
        }
    } else {
        ImGui::BeginDisabled(connecting_);
        if (ImGui::Button(connecting_ ? "Connecting..." : "Connect", ImVec2(120, 0))) {
            connecting_ = true;
            connection_error_.clear();

            spdlog::info("Attempting to connect to: {}", server_address_);

            if (client_->Connect(server_address_)) {
                spdlog::info("Successfully connected!");
                connecting_ = false;

                if (connection_callback_) {
                    connection_callback_(true);
                }
            } else {
                connection_error_ = client_->GetLastError();
                spdlog::error("Connection failed: {}", connection_error_);
                connecting_ = false;

                if (connection_callback_) {
                    connection_callback_(false);
                }
            }
        }
        ImGui::EndDisabled();
    }

    // Show connection error if any
    if (!connection_error_.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", connection_error_.c_str());
    }
}

void ConnectionDialog::RenderJobSubmitPanel() {
    ImGui::SeparatorText("Submit Job");

    ImGui::Text("Model Definition:");
    ImGui::InputTextMultiline("##model_def", model_definition_, sizeof(model_definition_),
                               ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);

    ImGui::Spacing();

    ImGui::Text("Dataset URI:");
    ImGui::InputText("##dataset_uri", dataset_uri_, sizeof(dataset_uri_));

    ImGui::Spacing();

    ImGui::BeginDisabled(submitting_job_);
    if (ImGui::Button(submitting_job_ ? "Submitting..." : "Submit Job", ImVec2(120, 0))) {
        submitting_job_ = true;

        std::string job_id;
        if (job_manager_->SubmitSimpleJob(model_definition_, dataset_uri_, job_id)) {
            last_submitted_job_id_ = job_id;
            spdlog::info("Job submitted successfully: {}", job_id);
        } else {
            spdlog::error("Failed to submit job");
        }

        submitting_job_ = false;
    }
    ImGui::EndDisabled();

    if (!last_submitted_job_id_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Last job: %s", last_submitted_job_id_.c_str());
    }
}

void ConnectionDialog::RenderActiveJobsPanel() {
    ImGui::SeparatorText("Active Jobs");

    const auto& active_jobs = job_manager_->GetActiveJobs();

    if (active_jobs.empty()) {
        ImGui::TextDisabled("No active jobs");
        return;
    }

    if (ImGui::BeginTable("active_jobs", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Job ID", ImGuiTableColumnFlags_WidthFixed, 200);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Progress", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableHeadersRow();

        for (const auto& job : active_jobs) {
            ImGui::TableNextRow();

            // Job ID
            ImGui::TableNextColumn();
            ImGui::Text("%s", job.job_id.c_str());

            // Status
            ImGui::TableNextColumn();
            std::string status_name = cyxwiz::protocol::StatusCode_Name(job.status.status());

            ImVec4 status_color;
            if (job.status.status() == cyxwiz::protocol::STATUS_SUCCESS) {
                status_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
            } else if (job.status.status() == cyxwiz::protocol::STATUS_FAILED ||
                       job.status.status() == cyxwiz::protocol::STATUS_CANCELLED) {
                status_color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            } else if (job.status.status() == cyxwiz::protocol::STATUS_IN_PROGRESS) {
                status_color = ImVec4(0.0f, 0.8f, 1.0f, 1.0f);
            } else {
                status_color = ImVec4(0.8f, 0.8f, 0.0f, 1.0f);
            }

            ImGui::TextColored(status_color, "%s", status_name.c_str());

            // Progress
            ImGui::TableNextColumn();
            float progress = static_cast<float>(job.status.progress());
            ImGui::ProgressBar(progress, ImVec2(-1, 0), nullptr);
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("%.1f%%", progress * 100.0f);

            // Actions
            ImGui::TableNextColumn();
            // Use fully qualified names to avoid Windows macro conflicts
            if (job.status.status() == cyxwiz::protocol::STATUS_IN_PROGRESS ||
                job.status.status() == (cyxwiz::protocol::StatusCode)3) { // STATUS_PENDING
                ImGui::PushID(job.job_id.c_str());
                if (ImGui::SmallButton("Cancel")) {
                    job_manager_->CancelJob(job.job_id);
                }
                ImGui::PopID();
            }
        }

        ImGui::EndTable();
    }
}

} // namespace cyxwiz
