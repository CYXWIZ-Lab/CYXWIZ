#include "connection_dialog.h"
#include "../icons.h"
#include "../node_editor.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"
#include "core/data_registry.h"
#include "common.pb.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>

// Undefine Windows macros that conflict with protobuf STATUS_* enums
#ifdef STATUS_PENDING
#undef STATUS_PENDING
#endif
#ifdef STATUS_ERROR
#undef STATUS_ERROR
#endif
#ifdef STATUS_IN_PROGRESS
#undef STATUS_IN_PROGRESS
#endif

namespace cyxwiz {

ConnectionDialog::ConnectionDialog(network::GRPCClient* client, network::JobManager* job_manager)
    : client_(client), job_manager_(job_manager), show_(false), connecting_(false), submitting_job_(false) {
    // Default server address
    std::strncpy(server_address_, "localhost:50051", sizeof(server_address_) - 1);
    server_address_[sizeof(server_address_) - 1] = '\0';

    // Initialize job submission fields
    std::strncpy(model_definition_, "# Python model definition\nimport cyxwiz\n", sizeof(model_definition_) - 1);
    model_definition_[sizeof(model_definition_) - 1] = '\0';

    std::strncpy(dataset_uri_, "remote://engine", sizeof(dataset_uri_) - 1);
    dataset_uri_[sizeof(dataset_uri_) - 1] = '\0';
}

ConnectionDialog::~ConnectionDialog() = default;

void ConnectionDialog::Render() {
    if (!show_) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(900, 700), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Server Connection", &show_, ImGuiWindowFlags_NoCollapse)) {
        RenderConnectionPanel();

        ImGui::Separator();

        if (client_ && client_->IsConnected()) {
            // Node Discovery section (new)
            RenderNodeDiscoveryPanel();
            ImGui::Separator();

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

    // Show model from node editor instead of text input
    ImGui::Text("Model Definition:");
    if (node_editor_) {
        std::string graph_json = node_editor_->GetGraphJson();
        if (!graph_json.empty() && graph_json != "{}") {
            // Count nodes and links in the graph for display
            size_t node_count = 0;
            size_t link_count = 0;
            // Simple counting by finding "type": occurrences for nodes
            size_t pos = 0;
            while ((pos = graph_json.find("\"type\":", pos)) != std::string::npos) {
                node_count++;
                pos++;
            }
            pos = 0;
            while ((pos = graph_json.find("\"id\":", pos)) != std::string::npos) {
                link_count++;
                pos++;
            }
            // Rough estimate: links are in the "links" array
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                ICON_FA_CHECK " Graph loaded: %zu nodes", node_count);

            // Show preview of graph JSON (first 200 chars)
            ImGui::BeginChild("graph_preview", ImVec2(-1, 80), true);
            ImGui::TextWrapped("%s", graph_json.substr(0, 500).c_str());
            if (graph_json.length() > 500) {
                ImGui::TextDisabled("... (%zu more characters)", graph_json.length() - 500);
            }
            ImGui::EndChild();
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                ICON_FA_TRIANGLE_EXCLAMATION " No model graph - create nodes in Node Editor first");
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
            ICON_FA_XMARK " Node Editor not connected");
    }

    ImGui::Spacing();

    ImGui::Text("Dataset URI:");
    ImGui::InputText("##dataset_uri", dataset_uri_, sizeof(dataset_uri_));

    ImGui::Spacing();

    // Disable submit if no node editor or empty graph
    bool can_submit = node_editor_ != nullptr;
    std::string graph_json;
    if (node_editor_) {
        graph_json = node_editor_->GetGraphJson();
        can_submit = !graph_json.empty() && graph_json != "{}";
    }

    ImGui::BeginDisabled(submitting_job_ || !can_submit);
    if (ImGui::Button(submitting_job_ ? "Submitting..." : "Submit Job", ImVec2(120, 0))) {
        submitting_job_ = true;

        std::string job_id;
        // Use the graph JSON from NodeEditor as model definition
        if (job_manager_->SubmitSimpleJob(graph_json, dataset_uri_, job_id)) {
            last_submitted_job_id_ = job_id;
            spdlog::info("Job submitted successfully: {}", job_id);

            // If using remote:// URI, register the loaded dataset for lazy streaming
            std::string uri_str(dataset_uri_);
            if (uri_str.find("remote://") == 0) {
                auto& registry = cyxwiz::DataRegistry::Instance();
                auto dataset_names = registry.GetDatasetNames();
                if (!dataset_names.empty()) {
                    // Get the first loaded dataset
                    auto dataset = registry.GetDataset(dataset_names[0]);
                    if (dataset.IsValid()) {
                        auto dataset_ptr = std::make_shared<cyxwiz::DatasetHandle>(dataset);
                        job_manager_->SetRemoteDataset(job_id, dataset_ptr);
                        spdlog::info("Registered dataset '{}' for remote streaming", dataset_names[0]);
                    }
                } else {
                    spdlog::warn("No datasets loaded - remote streaming will fail");
                }
            }
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
            ImGui::PushID(job.job_id.c_str());

            // Use fully qualified names to avoid Windows macro conflicts
            bool is_active = (job.status.status() == cyxwiz::protocol::STATUS_IN_PROGRESS ||
                              job.status.status() == cyxwiz::protocol::STATUS_PENDING);

            if (is_active) {
                // Cancel button for active jobs
                if (ImGui::SmallButton("Cancel")) {
                    job_manager_->CancelJob(job.job_id);
                }
            } else {
                // Delete button for completed/failed/cancelled jobs
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.5f, 0.1f, 0.1f, 1.0f));
                if (ImGui::SmallButton("Delete")) {
                    job_manager_->DeleteJob(job.job_id);
                }
                ImGui::PopStyleColor(3);
            }

            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

// ============================================================================
// Node Discovery
// ============================================================================

void ConnectionDialog::RenderNodeDiscoveryPanel() {
    ImGui::SeparatorText(ICON_FA_SERVER " Available Compute Nodes");

    // Auto-refresh node list (only based on time interval, not empty check)
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last_node_refresh_time_).count();
    if (elapsed > node_refresh_interval_seconds_) {
        RefreshNodeList();
        last_node_refresh_time_ = now;
    }

    // Toolbar
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh")) {
        RefreshNodeList();
    }
    ImGui::SameLine();
    if (ImGui::Button(show_search_filters_ ? ICON_FA_FILTER " Hide Filters" : ICON_FA_FILTER " Show Filters")) {
        show_search_filters_ = !show_search_filters_;
    }
    ImGui::SameLine();
    ImGui::Text("| %zu nodes available", discovered_nodes_.size());

    // Search filters (collapsible)
    if (show_search_filters_) {
        RenderNodeSearchFilters();
    }

    ImGui::Spacing();

    // Node table
    RenderNodeTable();

    // Selected node info
    if (selected_node_index_ >= 0 && selected_node_index_ < static_cast<int>(discovered_nodes_.size())) {
        ImGui::Spacing();
        RenderSelectedNodeInfo();
    }
}

void ConnectionDialog::RenderNodeTable() {
    ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
                                   ImGuiTableFlags_Resizable;

    float table_height = 200.0f;
    if (ImGui::BeginTable("node_table", 7, table_flags, ImVec2(0, table_height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("VRAM", ImGuiTableColumnFlags_WidthFixed, 70);
        ImGui::TableSetupColumn("Price/hr", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Reputation", ImGuiTableColumnFlags_WidthFixed, 90);
        ImGui::TableSetupColumn("Region", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableHeadersRow();

        for (int i = 0; i < static_cast<int>(discovered_nodes_.size()); i++) {
            const auto& node = discovered_nodes_[i];
            ImGui::TableNextRow();

            bool is_selected = (i == selected_node_index_);

            // Status
            ImGui::TableNextColumn();
            if (node.is_online) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), ICON_FA_CIRCLE " On");
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), ICON_FA_CIRCLE " Off");
            }

            // Name (selectable)
            ImGui::TableNextColumn();
            if (ImGui::Selectable(node.name.c_str(), is_selected, ImGuiSelectableFlags_SpanAllColumns)) {
                selected_node_index_ = i;
                selected_node_id_ = node.node_id;
            }

            // Device type
            ImGui::TableNextColumn();
            if (node.device_type == "CUDA") {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.2f, 1.0f), "%s", node.device_type.c_str());
            } else if (node.device_type == "OpenCL") {
                ImGui::TextColored(ImVec4(0.2f, 0.6f, 0.9f, 1.0f), "%s", node.device_type.c_str());
            } else {
                ImGui::Text("%s", node.device_type.c_str());
            }

            // VRAM
            ImGui::TableNextColumn();
            if (node.vram_bytes > 0) {
                double vram_gb = node.vram_bytes / (1024.0 * 1024.0 * 1024.0);
                ImGui::Text("%.1f GB", vram_gb);
            } else {
                ImGui::TextDisabled("-");
            }

            // Price per hour
            ImGui::TableNextColumn();
            if (node.free_tier_available) {
                ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.4f, 1.0f), ICON_FA_GIFT " Free");
            } else if (node.price_usd_equivalent > 0) {
                ImGui::Text("$%.3f", node.price_usd_equivalent);
            } else {
                ImGui::Text("%.4f CYX", node.price_per_hour);
            }

            // Reputation
            ImGui::TableNextColumn();
            float rep = static_cast<float>(node.reputation_score);
            ImVec4 rep_color;
            if (rep >= 0.9f) {
                rep_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
            } else if (rep >= 0.7f) {
                rep_color = ImVec4(0.8f, 0.8f, 0.0f, 1.0f);
            } else {
                rep_color = ImVec4(1.0f, 0.4f, 0.0f, 1.0f);
            }
            ImGui::TextColored(rep_color, "%.0f%%", rep * 100);

            // Region
            ImGui::TableNextColumn();
            ImGui::Text("%s", node.region.c_str());
        }

        ImGui::EndTable();
    }
}

void ConnectionDialog::RenderNodeSearchFilters() {
    ImGui::BeginChild("search_filters", ImVec2(0, 100), true);
    ImGui::Text(ICON_FA_MAGNIFYING_GLASS " Search Filters");
    ImGui::Separator();

    // Row 1: Device type, VRAM, Price
    ImGui::SetNextItemWidth(100);
    const char* device_types[] = {"Any", "CUDA", "OpenCL", "CPU"};
    ImGui::Combo("Device", &filter_device_type_, device_types, IM_ARRAYSIZE(device_types));
    ImGui::SameLine();

    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Min VRAM (GB)", &filter_min_vram_gb_, 0.5f, 0.0f, 48.0f, "%.1f");
    ImGui::SameLine();

    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Max $/hr", &filter_max_price_, 0.01f, 0.0f, 10.0f, "%.2f");

    // Row 2: Reputation, Region, Sort, Free tier
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Min Rep", &filter_min_reputation_, 0.05f, 0.0f, 1.0f, "%.0f%%");
    ImGui::SameLine();

    ImGui::SetNextItemWidth(80);
    ImGui::InputText("Region", filter_region_, sizeof(filter_region_));
    ImGui::SameLine();

    ImGui::Checkbox("Free Only", &filter_free_tier_only_);
    ImGui::SameLine();

    ImGui::SetNextItemWidth(100);
    const char* sort_options[] = {"Price", "Performance", "Reputation", "Availability"};
    ImGui::Combo("Sort", &filter_sort_by_, sort_options, IM_ARRAYSIZE(sort_options));

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Search")) {
        SearchNodes();
    }

    ImGui::EndChild();
}

void ConnectionDialog::RenderSelectedNodeInfo() {
    if (selected_node_index_ < 0 || selected_node_index_ >= static_cast<int>(discovered_nodes_.size())) {
        return;
    }

    const auto& node = discovered_nodes_[selected_node_index_];

    ImGui::BeginChild("selected_node_info", ImVec2(0, 120), true);
    ImGui::Text(ICON_FA_CIRCLE_INFO " Selected Node: %s", node.name.c_str());
    ImGui::Separator();

    // Two columns of info
    ImGui::Columns(2, nullptr, false);

    // Left column
    ImGui::Text("Node ID: %s", node.node_id.c_str());
    ImGui::Text("Device: %s", node.device_type.c_str());
    if (node.vram_bytes > 0) {
        double vram_gb = node.vram_bytes / (1024.0 * 1024.0 * 1024.0);
        ImGui::Text("VRAM: %.1f GB", vram_gb);
    }
    ImGui::Text("CPU Cores: %d", node.cpu_cores);
    ImGui::Text("Compute Score: %.0f", node.compute_score);

    ImGui::NextColumn();

    // Right column
    ImGui::Text("Region: %s", node.region.c_str());
    ImGui::Text("Reputation: %.1f%%", node.reputation_score * 100);
    ImGui::Text("Jobs Completed: %d", node.total_jobs_completed);
    ImGui::Text("Billing: %s", node.billing_model.c_str());
    if (node.free_tier_available) {
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.4f, 1.0f), ICON_FA_GIFT " Free tier available");
    } else {
        ImGui::Text("Price: $%.4f/hr (%.4f CYX)", node.price_usd_equivalent, node.price_per_hour);
    }
    if (node.staked_amount > 0) {
        ImGui::Text("Staked: %.2f CYX", node.staked_amount);
    }

    ImGui::Columns(1);
    ImGui::EndChild();
}

void ConnectionDialog::RefreshNodeList() {
    if (!client_ || !client_->IsConnected()) {
        spdlog::warn("Cannot refresh nodes: not connected to server");
        return;
    }

    spdlog::debug("Refreshing node list...");
    if (client_->ListNodes(discovered_nodes_, true, 50)) {
        spdlog::info("Discovered {} nodes", discovered_nodes_.size());
    } else {
        spdlog::error("Failed to list nodes: {}", client_->GetLastError());
    }
}

void ConnectionDialog::SearchNodes() {
    if (!client_ || !client_->IsConnected()) {
        spdlog::warn("Cannot search nodes: not connected to server");
        return;
    }

    // Build search criteria from filter UI
    network::NodeSearchCriteria criteria;

    // Device type
    switch (filter_device_type_) {
        case 1: criteria.required_device = "CUDA"; break;
        case 2: criteria.required_device = "OpenCL"; break;
        case 3: criteria.required_device = "CPU"; break;
        default: criteria.required_device = ""; break;
    }

    // VRAM (convert GB to bytes)
    if (filter_min_vram_gb_ > 0) {
        criteria.min_vram = static_cast<int64_t>(filter_min_vram_gb_ * 1024 * 1024 * 1024);
    }

    // Price
    criteria.max_price_per_hour = filter_max_price_;
    criteria.require_free_tier = filter_free_tier_only_;

    // Reputation
    criteria.min_reputation = filter_min_reputation_;

    // Region
    criteria.preferred_region = filter_region_;

    // Sorting
    criteria.sort_by = filter_sort_by_;

    criteria.max_results = 50;

    spdlog::info("Searching for nodes with filters...");
    if (client_->FindNodes(criteria, discovered_nodes_)) {
        spdlog::info("Found {} matching nodes", discovered_nodes_.size());
        // Clear selection since list changed
        selected_node_index_ = -1;
        selected_node_id_.clear();
    } else {
        spdlog::error("Failed to search nodes: {}", client_->GetLastError());
    }
}

} // namespace cyxwiz
