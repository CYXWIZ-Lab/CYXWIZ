#include "connection_dialog.h"
#include "wallet_panel.h"
#include "../icons.h"
#include "../node_editor.h"
#include "network/grpc_client.h"
#include "network/job_manager.h"
#include "network/reservation_client.h"
#include "network/p2p_client.h"
#include "core/data_registry.h"
#include "auth/auth_client.h"
#include "common.pb.h"
#include "job.pb.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <ctime>

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
    : client_(client), job_manager_(job_manager), show_(false), connecting_(false) {
    // Default server address
    std::strncpy(server_address_, "localhost:50051", sizeof(server_address_) - 1);
    server_address_[sizeof(server_address_) - 1] = '\0';

    // Initialize dataset URI for P2P training
    std::strncpy(dataset_uri_, "remote://engine", sizeof(dataset_uri_) - 1);
    dataset_uri_[sizeof(dataset_uri_) - 1] = '\0';
}

ConnectionDialog::~ConnectionDialog() = default;

void ConnectionDialog::Render() {
    if (!show_) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(900, 750), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Server Connection", &show_, ImGuiWindowFlags_NoCollapse)) {
        RenderConnectionPanel();

        ImGui::Separator();

        if (client_ && client_->IsConnected()) {
            // Show active reservation if we have one
            if (has_active_reservation_) {
                RenderActiveReservationPanel();
                ImGui::Separator();
            }

            // Node Discovery section
            RenderNodeDiscoveryPanel();

            // Show reservation panel when a node is selected
            if (selected_node_index_ >= 0 && !has_active_reservation_) {
                ImGui::Separator();
                RenderReservationPanel();
            }
            // Job history removed - jobs are tracked via P2P Training Progress panel
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.0f, 1.0f), "Not connected to server");
        }

        // Render reservation confirmation dialog (modal popup)
        RenderReservationConfirmDialog();
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

// RenderActiveJobsPanel removed - jobs are now tracked via P2P Training Progress panel

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

            // Price per hour (hourly pricing like AWS/GCP/Azure)
            ImGui::TableNextColumn();
            // Check if node is in free work tier (poor reputation < 50, displayed as < 0.5)
            float rep_check = static_cast<float>(node.reputation_score);
            if (rep_check < 0.5f) {
                // Low reputation node - must do free work to rebuild
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION " Free*");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Low reputation node - free work tier");
                }
            } else if (node.price_usd_equivalent > 0) {
                // Show USD price per hour like cloud providers
                ImGui::Text("$%.2f/hr", node.price_usd_equivalent);
            } else if (node.price_per_hour > 0) {
                // Show CYX token price per hour
                ImGui::Text("%.2f CYX/hr", node.price_per_hour);
            } else {
                ImGui::TextDisabled("N/A");
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

    ImGui::Checkbox("Low-Rep (Free)", &filter_free_tier_only_);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Show only low-reputation nodes\noffering free compute to rebuild trust");
    }
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
    ImGui::Text("Reputation: %.0f%%", node.reputation_score * 100);
    ImGui::Text("Jobs Completed: %d", node.total_jobs_completed);
    ImGui::Text("Billing: Hourly");

    // Price display based on reputation tier
    float rep_score = static_cast<float>(node.reputation_score);
    if (rep_score < 0.5f) {
        // Low reputation - free work tier (must rebuild reputation)
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " Free Tier (Reputation Recovery)");
        ImGui::TextDisabled("This node has low reputation and must complete");
        ImGui::TextDisabled("free jobs to rebuild trust.");
    } else if (node.price_usd_equivalent > 0) {
        // Normal pricing - show hourly rate like cloud providers
        ImGui::Text("Price: $%.2f/hr (%.2f CYX)", node.price_usd_equivalent, node.price_per_hour);

        // Show discount info
        ImGui::TextDisabled("10%% off for >1hr, 20%% off for >24hr");
    } else if (node.price_per_hour > 0) {
        ImGui::Text("Price: %.2f CYX/hr", node.price_per_hour);
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

// ============================================================================
// Reservation System
// ============================================================================

bool ConnectionDialog::HasActiveReservation() const {
    return has_active_reservation_;
}

const network::ReservationInfo& ConnectionDialog::GetReservation() const {
    return active_reservation_;
}

void ConnectionDialog::RenderReservationPanel() {
    if (selected_node_index_ < 0 || selected_node_index_ >= static_cast<int>(discovered_nodes_.size())) {
        return;
    }

    const auto& node = discovered_nodes_[selected_node_index_];

    ImGui::SeparatorText(ICON_FA_CLOCK " Reserve Node");

    // Duration selection
    ImGui::Text("Reservation Duration:");
    ImGui::SetNextItemWidth(200);
    ImGui::SliderInt("##duration_minutes", &reservation_duration_minutes_, 10, 480, "%d minutes");
    ImGui::SameLine();
    ImGui::TextDisabled("(%.1f hours)", reservation_duration_minutes_ / 60.0f);

    // Training hyperparameters
    ImGui::Spacing();
    ImGui::Text("Training Settings:");
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("Epochs##res", &reservation_epochs_);
    if (reservation_epochs_ < 1) reservation_epochs_ = 1;
    if (reservation_epochs_ > 1000) reservation_epochs_ = 1000;

    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::InputInt("Batch Size##res", &reservation_batch_size_);
    if (reservation_batch_size_ < 1) reservation_batch_size_ = 1;
    if (reservation_batch_size_ > 512) reservation_batch_size_ = 512;

    // Cost estimate
    double hourly_rate = node.price_per_hour;
    double estimated_cost = hourly_rate * (reservation_duration_minutes_ / 60.0);
    double estimated_usd = node.price_usd_equivalent * (reservation_duration_minutes_ / 60.0);

    ImGui::Spacing();
    ImGui::Text("Estimated Cost:");
    if (node.free_tier_available) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.4f, 1.0f), ICON_FA_GIFT " FREE");
    } else {
        ImGui::SameLine();
        ImGui::Text("%.4f CYX", estimated_cost);
        if (estimated_usd > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("($%.4f)", estimated_usd);
        }
    }

    // Wallet address (from AuthClient user profile)
    ImGui::Spacing();
    ImGui::Text("Your Wallet Address:");

    std::string wallet_address;
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    if (auth.IsAuthenticated()) {
        wallet_address = auth.GetUserInfo().wallet_address;
    }

    if (!wallet_address.empty()) {
        // Show truncated wallet address
        std::string display_addr = wallet_address;
        if (display_addr.length() > 20) {
            display_addr = display_addr.substr(0, 8) + "..." + display_addr.substr(display_addr.length() - 8);
        }
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), ICON_FA_WALLET " %s", display_addr.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
            ICON_FA_TRIANGLE_EXCLAMATION " Connect wallet in Wallet Panel first (View > Panels > Wallet)");
    }

    ImGui::Spacing();

    // Reserve button - enabled if wallet address exists, node is online, and not already reserving
    bool can_reserve = !wallet_address.empty() && node.is_online && !reserving_;

    ImGui::BeginDisabled(!can_reserve);
    if (ImGui::Button(reserving_ ? ICON_FA_SPINNER " Reserving..." : ICON_FA_CALENDAR_CHECK " Reserve Node", ImVec2(150, 0))) {
        show_reservation_confirm_ = true;
    }
    ImGui::EndDisabled();

    // Show reservation error if any
    if (!reservation_error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ICON_FA_XMARK " %s", reservation_error_.c_str());
    }
}

void ConnectionDialog::RenderReservationConfirmDialog() {
    if (!show_reservation_confirm_) {
        return;
    }

    ImGui::OpenPopup("Confirm Reservation");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Confirm Reservation", &show_reservation_confirm_, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (selected_node_index_ < 0 || selected_node_index_ >= static_cast<int>(discovered_nodes_.size())) {
            ImGui::Text("Error: No node selected");
            if (ImGui::Button("Close")) {
                show_reservation_confirm_ = false;
            }
            ImGui::EndPopup();
            return;
        }

        const auto& node = discovered_nodes_[selected_node_index_];

        ImGui::Text(ICON_FA_SERVER " Reserving Node: %s", node.name.c_str());
        ImGui::Separator();

        // Reservation details
        ImGui::BulletText("Duration: %d minutes (%.1f hours)", reservation_duration_minutes_, reservation_duration_minutes_ / 60.0);
        ImGui::BulletText("Device: %s", node.device_type.c_str());
        if (node.vram_bytes > 0) {
            ImGui::BulletText("VRAM: %.1f GB", node.vram_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        ImGui::BulletText("Reputation: %.0f%%", node.reputation_score * 100);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Cost breakdown
        double hourly_rate = node.price_per_hour;
        double estimated_cost = hourly_rate * (reservation_duration_minutes_ / 60.0);

        ImGui::Text(ICON_FA_COINS " Cost Breakdown:");
        if (node.free_tier_available) {
            ImGui::BulletText("Node cost: FREE");
            ImGui::BulletText("Platform fee: FREE");
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.4f, 1.0f), "Total: FREE");
        } else {
            double platform_fee = estimated_cost * 0.10;
            double total = estimated_cost + platform_fee;
            ImGui::BulletText("Node cost: %.4f CYX", estimated_cost);
            ImGui::BulletText("Platform fee (10%%): %.4f CYX", platform_fee);
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Total (Escrow): %.4f CYX", total);
        }

        ImGui::Spacing();
        ImGui::Text("Payment will be locked in escrow until job completes.");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Confirm/Cancel buttons
        if (ImGui::Button(ICON_FA_CHECK " Confirm Reservation", ImVec2(160, 0))) {
            StartReservation();
            show_reservation_confirm_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_XMARK " Cancel", ImVec2(100, 0))) {
            show_reservation_confirm_ = false;
        }

        ImGui::EndPopup();
    }
}

void ConnectionDialog::RenderActiveReservationPanel() {
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));

    // Active Reservation Card
    if (ImGui::BeginChild("ActiveReservationCard", ImVec2(-1, 220), true)) {
        // Header with icon
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::Text(ICON_FA_BOLT " ACTIVE RESERVATION");
        ImGui::PopStyleColor();

        ImGui::Separator();
        ImGui::Spacing();

        // Calculate time remaining
        int64_t now = static_cast<int64_t>(std::time(nullptr));
        int64_t remaining_seconds = active_reservation_.end_time - now;
        int64_t total_seconds = active_reservation_.end_time - active_reservation_.start_time;
        if (remaining_seconds < 0) remaining_seconds = 0;

        // Time display with hours:minutes:seconds
        int hours = static_cast<int>(remaining_seconds / 3600);
        int minutes = static_cast<int>((remaining_seconds % 3600) / 60);
        int seconds = static_cast<int>(remaining_seconds % 60);

        // Progress calculation
        float progress = (total_seconds > 0) ? (1.0f - static_cast<float>(remaining_seconds) / total_seconds) : 1.0f;

        // Time color based on urgency
        ImVec4 time_color;
        ImVec4 progress_color;
        if (remaining_seconds < 300) { // < 5 minutes - Red
            time_color = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
            progress_color = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
        } else if (remaining_seconds < 600) { // < 10 minutes - Orange
            time_color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
            progress_color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
        } else { // Normal - Green
            time_color = ImVec4(0.2f, 1.0f, 0.4f, 1.0f);
            progress_color = ImVec4(0.2f, 0.8f, 0.4f, 1.0f);
        }

        // Large countdown timer display
        ImGui::BeginGroup();
        {
            ImGui::PushStyleColor(ImGuiCol_Text, time_color);
            ImGui::Text(ICON_FA_STOPWATCH);
            ImGui::SameLine();

            // Format time as HH:MM:SS or MM:SS
            char time_str[32];
            if (hours > 0) {
                snprintf(time_str, sizeof(time_str), "%d:%02d:%02d", hours, minutes, seconds);
            } else {
                snprintf(time_str, sizeof(time_str), "%02d:%02d", minutes, seconds);
            }

            // Display time in larger format
            ImGui::SetWindowFontScale(1.5f);
            ImGui::Text("%s", time_str);
            ImGui::SetWindowFontScale(1.0f);

            ImGui::SameLine();
            ImGui::TextDisabled("remaining");
            ImGui::PopStyleColor();
        }
        ImGui::EndGroup();

        // Progress bar
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, progress_color);
        ImGui::ProgressBar(progress, ImVec2(-1, 6), "");
        ImGui::PopStyleColor();

        ImGui::Spacing();

        // Two-column layout for details
        float col_width = ImGui::GetContentRegionAvail().x * 0.5f;

        // Left column - Node info
        ImGui::BeginGroup();
        ImGui::TextDisabled("Node");
        ImGui::Text(ICON_FA_SERVER " %s", active_reservation_.node_endpoint.c_str());
        ImGui::EndGroup();

        ImGui::SameLine(col_width);

        // Right column - P2P Status
        ImGui::BeginGroup();
        ImGui::TextDisabled("Connection");
        bool p2p_connected = p2p_client_ && p2p_client_->IsConnected();
        if (p2p_connected) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), ICON_FA_LINK " Connected");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), ICON_FA_LINK_SLASH " Disconnected");
        }
        ImGui::EndGroup();

        ImGui::Spacing();

        // Action buttons
        if (p2p_connected) {
            // Check training state
            bool is_streaming = p2p_client_ && p2p_client_->IsStreaming();
            bool is_waiting_for_new_job = p2p_client_ && p2p_client_->IsWaitingForNewJob();

            if (is_waiting_for_new_job) {
                // Job complete - show "Ready for new job" state
                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
                ImGui::Text(ICON_FA_CIRCLE_CHECK " Job Complete - Ready for New Training");
                ImGui::PopStyleColor();
                ImGui::Spacing();

                // Training configuration for new job
                ImGui::Text("Configure Next Training:");
                ImGui::SetNextItemWidth(100);
                ImGui::InputInt("Epochs##new", &reservation_epochs_);
                if (reservation_epochs_ < 1) reservation_epochs_ = 1;
                if (reservation_epochs_ > 1000) reservation_epochs_ = 1000;
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputInt("Batch Size##new", &reservation_batch_size_);
                if (reservation_batch_size_ < 1) reservation_batch_size_ = 1;
                if (reservation_batch_size_ > 512) reservation_batch_size_ = 512;
                ImGui::Spacing();

                // Start New Training button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
                if (ImGui::Button(ICON_FA_PLAY " Start New Training", ImVec2(160, 28))) {
                    StartNewP2PTraining();
                }
                ImGui::PopStyleColor(2);
                ImGui::SameLine();
            }
            else if (!is_streaming) {
                // Not streaming, not waiting - initial state
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
                if (ImGui::Button(ICON_FA_PLAY " Start Training", ImVec2(140, 28))) {
                    StartP2PTraining();
                }
                ImGui::PopStyleColor(2);
                ImGui::SameLine();
            } else {
                // Training in progress - show stop button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.3f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.4f, 0.3f, 1.0f));
                if (ImGui::Button(ICON_FA_STOP " Stop Training", ImVec2(140, 28))) {
                    if (p2p_client_) {
                        p2p_client_->StopTraining();
                    }
                }
                ImGui::PopStyleColor(2);
                ImGui::SameLine();
            }

            if (ImGui::Button(ICON_FA_LINK_SLASH " Disconnect", ImVec2(120, 28))) {
                if (p2p_client_) {
                    p2p_client_->StopTrainingStream();
                    p2p_client_->Disconnect();
                }
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
            if (ImGui::Button(ICON_FA_LINK " Connect to Node", ImVec2(150, 28))) {
                ConnectToReservedNode();
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button(ICON_FA_XMARK " Release", ImVec2(100, 28))) {
            CancelReservation();
        }
        ImGui::PopStyleColor(2);

        // Warning for low time
        if (remaining_seconds < 300 && remaining_seconds > 0) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION " Reservation ending soon!");
            ImGui::PopStyleColor();
        }

        // Check if reservation has expired
        if (remaining_seconds <= 0) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f),
                ICON_FA_CIRCLE_XMARK " Reservation has expired");

            // Send reservation end signal to Server Node
            if (p2p_client_ && p2p_client_->IsConnected()) {
                spdlog::info("Reservation timer expired - sending reservation end signal");
                p2p_client_->SendReservationEnd();
                p2p_client_->Disconnect();
            }

            has_active_reservation_ = false;
            if (reservation_client_) {
                reservation_client_->StopHeartbeat();
            }
        }
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void ConnectionDialog::StartReservation() {
    if (selected_node_index_ < 0 || selected_node_index_ >= static_cast<int>(discovered_nodes_.size())) {
        reservation_error_ = "No node selected";
        return;
    }

    if (!reservation_client_) {
        reservation_error_ = "Reservation client not configured";
        return;
    }

    // Get wallet address from AuthClient user profile
    std::string wallet_address;
    auto& auth = cyxwiz::auth::AuthClient::Instance();
    if (auth.IsAuthenticated()) {
        wallet_address = auth.GetUserInfo().wallet_address;
    }

    if (wallet_address.empty()) {
        reservation_error_ = "No wallet address in profile. Please set wallet in your account settings.";
        return;
    }

    const auto& node = discovered_nodes_[selected_node_index_];

    reserving_ = true;
    reservation_error_.clear();

    spdlog::info("Reserving node {} for {} minutes with wallet {}...",
        node.node_id, reservation_duration_minutes_, wallet_address);

    // Build job config from node editor
    cyxwiz::protocol::JobConfig job_config;
    if (node_editor_) {
        std::string graph_json = node_editor_->GetGraphJson();
        if (!graph_json.empty() && graph_json != "{}") {
            job_config.set_model_definition(graph_json);
        }
    }
    job_config.set_dataset_uri(dataset_uri_);

    // Make async reservation request
    reservation_client_->ReserveNodeAsync(
        node.node_id,
        wallet_address,
        reservation_duration_minutes_,
        job_config,
        [this](bool success, const network::ReservationInfo& info, const std::string& error) {
            reserving_ = false;

            if (success) {
                active_reservation_ = info;
                has_active_reservation_ = true;
                reservation_error_.clear();

                spdlog::info("Node reserved successfully!");
                spdlog::info("  Reservation ID: {}", info.reservation_id);
                spdlog::info("  Job ID: {}", info.job_id);
                spdlog::info("  Node endpoint: {}", info.node_endpoint);

                // Start heartbeat
                reservation_client_->StartHeartbeat(info.reservation_id, info.job_id);

                // Update P2P training panel if available
                if (p2p_training_panel_) {
                    // P2P panel will be updated when we connect
                }
            } else {
                reservation_error_ = error;
                spdlog::error("Reservation failed: {}", error);
            }
        }
    );
}

void ConnectionDialog::CancelReservation() {
    if (!has_active_reservation_) {
        return;
    }

    if (!reservation_client_) {
        reservation_error_ = "Reservation client not configured";
        return;
    }

    spdlog::info("Releasing reservation {}...", active_reservation_.reservation_id);

    int64_t time_used = 0;
    int64_t payment_released = 0;
    int64_t refund_amount = 0;

    bool success = reservation_client_->ReleaseReservation(
        active_reservation_.reservation_id,
        "User requested cancellation",
        time_used,
        payment_released,
        refund_amount
    );

    if (success) {
        spdlog::info("Reservation released!");
        spdlog::info("  Time used: {} seconds", time_used);
        spdlog::info("  Payment to node: {} lamports", payment_released);
        spdlog::info("  Refund: {} lamports", refund_amount);

        // Disconnect P2P if connected
        if (p2p_client_ && p2p_client_->IsConnected()) {
            p2p_client_->Disconnect();
        }

        // Stop heartbeat
        reservation_client_->StopHeartbeat();

        // Clear reservation state
        has_active_reservation_ = false;
        active_reservation_ = network::ReservationInfo{};
        reservation_error_.clear();
    } else {
        reservation_error_ = reservation_client_->GetLastError();
        spdlog::error("Failed to release reservation: {}", reservation_error_);
    }
}

void ConnectionDialog::ConnectToReservedNode() {
    if (!has_active_reservation_) {
        reservation_error_ = "No active reservation";
        return;
    }

    if (!p2p_client_) {
        reservation_error_ = "P2P client not configured";
        return;
    }

    spdlog::info("Connecting to reserved node at {}...", active_reservation_.node_endpoint);

    // Connect to node with auth token
    if (p2p_client_->ConnectToNode(
            active_reservation_.node_endpoint,
            active_reservation_.job_id,
            active_reservation_.p2p_auth_token)) {
        spdlog::info("P2P connected to reserved node!");

        // Start monitoring with P2P training panel if available
        if (p2p_training_panel_) {
            p2p_training_panel_->StartMonitoring(
                active_reservation_.job_id,
                active_reservation_.node_endpoint
            );
        }

        reservation_error_.clear();
    } else {
        reservation_error_ = "Failed to connect to node: " + p2p_client_->GetLastError();
        spdlog::error("{}", reservation_error_);
    }
}

void ConnectionDialog::StartP2PTraining() {
    if (!p2p_client_ || !p2p_client_->IsConnected()) {
        reservation_error_ = "Not connected to node. Please connect first.";
        return;
    }

    if (!node_editor_) {
        reservation_error_ = "Node editor not configured";
        return;
    }

    // Get the model definition from NodeEditor
    std::string graph_json = node_editor_->GetGraphJson();
    if (graph_json.empty() || graph_json == "{}") {
        reservation_error_ = "No model defined. Please create a model in the Node Editor.";
        return;
    }

    spdlog::info("Starting P2P training (direct to Server Node)...");
    spdlog::info("  Model definition: {} chars", graph_json.size());
    spdlog::info("  Dataset URI: {}", dataset_uri_);
    spdlog::info("  Epochs: {}", reservation_epochs_);
    spdlog::info("  Batch Size: {}", reservation_batch_size_);

    // Build JobConfig for P2P transmission to Server Node
    // Note: No wallet/payment_address needed - escrow was handled during reservation
    cyxwiz::protocol::JobConfig config;
    config.set_job_id(active_reservation_.job_id);
    config.set_job_type(cyxwiz::protocol::JOB_TYPE_TRAINING);
    config.set_priority(cyxwiz::protocol::PRIORITY_NORMAL);
    config.set_model_definition(graph_json);
    config.set_dataset_uri(dataset_uri_);
    config.set_batch_size(reservation_batch_size_);  // From user input
    config.set_epochs(reservation_epochs_);          // From user input
    config.set_required_device(cyxwiz::protocol::DEVICE_CUDA);

    // Send job config directly to Server Node via P2P
    std::string uri_str(dataset_uri_);
    bool send_success = false;

    if (!uri_str.empty()) {
        send_success = p2p_client_->SendJobWithDatasetURI(config, uri_str);
    } else {
        send_success = p2p_client_->SendJob(config);
    }

    if (!send_success) {
        reservation_error_ = "Failed to send job to node: " + p2p_client_->GetLastError();
        spdlog::error("{}", reservation_error_);
        return;
    }

    spdlog::info("Job config sent to Server Node successfully via P2P");

    // Register dataset for lazy streaming if using remote://
    if (uri_str.find("remote://") == 0) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        auto dataset_names = registry.GetDatasetNames();
        if (!dataset_names.empty()) {
            auto dataset = registry.GetDataset(dataset_names[0]);
            if (dataset.IsValid()) {
                p2p_client_->RegisterDatasetForJob(active_reservation_.job_id, dataset);
                spdlog::info("Registered dataset '{}' for lazy streaming", dataset_names[0]);
            }
        }
    }

    // Start monitoring on the P2P training panel
    if (p2p_training_panel_) {
        // Set the P2P client on the panel
        p2p_training_panel_->SetP2PClient(p2p_client_);

        // Start monitoring
        p2p_training_panel_->StartMonitoring(
            active_reservation_.job_id,
            active_reservation_.node_endpoint
        );

        // Make the panel visible
        p2p_training_panel_->Show();

        spdlog::info("P2P Training Panel: Started monitoring for job {}", active_reservation_.job_id);
    }

    // Set up progress callbacks - forward to P2P training panel only (no console log)
    p2p_client_->SetProgressCallback([this](const network::TrainingProgress& progress) {
        // Forward to P2P training panel if available
        if (p2p_training_panel_) {
            p2p_training_panel_->OnProgressUpdate(progress);
        }
    });

    p2p_client_->SetCompletionCallback([this](const network::TrainingComplete& complete) {
        spdlog::info("Training completed! Success: {}", complete.success);
        if (p2p_training_panel_) {
            p2p_training_panel_->OnTrainingComplete(complete);
        }
    });

    p2p_client_->SetErrorCallback([this](const std::string& error, bool is_fatal) {
        spdlog::error("Training error: {} (fatal: {})", error, is_fatal);
        if (is_fatal) {
            reservation_error_ = "Training error: " + error;
        }
    });

    // Start bidirectional training stream with Server Node
    if (!p2p_client_->StartTrainingStream(active_reservation_.job_id)) {
        reservation_error_ = "Failed to start training stream: " + p2p_client_->GetLastError();
        spdlog::error("{}", reservation_error_);
        return;
    }

    spdlog::info("P2P training started successfully!");
    spdlog::info("  Job ID: {}", active_reservation_.job_id);
    spdlog::info("  Training on Server Node: {}", active_reservation_.node_endpoint);

    reservation_error_.clear();
}

void ConnectionDialog::StartNewP2PTraining() {
    if (!p2p_client_ || !p2p_client_->IsConnected()) {
        reservation_error_ = "Not connected to node. Please connect first.";
        return;
    }

    if (!p2p_client_->IsWaitingForNewJob()) {
        reservation_error_ = "Server Node is not ready for a new job.";
        return;
    }

    if (!node_editor_) {
        reservation_error_ = "Node editor not configured";
        return;
    }

    // Get the model definition from NodeEditor
    std::string graph_json = node_editor_->GetGraphJson();
    if (graph_json.empty() || graph_json == "{}") {
        reservation_error_ = "No model defined. Please create a model in the Node Editor.";
        return;
    }

    spdlog::info("Starting NEW P2P training within existing reservation...");
    spdlog::info("  Model definition: {} chars", graph_json.size());
    spdlog::info("  Dataset URI: {}", dataset_uri_);
    spdlog::info("  Epochs: {}", reservation_epochs_);
    spdlog::info("  Batch Size: {}", reservation_batch_size_);

    // Build new JobConfig for the new training run
    // Generate a new job ID for this run (within same reservation)
    std::string new_job_id = active_reservation_.job_id + "_" +
        std::to_string(std::time(nullptr));

    cyxwiz::protocol::JobConfig config;
    config.set_job_id(new_job_id);
    config.set_job_type(cyxwiz::protocol::JOB_TYPE_TRAINING);
    config.set_priority(cyxwiz::protocol::PRIORITY_NORMAL);
    config.set_model_definition(graph_json);
    config.set_dataset_uri(dataset_uri_);
    config.set_batch_size(reservation_batch_size_);
    config.set_epochs(reservation_epochs_);
    config.set_required_device(cyxwiz::protocol::DEVICE_CUDA);

    // Check if streaming is active - if not, we need to use SendJob + StartTrainingStream
    // This happens when user stopped previous training and wants to start a new one
    bool send_success = false;
    if (p2p_client_->IsStreaming()) {
        // Stream is active, use SendNewJobConfig to send via existing stream
        spdlog::info("Stream active - sending new job config via existing stream");
        send_success = p2p_client_->SendNewJobConfig(config);
    } else {
        // Stream is not active (was stopped), need to use SendJob + StartTrainingStream
        spdlog::info("Stream not active - using SendJob + StartTrainingStream");

        // First, send the job via RPC
        std::string uri_str(dataset_uri_);
        if (!uri_str.empty()) {
            send_success = p2p_client_->SendJobWithDatasetURI(config, uri_str);
        } else {
            send_success = p2p_client_->SendJob(config);
        }

        if (send_success) {
            // Set up callbacks for the new training session
            p2p_client_->SetProgressCallback([this](const network::TrainingProgress& progress) {
                if (p2p_training_panel_) {
                    p2p_training_panel_->OnProgressUpdate(progress);
                }
            });

            p2p_client_->SetErrorCallback([this](const std::string& error, bool is_fatal) {
                spdlog::error("Training error (fatal={}): {}", is_fatal, error);
                if (is_fatal) {
                    reservation_error_ = "Training error: " + error;
                }
            });

            // Start the training stream
            if (!p2p_client_->StartTrainingStream(new_job_id)) {
                reservation_error_ = "Failed to start training stream: " + p2p_client_->GetLastError();
                spdlog::error("{}", reservation_error_);
                return;
            }
            spdlog::info("Training stream restarted successfully");
        }
    }

    if (!send_success) {
        reservation_error_ = "Failed to send new job config: " + p2p_client_->GetLastError();
        spdlog::error("{}", reservation_error_);
        return;
    }

    spdlog::info("New job config sent to Server Node successfully!");

    // Reset waiting state
    p2p_client_->SetWaitingForNewJob(false);

    // Register dataset for lazy streaming if using remote://
    std::string uri_str(dataset_uri_);
    if (uri_str.find("remote://") == 0) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        auto dataset_names = registry.GetDatasetNames();
        if (!dataset_names.empty()) {
            auto dataset = registry.GetDataset(dataset_names[0]);
            if (dataset.IsValid()) {
                p2p_client_->RegisterDatasetForJob(new_job_id, dataset);
                spdlog::info("Registered dataset '{}' for lazy streaming", dataset_names[0]);
            }
        }
    }

    // Update P2P training panel with new job ID
    if (p2p_training_panel_) {
        p2p_training_panel_->StartMonitoring(
            new_job_id,
            active_reservation_.node_endpoint
        );
        p2p_training_panel_->Show();
        spdlog::info("P2P Training Panel: Started monitoring for new job {}", new_job_id);
    }

    spdlog::info("New P2P training started successfully!");
    spdlog::info("  New Job ID: {}", new_job_id);
    spdlog::info("  Training on Server Node: {}", active_reservation_.node_endpoint);

    reservation_error_.clear();
}

} // namespace cyxwiz
