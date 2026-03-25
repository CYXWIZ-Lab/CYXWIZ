#include "cloud_browser.h"
#include "../../auth/auth_client.h"
#include "../../core/engine_config.h"
#include <spdlog/spdlog.h>
#include "../icons.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <thread>

namespace gui {

// Trust level colors
static ImVec4 GetTrustLevelColor(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self:
            return ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green
        case network::TrustLevel::Signed:
            return ImVec4(0.2f, 0.6f, 0.9f, 1.0f);  // Blue
        case network::TrustLevel::Verified:
            return ImVec4(0.4f, 0.8f, 0.4f, 1.0f);  // Light green
        case network::TrustLevel::Attested:
            return ImVec4(0.9f, 0.7f, 0.2f, 1.0f);  // Gold
        case network::TrustLevel::Untrusted:
            return ImVec4(0.9f, 0.3f, 0.3f, 1.0f);  // Red
        default:
            return ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray
    }
}

static const char* GetTrustLevelIcon(network::TrustLevel level) {
    switch (level) {
        case network::TrustLevel::Self: return ICON_FA_USER_CHECK;
        case network::TrustLevel::Signed: return ICON_FA_SIGNATURE;
        case network::TrustLevel::Verified: return ICON_FA_CIRCLE_CHECK;
        case network::TrustLevel::Attested: return ICON_FA_SHIELD_HALVED;
        case network::TrustLevel::Untrusted: return ICON_FA_TRIANGLE_EXCLAMATION;
        default: return ICON_FA_QUESTION;
    }
}

CloudBrowserPanel::CloudBrowserPanel()
    : Panel("Cloud Browser", true) {
}

CloudBrowserPanel::~CloudBrowserPanel() {
    // Signal shutdown and wait for any running async thread
    shutdown_requested_.store(true);
    if (async_thread_.joinable()) {
        async_thread_.join();
    }
}

void CloudBrowserPanel::Render() {
    if (!visible_) return;
    
    // Try auto-connect on first render
    TryAutoConnect();

    ImGui::SetNextWindowSize(ImVec2(700, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CLOUD " Cloud Browser###CloudBrowser", &visible_)) {
        focused_ = ImGui::IsWindowFocused();

        // Check connection
        if (!datastream_client_ || !datastream_client_->IsConnected()) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                ICON_FA_TRIANGLE_EXCLAMATION " Not connected to CyxCloud Gateway");

            if (ImGui::Button(ICON_FA_PLUG " Connect")) {
                // Reset and try again
                auto_connect_attempted_ = false;
                TryAutoConnect();
                
            }
            ImGui::End();
            return;
        }

        RenderToolbar();
        ImGui::Separator();
        RenderTabs();

        // Render popups
        RenderDatasetInfoPopup();
        RenderVerificationPopup();
        RenderSharePopup();
        RenderDeleteConfirmPopup();

        // Show error if any
        if (!last_error_.empty() && error_time_ > 0) {
            error_time_ -= ImGui::GetIO().DeltaTime;
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                ICON_FA_CIRCLE_EXCLAMATION " %s", last_error_.c_str());
        }
    }
    ImGui::End();
}

void CloudBrowserPanel::RenderToolbar() {
    // Refresh button
    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh")) {
        RefreshDatasets();
    }
    ImGui::SameLine();

    // Upload button
    if (ImGui::Button(ICON_FA_CLOUD_ARROW_UP " Upload")) {
        if (on_upload_requested_) {
            on_upload_requested_();
        }
    }
    ImGui::SameLine();

    // Search bar
    ImGui::SetNextItemWidth(200);
    if (ImGui::InputTextWithHint("##search", ICON_FA_SEARCH " Search datasets...",
                                  search_buffer_, sizeof(search_buffer_))) {
        ApplyFilter();
    }
    ImGui::SameLine();

    // Trust level filter
    ImGui::SetNextItemWidth(120);
    const char* trust_items[] = {"All", "Self", "Signed", "Verified", "Attested", "Untrusted"};
    int trust_index = static_cast<int>(trust_filter_) + 1;  // +1 for "All"
    if (trust_filter_ == network::TrustLevel::Untrusted && trust_index == 5) {
        trust_index = 0;  // "All" when set to untrusted
    }
    if (ImGui::Combo("##trust_filter", &trust_index, trust_items, IM_ARRAYSIZE(trust_items))) {
        if (trust_index == 0) {
            trust_filter_ = network::TrustLevel::Untrusted;  // "All"
        } else {
            trust_filter_ = static_cast<network::TrustLevel>(trust_index - 1);
        }
        ApplyFilter();
    }
    ImGui::SameLine();

    // Show shared checkbox
    if (ImGui::Checkbox("Show Shared", &show_shared_)) {
        RefreshDatasets();
    }

    // Loading indicator
    if (is_loading_.load()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
            ICON_FA_SPINNER " %s", loading_message_.c_str());
    }
}

void CloudBrowserPanel::RenderTabs() {
    if (ImGui::BeginTabBar("##CloudBrowserTabs")) {
        if (ImGui::BeginTabItem(ICON_FA_FOLDER " My Datasets")) {
            current_tab_ = 0;
            RenderMyDatasetsTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem(ICON_FA_DATABASE " Public Datasets")) {
            current_tab_ = 1;
            RenderPublicDatasetsTab();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
}

void CloudBrowserPanel::RenderMyDatasetsTab() {
    if (filtered_datasets_.empty() && !is_loading_.load()) {
        if (my_datasets_.empty()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "No datasets found. Click 'Upload' to add a dataset.");
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "No datasets match your filter.");
        }
        return;
    }

    RenderDatasetTable(filtered_datasets_);
}

void CloudBrowserPanel::RenderPublicDatasetsTab() {
    // Auto-fetch on first view
    static bool first_view = true;
    if (first_view && public_datasets_.empty()) {
        first_view = false;
        RefreshPublicDatasets();
    }

    if (filtered_public_datasets_.empty() && !is_loading_.load()) {
        if (public_datasets_.empty()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "Loading public datasets...");
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                "No public datasets match your filter.");
        }
        return;
    }

    RenderPublicDatasetTable(filtered_public_datasets_);
}

void CloudBrowserPanel::RenderDatasetTable(const std::vector<network::CloudDatasetInfo>& datasets, bool is_public) {
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_ScrollY;

    if (ImGui::BeginTable("##datasets", 5, flags, ImVec2(0, 0))) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort, 200);
        ImGui::TableSetupColumn("Trust", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupColumn("Files", ImGuiTableColumnFlags_WidthFixed, 50);
        ImGui::TableSetupColumn("Created", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        // Handle sorting
        ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs();
        if (sort_specs && sort_specs->SpecsDirty) {
            sort_column_ = sort_specs->Specs[0].ColumnIndex;
            sort_ascending_ = (sort_specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending);
            sort_specs->SpecsDirty = false;
            // TODO: Re-sort the data
        }

        for (size_t i = 0; i < datasets.size(); i++) {
            const auto& ds = datasets[i];
            ImGui::TableNextRow();

            bool is_selected = (selected_index_ == static_cast<int>(i));

            // Name column
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(ds.name.c_str(), is_selected,
                                   ImGuiSelectableFlags_SpanAllColumns)) {
                selected_index_ = static_cast<int>(i);
                selected_dataset_ = const_cast<network::CloudDatasetInfo*>(&filtered_datasets_[i]);

                // Double-click to use dataset
                if (ImGui::IsMouseDoubleClicked(0)) {
                    if (on_dataset_selected_ && selected_dataset_) {
                        on_dataset_selected_(*selected_dataset_);
                    }
                }
            }

            // Context menu
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                selected_index_ = static_cast<int>(i);
                selected_dataset_ = const_cast<network::CloudDatasetInfo*>(&filtered_datasets_[i]);
                ImGui::OpenPopup("DatasetContextMenu");
            }

            // Trust column
            ImGui::TableSetColumnIndex(1);
            RenderTrustBadge(ds.trust_level);

            // Size column
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", ds.GetSizeString().c_str());

            // Files column
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%d", ds.file_count);

            // Created column
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%s", ds.GetCreatedAtString().c_str());
        }

        // Context menu popup
        if (ImGui::BeginPopup("DatasetContextMenu")) {
            if (selected_dataset_) {
                RenderDatasetContextMenu(*selected_dataset_);
            }
            ImGui::EndPopup();
        }

        ImGui::EndTable();
    }
}

void CloudBrowserPanel::RenderPublicDatasetTable(const std::vector<network::PublicDatasetInfo>& datasets) {
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY;

    if (ImGui::BeginTable("##public_datasets", 5, flags, ImVec2(0, 0))) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_DefaultSort, 150);
        ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, 60);
        ImGui::TableSetupColumn("License", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableSetupColumn("Verified By", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, 80);
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < datasets.size(); i++) {
            const auto& ds = datasets[i];
            ImGui::TableNextRow();

            bool is_selected = (selected_public_index_ == static_cast<int>(i));

            // Name column
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable(ds.name.c_str(), is_selected,
                                   ImGuiSelectableFlags_SpanAllColumns)) {
                selected_public_index_ = static_cast<int>(i);
                selected_public_dataset_ = const_cast<network::PublicDatasetInfo*>(&filtered_public_datasets_[i]);
            }

            // Context menu
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                selected_public_index_ = static_cast<int>(i);
                selected_public_dataset_ = const_cast<network::PublicDatasetInfo*>(&filtered_public_datasets_[i]);
                ImGui::OpenPopup("PublicDatasetContextMenu");
            }

            // Version column
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", ds.version.c_str());

            // License column
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%s", ds.license.c_str());

            // Verified By column
            ImGui::TableSetColumnIndex(3);
            if (!ds.verified_by.empty()) {
                std::string verifiers;
                for (size_t j = 0; j < ds.verified_by.size(); j++) {
                    if (j > 0) verifiers += ", ";
                    verifiers += ds.verified_by[j];
                }
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f),
                    ICON_FA_CIRCLE_CHECK " %s", verifiers.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "-");
            }

            // Status column
            ImGui::TableSetColumnIndex(4);
            if (ds.cached) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                    ICON_FA_CHECK " Cached");
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f),
                    ICON_FA_DOWNLOAD " Download");
            }
        }

        // Context menu popup
        if (ImGui::BeginPopup("PublicDatasetContextMenu")) {
            if (selected_public_dataset_) {
                RenderPublicDatasetContextMenu(*selected_public_dataset_);
            }
            ImGui::EndPopup();
        }

        ImGui::EndTable();
    }
}

void CloudBrowserPanel::RenderDatasetContextMenu(const network::CloudDatasetInfo& dataset) {
    if (ImGui::MenuItem(ICON_FA_PLAY " Use for Training")) {
        if (on_dataset_selected_) {
            on_dataset_selected_(dataset);
        }
    }

    if (ImGui::MenuItem(ICON_FA_CIRCLE_INFO " View Details")) {
        show_info_popup_ = true;
        // Fetch full details including files
        if (datastream_client_) {
            network::CloudDatasetInfo info;
            datastream_client_->GetDatasetInfo(dataset.id, info, selected_dataset_files_);
        }
    }

    if (ImGui::MenuItem(ICON_FA_SHIELD_HALVED " Verify Integrity")) {
        show_verify_popup_ = true;
        VerifySelectedDataset();
    }

    ImGui::Separator();

    if (ImGui::MenuItem(ICON_FA_SHARE_NODES " Share...")) {
        show_share_popup_ = true;
        std::memset(share_user_id_buffer_, 0, sizeof(share_user_id_buffer_));
        share_permission_flags_ = 0x3;  // read + stream by default
    }

    ImGui::Separator();

    if (ImGui::MenuItem(ICON_FA_TRASH " Delete", nullptr, false,
                        dataset.trust_level == network::TrustLevel::Self)) {
        show_delete_popup_ = true;
    }
}

void CloudBrowserPanel::RenderPublicDatasetContextMenu(const network::PublicDatasetInfo& dataset) {
    if (dataset.cached) {
        if (ImGui::MenuItem(ICON_FA_PLAY " Use for Training")) {
            // Use cached version
            if (on_dataset_selected_ && !dataset.cached_dataset_id.empty()) {
                network::CloudDatasetInfo info;
                info.id = dataset.cached_dataset_id;
                info.name = dataset.name;
                info.trust_level = network::TrustLevel::Verified;
                on_dataset_selected_(info);
            }
        }
    } else {
        if (ImGui::MenuItem(ICON_FA_DOWNLOAD " Download to CyxCloud")) {
            // TODO: Trigger download
        }
    }

    ImGui::Separator();

    if (ImGui::MenuItem(ICON_FA_LINK " View Official Page")) {
        // TODO: Open URL in browser
    }

    if (!dataset.paper_url.empty()) {
        if (ImGui::MenuItem(ICON_FA_FILE_LINES " View Paper")) {
            // TODO: Open URL in browser
        }
    }
}

void CloudBrowserPanel::RenderTrustBadge(network::TrustLevel level) {
    ImVec4 color = GetTrustLevelColor(level);
    const char* icon = GetTrustLevelIcon(level);
    const char* label = "";

    switch (level) {
        case network::TrustLevel::Self: label = "Self"; break;
        case network::TrustLevel::Signed: label = "Signed"; break;
        case network::TrustLevel::Verified: label = "Verified"; break;
        case network::TrustLevel::Attested: label = "Attested"; break;
        case network::TrustLevel::Untrusted: label = "Untrusted"; break;
        default: label = "Unknown"; break;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Text("%s %s", icon, label);
    ImGui::PopStyleColor();

    // Tooltip with explanation
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        switch (level) {
            case network::TrustLevel::Self:
                ImGui::Text("Your own upload - fully trusted");
                break;
            case network::TrustLevel::Signed:
                ImGui::Text("Cryptographically signed by a trusted party");
                break;
            case network::TrustLevel::Verified:
                ImGui::Text("Hash verified against a known good source");
                break;
            case network::TrustLevel::Attested:
                ImGui::Text("Hardware attestation (TEE/SGX)");
                break;
            case network::TrustLevel::Untrusted:
                ImGui::Text("Unknown source - verification recommended");
                break;
            default:
                ImGui::Text("Unknown trust level");
                break;
        }
        ImGui::EndTooltip();
    }
}

void CloudBrowserPanel::RenderDatasetInfoPopup() {
    if (!show_info_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Dataset Details", &show_info_popup_)) {
        if (selected_dataset_) {
            const auto& ds = *selected_dataset_;

            // Header
            ImGui::Text(ICON_FA_DATABASE " %s", ds.name.c_str());
            ImGui::SameLine();
            RenderTrustBadge(ds.trust_level);

            ImGui::Separator();

            // Basic info
            ImGui::Text("ID: %s", ds.id.c_str());
            ImGui::Text("Size: %s", ds.GetSizeString().c_str());
            ImGui::Text("Files: %d", ds.file_count);
            ImGui::Text("Version: %d", ds.version);
            ImGui::Text("Created: %s", ds.GetCreatedAtString().c_str());

            if (!ds.description.empty()) {
                ImGui::Separator();
                ImGui::TextWrapped("Description: %s", ds.description.c_str());
            }

            // Files list
            if (!selected_dataset_files_.empty()) {
                ImGui::Separator();
                ImGui::Text("Files:");

                if (ImGui::BeginChild("##files_list", ImVec2(0, 150), true)) {
                    for (const auto& f : selected_dataset_files_) {
                        ImGui::Text("%s (%lld bytes)", f.path.c_str(),
                                    static_cast<long long>(f.size_bytes));
                    }
                }
                ImGui::EndChild();
            }

            ImGui::Separator();
            if (ImGui::Button("Use for Training")) {
                if (on_dataset_selected_) {
                    on_dataset_selected_(ds);
                }
                show_info_popup_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Close")) {
                show_info_popup_ = false;
            }
        }
    }
    ImGui::End();
}

void CloudBrowserPanel::RenderVerificationPopup() {
    if (!show_verify_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Dataset Verification", &show_verify_popup_)) {
        if (verification_in_progress_) {
            ImGui::Text(ICON_FA_SPINNER " Verifying dataset...");
        } else {
            const auto& r = verification_result_;

            // Status header
            if (r.manifest_valid && r.all_files_valid) {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                    ICON_FA_CIRCLE_CHECK " Dataset verified successfully");
            } else {
                ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f),
                    ICON_FA_CIRCLE_XMARK " Verification failed");
            }

            ImGui::Separator();

            ImGui::Text("Manifest Valid: %s", r.manifest_valid ? "Yes" : "No");
            ImGui::Text("All Files Valid: %s", r.all_files_valid ? "Yes" : "No");
            ImGui::Text("Computed Trust Level: ");
            ImGui::SameLine();
            RenderTrustBadge(r.computed_trust_level);

            if (!r.public_match_name.empty()) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f),
                    ICON_FA_CHECK " Matches public dataset: %s (%.0f%% confidence)",
                    r.public_match_name.c_str(), r.public_match_confidence * 100);
            }

            if (!r.message.empty()) {
                ImGui::Separator();
                ImGui::TextWrapped("%s", r.message.c_str());
            }
        }

        ImGui::Separator();
        if (ImGui::Button("Close")) {
            show_verify_popup_ = false;
        }
    }
    ImGui::End();
}

void CloudBrowserPanel::RenderSharePopup() {
    if (!show_share_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(350, 200), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Share Dataset", &show_share_popup_)) {
        ImGui::Text("Share with user ID:");
        ImGui::InputText("##share_user", share_user_id_buffer_, sizeof(share_user_id_buffer_));

        ImGui::Text("Permissions:");
        bool can_read = share_permission_flags_ & 0x1;
        bool can_stream = share_permission_flags_ & 0x2;
        bool can_reshare = share_permission_flags_ & 0x4;

        if (ImGui::Checkbox("Read", &can_read)) {
            if (can_read) share_permission_flags_ |= 0x1;
            else share_permission_flags_ &= ~0x1;
        }
        if (ImGui::Checkbox("Stream", &can_stream)) {
            if (can_stream) share_permission_flags_ |= 0x2;
            else share_permission_flags_ &= ~0x2;
        }
        if (ImGui::Checkbox("Re-share", &can_reshare)) {
            if (can_reshare) share_permission_flags_ |= 0x4;
            else share_permission_flags_ &= ~0x4;
        }

        ImGui::Separator();

        if (ImGui::Button("Share")) {
            ShareSelectedDataset();
            show_share_popup_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            show_share_popup_ = false;
        }
    }
    ImGui::End();
}

void CloudBrowserPanel::RenderDeleteConfirmPopup() {
    if (!show_delete_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(300, 120), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Delete Dataset", &show_delete_popup_)) {
        ImGui::TextWrapped(ICON_FA_TRIANGLE_EXCLAMATION
            " Are you sure you want to delete this dataset? This cannot be undone.");

        ImGui::Separator();

        if (ImGui::Button("Delete", ImVec2(80, 0))) {
            DeleteSelectedDataset();
            show_delete_popup_ = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(80, 0))) {
            show_delete_popup_ = false;
        }
    }
    ImGui::End();
}

void CloudBrowserPanel::RefreshDatasets() {
    if (!datastream_client_ || !datastream_client_->IsConnected()) return;
    if (shutdown_requested_.load()) return;

    // Wait for any existing async operation
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    is_loading_.store(true);
    loading_message_ = "Loading datasets...";

    async_thread_ = std::thread([this]() {
        if (!shutdown_requested_.load()) {
            FetchDatasets();
        }
    });
}

void CloudBrowserPanel::RefreshPublicDatasets() {
    if (!datastream_client_ || !datastream_client_->IsConnected()) return;
    if (shutdown_requested_.load()) return;

    // Wait for any existing async operation
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    is_loading_.store(true);
    loading_message_ = "Loading public datasets...";

    async_thread_ = std::thread([this]() {
        if (!shutdown_requested_.load()) {
            FetchPublicDatasets();
        }
    });
}

void CloudBrowserPanel::FetchDatasets() {
    std::vector<network::CloudDatasetInfo> datasets;
    bool success = datastream_client_->ListDatasets(datasets, 100, 0, show_shared_);

    {
        std::lock_guard<std::mutex> lock(datasets_mutex_);
        if (success) {
            my_datasets_ = std::move(datasets);
            ApplyFilter();
        } else {
            last_error_ = datastream_client_->GetLastError();
            error_time_ = 5.0f;
        }
    }

    is_loading_.store(false);
}

void CloudBrowserPanel::FetchPublicDatasets() {
    std::vector<network::PublicDatasetInfo> datasets;
    bool success = datastream_client_->ListPublicDatasets(datasets);

    {
        std::lock_guard<std::mutex> lock(datasets_mutex_);
        if (success) {
            public_datasets_ = std::move(datasets);

            // Apply filter
            filtered_public_datasets_.clear();
            std::string filter = search_buffer_;
            std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);

            for (const auto& ds : public_datasets_) {
                if (filter.empty()) {
                    filtered_public_datasets_.push_back(ds);
                } else {
                    std::string name = ds.name;
                    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
                    if (name.find(filter) != std::string::npos) {
                        filtered_public_datasets_.push_back(ds);
                    }
                }
            }
        } else {
            last_error_ = datastream_client_->GetLastError();
            error_time_ = 5.0f;
        }
    }

    is_loading_.store(false);
}

void CloudBrowserPanel::ApplyFilter() {
    std::lock_guard<std::mutex> lock(datasets_mutex_);
    filtered_datasets_.clear();

    std::string filter = search_buffer_;
    std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);

    for (const auto& ds : my_datasets_) {
        // Trust level filter (Untrusted means show all)
        if (trust_filter_ != network::TrustLevel::Untrusted &&
            ds.trust_level > trust_filter_) {
            continue;
        }

        // Text filter
        if (!filter.empty()) {
            std::string name = ds.name;
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            std::string desc = ds.description;
            std::transform(desc.begin(), desc.end(), desc.begin(), ::tolower);

            if (name.find(filter) == std::string::npos &&
                desc.find(filter) == std::string::npos) {
                continue;
            }
        }

        filtered_datasets_.push_back(ds);
    }

    // Reset selection if it's now invalid
    if (selected_index_ >= static_cast<int>(filtered_datasets_.size())) {
        selected_index_ = -1;
        selected_dataset_ = nullptr;
    }
}

void CloudBrowserPanel::VerifySelectedDataset() {
    if (!selected_dataset_ || !datastream_client_) return;
    if (shutdown_requested_.load()) return;

    // Wait for any existing async operation
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    verification_in_progress_ = true;

    std::string dataset_id = selected_dataset_->id;
    async_thread_ = std::thread([this, dataset_id]() {
        if (shutdown_requested_.load()) {
            verification_in_progress_ = false;
            return;
        }

        bool success = datastream_client_->VerifyDataset(dataset_id, verification_result_);

        if (!success && !shutdown_requested_.load()) {
            last_error_ = datastream_client_->GetLastError();
            error_time_ = 5.0f;
        }

        verification_in_progress_ = false;
    });
}

void CloudBrowserPanel::DeleteSelectedDataset() {
    if (!selected_dataset_ || !datastream_client_) return;
    if (shutdown_requested_.load()) return;

    // Wait for any existing async operation
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    std::string dataset_id = selected_dataset_->id;
    async_thread_ = std::thread([this, dataset_id]() {
        if (shutdown_requested_.load()) return;

        bool success = datastream_client_->DeleteDataset(dataset_id);

        if (shutdown_requested_.load()) return;

        if (success) {
            RefreshDatasets();
        } else {
            last_error_ = datastream_client_->GetLastError();
            error_time_ = 5.0f;
        }
    });
}

void CloudBrowserPanel::ShareSelectedDataset() {
    if (!selected_dataset_ || !datastream_client_) return;
    if (strlen(share_user_id_buffer_) == 0) return;
    if (shutdown_requested_.load()) return;

    // Wait for any existing async operation
    if (async_thread_.joinable()) {
        async_thread_.join();
    }

    std::vector<std::string> permissions;
    if (share_permission_flags_ & 0x1) permissions.push_back("read");
    if (share_permission_flags_ & 0x2) permissions.push_back("stream");
    if (share_permission_flags_ & 0x4) permissions.push_back("reshare");

    std::string dataset_id = selected_dataset_->id;
    std::string user_id = share_user_id_buffer_;

    async_thread_ = std::thread([this, dataset_id, user_id, permissions]() {
        if (shutdown_requested_.load()) return;

        bool success = datastream_client_->ShareDataset(dataset_id, user_id, permissions);

        if (!success && !shutdown_requested_.load()) {
            last_error_ = datastream_client_->GetLastError();
            error_time_ = 5.0f;
        }
    });
}

const network::CloudDatasetInfo* CloudBrowserPanel::GetSelectedDataset() const {
    return selected_dataset_;
}


void CloudBrowserPanel::TryAutoConnect() {
    if (auto_connect_attempted_) return;
    auto_connect_attempted_ = true;

    auto& auth = cyxwiz::auth::AuthClient::Instance();
    if (!auth.IsAuthenticated()) {
        spdlog::debug("CloudBrowser: Not authenticated, skipping auto-connect");
        return;
    }

    // Create owned client if needed
    if (!owned_client_) {
        owned_client_ = std::make_unique<network::DataStreamClient>();
    }

    // Get gateway address - use central server address with gRPC port
    std::string gateway_addr = cyxwiz::core::EngineConfig::Instance().GetCentralServerAddress();
    // CyxCloud gateway uses port 50052 for DataStream, override if needed
    if (gateway_addr.find(":50051") != std::string::npos) {
        gateway_addr.replace(gateway_addr.find(":50051"), 6, ":50052");
    };
    if (gateway_addr.empty()) {
        gateway_addr = "localhost:50052";  // CyxCloud gateway gRPC port  // Default
    }

    spdlog::info("CloudBrowser: Connecting to CyxCloud gateway at {}", gateway_addr);

    if (owned_client_->Connect(gateway_addr)) {
        // Set auth token
        owned_client_->SetAuthToken(auth.GetJwtToken());
        datastream_client_ = owned_client_.get();
        spdlog::info("CloudBrowser: Connected to CyxCloud successfully");

        // Auto-refresh datasets
        RefreshDatasets();
        RefreshPublicDatasets();
    } else {
        spdlog::warn("CloudBrowser: Failed to connect to CyxCloud gateway");
    }
}
} // namespace gui
