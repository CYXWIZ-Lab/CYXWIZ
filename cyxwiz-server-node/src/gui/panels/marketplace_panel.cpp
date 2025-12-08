// marketplace_panel.cpp - Model marketplace with daemon integration
#include "gui/panels/marketplace_panel.h"
#include "gui/icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz::servernode::gui {

void MarketplacePanel::Render() {
    ImGui::PushFont(GetSafeFont(FONT_LARGE));
    ImGui::Text("%s Marketplace", ICON_FA_STORE);
    ImGui::PopFont();
    ImGui::Separator();

    // Connection status
    if (IsDaemonConnected()) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Daemon Connected", ICON_FA_LINK);
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Daemon Disconnected", ICON_FA_LINK_SLASH);
        ImGui::TextDisabled("Connect to daemon to browse the marketplace.");
        return;
    }

    ImGui::Spacing();

    // Load listings on first render
    if (!listings_loaded_) {
        RefreshListings();
        listings_loaded_ = true;
    }

    RenderSearchBar();
    ImGui::Spacing();

    RenderCategories();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    RenderListings();

    // Detail dialog
    RenderDetailDialog();

    // Error popup
    if (show_error_popup_) {
        ImGui::OpenPopup("Marketplace Error");
        show_error_popup_ = false;
    }

    if (ImGui::BeginPopupModal("Marketplace Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, error_message_.c_str());
        ImGui::Spacing();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            error_message_.clear();
        }
        ImGui::EndPopup();
    }
}

void MarketplacePanel::RefreshListings() {
    auto* client = GetDaemonClient();
    if (!client || !client->IsConnected()) return;

    loading_ = true;

    const char* sort_options[] = {"rating", "downloads", "newest"};
    std::string sort_by = sort_options[sort_by_];

    if (client->ListMarketplaceModels(listings_, search_query_, selected_category_,
                                       PAGE_SIZE, current_page_ * PAGE_SIZE, sort_by)) {
        spdlog::debug("Loaded {} marketplace listings", listings_.size());
    } else {
        spdlog::warn("Failed to load marketplace listings");
    }

    loading_ = false;
}

void MarketplacePanel::RenderSearchBar() {
    // Search input
    ImGui::SetNextItemWidth(400);
    bool search_enter = ImGui::InputTextWithHint("##Search", "Search models...",
                                                  search_query_, sizeof(search_query_),
                                                  ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_MAGNIFYING_GLASS " Search") || search_enter) {
        current_page_ = 0;
        RefreshListings();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_ROTATE " Refresh")) {
        RefreshListings();
    }

    // Sort options
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 200);
    ImGui::Text("Sort:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    const char* sort_items[] = {"Rating", "Downloads", "Newest"};
    if (ImGui::Combo("##Sort", &sort_by_, sort_items, IM_ARRAYSIZE(sort_items))) {
        current_page_ = 0;
        RefreshListings();
    }
}

void MarketplacePanel::RenderCategories() {
    const struct {
        ipc::ModelCategory cat;
        const char* name;
        const char* icon;
    } categories[] = {
        {ipc::ModelCategory::All, "All", ICON_FA_GLOBE},
        {ipc::ModelCategory::LLM, "LLMs", ICON_FA_COMMENTS},
        {ipc::ModelCategory::Vision, "Vision", ICON_FA_EYE},
        {ipc::ModelCategory::Audio, "Audio", ICON_FA_MUSIC},
        {ipc::ModelCategory::Embedding, "Embedding", ICON_FA_DATABASE},
        {ipc::ModelCategory::Multimodal, "Multimodal", ICON_FA_LAYER_GROUP},
    };

    for (int i = 0; i < IM_ARRAYSIZE(categories); i++) {
        if (i > 0) ImGui::SameLine();

        bool selected = selected_category_ == categories[i].cat;
        if (selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.5f, 0.7f, 1.0f));
        }

        char label[64];
        snprintf(label, sizeof(label), "%s %s", categories[i].icon, categories[i].name);
        if (ImGui::Button(label, ImVec2(100, 0))) {
            selected_category_ = categories[i].cat;
            current_page_ = 0;
            RefreshListings();
        }

        if (selected) {
            ImGui::PopStyleColor();
        }
    }
}

void MarketplacePanel::RenderListings() {
    if (loading_) {
        ImGui::TextDisabled("Loading...");
        return;
    }

    if (listings_.empty()) {
        ImGui::TextDisabled("No models found.");
        ImGui::TextDisabled("Try a different search query or category.");
        return;
    }

    ImGui::Text("%zu models found", listings_.size());
    ImGui::Spacing();

    ImGui::BeginChild("Listings", ImVec2(0, 0), false);

    for (const auto& listing : listings_) {
        RenderListingCard(listing);
    }

    // Pagination
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (current_page_ > 0) {
        if (ImGui::Button(ICON_FA_CHEVRON_LEFT " Previous")) {
            current_page_--;
            RefreshListings();
        }
        ImGui::SameLine();
    }

    ImGui::Text("Page %d", current_page_ + 1);

    if (listings_.size() == PAGE_SIZE) {
        ImGui::SameLine();
        if (ImGui::Button("Next " ICON_FA_CHEVRON_RIGHT)) {
            current_page_++;
            RefreshListings();
        }
    }

    ImGui::EndChild();
}

void MarketplacePanel::RenderListingCard(const ipc::MarketplaceListing& listing) {
    ImGui::PushID(listing.id.c_str());

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild(("Card_" + listing.id).c_str(), ImVec2(0, 140), true);

    // Check if download in progress
    bool downloading = false;
    ipc::DownloadProgress download_progress;
    {
        std::lock_guard<std::mutex> lock(download_mutex_);
        auto it = active_downloads_.find(listing.id);
        if (it != active_downloads_.end()) {
            downloading = true;
            download_progress = it->second;
        }
    }

    // Header: Name and price
    ImGui::Text("%s %s", GetCategoryIcon(listing.category), listing.name.c_str());
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - 120);
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "%.4f CYXWIZ/req", listing.price_per_request);

    // Metadata line
    ImGui::TextDisabled("Format: %s | Size: %s | %s | %s downloads",
                        listing.format.c_str(),
                        FormatSize(listing.size_bytes).c_str(),
                        FormatRating(listing.rating).c_str(),
                        FormatCount(listing.download_count).c_str());

    // Architecture and params if available
    if (!listing.architecture.empty() || listing.parameter_count > 0) {
        ImGui::TextDisabled("Arch: %s | Params: %s",
                            listing.architecture.empty() ? "N/A" : listing.architecture.c_str(),
                            listing.parameter_count > 0 ? FormatCount(listing.parameter_count).c_str() : "N/A");
    }

    // Description (truncated)
    std::string desc = listing.description;
    if (desc.length() > 150) {
        desc = desc.substr(0, 147) + "...";
    }
    ImGui::TextWrapped("%s", desc.c_str());

    ImGui::Spacing();

    // Download progress or buttons
    if (downloading) {
        RenderDownloadProgress(listing.id);
    } else {
        if (ImGui::Button(ICON_FA_DOWNLOAD " Download")) {
            StartDownload(listing);
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_CIRCLE_INFO " Details")) {
            selected_listing_ = listing;
            show_detail_dialog_ = true;
        }
    }

    // Tags
    if (!listing.tags.empty()) {
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 200);
        for (size_t i = 0; i < std::min(listing.tags.size(), (size_t)3); i++) {
            if (i > 0) ImGui::SameLine();
            ImGui::TextDisabled("[%s]", listing.tags[i].c_str());
        }
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();

    ImGui::Spacing();
    ImGui::PopID();
}

void MarketplacePanel::RenderDownloadProgress(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(download_mutex_);
    auto it = active_downloads_.find(model_id);
    if (it == active_downloads_.end()) return;

    const auto& progress = it->second;

    if (!progress.error_message.empty()) {
        ImGui::TextColored(ImVec4(0.8f, 0.3f, 0.3f, 1.0f), "%s Download failed: %s",
                           ICON_FA_TRIANGLE_EXCLAMATION, progress.error_message.c_str());
        if (ImGui::Button("Dismiss")) {
            active_downloads_.erase(it);
        }
    } else if (progress.completed) {
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s Download complete!",
                           ICON_FA_CIRCLE_CHECK);
        ImGui::TextDisabled("Saved to: %s", progress.local_path.c_str());
        if (ImGui::Button("OK")) {
            active_downloads_.erase(it);
        }
    } else {
        // Progress bar
        ImGui::ProgressBar(progress.progress, ImVec2(300, 20));
        ImGui::SameLine();
        ImGui::Text("%s / %s",
                    FormatSize(progress.bytes_downloaded).c_str(),
                    FormatSize(progress.total_bytes).c_str());

        if (ImGui::Button(ICON_FA_STOP " Cancel")) {
            auto* client = GetDaemonClient();
            if (client) {
                client->CancelMarketplaceDownload(model_id);
            }
            active_downloads_.erase(it);
        }
    }
}

void MarketplacePanel::RenderDetailDialog() {
    if (!show_detail_dialog_) return;

    ImGui::OpenPopup("Model Details");

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    if (ImGui::BeginPopupModal("Model Details", &show_detail_dialog_)) {
        const auto& listing = selected_listing_;

        // Header
        ImGui::PushFont(GetSafeFont(FONT_LARGE));
        ImGui::Text("%s %s", GetCategoryIcon(listing.category), listing.name.c_str());
        ImGui::PopFont();

        ImGui::TextDisabled("ID: %s", listing.id.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Info columns
        ImGui::Columns(2, nullptr, false);

        ImGui::Text("Format:");
        ImGui::Text("Size:");
        ImGui::Text("Architecture:");
        ImGui::Text("Parameters:");
        ImGui::Text("Rating:");
        ImGui::Text("Downloads:");
        ImGui::Text("Price:");

        ImGui::NextColumn();

        ImGui::TextColored(ImVec4(0.3f, 0.7f, 0.9f, 1.0f), "%s", listing.format.c_str());
        ImGui::Text("%s", FormatSize(listing.size_bytes).c_str());
        ImGui::Text("%s", listing.architecture.empty() ? "N/A" : listing.architecture.c_str());
        ImGui::Text("%s", listing.parameter_count > 0 ? FormatCount(listing.parameter_count).c_str() : "N/A");
        ImGui::Text("%s", FormatRating(listing.rating).c_str());
        ImGui::Text("%s", FormatCount(listing.download_count).c_str());
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "%.4f CYXWIZ/req", listing.price_per_request);

        ImGui::Columns(1);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Description
        ImGui::Text("Description:");
        ImGui::TextWrapped("%s", listing.description.c_str());

        ImGui::Spacing();

        // Tags
        if (!listing.tags.empty()) {
            ImGui::Text("Tags:");
            ImGui::SameLine();
            for (size_t i = 0; i < listing.tags.size(); i++) {
                if (i > 0) ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.8f, 1.0f), "[%s]", listing.tags[i].c_str());
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Actions
        if (ImGui::Button(ICON_FA_DOWNLOAD " Download", ImVec2(150, 30))) {
            StartDownload(listing);
            show_detail_dialog_ = false;
        }

        ImGui::SameLine();
        if (ImGui::Button("Close", ImVec2(100, 30))) {
            show_detail_dialog_ = false;
        }

        ImGui::EndPopup();
    }
}

void MarketplacePanel::StartDownload(const ipc::MarketplaceListing& listing) {
    auto* client = GetDaemonClient();
    if (!client || !client->IsConnected()) {
        error_message_ = "Not connected to daemon";
        show_error_popup_ = true;
        return;
    }

    // Initialize progress tracking
    {
        std::lock_guard<std::mutex> lock(download_mutex_);
        ipc::DownloadProgress progress;
        progress.model_id = listing.id;
        progress.total_bytes = listing.size_bytes;
        active_downloads_[listing.id] = progress;
    }

    // Start async download
    client->DownloadMarketplaceModel(listing.id, "", [this](const ipc::DownloadProgress& progress) {
        std::lock_guard<std::mutex> lock(download_mutex_);
        active_downloads_[progress.model_id] = progress;
    });

    spdlog::info("Started download: {} ({})", listing.name, FormatSize(listing.size_bytes));
}

std::string MarketplacePanel::FormatSize(int64_t bytes) {
    if (bytes < 0) return "N/A";
    if (bytes < 1024) return std::to_string(bytes) + " B";
    if (bytes < 1024 * 1024) return std::to_string(bytes / 1024) + " KB";
    if (bytes < 1024 * 1024 * 1024) {
        double mb = bytes / (1024.0 * 1024.0);
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1f MB", mb);
        return buf;
    }
    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f GB", gb);
    return buf;
}

std::string MarketplacePanel::FormatRating(float rating) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%s %.1f/5", ICON_FA_STAR, rating);
    return buf;
}

std::string MarketplacePanel::FormatCount(int64_t count) {
    if (count < 1000) return std::to_string(count);
    if (count < 1000000) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1fK", count / 1000.0);
        return buf;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1fM", count / 1000000.0);
    return buf;
}

const char* MarketplacePanel::GetCategoryName(ipc::ModelCategory category) {
    switch (category) {
        case ipc::ModelCategory::All: return "All";
        case ipc::ModelCategory::LLM: return "LLM";
        case ipc::ModelCategory::Vision: return "Vision";
        case ipc::ModelCategory::Audio: return "Audio";
        case ipc::ModelCategory::Embedding: return "Embedding";
        case ipc::ModelCategory::Multimodal: return "Multimodal";
        default: return "Unknown";
    }
}

const char* MarketplacePanel::GetCategoryIcon(ipc::ModelCategory category) {
    switch (category) {
        case ipc::ModelCategory::All: return ICON_FA_GLOBE;
        case ipc::ModelCategory::LLM: return ICON_FA_COMMENTS;
        case ipc::ModelCategory::Vision: return ICON_FA_EYE;
        case ipc::ModelCategory::Audio: return ICON_FA_MUSIC;
        case ipc::ModelCategory::Embedding: return ICON_FA_DATABASE;
        case ipc::ModelCategory::Multimodal: return ICON_FA_LAYER_GROUP;
        default: return ICON_FA_CUBE;
    }
}

} // namespace cyxwiz::servernode::gui
