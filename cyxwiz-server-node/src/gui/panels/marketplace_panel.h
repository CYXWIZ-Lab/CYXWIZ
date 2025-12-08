// marketplace_panel.h - Model marketplace with daemon integration
#pragma once
#include "gui/server_panel.h"
#include "ipc/daemon_client.h"
#include <vector>
#include <string>
#include <map>
#include <mutex>

namespace cyxwiz::servernode::gui {

class MarketplacePanel : public ServerPanel {
public:
    MarketplacePanel() : ServerPanel("Marketplace") {}
    void Render() override;

private:
    void RefreshListings();
    void RenderSearchBar();
    void RenderCategories();
    void RenderListings();
    void RenderListingCard(const ipc::MarketplaceListing& listing);
    void RenderDetailDialog();
    void RenderDownloadProgress(const std::string& model_id);
    void StartDownload(const ipc::MarketplaceListing& listing);
    std::string FormatSize(int64_t bytes);
    std::string FormatRating(float rating);
    std::string FormatCount(int64_t count);
    const char* GetCategoryName(ipc::ModelCategory category);
    const char* GetCategoryIcon(ipc::ModelCategory category);

    // Search state
    char search_query_[256] = "";
    ipc::ModelCategory selected_category_ = ipc::ModelCategory::All;
    int sort_by_ = 0;  // 0=rating, 1=downloads, 2=newest

    // Listings
    std::vector<ipc::MarketplaceListing> listings_;
    bool listings_loaded_ = false;
    bool loading_ = false;
    int total_count_ = 0;
    int current_page_ = 0;
    static constexpr int PAGE_SIZE = 20;

    // Download state
    std::map<std::string, ipc::DownloadProgress> active_downloads_;
    std::mutex download_mutex_;

    // Detail view
    bool show_detail_dialog_ = false;
    ipc::MarketplaceListing selected_listing_;

    // Error handling
    std::string error_message_;
    bool show_error_popup_ = false;
};

} // namespace cyxwiz::servernode::gui
