#pragma once

#include "../panel.h"
#include "../../network/datastream_client.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>

namespace gui {

/**
 * Cloud Browser Panel
 *
 * Browse and manage datasets stored in CyxCloud.
 * Features:
 * - View user's datasets and shared datasets
 * - Browse public datasets (MNIST, CIFAR, ImageNet, etc.)
 * - Trust level visualization with badges
 * - Quick dataset info and verification
 * - Integration with local training
 */
class CloudBrowserPanel : public cyxwiz::Panel {
public:
    CloudBrowserPanel();
    ~CloudBrowserPanel() override;

    void Render() override;

    // Set DataStream client (must be connected)
    void SetDataStreamClient(network::DataStreamClient* client) { datastream_client_ = client; }

    // Callback when user selects a dataset for training
    using DatasetSelectedCallback = std::function<void(const network::CloudDatasetInfo& dataset)>;
    void SetDatasetSelectedCallback(DatasetSelectedCallback callback) { on_dataset_selected_ = callback; }

    // Callback when user wants to upload a new dataset
    using UploadRequestedCallback = std::function<void()>;
    void SetUploadRequestedCallback(UploadRequestedCallback callback) { on_upload_requested_ = callback; }

    // Refresh dataset list from server
    void RefreshDatasets();
    void RefreshPublicDatasets();

    // Get selected dataset (if any)
    const network::CloudDatasetInfo* GetSelectedDataset() const;

private:
    void RenderToolbar();
    void RenderTabs();
    void RenderMyDatasetsTab();
    void RenderPublicDatasetsTab();
    void RenderDatasetContextMenu(const network::CloudDatasetInfo& dataset);
    void RenderPublicDatasetContextMenu(const network::PublicDatasetInfo& dataset);
    void RenderDatasetTable(const std::vector<network::CloudDatasetInfo>& datasets, bool is_public = false);
    void RenderPublicDatasetTable(const std::vector<network::PublicDatasetInfo>& datasets);
    void RenderDatasetInfoPopup();
    void RenderVerificationPopup();
    void RenderSharePopup();
    void RenderDeleteConfirmPopup();
    void RenderTrustBadge(network::TrustLevel level);
    void RenderSearchBar();

    // Async operations
    void FetchDatasets();
    void FetchPublicDatasets();
    void VerifySelectedDataset();
    void DeleteSelectedDataset();
    void ShareSelectedDataset();

    // Filtering and sorting
    void ApplyFilter();
    void SortDatasets(int column, bool ascending);

    network::DataStreamClient* datastream_client_ = nullptr;
    DatasetSelectedCallback on_dataset_selected_;
    UploadRequestedCallback on_upload_requested_;

    // Dataset lists
    std::vector<network::CloudDatasetInfo> my_datasets_;
    std::vector<network::CloudDatasetInfo> filtered_datasets_;
    std::vector<network::PublicDatasetInfo> public_datasets_;
    std::vector<network::PublicDatasetInfo> filtered_public_datasets_;

    // Selected dataset
    int selected_index_ = -1;
    int selected_public_index_ = -1;
    network::CloudDatasetInfo* selected_dataset_ = nullptr;
    network::PublicDatasetInfo* selected_public_dataset_ = nullptr;

    // Detailed info for selected dataset
    std::vector<network::CloudFileInfo> selected_dataset_files_;

    // UI state
    int current_tab_ = 0;  // 0=My Datasets, 1=Public Datasets
    char search_buffer_[256] = "";
    int sort_column_ = 0;
    bool sort_ascending_ = true;
    bool show_shared_ = true;
    network::TrustLevel trust_filter_ = network::TrustLevel::Untrusted;  // Show all

    // Loading state
    std::atomic<bool> is_loading_{false};
    std::string loading_message_;

    // Popup state
    bool show_info_popup_ = false;
    bool show_verify_popup_ = false;
    bool show_share_popup_ = false;
    bool show_delete_popup_ = false;
    char share_user_id_buffer_[256] = "";
    int share_permission_flags_ = 0;  // 0x1=read, 0x2=stream, 0x4=reshare

    // Verification result (for popup)
    network::DatasetVerificationResult verification_result_;
    bool verification_in_progress_ = false;

    // Error display
    std::string last_error_;
    float error_time_ = 0.0f;

    // Thread safety
    std::mutex datasets_mutex_;
    
    // Owned DataStreamClient for auto-connection
    std::unique_ptr<network::DataStreamClient> owned_client_;
    bool auto_connect_attempted_ = false;
    
    // Auto-connect to CyxCloud when authenticated
    void TryAutoConnect();
};

} // namespace gui
