#pragma once

#include "../panel.h"
#include "../../network/datastream_client.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

namespace gui {

/**
 * Cloud Dataset Manager Panel
 *
 * Manage cloud datasets with advanced features:
 * - Create datasets from uploaded files
 * - Configure dataset schema
 * - Manage access tokens
 * - View streaming configuration
 * - Monitor batch streaming
 */
class CloudDatasetManagerPanel : public cyxwiz::Panel {
public:
    CloudDatasetManagerPanel();
    ~CloudDatasetManagerPanel() override;

    void Render() override;

    // Set DataStream client
    void SetDataStreamClient(network::DataStreamClient* client) { datastream_client_ = client; }

    // Set dataset to manage
    void SetDataset(const network::CloudDatasetInfo& dataset);
    void ClearDataset();

    // Callbacks
    using DatasetCreatedCallback = std::function<void(const network::CloudDatasetInfo& dataset)>;
    void SetDatasetCreatedCallback(DatasetCreatedCallback callback) { on_dataset_created_ = callback; }

private:
    void RenderDatasetInfo();
    void RenderSchemaEditor();
    void RenderAccessTokens();
    void RenderStreamingConfig();
    void RenderStreamingStatus();
    void RenderCreateDatasetWizard();

    // Token management
    void CreateNewToken();
    void RevokeToken(const std::string& token_id);

    // Dataset creation
    void CreateDataset();

    network::DataStreamClient* datastream_client_ = nullptr;
    DatasetCreatedCallback on_dataset_created_;

    // Current dataset
    network::CloudDatasetInfo current_dataset_;
    std::vector<network::CloudFileInfo> current_files_;
    bool has_dataset_ = false;

    // Access tokens
    struct ManagedToken {
        std::string token_id;
        std::string token_preview;  // First 8 chars
        int64_t expires_at;
        std::vector<std::string> scopes;
    };
    std::vector<ManagedToken> tokens_;

    // Create dataset wizard state
    bool show_create_wizard_ = false;
    int wizard_step_ = 0;
    char new_dataset_name_[256] = "";
    char new_dataset_description_[1024] = "";
    char new_dataset_schema_[4096] = "";
    std::vector<std::string> selected_file_ids_;

    // Streaming configuration
    int stream_batch_size_ = 32;
    int stream_prefetch_ = 4;
    bool stream_shuffle_ = true;
    int64_t stream_seed_ = 0;
    int stream_max_trust_ = 2;  // Verified

    // Streaming status
    bool is_streaming_ = false;
    uint64_t streamed_batches_ = 0;
    uint64_t total_stream_batches_ = 0;
    float streaming_progress_ = 0.0f;

    // Token creation dialog
    bool show_token_dialog_ = false;
    char token_node_id_[128] = "";
    int token_ttl_hours_ = 24;
    bool token_scope_read_ = true;
    bool token_scope_stream_ = true;

    // Error handling
    std::string last_error_;
    
    // Owned DataStreamClient for auto-connection
    std::unique_ptr<network::DataStreamClient> owned_client_;
    bool auto_connect_attempted_ = false;
    void TryAutoConnect();
    float error_time_ = 0.0f;
};

} // namespace gui
