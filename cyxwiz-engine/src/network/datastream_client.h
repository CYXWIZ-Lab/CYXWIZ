#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "datastream.grpc.pb.h"

namespace network {

// Trust level enum matching proto definition
enum class TrustLevel {
    Self = 0,       // User's own uploads
    Signed = 1,     // Cryptographically signed
    Verified = 2,   // Hash verified
    Attested = 3,   // TEE/SGX attested
    Untrusted = 4   // Unknown source
};

// Display-friendly dataset info
struct CloudDatasetInfo {
    std::string id;
    std::string name;
    std::string owner_id;
    std::string description;
    int64_t total_size_bytes = 0;
    int32_t file_count = 0;
    TrustLevel trust_level = TrustLevel::Untrusted;
    int32_t version = 1;
    int64_t created_at = 0;
    int64_t updated_at = 0;
    std::string schema_json;
    std::vector<uint8_t> content_hash;

    // Computed fields
    std::string GetSizeString() const;
    std::string GetTrustLevelString() const;
    std::string GetCreatedAtString() const;
};

// File info within a dataset
struct CloudFileInfo {
    std::string id;
    std::string path;
    int64_t size_bytes = 0;
    int32_t file_index = 0;
    std::vector<uint8_t> content_hash;
};

// Public dataset info (MNIST, CIFAR, etc.)
struct PublicDatasetInfo {
    std::string id;
    std::string name;
    std::string version;
    std::string official_url;
    std::string paper_url;
    std::string license;
    std::vector<std::string> verified_by;
    bool cached = false;
    std::string cached_dataset_id;
};

// Verification result
struct DatasetVerificationResult {
    std::string dataset_id;
    bool manifest_valid = false;
    bool all_files_valid = false;
    TrustLevel computed_trust_level = TrustLevel::Untrusted;
    std::string message;
    std::string public_match_name;
    double public_match_confidence = 0.0;
};

// Access token info
struct DataStreamAccessToken {
    std::string token;
    std::string token_id;
    int64_t expires_at = 0;
    std::vector<std::string> scopes;
};

// Streaming batch for training
struct StreamingBatch {
    uint64_t batch_index = 0;
    std::vector<std::vector<uint8_t>> items;
    std::vector<std::vector<uint8_t>> item_hashes;
    std::vector<uint8_t> batch_hash;
    uint64_t total_batches = 0;
    bool is_last = false;
};

// Callbacks for streaming
using BatchCallback = std::function<void(const StreamingBatch& batch)>;
using StreamErrorCallback = std::function<void(const std::string& error)>;
using StreamCompleteCallback = std::function<void(uint64_t total_batches)>;

// Upload progress callback
using UploadProgressCallback = std::function<void(int64_t bytes_sent, int64_t total_bytes)>;

/**
 * DataStream client for CyxCloud dataset streaming
 *
 * Provides access to cloud-stored datasets for ML training with:
 * - Dataset browsing and management
 * - Zero-copy batch streaming
 * - Blake3 hash verification
 * - Trust level enforcement
 * - Public dataset integration
 */
class DataStreamClient {
public:
    DataStreamClient();
    ~DataStreamClient();

    // Connection management
    bool Connect(const std::string& gateway_address);
    void Disconnect();
    bool IsConnected() const { return connected_; }
    std::string GetGatewayAddress() const { return gateway_address_; }

    // Authentication
    void SetAuthToken(const std::string& token) { auth_token_ = token; }
    void ClearAuthToken() { auth_token_.clear(); }
    bool HasAuthToken() const { return !auth_token_.empty(); }

    // ========================================================================
    // Dataset Browsing
    // ========================================================================

    // List user's datasets
    bool ListDatasets(std::vector<CloudDatasetInfo>& out_datasets,
                      int limit = 100,
                      int offset = 0,
                      bool include_shared = true,
                      TrustLevel max_trust_level = TrustLevel::Untrusted);

    // Get detailed dataset info
    bool GetDatasetInfo(const std::string& dataset_id,
                        CloudDatasetInfo& out_info,
                        std::vector<CloudFileInfo>& out_files);

    // List public datasets (MNIST, CIFAR, ImageNet, etc.)
    bool ListPublicDatasets(std::vector<PublicDatasetInfo>& out_datasets,
                            const std::string& name_filter = "");

    // ========================================================================
    // Dataset Management
    // ========================================================================

    // Create dataset from uploaded files
    bool CreateDataset(const std::string& name,
                       const std::string& description,
                       const std::vector<std::string>& file_ids,
                       const std::string& schema_json,
                       CloudDatasetInfo& out_dataset);

    // Delete a dataset
    bool DeleteDataset(const std::string& dataset_id);

    // Share dataset with another user
    bool ShareDataset(const std::string& dataset_id,
                      const std::string& share_with_user_id,
                      const std::vector<std::string>& permissions,
                      int64_t expires_at = 0);

    // ========================================================================
    // File Upload
    // ========================================================================

    // Upload a file to CyxCloud
    bool UploadFile(const std::string& filepath,
                    std::string& out_file_id,
                    UploadProgressCallback progress_callback = nullptr);

    // Upload data from memory
    bool UploadFileData(const std::string& filename,
                        const std::vector<uint8_t>& data,
                        const std::string& content_type,
                        std::string& out_file_id,
                        UploadProgressCallback progress_callback = nullptr);

    // ========================================================================
    // Verification
    // ========================================================================

    // Verify dataset integrity
    bool VerifyDataset(const std::string& dataset_id,
                       DatasetVerificationResult& out_result,
                       bool check_public_registry = true,
                       bool full_verification = false);

    // ========================================================================
    // Access Tokens
    // ========================================================================

    // Create access token for node streaming
    bool CreateAccessToken(const std::string& dataset_id,
                           DataStreamAccessToken& out_token,
                           const std::string& node_id = "",
                           int64_t ttl_seconds = 86400);

    // Revoke an access token
    bool RevokeAccessToken(const std::string& token_id);

    // ========================================================================
    // Batch Streaming
    // ========================================================================

    // Start streaming batches (non-blocking)
    bool StartStreaming(const std::string& dataset_id,
                        int32_t batch_size,
                        BatchCallback on_batch,
                        StreamErrorCallback on_error = nullptr,
                        StreamCompleteCallback on_complete = nullptr,
                        int64_t start_index = 0,
                        int32_t prefetch_batches = 4,
                        TrustLevel max_trust_level = TrustLevel::Verified,
                        bool shuffle = true,
                        int64_t seed = 0,
                        const std::string& access_token = "");

    // Stop streaming
    void StopStreaming();

    // Check if streaming is active
    bool IsStreaming() const { return streaming_.load(); }

    // Get streaming progress (0.0 - 1.0)
    float GetStreamingProgress() const;

    // Get total batches (available after first batch received)
    uint64_t GetTotalBatches() const { return total_batches_.load(); }

    // ========================================================================
    // Error Handling
    // ========================================================================

    std::string GetLastError() const { return last_error_; }

private:
    // Helper to add auth metadata
    void AddAuthMetadata(grpc::ClientContext& context);

    // Convert proto types to local types
    static CloudDatasetInfo ConvertDatasetInfo(const cyxwiz::protocol::DatasetInfo& proto);
    static CloudFileInfo ConvertFileInfo(const cyxwiz::protocol::DatasetFileInfo& proto);
    static PublicDatasetInfo ConvertPublicDatasetInfo(const cyxwiz::protocol::PublicDatasetInfo& proto);
    static TrustLevel ConvertTrustLevel(cyxwiz::protocol::TrustLevel proto);

    // Streaming thread function
    void StreamingThread(const std::string& dataset_id,
                         int32_t batch_size,
                         BatchCallback on_batch,
                         StreamErrorCallback on_error,
                         StreamCompleteCallback on_complete,
                         int64_t start_index,
                         int32_t prefetch_batches,
                         TrustLevel max_trust_level,
                         bool shuffle,
                         int64_t seed,
                         const std::string& access_token);

    bool connected_ = false;
    std::string gateway_address_;
    std::string auth_token_;
    std::string last_error_;

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<cyxwiz::protocol::DataStreamService::Stub> stub_;

    // Streaming state
    std::atomic<bool> streaming_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<uint64_t> current_batch_{0};
    std::atomic<uint64_t> total_batches_{0};
    std::unique_ptr<std::thread> streaming_thread_;
    std::unique_ptr<grpc::ClientContext> stream_context_;
};

} // namespace network
