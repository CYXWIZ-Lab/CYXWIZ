#pragma once

#include "../network/datastream_client.h"
#include "data_registry.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

namespace cyxwiz {

/**
 * Cloud Data Registry
 *
 * Extends the local DataRegistry to support cloud-stored datasets.
 * Provides seamless integration between local and cloud datasets for training.
 *
 * Features:
 * - Cloud dataset caching
 * - Streaming data loader integration
 * - Automatic local/cloud dataset resolution
 * - Trust level management
 */
class CloudDataRegistry {
public:
    CloudDataRegistry();
    ~CloudDataRegistry();

    // Singleton access
    static CloudDataRegistry& Instance();

    // Set the DataStream client for cloud access
    void SetDataStreamClient(network::DataStreamClient* client);
    network::DataStreamClient* GetDataStreamClient() const { return datastream_client_; }

    // ========================================================================
    // Cloud Dataset Management
    // ========================================================================

    // Register a cloud dataset for use in training
    // Returns a local handle that can be used with DataRegistry
    DatasetHandle RegisterCloudDataset(const network::CloudDatasetInfo& cloud_dataset);

    // Get cloud info for a registered dataset
    bool GetCloudDatasetInfo(DatasetHandle handle, network::CloudDatasetInfo& out_info) const;

    // Check if a dataset handle refers to a cloud dataset
    bool IsCloudDataset(DatasetHandle handle) const;

    // Get dataset ID for a cloud dataset handle
    std::string GetCloudDatasetId(DatasetHandle handle) const;

    // ========================================================================
    // Streaming Integration
    // ========================================================================

    // Start streaming a cloud dataset
    // This creates a streaming loader that can be used for training
    bool StartCloudStreaming(DatasetHandle handle,
                             int batch_size,
                             network::BatchCallback on_batch,
                             network::StreamErrorCallback on_error = nullptr,
                             network::StreamCompleteCallback on_complete = nullptr,
                             bool shuffle = true,
                             int64_t seed = 0);

    // Stop streaming
    void StopCloudStreaming(DatasetHandle handle);

    // Check if streaming is active for a dataset
    bool IsStreaming(DatasetHandle handle) const;

    // Get streaming progress
    float GetStreamingProgress(DatasetHandle handle) const;

    // ========================================================================
    // Trust Level Management
    // ========================================================================

    // Get the trust level for a dataset
    network::TrustLevel GetTrustLevel(DatasetHandle handle) const;

    // Set minimum trust level requirement
    void SetMinTrustLevel(network::TrustLevel level) { min_trust_level_ = level; }
    network::TrustLevel GetMinTrustLevel() const { return min_trust_level_; }

    // Verify a cloud dataset
    bool VerifyCloudDataset(DatasetHandle handle, network::DatasetVerificationResult& out_result);

    // ========================================================================
    // Public Dataset Integration
    // ========================================================================

    // List available public datasets
    bool ListPublicDatasets(std::vector<network::PublicDatasetInfo>& out_datasets,
                            const std::string& filter = "");

    // Use a public dataset (downloads/caches if needed)
    DatasetHandle UsePublicDataset(const network::PublicDatasetInfo& public_dataset);

    // ========================================================================
    // Caching
    // ========================================================================

    // Cache cloud data locally for offline use
    bool CacheCloudDataset(DatasetHandle handle, const std::string& cache_path);

    // Check if a dataset is cached locally
    bool IsCachedLocally(DatasetHandle handle) const;

    // Clear cache for a dataset
    void ClearCache(DatasetHandle handle);

    // Get cache directory
    std::string GetCacheDirectory() const { return cache_directory_; }
    void SetCacheDirectory(const std::string& path) { cache_directory_ = path; }

private:
    // Cloud dataset entry
    struct CloudDatasetEntry {
        DatasetHandle handle;
        network::CloudDatasetInfo info;
        std::string dataset_id;
        bool is_streaming = false;
        std::string cache_path;
    };

    network::DataStreamClient* datastream_client_ = nullptr;

    // Mapping from DatasetHandle to cloud info
    std::unordered_map<uint64_t, CloudDatasetEntry> cloud_datasets_;
    uint64_t next_cloud_handle_id_ = 1000000;  // Start high to avoid conflicts with local

    // Trust level settings
    network::TrustLevel min_trust_level_ = network::TrustLevel::Verified;

    // Cache settings
    std::string cache_directory_;
};

/**
 * Unified Dataset Handle
 *
 * Works with both local and cloud datasets transparently.
 */
class UnifiedDatasetHandle {
public:
    UnifiedDatasetHandle() = default;
    UnifiedDatasetHandle(DatasetHandle local_handle);
    UnifiedDatasetHandle(const network::CloudDatasetInfo& cloud_info);

    bool IsLocal() const { return is_local_; }
    bool IsCloud() const { return !is_local_; }
    bool IsValid() const { return is_valid_; }

    DatasetHandle GetLocalHandle() const { return local_handle_; }
    std::string GetCloudDatasetId() const { return cloud_dataset_id_; }
    std::string GetName() const { return name_; }

    // Get appropriate loader config based on source
    DataLoaderConfig GetLoaderConfig(int batch_size, bool shuffle = true) const;

private:
    bool is_valid_ = false;
    bool is_local_ = true;
    DatasetHandle local_handle_;
    std::string cloud_dataset_id_;
    std::string name_;
    network::CloudDatasetInfo cloud_info_;
};

} // namespace cyxwiz
