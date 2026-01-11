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
 * Data Loader Configuration
 * Settings for loading batches from a dataset
 */
struct DataLoaderConfig {
    int batch_size = 32;
    bool shuffle = true;
    int num_workers = 0;
    bool drop_last = false;
};

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
    // Returns a dataset ID string for tracking
    std::string RegisterCloudDataset(const network::CloudDatasetInfo& cloud_dataset);

    // Get cloud info for a registered dataset by ID
    bool GetCloudDatasetInfo(const std::string& dataset_id, network::CloudDatasetInfo& out_info) const;

    // Check if a dataset ID refers to a registered cloud dataset
    bool IsCloudDataset(const std::string& dataset_id) const;

    // ========================================================================
    // Streaming Integration
    // ========================================================================

    // Start streaming a cloud dataset
    // This creates a streaming loader that can be used for training
    bool StartCloudStreaming(const std::string& dataset_id,
                             int batch_size,
                             network::BatchCallback on_batch,
                             network::StreamErrorCallback on_error = nullptr,
                             network::StreamCompleteCallback on_complete = nullptr,
                             bool shuffle = true,
                             int64_t seed = 0);

    // Stop streaming
    void StopCloudStreaming(const std::string& dataset_id);

    // Check if streaming is active for a dataset
    bool IsStreaming(const std::string& dataset_id) const;

    // Get streaming progress
    float GetStreamingProgress(const std::string& dataset_id) const;

    // ========================================================================
    // Trust Level Management
    // ========================================================================

    // Get the trust level for a dataset
    network::TrustLevel GetTrustLevel(const std::string& dataset_id) const;

    // Set minimum trust level requirement
    void SetMinTrustLevel(network::TrustLevel level) { min_trust_level_ = level; }
    network::TrustLevel GetMinTrustLevel() const { return min_trust_level_; }

    // Verify a cloud dataset
    bool VerifyCloudDataset(const std::string& dataset_id, network::DatasetVerificationResult& out_result);

    // ========================================================================
    // Public Dataset Integration
    // ========================================================================

    // List available public datasets
    bool ListPublicDatasets(std::vector<network::PublicDatasetInfo>& out_datasets,
                            const std::string& filter = "");

    // Use a public dataset (downloads/caches if needed)
    std::string UsePublicDataset(const network::PublicDatasetInfo& public_dataset);

    // ========================================================================
    // Caching
    // ========================================================================

    // Cache cloud data locally for offline use
    bool CacheCloudDataset(const std::string& dataset_id, const std::string& cache_path);

    // Check if a dataset is cached locally
    bool IsCachedLocally(const std::string& dataset_id) const;

    // Clear cache for a dataset
    void ClearCache(const std::string& dataset_id);

    // Get cache directory
    std::string GetCacheDirectory() const { return cache_directory_; }
    void SetCacheDirectory(const std::string& path) { cache_directory_ = path; }

private:
    // Cloud dataset entry
    struct CloudDatasetEntry {
        network::CloudDatasetInfo info;
        std::string dataset_id;
        bool is_streaming = false;
        std::string cache_path;
    };

    network::DataStreamClient* datastream_client_ = nullptr;

    // Mapping from dataset_id to cloud info
    std::unordered_map<std::string, CloudDatasetEntry> cloud_datasets_;

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
    explicit UnifiedDatasetHandle(DatasetHandle local_handle);
    explicit UnifiedDatasetHandle(const network::CloudDatasetInfo& cloud_info);

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
