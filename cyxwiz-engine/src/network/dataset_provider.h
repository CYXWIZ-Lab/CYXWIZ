#pragma once

#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <vector>
#include "execution.pb.h"
#include "../core/data_registry.h"

namespace network {

/**
 * DatasetProvider - Handles dataset streaming requests from Server Node
 *
 * When the Engine submits a job with a dataset, the Server Node can request
 * batches on-demand through the P2P bidirectional stream. This class:
 *
 * 1. Registers datasets when jobs are submitted
 * 2. Handles DatasetInfoRequest -> returns split sizes, shapes, etc.
 * 3. Handles BatchRequest -> returns actual batch data as raw bytes
 *
 * Usage:
 *   DatasetProvider provider;
 *   provider.RegisterDataset("job_123", dataset_handle);
 *
 *   // In P2P stream handler:
 *   if (update.has_dataset_info_request()) {
 *       auto response = provider.HandleDatasetInfoRequest(update.dataset_info_request());
 *       // Send response through stream
 *   }
 */
class DatasetProvider {
public:
    DatasetProvider();
    ~DatasetProvider() = default;

    /**
     * Register a dataset for a job before submission
     * The dataset will be available for streaming until unregistered
     */
    void RegisterDataset(const std::string& job_id,
                         cyxwiz::DatasetHandle dataset);

    /**
     * Unregister a dataset when job completes or is cancelled
     */
    void UnregisterDataset(const std::string& job_id);

    /**
     * Check if a dataset is registered for a job
     */
    bool HasDataset(const std::string& job_id) const;

    /**
     * Handle a DatasetInfoRequest from Server Node
     * Returns metadata about the dataset (split sizes, shape, etc.)
     */
    cyxwiz::protocol::DatasetInfoResponse HandleDatasetInfoRequest(
        const cyxwiz::protocol::DatasetInfoRequest& request);

    /**
     * Handle a BatchRequest from Server Node
     * Returns actual batch data as raw float32 bytes
     */
    cyxwiz::protocol::BatchResponse HandleBatchRequest(
        const cyxwiz::protocol::BatchRequest& request);

    /**
     * Get count of registered datasets
     */
    size_t GetRegisteredCount() const;

    /**
     * Get list of registered job IDs
     */
    std::vector<std::string> GetRegisteredJobIds() const;

private:
    // Convert engine DatasetSplit to proto DatasetSplit
    static cyxwiz::protocol::DatasetSplit ToProtoSplit(cyxwiz::DatasetSplit split);

    // Convert proto DatasetSplit to engine DatasetSplit
    static cyxwiz::DatasetSplit FromProtoSplit(cyxwiz::protocol::DatasetSplit split);

    // Serialize batch data to raw bytes (float32)
    static std::string SerializeImages(const std::vector<std::vector<float>>& images);
    static std::string SerializeLabels(const std::vector<int>& labels, bool one_hot, int num_classes);

    struct RegisteredDataset {
        cyxwiz::DatasetHandle handle;
        cyxwiz::DatasetInfo info;
    };

    std::map<std::string, RegisteredDataset> datasets_;  // job_id -> dataset
    mutable std::mutex mutex_;
};

} // namespace network
