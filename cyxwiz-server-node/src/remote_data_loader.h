#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <grpcpp/grpcpp.h>
#include "execution.pb.h"

namespace cyxwiz {
namespace server_node {

/**
 * Batch data structure containing images and labels
 */
struct Batch {
    std::vector<float> images;   // Flattened image data
    std::vector<int32_t> labels; // Label indices
    std::vector<int64_t> image_shape;  // [batch_size, C, H, W]
    std::vector<int64_t> label_shape;  // [batch_size] or [batch_size, num_classes]
    int32_t batch_size = 0;
};

/**
 * Dataset metadata from Engine
 */
struct DatasetMetadata {
    int64_t train_samples = 0;
    int64_t val_samples = 0;
    int64_t test_samples = 0;
    bool train_available = false;
    bool val_available = false;
    bool test_available = false;
    std::vector<int32_t> sample_shape;  // [C, H, W] for images
    std::vector<int32_t> label_shape;
    std::string dtype;
    int32_t num_classes = 0;
    std::vector<std::string> class_names;
};

// Stream write callback type
using StreamWriteFunc = std::function<bool(const cyxwiz::protocol::TrainingUpdate&)>;

/**
 * RemoteDataLoader - Loads data lazily from Engine through P2P stream
 *
 * Instead of downloading the entire dataset, this loader requests batches
 * on-demand from the Engine during training. This enables:
 * - Memory efficient training (only current batch in memory)
 * - Large dataset support (datasets bigger than RAM)
 * - Test set privacy (Engine keeps test data private)
 *
 * Usage:
 *   RemoteDataLoader loader(write_func, job_id, SPLIT_TRAIN, batch_size);
 *   loader.Initialize(metadata);
 *
 *   while (loader.HasNextBatch()) {
 *       Batch batch = loader.GetNextBatch();
 *       // Use batch for training
 *   }
 *   loader.Reset();  // New epoch
 */
class RemoteDataLoader {
public:
    RemoteDataLoader(StreamWriteFunc write_func,
                     const std::string& job_id,
                     cyxwiz::protocol::DatasetSplit split,
                     int batch_size,
                     bool shuffle = true);
    ~RemoteDataLoader();

    /**
     * Initialize with dataset metadata from Engine
     */
    void Initialize(const DatasetMetadata& metadata);

    /**
     * Request dataset info from Engine
     * Returns true if request was sent (wait for OnDatasetInfoResponse)
     */
    bool RequestDatasetInfo();

    /**
     * Check if more batches available in current epoch
     */
    bool HasNextBatch() const;

    /**
     * Get the next batch (sends request to Engine, waits for response)
     * Returns empty batch on error
     */
    Batch GetNextBatch();

    /**
     * Reset for new epoch (reshuffles indices)
     */
    void Reset();

    /**
     * Handle incoming BatchResponse from Engine
     * Called by JobExecutionService when response arrives
     */
    void OnBatchResponse(const cyxwiz::protocol::BatchResponse& response);

    /**
     * Handle incoming DatasetInfoResponse from Engine
     * Called by JobExecutionService when response arrives
     */
    void OnDatasetInfoResponse(const cyxwiz::protocol::DatasetInfoResponse& response);

    // Getters
    int64_t NumSamples() const { return num_samples_; }
    int64_t NumBatches() const;
    int CurrentEpoch() const { return current_epoch_; }
    const DatasetMetadata& GetMetadata() const { return metadata_; }
    bool IsInitialized() const { return initialized_; }

    // Set timeout for batch requests (milliseconds)
    void SetTimeout(int timeout_ms) { timeout_ms_ = timeout_ms; }

private:
    // Send batch request through the stream
    bool SendBatchRequest(const std::vector<int64_t>& indices);

    // Wait for response with matching request_id
    bool WaitForResponse(int32_t request_id);

    // Parse batch data from protobuf
    Batch ParseBatchResponse(const cyxwiz::protocol::BatchResponse& response);

    // Stream writer function
    StreamWriteFunc write_func_;

    // Job and split configuration
    std::string job_id_;
    cyxwiz::protocol::DatasetSplit split_;
    int batch_size_;
    bool shuffle_;

    // Dataset metadata
    DatasetMetadata metadata_;
    int64_t num_samples_ = 0;
    bool initialized_ = false;

    // Epoch management
    std::vector<int64_t> indices_;
    int current_batch_idx_ = 0;
    int current_epoch_ = 0;

    // Request/response tracking
    std::atomic<int32_t> next_request_id_{1};
    int32_t pending_request_id_ = 0;

    // Response queue
    std::queue<cyxwiz::protocol::BatchResponse> batch_responses_;
    std::mutex response_mutex_;
    std::condition_variable response_cv_;

    // Dataset info response
    bool dataset_info_received_ = false;
    cyxwiz::protocol::DatasetInfoResponse dataset_info_response_;
    std::mutex info_mutex_;
    std::condition_variable info_cv_;

    // Timeout
    int timeout_ms_ = 30000;  // 30 seconds default
};

} // namespace server_node
} // namespace cyxwiz
