#include "remote_data_loader.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <random>
#include <cstring>
#include <chrono>

namespace cyxwiz {
namespace server_node {

RemoteDataLoader::RemoteDataLoader(StreamWriteFunc write_func,
                                   const std::string& job_id,
                                   cyxwiz::protocol::DatasetSplit split,
                                   int batch_size,
                                   bool shuffle)
    : write_func_(write_func)
    , job_id_(job_id)
    , split_(split)
    , batch_size_(batch_size)
    , shuffle_(shuffle)
{
    spdlog::info("RemoteDataLoader: Created for job {}, split={}, batch_size={}",
                 job_id, static_cast<int>(split), batch_size);
}

RemoteDataLoader::~RemoteDataLoader() {
    spdlog::debug("RemoteDataLoader: Destroyed for job {}", job_id_);
}

void RemoteDataLoader::Initialize(const DatasetMetadata& metadata) {
    metadata_ = metadata;

    // Get sample count for this split
    switch (split_) {
        case cyxwiz::protocol::SPLIT_TRAIN:
            num_samples_ = metadata.train_samples;
            break;
        case cyxwiz::protocol::SPLIT_VALIDATION:
            num_samples_ = metadata.val_samples;
            break;
        case cyxwiz::protocol::SPLIT_TEST:
            num_samples_ = metadata.test_samples;
            break;
        default:
            num_samples_ = metadata.train_samples;
            break;
    }

    // Create indices
    indices_.resize(num_samples_);
    for (int64_t i = 0; i < num_samples_; ++i) {
        indices_[i] = i;
    }

    // Shuffle if enabled
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }

    current_batch_idx_ = 0;
    current_epoch_ = 0;
    initialized_ = true;

    spdlog::info("RemoteDataLoader: Initialized with {} samples, {} batches",
                 num_samples_, NumBatches());
}

bool RemoteDataLoader::RequestDatasetInfo() {
    cyxwiz::protocol::TrainingUpdate update;
    update.set_job_id(job_id_);
    update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

    auto* request = update.mutable_dataset_info_request();
    request->set_job_id(job_id_);
    // auth_token is added at the connection level

    spdlog::info("RemoteDataLoader: Requesting dataset info for job {}", job_id_);

    dataset_info_received_ = false;
    if (!write_func_(update)) {
        spdlog::error("RemoteDataLoader: Failed to send DatasetInfoRequest");
        return false;
    }

    // Wait for response
    std::unique_lock<std::mutex> lock(info_mutex_);
    bool received = info_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_),
        [this]() { return dataset_info_received_; });

    if (!received) {
        spdlog::error("RemoteDataLoader: Timeout waiting for DatasetInfoResponse");
        return false;
    }

    return true;
}

bool RemoteDataLoader::HasNextBatch() const {
    if (!initialized_) return false;
    return current_batch_idx_ * batch_size_ < num_samples_;
}

Batch RemoteDataLoader::GetNextBatch() {
    Batch batch;

    if (!initialized_ || !HasNextBatch()) {
        spdlog::warn("RemoteDataLoader: No more batches available");
        return batch;
    }

    // Calculate indices for this batch
    int64_t start_idx = current_batch_idx_ * batch_size_;
    int64_t end_idx = std::min(start_idx + batch_size_, num_samples_);
    std::vector<int64_t> batch_indices(indices_.begin() + start_idx,
                                        indices_.begin() + end_idx);

    // Send batch request
    int32_t request_id = next_request_id_++;
    pending_request_id_ = request_id;

    if (!SendBatchRequest(batch_indices)) {
        spdlog::error("RemoteDataLoader: Failed to send batch request");
        return batch;
    }

    // Wait for response
    if (!WaitForResponse(request_id)) {
        spdlog::error("RemoteDataLoader: Failed to receive batch response");
        return batch;
    }

    // Get response from queue
    cyxwiz::protocol::BatchResponse response;
    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        if (batch_responses_.empty()) {
            spdlog::error("RemoteDataLoader: Response queue empty after wait");
            return batch;
        }
        response = std::move(batch_responses_.front());
        batch_responses_.pop();
    }

    // Parse response
    batch = ParseBatchResponse(response);
    current_batch_idx_++;

    spdlog::debug("RemoteDataLoader: Got batch {} with {} samples",
                  current_batch_idx_ - 1, batch.batch_size);

    return batch;
}

void RemoteDataLoader::Reset() {
    current_batch_idx_ = 0;
    current_epoch_++;

    // Reshuffle for new epoch
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }

    spdlog::info("RemoteDataLoader: Reset for epoch {}", current_epoch_);
}

void RemoteDataLoader::OnBatchResponse(const cyxwiz::protocol::BatchResponse& response) {
    spdlog::debug("RemoteDataLoader: Received BatchResponse, request_id={}",
                  response.request_id());

    {
        std::lock_guard<std::mutex> lock(response_mutex_);
        batch_responses_.push(response);
    }
    response_cv_.notify_all();
}

void RemoteDataLoader::OnDatasetInfoResponse(const cyxwiz::protocol::DatasetInfoResponse& response) {
    spdlog::info("RemoteDataLoader: Received DatasetInfoResponse, status={}",
                 static_cast<int>(response.status()));

    {
        std::lock_guard<std::mutex> lock(info_mutex_);
        dataset_info_response_ = response;
        dataset_info_received_ = true;

        // Parse metadata from response
        if (response.status() == cyxwiz::protocol::STATUS_SUCCESS) {
            metadata_.train_samples = response.train().num_samples();
            metadata_.val_samples = response.validation().num_samples();
            metadata_.test_samples = response.test().num_samples();
            metadata_.train_available = response.train().available();
            metadata_.val_available = response.validation().available();
            metadata_.test_available = response.test().available();

            for (int i = 0; i < response.sample_shape_size(); ++i) {
                metadata_.sample_shape.push_back(response.sample_shape(i));
            }
            for (int i = 0; i < response.label_shape_size(); ++i) {
                metadata_.label_shape.push_back(response.label_shape(i));
            }

            metadata_.dtype = response.dtype();
            metadata_.num_classes = response.num_classes();
            for (int i = 0; i < response.class_names_size(); ++i) {
                metadata_.class_names.push_back(response.class_names(i));
            }
        }
    }
    info_cv_.notify_all();
}

int64_t RemoteDataLoader::NumBatches() const {
    if (num_samples_ == 0) return 0;
    return (num_samples_ + batch_size_ - 1) / batch_size_;
}

bool RemoteDataLoader::SendBatchRequest(const std::vector<int64_t>& indices) {
    cyxwiz::protocol::TrainingUpdate update;
    update.set_job_id(job_id_);
    update.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());

    auto* request = update.mutable_batch_request();
    request->set_job_id(job_id_);
    request->set_split(split_);
    request->set_request_id(pending_request_id_);

    for (int64_t idx : indices) {
        request->add_sample_indices(idx);
    }

    return write_func_(update);
}

bool RemoteDataLoader::WaitForResponse(int32_t request_id) {
    std::unique_lock<std::mutex> lock(response_mutex_);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms_);

    while (batch_responses_.empty()) {
        if (response_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            spdlog::warn("RemoteDataLoader: Timeout waiting for response {}",
                         request_id);
            return false;
        }
    }

    // Check if response matches our request
    const auto& response = batch_responses_.front();
    if (response.request_id() != request_id) {
        spdlog::warn("RemoteDataLoader: Response mismatch (got {}, expected {})",
                     response.request_id(), request_id);
        // Continue anyway - could be out of order
    }

    return response.status() == cyxwiz::protocol::STATUS_SUCCESS;
}

Batch RemoteDataLoader::ParseBatchResponse(const cyxwiz::protocol::BatchResponse& response) {
    Batch batch;

    if (response.status() != cyxwiz::protocol::STATUS_SUCCESS) {
        spdlog::error("RemoteDataLoader: Batch response failed: {}",
                      response.error().message());
        return batch;
    }

    // Parse image data (float32)
    const std::string& image_bytes = response.images();
    size_t num_floats = image_bytes.size() / sizeof(float);
    batch.images.resize(num_floats);
    std::memcpy(batch.images.data(), image_bytes.data(), image_bytes.size());

    // Parse label data (int32)
    const std::string& label_bytes = response.labels();
    size_t num_labels = label_bytes.size() / sizeof(int32_t);
    batch.labels.resize(num_labels);
    std::memcpy(batch.labels.data(), label_bytes.data(), label_bytes.size());

    // Copy shape info
    for (int i = 0; i < response.batch_shape_size(); ++i) {
        batch.image_shape.push_back(response.batch_shape(i));
    }
    for (int i = 0; i < response.label_shape_size(); ++i) {
        batch.label_shape.push_back(response.label_shape(i));
    }

    // Set batch size from shape
    if (!batch.image_shape.empty()) {
        batch.batch_size = static_cast<int32_t>(batch.image_shape[0]);
    } else if (!batch.labels.empty()) {
        batch.batch_size = static_cast<int32_t>(batch.labels.size());
    }

    return batch;
}

} // namespace server_node
} // namespace cyxwiz
