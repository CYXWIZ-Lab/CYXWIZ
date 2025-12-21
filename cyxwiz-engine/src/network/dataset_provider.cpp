#include "dataset_provider.h"
#include <spdlog/spdlog.h>
#include <cstring>

namespace network {

DatasetProvider::DatasetProvider() {
    spdlog::info("DatasetProvider: Initialized");
}

void DatasetProvider::RegisterDataset(const std::string& job_id,
                                       cyxwiz::DatasetHandle dataset) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!dataset.IsValid()) {
        spdlog::warn("DatasetProvider: Cannot register invalid dataset for job {}", job_id);
        return;
    }

    RegisteredDataset reg;
    reg.handle = dataset;
    reg.info = dataset.GetInfo();

    datasets_[job_id] = reg;

    spdlog::info("DatasetProvider: Registered dataset '{}' for job {} "
                 "(train={}, val={}, test={}, shape={})",
                 reg.info.name,
                 job_id,
                 reg.info.train_count,
                 reg.info.val_count,
                 reg.info.test_count,
                 reg.info.GetShapeString());
}

void DatasetProvider::UnregisterDataset(const std::string& job_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = datasets_.find(job_id);
    if (it != datasets_.end()) {
        spdlog::info("DatasetProvider: Unregistered dataset for job {}", job_id);
        datasets_.erase(it);
    }
}

bool DatasetProvider::HasDataset(const std::string& job_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return datasets_.find(job_id) != datasets_.end();
}

size_t DatasetProvider::GetRegisteredCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return datasets_.size();
}

std::vector<std::string> DatasetProvider::GetRegisteredJobIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    for (const auto& [id, _] : datasets_) {
        ids.push_back(id);
    }
    return ids;
}

cyxwiz::protocol::DatasetInfoResponse DatasetProvider::HandleDatasetInfoRequest(
    const cyxwiz::protocol::DatasetInfoRequest& request) {

    cyxwiz::protocol::DatasetInfoResponse response;
    std::lock_guard<std::mutex> lock(mutex_);

    spdlog::debug("DatasetProvider: HandleDatasetInfoRequest for job {}", request.job_id());

    auto it = datasets_.find(request.job_id());
    if (it == datasets_.end()) {
        spdlog::warn("DatasetProvider: No dataset registered for job {}", request.job_id());
        response.set_status(cyxwiz::protocol::STATUS_ERROR);
        response.mutable_error()->set_code(1001);  // DATASET_NOT_FOUND
        response.mutable_error()->set_message("No dataset registered for job: " + request.job_id());
        return response;
    }

    const auto& reg = it->second;
    const auto& info = reg.info;

    response.set_status(cyxwiz::protocol::STATUS_SUCCESS);

    // Train split info
    auto* train = response.mutable_train();
    train->set_num_samples(info.train_count);
    train->set_available(info.train_count > 0);
    train->set_shuffle_enabled(true);

    // Validation split info
    auto* val = response.mutable_validation();
    val->set_num_samples(info.val_count);
    val->set_available(info.val_count > 0);
    val->set_shuffle_enabled(false);

    // Test split info - mark as NOT available (stays on Engine)
    auto* test = response.mutable_test();
    test->set_num_samples(info.test_count);
    test->set_available(false);  // Test set stays private on Engine
    test->set_shuffle_enabled(false);

    // Sample shape (convert size_t to int32)
    for (size_t dim : info.shape) {
        response.add_sample_shape(static_cast<int32_t>(dim));
    }

    // Label shape (for now, just [1] for class index)
    response.add_label_shape(1);

    response.set_dtype("float32");
    response.set_num_classes(static_cast<int32_t>(info.num_classes));

    // Class names
    for (const auto& name : info.class_names) {
        response.add_class_names(name);
    }

    spdlog::info("DatasetProvider: Sent dataset info for job {} "
                 "(train={}, val={}, test={} [private])",
                 request.job_id(),
                 info.train_count,
                 info.val_count,
                 info.test_count);

    return response;
}

cyxwiz::protocol::BatchResponse DatasetProvider::HandleBatchRequest(
    const cyxwiz::protocol::BatchRequest& request) {

    cyxwiz::protocol::BatchResponse response;
    response.set_request_id(request.request_id());

    std::lock_guard<std::mutex> lock(mutex_);

    spdlog::debug("DatasetProvider: HandleBatchRequest for job {}, split={}, indices={}",
                  request.job_id(),
                  static_cast<int>(request.split()),
                  request.sample_indices_size());

    auto it = datasets_.find(request.job_id());
    if (it == datasets_.end()) {
        spdlog::warn("DatasetProvider: No dataset registered for job {}", request.job_id());
        response.set_status(cyxwiz::protocol::STATUS_ERROR);
        response.mutable_error()->set_code(1001);  // DATASET_NOT_FOUND
        response.mutable_error()->set_message("No dataset registered for job: " + request.job_id());
        return response;
    }

    const auto& reg = it->second;

    // Check if requesting test set (not allowed)
    if (request.split() == cyxwiz::protocol::SPLIT_TEST) {
        spdlog::warn("DatasetProvider: Test set access denied for job {}", request.job_id());
        response.set_status(cyxwiz::protocol::STATUS_ERROR);
        response.mutable_error()->set_code(1002);  // TEST_SET_PRIVATE
        response.mutable_error()->set_message("Test set is private and not available for remote training");
        return response;
    }

    // Get the split indices
    cyxwiz::DatasetSplit engine_split = FromProtoSplit(request.split());
    const auto& split_indices = reg.handle.GetSplitIndices(engine_split);

    // Validate requested indices
    std::vector<size_t> batch_indices;
    for (int i = 0; i < request.sample_indices_size(); ++i) {
        int64_t requested_idx = request.sample_indices(i);
        if (requested_idx < 0 || requested_idx >= static_cast<int64_t>(split_indices.size())) {
            spdlog::warn("DatasetProvider: Invalid index {} for split with {} samples",
                         requested_idx, split_indices.size());
            continue;
        }
        // Map split-relative index to absolute dataset index
        batch_indices.push_back(split_indices[requested_idx]);
    }

    if (batch_indices.empty()) {
        spdlog::warn("DatasetProvider: No valid indices in batch request");
        response.set_status(cyxwiz::protocol::STATUS_ERROR);
        response.mutable_error()->set_code(1003);  // INVALID_INDICES
        response.mutable_error()->set_message("No valid indices in batch request");
        return response;
    }

    // Get the batch data
    auto [images, labels] = reg.handle.GetBatch(batch_indices);

    // Serialize to raw bytes
    std::string image_bytes = SerializeImages(images);
    std::string label_bytes = SerializeLabels(labels, false, reg.info.num_classes);

    response.set_status(cyxwiz::protocol::STATUS_SUCCESS);
    response.set_images(image_bytes);
    response.set_labels(label_bytes);

    // Set batch shape: [batch_size, ...sample_shape]
    response.add_batch_shape(static_cast<int64_t>(images.size()));
    for (size_t dim : reg.info.shape) {
        response.add_batch_shape(static_cast<int64_t>(dim));
    }

    // Label shape: [batch_size]
    response.add_label_shape(static_cast<int64_t>(labels.size()));

    spdlog::debug("DatasetProvider: Sent batch of {} samples ({} bytes images, {} bytes labels)",
                  images.size(), image_bytes.size(), label_bytes.size());

    return response;
}

cyxwiz::protocol::DatasetSplit DatasetProvider::ToProtoSplit(cyxwiz::DatasetSplit split) {
    switch (split) {
        case cyxwiz::DatasetSplit::Train:
            return cyxwiz::protocol::SPLIT_TRAIN;
        case cyxwiz::DatasetSplit::Validation:
            return cyxwiz::protocol::SPLIT_VALIDATION;
        case cyxwiz::DatasetSplit::Test:
            return cyxwiz::protocol::SPLIT_TEST;
        default:
            return cyxwiz::protocol::SPLIT_TRAIN;
    }
}

cyxwiz::DatasetSplit DatasetProvider::FromProtoSplit(cyxwiz::protocol::DatasetSplit split) {
    switch (split) {
        case cyxwiz::protocol::SPLIT_TRAIN:
            return cyxwiz::DatasetSplit::Train;
        case cyxwiz::protocol::SPLIT_VALIDATION:
            return cyxwiz::DatasetSplit::Validation;
        case cyxwiz::protocol::SPLIT_TEST:
            return cyxwiz::DatasetSplit::Test;
        default:
            return cyxwiz::DatasetSplit::Train;
    }
}

std::string DatasetProvider::SerializeImages(const std::vector<std::vector<float>>& images) {
    if (images.empty()) {
        return {};
    }

    // Calculate total size
    size_t total_floats = 0;
    for (const auto& img : images) {
        total_floats += img.size();
    }

    // Allocate buffer
    std::string buffer;
    buffer.resize(total_floats * sizeof(float));

    // Copy all image data sequentially
    float* ptr = reinterpret_cast<float*>(buffer.data());
    for (const auto& img : images) {
        std::memcpy(ptr, img.data(), img.size() * sizeof(float));
        ptr += img.size();
    }

    return buffer;
}

std::string DatasetProvider::SerializeLabels(const std::vector<int>& labels,
                                              bool one_hot,
                                              int num_classes) {
    if (labels.empty()) {
        return {};
    }

    if (one_hot && num_classes > 0) {
        // One-hot encoding: [batch_size, num_classes] as float32
        size_t total_floats = labels.size() * num_classes;
        std::string buffer;
        buffer.resize(total_floats * sizeof(float));

        float* ptr = reinterpret_cast<float*>(buffer.data());
        for (int label : labels) {
            for (int c = 0; c < num_classes; ++c) {
                *ptr++ = (c == label) ? 1.0f : 0.0f;
            }
        }
        return buffer;
    } else {
        // Class indices as int32
        std::string buffer;
        buffer.resize(labels.size() * sizeof(int32_t));
        std::memcpy(buffer.data(), labels.data(), buffer.size());
        return buffer;
    }
}

} // namespace network
