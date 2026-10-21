#include "cyxwiz/dataloader.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace cyxwiz {

// ============================================================================
// TrainingDataLoader Implementation
// ============================================================================

TrainingDataLoader::TrainingDataLoader(std::shared_ptr<DatasetBase> dataset,
                                       size_t batch_size,
                                       bool shuffle,
                                       bool drop_last,
                                       int seed)
    : dataset_(std::move(dataset))
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , drop_last_(drop_last)
    , rng_(seed)
{
    // Initialize indices
    size_t num_samples = dataset_->Size();
    indices_.resize(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        indices_[i] = i;
    }

    if (shuffle_) {
        ShuffleIndices();
    }
}

DataBatch TrainingDataLoader::GetNextBatch() {
    DataBatch batch;

    if (IsEpochComplete()) {
        return batch;
    }

    // Determine actual batch size
    size_t remaining = indices_.size() - current_index_;
    size_t actual_batch_size = std::min(batch_size_, remaining);

    if (drop_last_ && actual_batch_size < batch_size_) {
        return batch;
    }

    // Collect samples
    std::vector<std::vector<float>> batch_data;
    std::vector<int> batch_labels;
    batch_data.reserve(actual_batch_size);
    batch_labels.reserve(actual_batch_size);

    for (size_t i = 0; i < actual_batch_size; i++) {
        size_t idx = indices_[current_index_ + i];
        auto [data, label] = dataset_->GetItem(idx);

        // Apply normalization if enabled
        if (normalize_) {
            for (float& val : data) {
                val = (val - norm_mean_) / norm_std_;
            }
        }

        batch_data.push_back(std::move(data));
        batch_labels.push_back(label);
    }

    current_index_ += actual_batch_size;
    current_batch_++;

    // Build data tensor
    std::vector<size_t> data_shape = {actual_batch_size};
    auto sample_shape = dataset_->GetShape();
    data_shape.insert(data_shape.end(), sample_shape.begin(), sample_shape.end());

    // Flatten all data into single vector
    size_t sample_size = 1;
    for (size_t dim : sample_shape) {
        sample_size *= dim;
    }

    std::vector<float> flat_data;
    flat_data.reserve(actual_batch_size * sample_size);
    for (const auto& sample : batch_data) {
        flat_data.insert(flat_data.end(), sample.begin(), sample.end());
    }

    batch.data = VectorToTensor(flat_data, data_shape);

    // Build labels tensor
    if (one_hot_) {
        batch.labels = LabelsToOneHot(batch_labels);
    } else {
        batch.labels = LabelsToTensor(batch_labels);
    }

    batch.size = actual_batch_size;
    return batch;
}

void TrainingDataLoader::Reset() {
    current_index_ = 0;
    current_batch_ = 0;

    if (shuffle_) {
        ShuffleIndices();
    }
}

bool TrainingDataLoader::IsEpochComplete() const {
    if (drop_last_) {
        return current_index_ + batch_size_ > indices_.size();
    }
    return current_index_ >= indices_.size();
}

size_t TrainingDataLoader::NumBatches() const {
    if (drop_last_) {
        return indices_.size() / batch_size_;
    }
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}

void TrainingDataLoader::SetOneHotEncoding(bool enabled, size_t num_classes) {
    one_hot_ = enabled;
    one_hot_classes_ = (num_classes > 0) ? num_classes : dataset_->NumClasses();
}

void TrainingDataLoader::SetNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void TrainingDataLoader::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

Tensor TrainingDataLoader::VectorToTensor(const std::vector<float>& data,
                                          const std::vector<size_t>& shape) {
    return Tensor(shape, data.data(), DataType::Float32);
}

Tensor TrainingDataLoader::LabelsToTensor(const std::vector<int>& labels) {
    std::vector<size_t> shape = {labels.size()};

    // Convert int to int32_t
    std::vector<int32_t> labels_i32(labels.begin(), labels.end());
    return Tensor(shape, labels_i32.data(), DataType::Int32);
}

Tensor TrainingDataLoader::LabelsToOneHot(const std::vector<int>& labels) {
    size_t num_samples = labels.size();
    std::vector<size_t> shape = {num_samples, one_hot_classes_};

    std::vector<float> one_hot(num_samples * one_hot_classes_, 0.0f);
    for (size_t i = 0; i < num_samples; i++) {
        int label = labels[i];
        if (label >= 0 && static_cast<size_t>(label) < one_hot_classes_) {
            one_hot[i * one_hot_classes_ + label] = 1.0f;
        }
    }

    return Tensor(shape, one_hot.data(), DataType::Float32);
}

} // namespace cyxwiz
