#include "dataset_batcher.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#include "../preprocessing/normalization_transform.h"
#include "../preprocessing/scaling_transform.h"
#include "../preprocessing/image_transform.h"
#include "../transforms/transform.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

DatasetBatcher::DatasetBatcher(
    DatasetHandle dataset,
    size_t batch_size,
    DatasetSplit split,
    bool shuffle,
    bool drop_last)
    : dataset_(dataset)
    , batch_size_(batch_size)
    , split_(split)
    , shuffle_(shuffle)
    , drop_last_(drop_last)
    , rng_(std::random_device{}())
{
    if (!dataset_.IsValid()) {
        spdlog::error("DatasetBatcher: Invalid dataset handle");
        return;
    }

    // Get indices for the specified split
    indices_ = dataset_.GetSplitIndices(split);

    spdlog::info("DatasetBatcher: Created for {} samples, batch_size={}, shuffle={}",
                 indices_.size(), batch_size_, shuffle_);

    // Initial shuffle if enabled
    if (shuffle_) {
        ShuffleIndices();
    }
}

DatasetBatcher::~DatasetBatcher() {
    // Clean up preprocessing resources
    ClearPreprocessing();
}

Batch DatasetBatcher::GetNextBatch() {
    Batch batch;

    if (!dataset_.IsValid() || indices_.empty()) {
        return batch;
    }

    if (IsEpochComplete()) {
        return batch;
    }

    // Calculate batch bounds
    size_t batch_start = current_index_;
    size_t batch_end = std::min(current_index_ + batch_size_, indices_.size());
    size_t actual_batch_size = batch_end - batch_start;

    // Skip last incomplete batch if drop_last is enabled
    if (drop_last_ && actual_batch_size < batch_size_) {
        return batch;
    }

    // Get dataset info for shape
    DatasetInfo info = dataset_.GetInfo();

    // Collect batch data
    std::vector<float> batch_data;
    std::vector<int> batch_labels;

    size_t sample_size = 1;
    for (size_t dim : info.shape) {
        sample_size *= dim;
    }

    batch_data.reserve(actual_batch_size * sample_size);
    batch_labels.reserve(actual_batch_size);

    // Check if augmentation should be applied
    bool should_augment = augmentation_pipeline_ != nullptr &&
                          apply_augmentation_on_train_ &&
                          split_ == DatasetSplit::Train;

    for (size_t i = batch_start; i < batch_end; ++i) {
        size_t sample_idx = indices_[i];
        auto [sample, label] = dataset_.GetSample(sample_idx);

        // Apply augmentation if enabled (BEFORE preprocessing)
        if (should_augment) {
            // Convert sample to transforms::Image
            // Assume image format: [height, width, channels] or [width, height, channels]
            size_t height = info.shape[0];
            size_t width = info.shape[1];
            size_t channels = info.shape.size() > 2 ? info.shape[2] : 1;

            transforms::Image img(sample, width, height, channels);

            // Apply augmentation pipeline
            transforms::Image augmented = augmentation_pipeline_->apply(img);

            // Use augmented data
            sample = augmented.data;
        }

        // Append sample data
        batch_data.insert(batch_data.end(), sample.begin(), sample.end());
        batch_labels.push_back(label);
    }

    // Determine data shape
    std::vector<size_t> data_shape;
    if (flatten_) {
        // Flatten to [batch_size, flat_size]
        data_shape = {actual_batch_size, sample_size};
    } else {
        // Keep original shape: [batch_size, ...sample_shape]
        data_shape = {actual_batch_size};
        data_shape.insert(data_shape.end(), info.shape.begin(), info.shape.end());
    }

    // Convert to tensors
    batch.data = VectorToTensor(batch_data, data_shape);

    // Apply preprocessing pipeline (BEFORE old normalization)
    if (preprocessing_enabled_) {
        ApplyPreprocessing(batch.data);
    }

    // Apply old normalization if enabled (DEPRECATED - use preprocessing instead)
    if (normalize_ && !preprocessing_enabled_) {
        size_t num_elements = batch.data.NumElements();
        float* data_ptr = batch.data.Data<float>();
        std::vector<float> data_vec(data_ptr, data_ptr + num_elements);
        NormalizeData(data_vec);
        batch.data = VectorToTensor(data_vec, data_shape);
        spdlog::warn("DatasetBatcher: Using deprecated SetNormalization(). Consider using preprocessing pipeline instead.");
    }

    if (one_hot_) {
        batch.labels = LabelsToOneHot(batch_labels);
    } else {
        batch.labels = LabelsToTensor(batch_labels);
    }

    batch.size = actual_batch_size;

    // Advance position
    current_index_ = batch_end;
    current_batch_++;

    return batch;
}

void DatasetBatcher::Reset() {
    current_index_ = 0;
    current_batch_ = 0;

    if (shuffle_) {
        ShuffleIndices();
    }
}

bool DatasetBatcher::IsEpochComplete() const {
    if (drop_last_) {
        return current_index_ + batch_size_ > indices_.size();
    }
    return current_index_ >= indices_.size();
}

size_t DatasetBatcher::GetNumBatches() const {
    if (indices_.empty()) return 0;

    if (drop_last_) {
        return indices_.size() / batch_size_;
    }
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}

void DatasetBatcher::SetNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void DatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    one_hot_ = true;
    num_classes_ = num_classes;
}

Tensor DatasetBatcher::VectorToTensor(const std::vector<float>& data, const std::vector<size_t>& shape) {
    Tensor tensor(shape, DataType::Float32);

    // Copy data to tensor
    float* tensor_data = tensor.Data<float>();
    std::memcpy(tensor_data, data.data(), data.size() * sizeof(float));

    return tensor;
}

Tensor DatasetBatcher::LabelsToOneHot(const std::vector<int>& labels) {
    std::vector<size_t> shape = {labels.size(), num_classes_};
    Tensor tensor(shape, DataType::Float32);

    float* data = tensor.Data<float>();
    std::memset(data, 0, labels.size() * num_classes_ * sizeof(float));

    for (size_t i = 0; i < labels.size(); ++i) {
        int label = labels[i];
        if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
            data[i * num_classes_ + label] = 1.0f;
        }
    }

    return tensor;
}

Tensor DatasetBatcher::LabelsToTensor(const std::vector<int>& labels) {
    std::vector<size_t> shape = {labels.size()};
    Tensor tensor(shape, DataType::Int32);

    int* data = tensor.Data<int>();
    std::memcpy(data, labels.data(), labels.size() * sizeof(int));

    return tensor;
}

void DatasetBatcher::NormalizeData(std::vector<float>& data) {
    if (norm_std_ == 0.0f) {
        norm_std_ = 1.0f; // Avoid division by zero
    }

    for (float& val : data) {
        val = (val - norm_mean_) / norm_std_;
    }
}

void DatasetBatcher::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

// Preprocessing pipeline methods

void DatasetBatcher::SetPreprocessingConfig(const PreprocessingConfig& config) {
    // Clean up old pipeline
    ClearPreprocessing();

    // Create new config
    preprocessing_config_ = new PreprocessingConfig(config);
    spdlog::info("DatasetBatcher: Set preprocessing config (enabled={})", config.enabled);
}

const PreprocessingConfig& DatasetBatcher::GetPreprocessingConfig() const {
    static PreprocessingConfig empty_config;
    if (!preprocessing_config_) {
        return empty_config;
    }
    return *preprocessing_config_;
}

void DatasetBatcher::InitializePreprocessing(const DatasetStatistics& stats) {
    if (!preprocessing_config_ || !preprocessing_config_->enabled) {
        spdlog::warn("DatasetBatcher: Cannot initialize preprocessing - config not set or disabled");
        return;
    }

    // Clear existing transforms
    ClearPreprocessing();

    const auto& config = *preprocessing_config_;

    // Build preprocessing pipeline in order:
    // 1. Image preprocessing (resize, format conversion)
    // 2. Normalization (MNIST/CIFAR/ImageNet/Custom)
    // 3. Scaling (MinMax/Standard/Robust/PCA)

    // 1. Image preprocessing
    if (config.image_config.resize_mode != ResizeMode::None ||
        config.image_config.convert_to_grayscale ||
        config.image_config.convert_to_rgb) {

        auto* transform = new ImageTransform(config.image_config);
        image_transforms_.push_back(transform);
        spdlog::info("DatasetBatcher: Added ImageTransform to pipeline");
    }

    // 2. Normalization
    if (config.normalization_config.strategy != NormalizationStrategy::None) {
        auto* transform = new NormalizationTransform(config.normalization_config);
        transform->Initialize(stats);
        normalization_transforms_.push_back(transform);
        spdlog::info("DatasetBatcher: Added NormalizationTransform to pipeline (strategy={})",
                     static_cast<int>(config.normalization_config.strategy));
    }

    // 3. Scaling
    if (config.scaling_config.strategy != ScalingStrategy::None) {
        auto* transform = new ScalingTransform(config.scaling_config);
        transform->Initialize(stats);
        scaling_transforms_.push_back(transform);
        spdlog::info("DatasetBatcher: Added ScalingTransform to pipeline (strategy={})",
                     static_cast<int>(config.scaling_config.strategy));
    }

    preprocessing_enabled_ = !image_transforms_.empty() ||
                             !normalization_transforms_.empty() ||
                             !scaling_transforms_.empty();

    spdlog::info("DatasetBatcher: Preprocessing pipeline initialized with {} transforms",
                 image_transforms_.size() + normalization_transforms_.size() + scaling_transforms_.size());
}

void DatasetBatcher::ClearPreprocessing() {
    // Clean up image transforms
    for (auto* transform : image_transforms_) {
        delete transform;
    }
    image_transforms_.clear();

    // Clean up normalization transforms
    for (auto* transform : normalization_transforms_) {
        delete transform;
    }
    normalization_transforms_.clear();

    // Clean up scaling transforms
    for (auto* transform : scaling_transforms_) {
        delete transform;
    }
    scaling_transforms_.clear();

    // Clean up config
    if (preprocessing_config_) {
        delete preprocessing_config_;
        preprocessing_config_ = nullptr;
    }

    preprocessing_enabled_ = false;
}

void DatasetBatcher::ApplyPreprocessing(Tensor& batch) {
    if (!preprocessing_enabled_) {
        return;
    }

    // Apply transforms in order:
    // 1. Image preprocessing
    for (auto* transform : image_transforms_) {
        batch = transform->Apply(batch);
    }

    // 2. Normalization
    for (auto* transform : normalization_transforms_) {
        batch = transform->Apply(batch);
    }

    // 3. Scaling
    for (auto* transform : scaling_transforms_) {
        batch = transform->Apply(batch);
    }
}

// =============================================================================
// Augmentation Pipeline Management
// =============================================================================

void DatasetBatcher::SetAugmentationPipeline(std::shared_ptr<transforms::Compose> pipeline) {
    augmentation_pipeline_ = pipeline;
    if (pipeline) {
        spdlog::info("DatasetBatcher: Augmentation pipeline enabled for split={}",
                     static_cast<int>(split_));
    } else {
        spdlog::info("DatasetBatcher: Augmentation pipeline disabled");
    }
}

// =============================================================================
// DatasetIterator implementation
// =============================================================================

DatasetIterator::DatasetIterator(
    DatasetHandle dataset,
    size_t batch_size,
    bool shuffle)
{
    train_batcher_ = std::make_unique<DatasetBatcher>(
        dataset, batch_size, DatasetSplit::Train, shuffle, false);

    val_batcher_ = std::make_unique<DatasetBatcher>(
        dataset, batch_size, DatasetSplit::Validation, false, false);

    test_batcher_ = std::make_unique<DatasetBatcher>(
        dataset, batch_size, DatasetSplit::Test, false, false);
}

void DatasetIterator::ResetAll() {
    train_batcher_->Reset();
    val_batcher_->Reset();
    test_batcher_->Reset();
}

void DatasetIterator::SetNormalization(float mean, float std) {
    train_batcher_->SetNormalization(mean, std);
    val_batcher_->SetNormalization(mean, std);
    test_batcher_->SetNormalization(mean, std);
}

void DatasetIterator::SetOneHotEncoding(size_t num_classes) {
    train_batcher_->SetOneHotEncoding(num_classes);
    val_batcher_->SetOneHotEncoding(num_classes);
    test_batcher_->SetOneHotEncoding(num_classes);
}

void DatasetIterator::SetFlatten(bool flatten) {
    train_batcher_->SetFlatten(flatten);
    val_batcher_->SetFlatten(flatten);
    test_batcher_->SetFlatten(flatten);
}

} // namespace cyxwiz
