#include "dataset_batcher.h"
#include "annotation_manager.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#include "../preprocessing/normalization_transform.h"
#include "../preprocessing/scaling_transform.h"
#include "../preprocessing/image_transform.h"
#include "../transforms/transform.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

namespace cyxwiz {

DatasetBatcher::DatasetBatcher(
    DatasetHandle dataset,
    size_t batch_size,
    DatasetSplit split,
    bool shuffle,
    bool drop_last,
    int num_workers,
    uint32_t seed)
    : dataset_(dataset)
    , batch_size_(batch_size)
    , split_(split)
    , shuffle_(shuffle)
    , drop_last_(drop_last)
    , num_workers_(std::max(0, num_workers))
    , rng_(seed)
{
    if (!dataset_.IsValid()) {
        spdlog::error("DatasetBatcher: Invalid dataset handle");
        return;
    }

    // Get indices for the specified split
    indices_ = dataset_.GetSplitIndices(split);

    spdlog::info("DatasetBatcher: Created for {} samples, batch_size={}, shuffle={}, num_workers={}",
                 indices_.size(), batch_size_, shuffle_, num_workers_);

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

    std::vector<std::pair<std::vector<float>, int>> samples(actual_batch_size);

    auto load_range = [&](size_t begin, size_t end) {
        for (size_t offset = begin; offset < end; ++offset) {
            size_t sample_idx = indices_[batch_start + offset];
            samples[offset] = dataset_.GetSample(sample_idx);
        }
    };

    if (num_workers_ > 1 && actual_batch_size > 1) {
        size_t worker_count = std::min(static_cast<size_t>(num_workers_), actual_batch_size);
        size_t chunk_size = (actual_batch_size + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        for (size_t worker = 0; worker < worker_count; ++worker) {
            size_t begin = worker * chunk_size;
            size_t end = std::min(actual_batch_size, begin + chunk_size);
            if (begin >= end) break;
            workers.emplace_back(load_range, begin, end);
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        load_range(0, actual_batch_size);
    }

    for (size_t offset = 0; offset < actual_batch_size; ++offset) {
        auto [sample, label] = std::move(samples[offset]);

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
        static bool warned_once = false; if (!warned_once) { spdlog::warn("DatasetBatcher: Using deprecated SetNormalization(). Consider using preprocessing pipeline instead."); warned_once = true; }
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
    SetLegacyNormalization(mean, std);
}

void DatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    SetLegacyOneHotEncoding(num_classes);
}

void DatasetBatcher::SetLegacyNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void DatasetBatcher::SetLegacyOneHotEncoding(size_t num_classes) {
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
    preprocessing_config_ = std::make_unique<PreprocessingConfig>(config);
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

        image_transforms_.push_back(std::make_unique<ImageTransform>(config.image_config));
        spdlog::info("DatasetBatcher: Added ImageTransform to pipeline");
    }

    // 2. Normalization
    if (config.normalization_config.strategy != NormalizationStrategy::None) {
        auto transform = std::make_unique<NormalizationTransform>(config.normalization_config);
        transform->Initialize(stats);
        normalization_transforms_.push_back(std::move(transform));
        spdlog::info("DatasetBatcher: Added NormalizationTransform to pipeline (strategy={})",
                     static_cast<int>(config.normalization_config.strategy));
    }

    // 3. Scaling
    if (config.scaling_config.strategy != ScalingStrategy::None) {
        auto transform = std::make_unique<ScalingTransform>(config.scaling_config);
        transform->Initialize(stats);
        scaling_transforms_.push_back(std::move(transform));
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
    image_transforms_.clear();

    // Clean up normalization transforms
    normalization_transforms_.clear();

    // Clean up scaling transforms
    scaling_transforms_.clear();

    // Clean up config
    preprocessing_config_.reset();

    preprocessing_enabled_ = false;
}

void DatasetBatcher::ApplyPreprocessing(Tensor& batch) {
    if (!preprocessing_enabled_) {
        return;
    }

    // Apply transforms in order:
    // 1. Image preprocessing
    for (const auto& transform : image_transforms_) {
        batch = transform->Apply(batch);
    }

    // 2. Normalization
    for (const auto& transform : normalization_transforms_) {
        batch = transform->Apply(batch);
    }

    // 3. Scaling
    for (const auto& transform : scaling_transforms_) {
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
    train_batcher_->SetLegacyNormalization(mean, std);
    val_batcher_->SetLegacyNormalization(mean, std);
    test_batcher_->SetLegacyNormalization(mean, std);
}

void DatasetIterator::SetOneHotEncoding(size_t num_classes) {
    train_batcher_->SetLegacyOneHotEncoding(num_classes);
    val_batcher_->SetLegacyOneHotEncoding(num_classes);
    test_batcher_->SetLegacyOneHotEncoding(num_classes);
}

void DatasetIterator::SetFlatten(bool flatten) {
    train_batcher_->SetFlatten(flatten);
    val_batcher_->SetFlatten(flatten);
    test_batcher_->SetFlatten(flatten);
}


// =============================================================================
// Annotation-aware batch access (for segmentation training)
// =============================================================================

bool DatasetBatcher::HasAnnotations(const std::string& dataset_id) const {
    const auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    return ann_mgr.HasAnnotationSet(dataset_id);
}

AnnotatedBatch DatasetBatcher::GetNextAnnotatedBatch(const std::string& dataset_id) {
    AnnotatedBatch batch;

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

    // Collect indices for this batch
    std::vector<size_t> sample_indices(actual_batch_size);
    for (size_t i = 0; i < actual_batch_size; ++i) {
        sample_indices[i] = indices_[batch_start + i];
    }

    // Advance position
    current_index_ = batch_end;
    current_batch_++;

    return GetAnnotatedBatch(dataset_id, sample_indices);
}

AnnotatedBatch DatasetBatcher::GetAnnotatedBatch(const std::string& dataset_id,
                                                  const std::vector<size_t>& sample_indices) {
    AnnotatedBatch batch;

    if (!dataset_.IsValid() || sample_indices.empty()) {
        return batch;
    }

    // Get dataset info for shape
    DatasetInfo info = dataset_.GetInfo();

    // Determine dimensions from sample shape
    if (info.shape.size() >= 2) {
        batch.height = info.shape[0];
        batch.width = info.shape[1];
        batch.channels = info.shape.size() > 2 ? info.shape[2] : 1;
    } else {
        spdlog::warn("DatasetBatcher: Cannot determine image dimensions from shape");
        return batch;
    }

    batch.size = sample_indices.size();
    batch.indices = sample_indices;

    // Collect batch data
    size_t sample_size = batch.height * batch.width * batch.channels;
    std::vector<float> batch_data;
    std::vector<int> batch_labels;
    batch_data.reserve(batch.size * sample_size);
    batch_labels.reserve(batch.size);

    // Check if augmentation should be applied
    bool should_augment = augmentation_pipeline_ != nullptr &&
                          apply_augmentation_on_train_ &&
                          split_ == DatasetSplit::Train;

    std::vector<std::pair<std::vector<float>, int>> samples(batch.size);

    auto load_range = [&](size_t begin, size_t end) {
        for (size_t offset = begin; offset < end; ++offset) {
            samples[offset] = dataset_.GetSample(sample_indices[offset]);
        }
    };

    if (num_workers_ > 1 && batch.size > 1) {
        size_t worker_count = std::min(static_cast<size_t>(num_workers_), batch.size);
        size_t chunk_size = (batch.size + worker_count - 1) / worker_count;
        std::vector<std::thread> workers;
        workers.reserve(worker_count);

        for (size_t worker = 0; worker < worker_count; ++worker) {
            size_t begin = worker * chunk_size;
            size_t end = std::min(batch.size, begin + chunk_size);
            if (begin >= end) break;
            workers.emplace_back(load_range, begin, end);
        }

        for (auto& worker : workers) {
            worker.join();
        }
    } else {
        load_range(0, batch.size);
    }

    for (size_t offset = 0; offset < batch.size; ++offset) {
        auto [sample, label] = std::move(samples[offset]);

        // Apply augmentation if enabled
        if (should_augment && !sample.empty()) {
            transforms::Image img(sample, batch.width, batch.height, batch.channels);
            transforms::Image augmented = augmentation_pipeline_->apply(img);
            sample = augmented.data;
        }

        batch_data.insert(batch_data.end(), sample.begin(), sample.end());
        batch_labels.push_back(label);
    }

    // Convert to tensors
    std::vector<size_t> data_shape = {batch.size, batch.height, batch.width, batch.channels};
    batch.images = VectorToTensor(batch_data, data_shape);

    // Apply preprocessing if enabled
    if (preprocessing_enabled_) {
        ApplyPreprocessing(batch.images);
    }

    // Convert labels
    if (one_hot_) {
        batch.labels = LabelsToOneHot(batch_labels);
    } else {
        batch.labels = LabelsToTensor(batch_labels);
    }

    // Generate segmentation masks if annotations exist
    const auto& ann_mgr = DataRegistry::Instance().GetAnnotationManager();
    if (ann_mgr.HasAnnotationSet(dataset_id)) {
        int out_width = mask_width_ > 0 ? mask_width_ : static_cast<int>(batch.width);
        int out_height = mask_height_ > 0 ? mask_height_ : static_cast<int>(batch.height);
        size_t mask_size = static_cast<size_t>(out_width) * static_cast<size_t>(out_height);

        std::vector<float> masks_data;
        masks_data.reserve(batch.size * mask_size);

        for (size_t idx : sample_indices) {
            auto mask = ann_mgr.GetSegmentationMask(dataset_id, idx, out_width, out_height);
            if (mask.size() == mask_size) {
                masks_data.insert(masks_data.end(), mask.begin(), mask.end());
            } else {
                // No annotations for this image, fill with zeros (background)
                masks_data.resize(masks_data.size() + mask_size, 0.0f);
            }
        }

        std::vector<size_t> mask_shape = {batch.size, static_cast<size_t>(out_height),
                                           static_cast<size_t>(out_width)};
        batch.masks = VectorToTensor(masks_data, mask_shape);

        spdlog::debug("DatasetBatcher: Generated {} segmentation masks", batch.size);
    }

    return batch;
}


} // namespace cyxwiz
