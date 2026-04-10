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

    for (size_t idx : sample_indices) {
        auto [sample, label] = dataset_.GetSample(idx);

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

// =============================================================================
// ArrowDatasetBatcher Implementation
// =============================================================================

ArrowDatasetBatcher::ArrowDatasetBatcher(
    std::shared_ptr<ArrowDataset> dataset,
    const std::string& label_column,
    size_t batch_size,
    bool shuffle,
    float train_split,
    bool is_training)
    : dataset_(dataset)
    , label_column_(label_column)
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , is_training_(is_training)
    , rng_(std::random_device{}())
{
    if (!dataset_) {
        spdlog::error("ArrowDatasetBatcher: Invalid dataset");
        return;
    }

    int64_t num_rows = dataset_->GetNumRows();

    // Create train/val split
    int64_t train_count = static_cast<int64_t>(num_rows * train_split);

    if (is_training_) {
        // Training: first train_count rows
        indices_.reserve(train_count);
        for (int64_t i = 0; i < train_count; ++i) {
            indices_.push_back(i);
        }
    } else {
        // Validation: remaining rows
        indices_.reserve(num_rows - train_count);
        for (int64_t i = train_count; i < num_rows; ++i) {
            indices_.push_back(i);
        }
    }

    // Initialize feature/label column indices
    InitializeColumns();

    spdlog::info("ArrowDatasetBatcher: {} samples, {} features, batch_size={}, shuffle={}",
                 indices_.size(), num_features_, batch_size_, shuffle_);

    if (shuffle_) {
        ShuffleIndices();
    }
}

void ArrowDatasetBatcher::InitializeColumns() {
    if (!dataset_) return;

    auto schema = dataset_->GetSchema();
    if (!schema) return;

    feature_cols_.clear();
    label_col_idx_ = -1;

    // Common label column names for auto-detection
    static const std::vector<std::string> common_label_names = {
        "label", "Label", "LABEL", "class", "Class", "CLASS",
        "target", "Target", "TARGET", "y", "Y", "digit", "category"
    };

    // First pass: find explicit label column or auto-detect
    for (int i = 0; i < schema->num_fields(); ++i) {
        auto field = schema->field(i);
        std::string name = field->name();

        // Check for explicit match
        if (!label_column_.empty() && name == label_column_) {
            label_col_idx_ = i;
            spdlog::info("ArrowDatasetBatcher: Found explicit label column '{}' at index {}", name, i);
        }
        // Auto-detect common label names if no explicit label given
        else if (label_column_.empty() && label_col_idx_ < 0) {
            for (const auto& common : common_label_names) {
                if (name == common) {
                    label_col_idx_ = i;
                    spdlog::info("ArrowDatasetBatcher: Auto-detected label column '{}' at index {}", name, i);
                    break;
                }
            }
        }
    }

    // Second pass: collect feature columns (all numeric except label)
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (i == label_col_idx_) continue;  // Skip label column

        auto field = schema->field(i);
        auto type = field->type();
        if (type->id() == arrow::Type::DOUBLE ||
            type->id() == arrow::Type::FLOAT ||
            type->id() == arrow::Type::INT64 ||
            type->id() == arrow::Type::INT32 ||
            type->id() == arrow::Type::INT16 ||
            type->id() == arrow::Type::INT8 ||
            type->id() == arrow::Type::UINT64 ||
            type->id() == arrow::Type::UINT32 ||
            type->id() == arrow::Type::UINT16 ||
            type->id() == arrow::Type::UINT8) {
            feature_cols_.push_back(i);
        }
    }

    num_features_ = feature_cols_.size();
    spdlog::info("ArrowDatasetBatcher: {} feature columns, label_col={}",
                 num_features_, label_col_idx_);
}

Batch ArrowDatasetBatcher::GetNextBatch() {
    Batch batch;

    try {
        spdlog::debug("ArrowDatasetBatcher::GetNextBatch called, current_index_={}, indices_.size()={}",
                      current_index_, indices_.size());

        if (!dataset_ || indices_.empty()) {
            spdlog::warn("ArrowDatasetBatcher::GetNextBatch: No dataset or empty indices");
            return batch;
        }

        // Validate indices to catch potential memory corruption
        int64_t num_total_rows = dataset_->GetNumRows();
        for (size_t i = 0; i < std::min(indices_.size(), size_t(10)); ++i) {
            if (indices_[i] < 0 || indices_[i] >= num_total_rows) {
                spdlog::error("ArrowDatasetBatcher: Invalid index[{}]={} (max={})",
                              i, indices_[i], num_total_rows);
            }
        }

        if (IsEpochComplete()) {
            return batch;
        }

        // Validate we have features
        if (num_features_ == 0) {
            spdlog::error("ArrowDatasetBatcher::GetNextBatch: No feature columns found");
            return batch;
        }

        // Calculate batch bounds
        size_t batch_start = current_index_;
        size_t batch_end = std::min(current_index_ + batch_size_, indices_.size());
        size_t actual_batch_size = batch_end - batch_start;
        batch.size = actual_batch_size;

        // Get Arrow table
        auto table = dataset_->GetArrowTable();
        if (!table) {
            spdlog::error("ArrowDatasetBatcher: No Arrow table");
            return batch;
        }

        int64_t num_rows = table->num_rows();
        int num_cols = table->num_columns();

        // OPTIMIZED: Pre-allocate batch data as [batch_size, num_features] in row-major order
        std::vector<float> batch_data(actual_batch_size * num_features_, 0.0f);
        std::vector<int> batch_labels(actual_batch_size, 0);

        spdlog::debug("ArrowDatasetBatcher: Processing {} feature columns, batch_data.size()={}",
                      feature_cols_.size(), batch_data.size());
        spdlog::default_logger()->flush();  // Force flush to ensure logs are written before crash

        // OPTIMIZED: Process column by column (Arrow is columnar, this is much faster)
        for (size_t feat_idx = 0; feat_idx < feature_cols_.size(); ++feat_idx) {
            int col_idx = feature_cols_[feat_idx];

            // Log first column for debugging
            if (feat_idx == 0) {
                spdlog::debug("ArrowDatasetBatcher: First column idx={}, type={}",
                              col_idx, (col_idx < num_cols && table->column(col_idx)) ?
                              table->column(col_idx)->type()->ToString() : "invalid");
                spdlog::default_logger()->flush();
            }

            // Log progress every 100 columns — flushed so crash bisection survives
            if (feat_idx > 0 && feat_idx % 100 == 0) {
                spdlog::debug("ArrowDatasetBatcher: Processing column {}/{}", feat_idx, feature_cols_.size());
                spdlog::default_logger()->flush();
            }

            if (col_idx < 0 || col_idx >= num_cols) {
                spdlog::warn("ArrowDatasetBatcher: Invalid col_idx={} (num_cols={})", col_idx, num_cols);
                continue;
            }

            auto column = table->column(col_idx);
            if (!column || column->num_chunks() == 0) {
                spdlog::warn("ArrowDatasetBatcher: Null or empty column at idx={}", col_idx);
                continue;
            }

            // For single-chunk columns (most common case), use direct pointer access
            if (column->num_chunks() == 1) {
                auto chunk = column->chunk(0);

                // Get raw data pointer based on type
                switch (chunk->type_id()) {
                    case arrow::Type::DOUBLE: {
                        auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        const double* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::FLOAT: {
                        auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                        const float* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = data[row_idx];
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT64: {
                        auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        const int64_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT32: {
                        auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                        const int32_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT16: {
                        auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                        const int16_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT8: {
                        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                        const int8_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::UINT8: {
                        auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                        const uint8_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    default:
                        // Fallback to Value() method for other types
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < chunk->length()) {
                                // Use generic scalar access
                                auto result = chunk->GetScalar(row_idx);
                                if (result.ok()) {
                                    auto scalar = result.ValueOrDie();
                                    if (scalar->is_valid) {
                                        // Try to cast to double
                                        auto cast_result = scalar->CastTo(arrow::float64());
                                        if (cast_result.ok()) {
                                            auto dbl = std::static_pointer_cast<arrow::DoubleScalar>(cast_result.ValueOrDie());
                                            batch_data[b * num_features_ + feat_idx] = static_cast<float>(dbl->value);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                }
            } else {
                // Multi-chunk fallback (slower, but handles chunked data)
                for (size_t b = 0; b < actual_batch_size; ++b) {
                    int64_t row_idx = indices_[batch_start + b];
                    if (row_idx < 0 || row_idx >= num_rows) continue;

                    // Find the right chunk
                    int64_t local_idx = row_idx;
                    std::shared_ptr<arrow::Array> chunk = nullptr;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto ch = column->chunk(c);
                        if (local_idx < ch->length()) {
                            chunk = ch;
                            break;
                        }
                        local_idx -= ch->length();
                    }

                    if (chunk) {
                        float value = 0.0f;
                        switch (chunk->type_id()) {
                            case arrow::Type::DOUBLE:
                                value = static_cast<float>(std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::FLOAT:
                                value = std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(local_idx);
                                break;
                            case arrow::Type::INT64:
                                value = static_cast<float>(std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::INT32:
                                value = static_cast<float>(std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::UINT8:
                                value = static_cast<float>(std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(local_idx));
                                break;
                            default:
                                break;
                        }
                        batch_data[b * num_features_ + feat_idx] = value;
                    }
                }
            }
        }

        // CHECKPOINT: feature column loop completed successfully
        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT A - feature loop complete, all {} cols processed",
                      feature_cols_.size());
        spdlog::default_logger()->flush();

        // Extract labels - handles both single-chunk (fast path) and multi-chunk
        // (correct path) columns. The previous implementation always read chunk(0)
        // and crashed whenever Arrow's CSV reader split the table into multiple
        // chunks, which it does for any file >~1 MB block size.
        if (label_col_idx_ >= 0 && label_col_idx_ < num_cols) {
            auto column = table->column(label_col_idx_);
            if (column && column->num_chunks() > 0) {
                spdlog::debug("ArrowDatasetBatcher: label column has {} chunks, "
                              "first chunk length={}, table num_rows={}",
                              column->num_chunks(),
                              column->chunk(0) ? column->chunk(0)->length() : -1,
                              num_rows);
                spdlog::default_logger()->flush();

                // Generic per-row read that handles any number of chunks. Slower
                // than a type-specific raw_values() loop on a single chunk, but
                // correct regardless of chunking. Labels are small (1 element per
                // row), so per-batch cost here is trivial.
                auto read_label = [&](int64_t global_row_idx) -> int {
                    // Find which chunk holds this row
                    int64_t offset = 0;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto chk = column->chunk(c);
                        int64_t chk_len = chk->length();
                        if (global_row_idx < offset + chk_len) {
                            int64_t local_idx = global_row_idx - offset;
                            if (chk->IsNull(local_idx)) return 0;
                            switch (chk->type_id()) {
                                case arrow::Type::INT64:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::Int64Array>(chk)->Value(local_idx));
                                case arrow::Type::INT32:
                                    return std::static_pointer_cast<arrow::Int32Array>(chk)->Value(local_idx);
                                case arrow::Type::INT16:
                                    return std::static_pointer_cast<arrow::Int16Array>(chk)->Value(local_idx);
                                case arrow::Type::INT8:
                                    return std::static_pointer_cast<arrow::Int8Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT8:
                                    return std::static_pointer_cast<arrow::UInt8Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT16:
                                    return std::static_pointer_cast<arrow::UInt16Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT32:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::UInt32Array>(chk)->Value(local_idx));
                                case arrow::Type::DOUBLE:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::DoubleArray>(chk)->Value(local_idx));
                                case arrow::Type::FLOAT:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::FloatArray>(chk)->Value(local_idx));
                                default:
                                    return 0;
                            }
                        }
                        offset += chk_len;
                    }
                    return 0;  // row not found across any chunk
                };

                // Warn once per batcher if the label type isn't something we recognise
                if (column->num_chunks() > 0) {
                    auto first_type = column->chunk(0)->type_id();
                    if (first_type != arrow::Type::INT64 &&
                        first_type != arrow::Type::INT32 &&
                        first_type != arrow::Type::INT16 &&
                        first_type != arrow::Type::INT8 &&
                        first_type != arrow::Type::UINT8 &&
                        first_type != arrow::Type::UINT16 &&
                        first_type != arrow::Type::UINT32 &&
                        first_type != arrow::Type::DOUBLE &&
                        first_type != arrow::Type::FLOAT) {
                        spdlog::warn("ArrowDatasetBatcher: Unsupported label type: {}",
                                     column->chunk(0)->type()->ToString());
                    }
                }

                for (size_t b = 0; b < actual_batch_size; ++b) {
                    int64_t row_idx = indices_[batch_start + b];
                    if (row_idx >= 0 && row_idx < num_rows) {
                        batch_labels[b] = read_label(row_idx);
                    }
                }
            }
        }

        // CHECKPOINT: label extraction complete
        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT B - label extraction complete, "
                      "{} features, {} labels",
                      batch_data.size(), batch_labels.size());
        spdlog::default_logger()->flush();

        // Apply normalization if enabled
        if (normalize_ && norm_std_ != 0.0f) {
            for (float& val : batch_data) {
                val = (val - norm_mean_) / norm_std_;
            }
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT C - normalization applied "
                          "(mean={}, std={})", norm_mean_, norm_std_);
            spdlog::default_logger()->flush();
        }

        // Validate data before creating tensor
        if (batch_data.empty() || actual_batch_size == 0 || num_features_ == 0) {
            spdlog::error("ArrowDatasetBatcher: Empty batch data - size={}, features={}",
                          actual_batch_size, num_features_);
            return batch;
        }

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT D - about to construct data tensor [{}, {}]",
                      actual_batch_size, num_features_);
        spdlog::default_logger()->flush();

        // Create data tensor [batch_size, num_features]
        std::vector<size_t> data_shape = {actual_batch_size, num_features_};
        batch.data = Tensor(data_shape, batch_data.data());

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT E - data tensor constructed successfully");
        spdlog::default_logger()->flush();

        // Create labels tensor
        if (one_hot_ && !batch_labels.empty()) {
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT F - creating one-hot labels [{}, {}]",
                          actual_batch_size, num_classes_);
            spdlog::default_logger()->flush();
            // One-hot encoding
            std::vector<float> onehot_data(actual_batch_size * num_classes_, 0.0f);
            for (size_t i = 0; i < batch_labels.size(); ++i) {
                int label = batch_labels[i];
                if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                    onehot_data[i * num_classes_ + label] = 1.0f;
                }
            }
            std::vector<size_t> label_shape = {actual_batch_size, num_classes_};
            batch.labels = Tensor(label_shape, onehot_data.data());
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT G - one-hot labels tensor created");
            spdlog::default_logger()->flush();
        } else if (!batch_labels.empty()) {
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT F - creating integer labels [{}]",
                          actual_batch_size);
            spdlog::default_logger()->flush();
            // Integer labels
            std::vector<float> label_data(batch_labels.begin(), batch_labels.end());
            std::vector<size_t> label_shape = {actual_batch_size};
            batch.labels = Tensor(label_shape, label_data.data());
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT G - integer labels tensor created");
            spdlog::default_logger()->flush();
        }

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT H - returning batch to caller");
        spdlog::default_logger()->flush();
        current_index_ = batch_end;
        return batch;

    } catch (const std::exception& e) {
        spdlog::error("ArrowDatasetBatcher::GetNextBatch exception: {}", e.what());
        return batch;
    } catch (...) {
        spdlog::error("ArrowDatasetBatcher::GetNextBatch unknown exception");
        return batch;
    }
}

void ArrowDatasetBatcher::Reset() {
    current_index_ = 0;
    if (shuffle_) {
        ShuffleIndices();
    }
}

bool ArrowDatasetBatcher::IsEpochComplete() const {
    return current_index_ >= indices_.size();
}

size_t ArrowDatasetBatcher::GetNumBatches() const {
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}

void ArrowDatasetBatcher::SetNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void ArrowDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    one_hot_ = true;
    num_classes_ = num_classes;
}

void ArrowDatasetBatcher::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

} // namespace cyxwiz
