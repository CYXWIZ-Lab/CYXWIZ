#pragma once

#include "data_registry.h"
#include <cyxwiz/tensor.h>
#include <vector>
#include <random>
#include <memory>

namespace cyxwiz {

// Forward declarations
class AnnotationManager;

namespace transforms {
    class Compose;
    struct Image;
}

/**
 * A batch of data ready for training
 */
struct Batch {
    Tensor data;          // [batch_size, ...input_dims] - input features
    Tensor labels;        // [batch_size] or [batch_size, num_classes] if one-hot
    size_t size = 0;      // Actual batch size (may be < requested for last batch)

    bool IsValid() const { return size > 0; }
};

/**
 * A batch of images with segmentation masks from annotations
 * Used for training semantic segmentation models
 */
struct AnnotatedBatch {
    Tensor images;        // [batch_size, height, width, channels] - input images
    Tensor labels;        // [batch_size] - classification labels
    Tensor masks;         // [batch_size, height, width] - segmentation masks (class IDs per pixel)
    size_t size = 0;      // Actual batch size

    // Batch metadata
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;

    // Which images in the dataset
    std::vector<size_t> indices;

    bool IsValid() const { return size > 0; }
    bool HasMasks() const { return masks.NumElements() > 0; }
};

/**
 * DatasetBatcher - Provides batched iteration over a dataset
 *
 * Fetches samples from DataRegistry and converts them to Tensors
 * for training. Supports shuffling, one-hot encoding, and normalization.
 */
class DatasetBatcher {
public:
    /**
     * Create a batcher for the given dataset
     * @param dataset Handle to dataset from DataRegistry
     * @param batch_size Number of samples per batch
     * @param split Which split to iterate (Train, Validation, Test)
     * @param shuffle Whether to shuffle samples each epoch
     * @param drop_last Drop last batch if smaller than batch_size
     */
    DatasetBatcher(
        DatasetHandle dataset,
        size_t batch_size,
        DatasetSplit split = DatasetSplit::Train,
        bool shuffle = true,
        bool drop_last = false
    );

    /**
     * Destructor - cleans up preprocessing resources
     */
    ~DatasetBatcher();

    /**
     * Get the next batch
     * @return Batch with data and labels tensors
     */
    Batch GetNextBatch();

    /**
     * Reset to beginning of epoch (re-shuffles if shuffle=true)
     */
    void Reset();

    /**
     * Check if current epoch is complete
     */
    bool IsEpochComplete() const;

    /**
     * Get total number of batches per epoch
     */
    size_t GetNumBatches() const;

    /**
     * Get current batch index (0-based)
     */
    size_t GetCurrentBatchIndex() const { return current_batch_; }

    /**
     * Get total number of samples in this split
     */
    size_t GetNumSamples() const { return indices_.size(); }

    // Preprocessing options (DEPRECATED - use SetPreprocessingConfig instead)
    [[deprecated("Use SetPreprocessingConfig() with the new preprocessing pipeline instead")]]
    void SetNormalization(float mean, float std);

    [[deprecated("Use SetPreprocessingConfig() with the new preprocessing pipeline instead")]]
    void SetOneHotEncoding(size_t num_classes);

    void SetFlatten(bool flatten) { flatten_ = flatten; }

    // New preprocessing pipeline
    void SetPreprocessingConfig(const struct PreprocessingConfig& config);
    const struct PreprocessingConfig& GetPreprocessingConfig() const;
    void InitializePreprocessing(const struct DatasetStatistics& stats);
    void ClearPreprocessing();
    bool HasPreprocessing() const { return preprocessing_enabled_; }

    // Augmentation pipeline (applied before preprocessing during training only)
    void SetAugmentationPipeline(std::shared_ptr<transforms::Compose> pipeline);
    void SetApplyAugmentationOnTrain(bool enable) { apply_augmentation_on_train_ = enable; }
    bool HasAugmentation() const { return augmentation_pipeline_ != nullptr; }

    // =========================================================================
    // Annotation-aware batch access (for segmentation training)
    // =========================================================================

    /**
     * Get annotated batch for specific image indices
     * Includes segmentation masks from AnnotationManager
     * @param dataset_id Dataset identifier for annotations
     * @param sample_indices Image indices to include
     * @return AnnotatedBatch with images and segmentation masks
     */
    AnnotatedBatch GetAnnotatedBatch(const std::string& dataset_id,
                                      const std::vector<size_t>& sample_indices);

    /**
     * Get next annotated batch (like GetNextBatch but with masks)
     * @param dataset_id Dataset identifier for annotations
     * @return AnnotatedBatch with images and segmentation masks
     */
    AnnotatedBatch GetNextAnnotatedBatch(const std::string& dataset_id);

    /**
     * Check if dataset has annotations
     * @param dataset_id Dataset identifier
     * @return true if annotations exist
     */
    bool HasAnnotations(const std::string& dataset_id) const;

    /**
     * Set mask output dimensions (if different from image size)
     * @param width Mask width (0 = same as image)
     * @param height Mask height (0 = same as image)
     */
    void SetMaskSize(int width, int height) {
        mask_width_ = width;
        mask_height_ = height;
    }

private:
    DatasetHandle dataset_;
    size_t batch_size_;
    DatasetSplit split_;
    bool shuffle_;
    bool drop_last_;

    std::vector<size_t> indices_;     // Sample indices for current split
    size_t current_index_ = 0;        // Current position in indices_
    size_t current_batch_ = 0;        // Current batch number

    std::mt19937 rng_;

    // Preprocessing options
    bool normalize_ = false;
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;

    bool one_hot_ = false;
    size_t num_classes_ = 0;

    bool flatten_ = false;

    // Mask output size (0 = use image dimensions)
    int mask_width_ = 0;
    int mask_height_ = 0;

    // Preprocessing pipeline
    std::unique_ptr<struct PreprocessingConfig> preprocessing_config_;
    std::vector<std::unique_ptr<class NormalizationTransform>> normalization_transforms_;
    std::vector<std::unique_ptr<class ScalingTransform>> scaling_transforms_;
    std::vector<std::unique_ptr<class ImageTransform>> image_transforms_;
    bool preprocessing_enabled_ = false;

    // Augmentation pipeline (applied BEFORE preprocessing, only on training split)
    std::shared_ptr<transforms::Compose> augmentation_pipeline_;
    bool apply_augmentation_on_train_ = true;  // Only apply to training split

    // Convert float vector to Tensor
    Tensor VectorToTensor(const std::vector<float>& data, const std::vector<size_t>& shape);

    // Convert labels to one-hot encoded Tensor
    Tensor LabelsToOneHot(const std::vector<int>& labels);

    // Convert labels to integer Tensor
    Tensor LabelsToTensor(const std::vector<int>& labels);

    // Apply normalization to data
    void NormalizeData(std::vector<float>& data);

    // Apply preprocessing pipeline
    void ApplyPreprocessing(Tensor& batch);

    // Shuffle indices
    void ShuffleIndices();
};

/**
 * Helper class for iterating multiple batchers (train/val/test)
 */
class DatasetIterator {
public:
    DatasetIterator(
        DatasetHandle dataset,
        size_t batch_size,
        bool shuffle = true
    );

    DatasetBatcher& GetTrainBatcher() { return *train_batcher_; }
    DatasetBatcher& GetValBatcher() { return *val_batcher_; }
    DatasetBatcher& GetTestBatcher() { return *test_batcher_; }

    void ResetAll();

    [[deprecated("Use SetPreprocessingConfig() with the new preprocessing pipeline instead")]]
    void SetNormalization(float mean, float std);

    [[deprecated("Use SetPreprocessingConfig() with the new preprocessing pipeline instead")]]
    void SetOneHotEncoding(size_t num_classes);

    void SetFlatten(bool flatten);

private:
    std::unique_ptr<DatasetBatcher> train_batcher_;
    std::unique_ptr<DatasetBatcher> val_batcher_;
    std::unique_ptr<DatasetBatcher> test_batcher_;
};

} // namespace cyxwiz
