#pragma once

#include "data_registry.h"
#include "arrow_dataset.h"
#include <cyxwiz/layers/linear_sparse_csr.h>
#include <cyxwiz/tensor.h>
#include <cstdint>
#include <optional>
#include <vector>
#include <random>
#include <memory>
#include <string>

namespace cyxwiz {

/** Owning, batch-local CSR payload for a first Linear projection. */
struct SparseFeatureBatch {
    size_t rows = 0;
    size_t columns = 0;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<float> values;

    LinearSparseCsrBatchView View() const noexcept {
        return {
            rows,
            columns,
            values.size(),
            row_offsets.data(),
            column_indices.data(),
            values.data()};
    }
};

struct BatchColumnInspection {
    std::string name;
    std::string source_dtype;
};

struct BatchInspectionMetadata {
    bool available = false;
    size_t row_count = 0;
    size_t feature_column_count = 0;
    size_t label_column_count = 0;
    std::vector<BatchColumnInspection> feature_columns_preview;
    std::vector<BatchColumnInspection> label_columns_preview;
    bool feature_columns_truncated = false;
    bool label_columns_truncated = false;
    bool null_summary_available = false;
    uint64_t inspected_value_count = 0;
    uint64_t feature_null_count = 0;
    uint64_t label_null_count = 0;
    bool token_sequence_columns = false;
};

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
    // Present only when a sparse feature batch is routed directly into the
    // model's first Linear projection. In that mode data remains empty.
    std::optional<SparseFeatureBatch> sparse_features;
    Tensor labels;        // [batch_size] or [batch_size, num_classes] if one-hot
    size_t size = 0;      // Actual batch size (may be < requested for last batch)
    BatchInspectionMetadata inspection;

    bool IsValid() const { return size > 0; }
    bool HasSparseFeatures() const {
        return sparse_features.has_value();
    }
};

/**
 * Named sequence payload for token-level tasks such as NER.
 *
 * This is intentionally separate from Batch: token tagging needs multiple
 * named tensors with sequence shape, while Batch represents the current
 * single tensor/label runtime contract.
 */
struct SequenceBatch {
    Tensor word_ids;        // [batch, seq] int ids, required
    Tensor pos_ids;         // [batch, seq] int ids, optional
    Tensor attention_mask;  // [batch, seq] 1 for real tokens, 0 for padding
    Tensor tag_ids;         // [batch, seq] int tag ids, required for training
    Tensor target_ids;      // [batch, seq] next-token ids for causal LM training
    size_t size = 0;        // Actual batch size
    size_t sequence_length = 0;

    bool HasWordIds() const {
        return !word_ids.Shape().empty() && word_ids.NumElements() > 0;
    }
    bool HasPosIds() const {
        return !pos_ids.Shape().empty() && pos_ids.NumElements() > 0;
    }
    bool HasAttentionMask() const {
        return !attention_mask.Shape().empty() &&
               attention_mask.NumElements() > 0;
    }
    bool HasTagIds() const {
        return !tag_ids.Shape().empty() && tag_ids.NumElements() > 0;
    }
    bool HasTargetIds() const {
        return !target_ids.Shape().empty() && target_ids.NumElements() > 0;
    }
    bool IsValid() const {
        return size > 0 && sequence_length > 0 && HasWordIds();
    }
    bool IsSupervised() const { return IsValid() && (HasTagIds() || HasTargetIds()); }
    bool IsLanguageModeling() const { return IsValid() && HasTargetIds(); }
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
 * IBatcher - Abstract interface shared by every training batcher.
 *
 * Training code (TrainingExecutor::RunTrainingEpoch, RunValidation) holds
 * IBatcher references instead of concrete batcher types, so it can drive
 * either the in-memory Arrow path (ArrowDatasetBatcher) or the disk-backed
 * Parquet path (ParquetArrowBatcher) without branching per batch.
 *
 * Implementations must be safe to call GetNextBatch repeatedly until it
 * returns an invalid (size=0) Batch, at which point Reset starts a new
 * epoch. IsEpochComplete should become true after the last batch has
 * been consumed.
 *
 * Preprocessing setters (SetNormalization, SetOneHotEncoding, SetFlatten)
 * are exposed via this interface so the training executor can configure
 * any batcher uniformly. Subclasses that don't support a particular knob
 * can leave it as a no-op.
 */
// Which phase the batcher is currently iterating. Image/audio batchers
// use a single instance for both train and val (same underlying dataset);
// SetPhase flips between the index sets without reconstructing. Text now
// also uses a dedicated Test phase so held-out evaluation can reuse the
// same dataset entry after training.
// Arrow/Parquet/legacy batchers ignore this since they already instantiate
// separate train vs val batchers.
enum class BatcherPhase { Train, Val, Test };

class ISequenceBatcher {
public:
    virtual ~ISequenceBatcher() = default;

    virtual SequenceBatch GetNextSequenceBatch() = 0;
    virtual void Reset() = 0;
    virtual bool IsEpochComplete() const = 0;
    virtual size_t GetNumBatches() const = 0;
    virtual size_t GetNumSamples() const = 0;
    virtual void SetPhase(BatcherPhase /*phase*/) {}
    virtual bool HasPhase(BatcherPhase phase) const {
        return phase != BatcherPhase::Test;
    }
};

class IBatcher {
public:
    virtual ~IBatcher() = default;

    virtual Batch GetNextBatch() = 0;
    virtual void Reset() = 0;
    virtual bool IsEpochComplete() const = 0;
    virtual size_t GetNumBatches() const = 0;
    virtual size_t GetNumSamples() const = 0;

    virtual void SetNormalization(float mean, float std_dev) = 0;
    virtual void SetOneHotEncoding(size_t num_classes) = 0;
    virtual void SetScalarLabelMode(bool /*enable*/) {}
    virtual void SetClassIndexLabelMode(bool /*enable*/) {}
    virtual void SetFlatten(bool flatten) = 0;
    virtual void SetBatchInspectionEnabled(bool /*enable*/) {}
    virtual void SetDropLast(bool /*drop_last*/) {}
    virtual void SetSparseFeatureOutput(bool /*enable*/) {}

    // Switch between train/val/test index sets. Default: no-op (batchers
    // that already have a separate validation instance don't need this).
    // The caller should call Reset() afterwards to pick up the new phase's
    // epoch order. Image/audio batchers override this.
    virtual void SetPhase(BatcherPhase /*phase*/) {}
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
        bool drop_last = false,
        int num_workers = 0,
        uint32_t seed = 42
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

    // Legacy runtime knobs used by the current training/test paths.
    // These avoid self-triggering the public deprecation warning while
    // keeping SetNormalization/SetOneHotEncoding available for old callers.
    void SetLegacyNormalization(float mean, float std);
    void SetLegacyOneHotEncoding(size_t num_classes);
    void SetLegacyScalarLabelMode(bool enable) { scalar_label_mode_ = enable; }

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
    int num_workers_ = 0;

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
    bool scalar_label_mode_ = false;

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

/**
 * ArrowDatasetBatcher - Batched iteration over Arrow datasets
 *
 * Similar to DatasetBatcher but works with ArrowDataset (columnar data)
 * for Data Studio pipelines.
 */
class ArrowDatasetBatcher : public IBatcher {
public:
    /**
     * Create a batcher for Arrow dataset
     * @param dataset ArrowDataset containing data
     * @param label_column Name of the label column (empty for unsupervised)
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle samples each epoch
     * @param train_split Fraction of data for training (0.8 = 80% train)
     * @param is_training Whether this is a training batcher (affects split)
     */
    ArrowDatasetBatcher(
        std::shared_ptr<class ArrowDataset> dataset,
        const std::string& label_column,
        size_t batch_size,
        bool shuffle = true,
        float train_split = 0.8f,
        bool is_training = true,
        // Phase 4 Time-Series: if `partition_column` is non-empty, the
        // batcher filters rows where that column (int8) equals
        // `partition_value` and builds its index set from those rows
        // only — bypassing the train_split first-N% slicing entirely.
        // partition_value: 0=train, 1=val, 2=test. Legacy callers omit
        // these args and keep chronological first-N% slicing unchanged.
        const std::string& partition_column = "",
        int partition_value = 0,
        int num_workers = 0,
        BatcherPhase split_phase = BatcherPhase::Train,
        float val_split = 0.0f,
        uint32_t seed = 42,
        bool balance_classes = false,
        const std::string& balance_mode = "none",
        const std::string& balance_target = "max",
        uint32_t balance_seed = 42
    );

    // IBatcher interface
    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override { return indices_.size(); }

    // Preprocessing (IBatcher)
    void SetNormalization(float mean, float std) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetFlatten(bool flatten) override { flatten_ = flatten; }
    void SetDropLast(bool drop_last) override { drop_last_ = drop_last; }

    // Phase 4 Time-Series: opt into regression label handling. When true,
    // labels are read as float (not int-cast), one-hot encoding is
    // bypassed, and the label tensor is emitted as [batch_size, 1] float
    // instead of [batch_size] int or [batch_size, num_classes] one-hot.
    // The current feedforward MSELoss regression path consumes this shape
    // directly. Legacy classification callers never set this and keep the
    // existing behavior.
    void SetScalarLabelMode(bool enable) override { scalar_label_mode_ = enable; }
    void SetClassIndexLabelMode(bool enable) override {
        class_index_label_mode_ = enable;
        if (enable) one_hot_ = false;
    }
    void SetRegressionMode(bool enable) { SetScalarLabelMode(enable); }
    void SetBatchInspectionEnabled(bool enable) override {
        batch_inspection_enabled_ = enable;
    }

    // Configure ordered regression targets. Width one consumes the primary
    // label; wider targets consume `<base>`, `<base>_1`, ... and remove every
    // target from the feature tensor. Throws when the schema is incomplete.
    void SetRegressionTargetWidth(
        size_t width, const std::string& target_base = {});

private:
    std::shared_ptr<class ArrowDataset> dataset_;
    std::string label_column_;
    size_t batch_size_;
    bool shuffle_;
    bool is_training_;
    bool drop_last_ = false;
    int num_workers_ = 0;
    std::string partition_column_;   // Phase 4: time-series partition col name, "" = legacy slicing
    int partition_value_ = 0;        // which partition this batcher iterates (0=train, 1=val, 2=test)
    BatcherPhase split_phase_ = BatcherPhase::Train;
    float val_split_ = 0.0f;

    std::vector<int64_t> indices_;   // Row indices for this split
    std::vector<int64_t> base_indices_;  // Original split before balancing/shuffle
    size_t current_index_ = 0;
    std::mt19937 rng_;
    std::mt19937 balance_rng_;
    uint32_t balance_seed_ = 42;
    uint32_t balance_epoch_ = 0;
    bool balance_classes_ = false;
    std::string balance_mode_ = "none";
    std::string balance_target_ = "max";

    // Feature and label column indices
    std::vector<int> feature_cols_;
    int label_col_idx_ = -1;
    std::vector<int> label_col_indices_;
    size_t num_features_ = 0;

    // Preprocessing
    bool normalize_ = false;
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool one_hot_ = false;
    size_t num_classes_ = 10;  // Default for MNIST
    bool flatten_ = true;
    bool scalar_label_mode_ = false;  // float [batch, 1] labels, no one-hot
    bool class_index_label_mode_ = false;  // Int32 [batch] labels
    bool batch_inspection_enabled_ = false;

    void ShuffleIndices();
    void InitializeColumns();
    void RebuildBalancedIndices();
};

} // namespace cyxwiz
