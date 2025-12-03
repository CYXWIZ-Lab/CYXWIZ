#pragma once

#include "data_registry.h"
#include <cyxwiz/tensor.h>
#include <vector>
#include <random>
#include <memory>

namespace cyxwiz {

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

    // Preprocessing options
    void SetNormalization(float mean, float std);
    void SetOneHotEncoding(size_t num_classes);
    void SetFlatten(bool flatten) { flatten_ = flatten; }

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

    // Convert float vector to Tensor
    Tensor VectorToTensor(const std::vector<float>& data, const std::vector<size_t>& shape);

    // Convert labels to one-hot encoded Tensor
    Tensor LabelsToOneHot(const std::vector<int>& labels);

    // Convert labels to integer Tensor
    Tensor LabelsToTensor(const std::vector<int>& labels);

    // Apply normalization to data
    void NormalizeData(std::vector<float>& data);

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

    void SetNormalization(float mean, float std);
    void SetOneHotEncoding(size_t num_classes);
    void SetFlatten(bool flatten);

private:
    std::unique_ptr<DatasetBatcher> train_batcher_;
    std::unique_ptr<DatasetBatcher> val_batcher_;
    std::unique_ptr<DatasetBatcher> test_batcher_;
};

} // namespace cyxwiz
