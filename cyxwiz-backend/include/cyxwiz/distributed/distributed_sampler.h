#pragma once

#include "../api_export.h"
#include "process_group.h"
#include <vector>
#include <cstddef>
#include <random>

namespace cyxwiz {

/**
 * DistributedSampler - Shards dataset indices across distributed workers
 *
 * Each rank gets a non-overlapping subset of the dataset. The sampler ensures
 * that all ranks process the same number of samples by padding if necessary.
 *
 * Index distribution pattern (stride):
 *   With 4 ranks and 1000 samples:
 *     Rank 0: 0, 4, 8, 12, ...  (250 samples)
 *     Rank 1: 1, 5, 9, 13, ...  (250 samples)
 *     Rank 2: 2, 6, 10, 14, ... (250 samples)
 *     Rank 3: 3, 7, 11, 15, ... (250 samples)
 *
 * For shuffling, all ranks use the same seed (seed + epoch) to ensure
 * consistent global shuffling while each rank takes its stride.
 *
 * Usage:
 *   DistributedSampler sampler(dataset_size, true, 42);
 *
 *   for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *       sampler.SetEpoch(epoch);  // For deterministic shuffling
 *       auto indices = sampler.GetIndices();
 *
 *       for (size_t idx : indices) {
 *           // Process sample at idx
 *       }
 *   }
 */
class CYXWIZ_API DistributedSampler {
public:
    /**
     * Create a distributed sampler
     *
     * @param dataset_size Total number of samples in the dataset
     * @param shuffle Whether to shuffle indices each epoch
     * @param seed Base random seed (combined with epoch for reproducibility)
     * @param drop_last If true, drop samples that don't divide evenly
     */
    DistributedSampler(size_t dataset_size, bool shuffle = true,
                       unsigned int seed = 0, bool drop_last = false);

    /**
     * Set the current epoch for shuffling
     *
     * Must be called at the start of each epoch for deterministic shuffling.
     * The actual seed used is: base_seed + epoch
     *
     * @param epoch Current epoch number
     */
    void SetEpoch(int epoch);

    /**
     * Get current epoch
     */
    int GetEpoch() const { return epoch_; }

    /**
     * Get indices for this rank's portion of the dataset
     *
     * Returns the indices that this rank should process. If shuffle is enabled,
     * the global order is shuffled (deterministically based on seed + epoch),
     * then each rank takes every Nth sample starting from its rank.
     *
     * @return Vector of dataset indices for this rank
     */
    std::vector<size_t> GetIndices() const;

    /**
     * Get number of samples for this rank
     *
     * This is the number of samples this rank will process per epoch.
     * May be slightly larger than dataset_size / world_size due to padding.
     */
    size_t LocalSize() const;

    /**
     * Get total dataset size
     */
    size_t TotalSize() const { return dataset_size_; }

    /**
     * Get the padded total size (divisible by world_size)
     */
    size_t PaddedSize() const;

    /**
     * Get rank of this process
     */
    int GetRank() const { return rank_; }

    /**
     * Get world size
     */
    int GetWorldSize() const { return world_size_; }

private:
    size_t dataset_size_;
    bool shuffle_;
    unsigned int seed_;
    bool drop_last_;
    int epoch_ = 0;

    // Cached from process group
    int rank_;
    int world_size_;

    /**
     * Generate shuffled global indices based on current epoch
     */
    std::vector<size_t> GenerateGlobalIndices() const;
};

/**
 * Helper class to iterate over batches in distributed training
 *
 * Combines DistributedSampler with batch creation.
 */
class CYXWIZ_API DistributedBatchIterator {
public:
    /**
     * Create a batch iterator
     *
     * @param sampler The distributed sampler to use
     * @param batch_size Per-rank batch size
     */
    DistributedBatchIterator(DistributedSampler& sampler, size_t batch_size);

    /**
     * Reset iterator for new epoch
     * Automatically calls sampler.SetEpoch()
     */
    void Reset(int epoch);

    /**
     * Check if there are more batches
     */
    bool HasNext() const;

    /**
     * Get next batch of indices
     *
     * @return Vector of indices for next batch (size <= batch_size)
     */
    std::vector<size_t> Next();

    /**
     * Get total number of batches for this rank
     */
    size_t NumBatches() const;

    /**
     * Get current batch index
     */
    size_t CurrentBatch() const { return current_batch_; }

private:
    DistributedSampler& sampler_;
    size_t batch_size_;
    std::vector<size_t> indices_;
    size_t current_batch_ = 0;
    size_t num_batches_ = 0;
};

} // namespace cyxwiz
