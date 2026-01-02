#include "cyxwiz/distributed/distributed_sampler.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

// ========== DistributedSampler ==========

DistributedSampler::DistributedSampler(size_t dataset_size, bool shuffle,
                                       unsigned int seed, bool drop_last)
    : dataset_size_(dataset_size)
    , shuffle_(shuffle)
    , seed_(seed)
    , drop_last_(drop_last)
    , epoch_(0) {

    // Get rank info from process group
    ProcessGroup* pg = GetDefaultProcessGroup();
    if (pg && pg->IsInitialized()) {
        rank_ = pg->GetRank();
        world_size_ = pg->GetWorldSize();
    } else {
        // Single process mode
        rank_ = 0;
        world_size_ = 1;
    }

    spdlog::debug("DistributedSampler: dataset_size={}, rank={}/{}, shuffle={}, seed={}",
                  dataset_size_, rank_, world_size_, shuffle_, seed_);
}

void DistributedSampler::SetEpoch(int epoch) {
    epoch_ = epoch;
}

size_t DistributedSampler::PaddedSize() const {
    if (drop_last_) {
        // Round down to nearest multiple of world_size
        return (dataset_size_ / world_size_) * world_size_;
    } else {
        // Round up to nearest multiple of world_size
        return ((dataset_size_ + world_size_ - 1) / world_size_) * world_size_;
    }
}

size_t DistributedSampler::LocalSize() const {
    return PaddedSize() / world_size_;
}

std::vector<size_t> DistributedSampler::GenerateGlobalIndices() const {
    size_t padded_size = PaddedSize();
    std::vector<size_t> indices(padded_size);

    // Initialize with sequential indices
    for (size_t i = 0; i < padded_size; ++i) {
        // Wrap around for padding
        indices[i] = i % dataset_size_;
    }

    // Shuffle if enabled
    if (shuffle_) {
        // Use seed + epoch for deterministic shuffling across ranks
        std::mt19937 gen(seed_ + static_cast<unsigned int>(epoch_));
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    return indices;
}

std::vector<size_t> DistributedSampler::GetIndices() const {
    // Generate globally shuffled indices (same on all ranks due to same seed)
    std::vector<size_t> global_indices = GenerateGlobalIndices();

    // Extract this rank's portion using stride pattern
    // Rank 0: 0, N, 2N, ...
    // Rank 1: 1, N+1, 2N+1, ...
    // etc.
    size_t local_size = LocalSize();
    std::vector<size_t> local_indices;
    local_indices.reserve(local_size);

    for (size_t i = 0; i < local_size; ++i) {
        size_t global_idx = i * world_size_ + rank_;
        if (global_idx < global_indices.size()) {
            local_indices.push_back(global_indices[global_idx]);
        }
    }

    return local_indices;
}

// ========== DistributedBatchIterator ==========

DistributedBatchIterator::DistributedBatchIterator(DistributedSampler& sampler,
                                                   size_t batch_size)
    : sampler_(sampler)
    , batch_size_(batch_size)
    , current_batch_(0) {

    // Initialize with current epoch
    Reset(sampler_.GetEpoch());
}

void DistributedBatchIterator::Reset(int epoch) {
    sampler_.SetEpoch(epoch);
    indices_ = sampler_.GetIndices();
    current_batch_ = 0;
    num_batches_ = (indices_.size() + batch_size_ - 1) / batch_size_;
}

bool DistributedBatchIterator::HasNext() const {
    return current_batch_ < num_batches_;
}

std::vector<size_t> DistributedBatchIterator::Next() {
    if (!HasNext()) {
        return {};
    }

    size_t start = current_batch_ * batch_size_;
    size_t end = std::min(start + batch_size_, indices_.size());

    std::vector<size_t> batch(indices_.begin() + start, indices_.begin() + end);
    ++current_batch_;

    return batch;
}

size_t DistributedBatchIterator::NumBatches() const {
    return num_batches_;
}

} // namespace cyxwiz
