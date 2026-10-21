#pragma once

#include "dataset_batcher.h"
#include "materialization_memory_types.h"
#include "sparse_feature_dataset.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Estimate peak memory for controlled batch-local CSR densification.
 * raw_output_bytes is the row-major host buffer; temporary_bytes accounts for
 * the ArrayFire-backed Tensor receiving the same values.
 */
MaterializationMemoryEstimate EstimateSparseBatchDensificationMemory(
    uint64_t rows,
    uint64_t features);

/**
 * Batches an immutable SparseFeatureDataset into the existing dense Batch
 * contract without ever allocating a full rows-by-features dataset matrix.
 */
class SparseFeatureDatasetBatcher final : public IBatcher {
public:
    SparseFeatureDatasetBatcher(
        std::shared_ptr<const SparseFeatureDataset> dataset,
        size_t batch_size,
        bool shuffle = true,
        float train_split = 0.8f,
        bool is_training = true,
        BatcherPhase split_phase = BatcherPhase::Train,
        float val_split = 0.0f,
        uint32_t seed = 42,
        bool balance_classes = false,
        const std::string& balance_mode = "none",
        const std::string& balance_target = "max",
        uint32_t balance_seed = 42,
        MaterializationMemoryContext memory_context = {},
        bool stratified = false,
        bool full_dataset = false,
        bool sparse_feature_output = false);

    Batch GetNextBatch() override;
    void Reset() override;
    bool IsEpochComplete() const override;
    size_t GetNumBatches() const override;
    size_t GetNumSamples() const override { return indices_.size(); }

    void SetNormalization(float mean, float std_dev) override;
    void SetOneHotEncoding(size_t num_classes) override;
    void SetScalarLabelMode(bool enable) override;
    void SetClassIndexLabelMode(bool enable) override;
    void SetFlatten(bool /*flatten*/) override {}
    void SetBatchInspectionEnabled(bool enable) override {
        batch_inspection_enabled_ = enable;
    }
    void SetDropLast(bool drop_last) override { drop_last_ = drop_last; }
    void SetSparseFeatureOutput(bool enable) override;

    const MaterializationMemoryEstimate& GetDenseBatchMemoryEstimate() const {
        return dense_batch_memory_estimate_;
    }
    const MaterializationMemoryDecision& GetDenseBatchMemoryDecision() const {
        return dense_batch_memory_decision_;
    }

private:
    struct LabelValue {
        double scalar = 0.0;
        int64_t class_index = 0;
    };

    LabelValue ReadLabel(int64_t row, bool require_class_index) const;
    void ValidateDenseBatchMemory(size_t rows) const;
    void ShuffleIndices();
    void RebuildBalancedIndices();
    void PopulateInspection(Batch& batch,
                            size_t batch_start,
                            size_t actual_batch_size) const;

    std::shared_ptr<const SparseFeatureDataset> dataset_;
    size_t batch_size_ = 0;
    bool shuffle_ = true;
    bool is_training_ = true;
    bool drop_last_ = false;
    BatcherPhase split_phase_ = BatcherPhase::Train;

    std::vector<int64_t> indices_;
    std::vector<int64_t> base_indices_;
    size_t current_index_ = 0;
    std::mt19937 rng_;
    std::mt19937 balance_rng_;
    uint32_t balance_seed_ = 42;
    uint32_t balance_epoch_ = 0;
    bool balance_classes_ = false;
    std::string balance_mode_;
    std::string balance_target_;

    bool normalize_ = false;
    float norm_mean_ = 0.0f;
    float norm_std_ = 1.0f;
    bool one_hot_ = false;
    size_t num_classes_ = 0;
    bool scalar_label_mode_ = false;
    bool class_index_label_mode_ = false;
    bool batch_inspection_enabled_ = false;
    bool sparse_feature_output_ = false;

    MaterializationMemoryContext memory_context_;
    MaterializationMemoryEstimate dense_batch_memory_estimate_;
    MaterializationMemoryDecision dense_batch_memory_decision_;
};

} // namespace cyxwiz
