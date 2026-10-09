#pragma once

// Pair and Triplet Dataset Builders (TOFIX140 A5): metric learning runs the
// ordinary model once over a stacked batch. MetricBatchSampler wraps any
// batcher of class-labelled rows and, per batch, hands out
//   Triplets: data = [anchors; positives; negatives] (3T rows), labels = the
//             anchors' class ids (T), size = T;
//   Pairs:    data = [firsts; seconds] (2P rows), labels = 1 for a same-class
//             pair and 0 otherwise (P), size = P.
// For each row it picks a positive from the other rows of its class and a
// negative from the rows of the other classes, from the same batch; a pair
// row takes a positive or a negative in turn. Rows without a candidate are
// left out, and a batch with nothing to pick is skipped. The picks depend
// only on (key, batch index, row), through SplitMix64, so a run replays them
// and the PyTorch fixture reproduces them.

#include "dataset_batcher.h"
#include "metric_learning_batch.h"

#include <array>
#include <cstdint>
#include <vector>

namespace cyxwiz {

uint64_t MetricSamplingMix(uint64_t value);

// One triplet per row that has a positive and a negative candidate, in row
// order: {anchor, positive, negative} row indices into the batch.
std::vector<std::array<int, 3>> SelectBatchTriplets(
    const std::vector<int64_t>& class_ids,
    uint64_t key,
    uint64_t batch_index);

// One pair per row that has a partner, in row order: {row, partner, similar}.
// Row r asks for a same-class partner when (r + batch key) is even and for an
// other-class one otherwise; it takes the other kind when its kind is absent.
std::vector<std::array<int, 3>> SelectBatchPairs(
    const std::vector<int64_t>& class_ids,
    uint64_t key,
    uint64_t batch_index);

// Class ids of a batch's labels: Int32/Int64/Float32, shape [N] or [N, 1].
std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows);

class MetricBatchSampler final : public IBatcher {
public:
    // The picks are keyed by the executor's epoch shuffle seed (training: one
    // key per epoch, from the run's DataLoader seed); a batcher that is never
    // given one (validation) keeps `seed`, so it scores the same picks.
    MetricBatchSampler(IBatcher& source, MetricSampling sampling, uint64_t seed);

    Batch GetNextBatch() override;
    void Reset() override;
    bool SetEpochShuffleSeed(uint64_t seed) override;
    bool IsEpochComplete() const override { return source_->IsEpochComplete(); }
    size_t GetNumBatches() const override { return source_->GetNumBatches(); }
    size_t GetNumSamples() const override { return source_->GetNumSamples(); }

    void SetNormalization(float mean, float std_dev) override { source_->SetNormalization(mean, std_dev); }
    void SetOneHotEncoding(size_t num_classes) override { source_->SetOneHotEncoding(num_classes); }
    void SetScalarLabelMode(bool enable) override { source_->SetScalarLabelMode(enable); }
    void SetClassIndexLabelMode(bool enable) override { source_->SetClassIndexLabelMode(enable); }
    void SetFlatten(bool flatten) override { source_->SetFlatten(flatten); }
    void SetBatchInspectionEnabled(bool enable) override { source_->SetBatchInspectionEnabled(enable); }
    void SetDropLast(bool drop_last) override { source_->SetDropLast(drop_last); }
    // Rows are picked by index (IndexSelect), so the source stays dense.
    void SetSparseFeatureOutput(bool /*enable*/) override { source_->SetSparseFeatureOutput(false); }
    void SetPhase(BatcherPhase phase) override { source_->SetPhase(phase); }

    // Batches of this epoch that gave nothing to pick (skipped).
    size_t SkippedBatches() const { return skipped_batches_; }

private:
    IBatcher* source_;
    MetricSampling sampling_;
    uint64_t seed_;
    uint64_t epoch_key_;
    uint64_t batch_index_ = 0;
    size_t skipped_batches_ = 0;
};

}  // namespace cyxwiz
