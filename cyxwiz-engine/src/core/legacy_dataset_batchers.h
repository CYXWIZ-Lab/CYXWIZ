#pragma once

// Engine-side batchers for the pre-Arrow dataset types: DataRegistry
// DatasetHandle datasets and registered text datasets. The shared training and
// test executors only run IBatchers; these factories adapt the Engine's own
// dataset types to them (TOFIX118 P2), with the same setup the executors
// used to do inline - registry preprocessing and statistics, augmentation,
// legacy normalization, label modes, flattening.

#include "data_registry.h"
#include "dataset_batcher.h"
#include "graph_compiler.h"
#include "test_executor.h"
#include "training_executor.h"

#include <memory>

namespace cyxwiz {

// IBatcher view of a DatasetBatcher (one fixed split).
class DatasetBatcherAdapter final : public IBatcher {
public:
    explicit DatasetBatcherAdapter(std::unique_ptr<DatasetBatcher> batcher);

    Batch GetNextBatch() override { return batcher_->GetNextBatch(); }
    void Reset() override { batcher_->Reset(); }
    bool IsEpochComplete() const override { return batcher_->IsEpochComplete(); }
    size_t GetNumBatches() const override { return batcher_->GetNumBatches(); }
    size_t GetNumSamples() const override { return batcher_->GetNumSamples(); }
    void SetNormalization(float mean, float std_dev) override { batcher_->SetLegacyNormalization(mean, std_dev); }
    void SetOneHotEncoding(size_t num_classes) override { batcher_->SetLegacyOneHotEncoding(num_classes); }
    void SetFlatten(bool flatten) override { batcher_->SetFlatten(flatten); }

    DatasetBatcher& Inner() { return *batcher_; }

private:
    std::unique_ptr<DatasetBatcher> batcher_;
};

// Train + validation batchers for a DatasetHandle (validation never shuffles
// or drops the last batch).
ResolvedExternalBatchers BuildLegacyTrainingBatchers(const TrainingConfiguration& config,
                                                     const DatasetHandle& dataset, int batch_size);
ExternalBatcherFactory LegacyTrainingBatcherFactory(DatasetHandle dataset);

// Test-split batcher for a DatasetHandle.
std::unique_ptr<IBatcher> BuildLegacyTestBatcher(const TrainingConfiguration& config, const DatasetHandle& dataset,
                                                 int batch_size);
ExternalTestSource LegacyTestSource(DatasetHandle dataset);

// Test-phase batcher for a registered text dataset (carries its class names).
std::unique_ptr<IBatcher> BuildTextTestBatcher(const TrainingConfiguration& config,
                                               const DataRegistry::TextDatasetEntry& entry, int batch_size);
ExternalTestSource TextTestSource(DataRegistry::TextDatasetEntry entry);

}  // namespace cyxwiz
