#include "training_batcher_setup.h"

#include "classification_decision.h"
#include "sparse_feature_dataset.h"

#include <arrow/type.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace cyxwiz {
namespace {

bool RequiresClassIndexTargetsForSparse(
    const TrainingConfiguration& config) {
    if (!UsesClassIndexTargets(config.loss_type) ||
        config.preprocessing.has_onehot) {
        return false;
    }
    const auto weights = config.loss_params.find("class_weights");
    const bool has_weights = weights != config.loss_params.end() &&
                             !weights->second.empty();
    return has_weights || config.loss_params.contains("ignore_index");
}

void ValidateSparseTrainingContract(
    const TrainingConfiguration& config,
    const SparseFeatureDataset& dataset) {
    if (config.sequence_batch.enabled || config.is_time_series) {
        throw std::invalid_argument(
            "Sparse text feature training does not support sequence or "
            "time-series batch contracts");
    }
    if (!dataset.GetLabels()) {
        throw std::invalid_argument(
            "Sparse text feature training requires labels");
    }
    if (UsesContinuousTargetMetrics(config) && config.target.width > 1) {
        throw std::invalid_argument(
            "Sparse text feature training currently supports one target "
            "column per dataset");
    }
}

void ApplySparseBatcherTransforms(
    SparseFeatureDatasetBatcher& batcher,
    const TrainingConfiguration& config) {
    if (config.preprocessing.has_normalization) {
        batcher.SetNormalization(
            config.preprocessing.norm_mean,
            config.preprocessing.norm_std);
    }
    if (UsesContinuousTargetMetrics(config) ||
        UsesScalarBinaryTargets(config.loss_type)) {
        batcher.SetScalarLabelMode(true);
    } else if (RequiresClassIndexTargetsForSparse(config)) {
        batcher.SetClassIndexLabelMode(true);
    } else if (config.preprocessing.has_onehot) {
        batcher.SetOneHotEncoding(config.preprocessing.num_classes);
    } else {
        batcher.SetOneHotEncoding(config.output_size);
    }
}

std::unique_ptr<SparseFeatureDatasetBatcher> MakeExternalSparseRole(
    const TrainingConfiguration& config,
    const std::shared_ptr<const SparseFeatureDataset>& dataset,
    int batch_size,
    BatcherPhase phase) {
    ValidateSparseTrainingContract(config, *dataset);
    auto batcher = std::make_unique<SparseFeatureDatasetBatcher>(
        dataset, static_cast<size_t>(batch_size), false, 1.0f, false,
        phase, 0.0f,
        static_cast<uint32_t>(config.dataloader_seed), false, "none", "max",
        static_cast<uint32_t>(config.dataloader_seed),
        MaterializationMemoryContext{}, false, true, true);
    ApplySparseBatcherTransforms(*batcher, config);
    return batcher;
}

std::string SparseRoleCompatibilityError(
    const SparseFeatureDataset& train,
    const SparseFeatureDataset& role,
    const char* role_name) {
    if (train.GetNumFeatures() != role.GetNumFeatures()) {
        return std::string(role_name) + " sparse dataset has " +
            std::to_string(role.GetNumFeatures()) +
            " features, but Training has " +
            std::to_string(train.GetNumFeatures());
    }
    if (train.GetFeatureNames() != role.GetFeatureNames()) {
        return std::string(role_name) +
            " sparse vocabulary/order does not match Training; reuse the "
            "Training vectorizer fitted state in Transform Only mode";
    }
    if (!train.GetLabels() || !role.GetLabels()) {
        return std::string(role_name) +
            " sparse dataset and Training must both contain labels";
    }
    if (!train.GetLabels()->type()->Equals(role.GetLabels()->type())) {
        return std::string(role_name) + " sparse label type " +
            role.GetLabels()->type()->ToString() +
            " does not match Training label type " +
            train.GetLabels()->type()->ToString();
    }
    return {};
}

} // namespace

TrainingBatcherSet BuildSparseTrainingBatchers(
    const TrainingConfiguration& config,
    std::shared_ptr<const SparseFeatureDataset> dataset,
    int batch_size) {
    if (!dataset) {
        throw std::invalid_argument(
            "BuildSparseTrainingBatchers requires a dataset");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "BuildSparseTrainingBatchers batch_size must be positive");
    }
    ValidateSparseTrainingContract(config, *dataset);

    TrainingBatcherSet result;
    const float effective_val_ratio =
        config.has_data_split ? config.val_ratio : 0.0f;
    const auto make = [&](bool shuffle, bool is_training,
                          BatcherPhase phase,
                          bool balance) {
        return std::make_unique<SparseFeatureDatasetBatcher>(
            dataset, static_cast<size_t>(batch_size), shuffle,
            config.train_ratio, is_training, phase, effective_val_ratio,
            static_cast<uint32_t>(config.dataloader_seed),
            balance, config.balance_mode, config.balance_target,
            static_cast<uint32_t>(std::max(0, config.balance_seed)),
            MaterializationMemoryContext{}, config.stratified, false, true);
    };
    const bool continuous_target = UsesContinuousTargetMetrics(config);
    result.sparse_train = make(
        config.shuffle, true, BatcherPhase::Train,
        config.balance_classes && !continuous_target);
    result.sparse_val = make(false, false, BatcherPhase::Val, false);
    result.sparse_test = make(false, false, BatcherPhase::Test, false);

    if (config.drop_last) {
        result.sparse_train->SetDropLast(true);
    }
    ApplySparseBatcherTransforms(*result.sparse_train, config);
    ApplySparseBatcherTransforms(*result.sparse_val, config);
    ApplySparseBatcherTransforms(*result.sparse_test, config);

    result.num_train_samples = result.sparse_train->GetNumSamples();
    result.num_val_samples = result.sparse_val->GetNumSamples();
    result.num_test_samples = result.sparse_test->GetNumSamples();
    result.train = result.sparse_train.get();
    result.val = result.sparse_val.get();
    result.test = result.sparse_test.get();
    AttachTrainingBatcherPrefetchWrappers(result, config, "sparse CSR");
    return result;
}

ResolvedTabularBatcherBuildResult BuildResolvedSparseTrainingBatchers(
    TrainingConfiguration& config,
    const ResolvedTabularDatasets& datasets,
    int batch_size) {
    ResolvedTabularBatcherBuildResult result;
    const auto& partitions = config.dataset_roles;
    if (!datasets.train_sparse) {
        result.error_message =
            "Sparse external roles cannot be mixed with an Arrow/Parquet "
            "Training dataset";
        return result;
    }
    if ((partitions.dev.IsSupplied() && !datasets.dev_sparse) ||
        (partitions.test.IsSupplied() && !datasets.test_sparse)) {
        result.error_message =
            "A sparse Training dataset requires supplied Dev/Test roles to "
            "use the same sparse feature representation";
        return result;
    }
    if (datasets.dev_sparse) {
        result.error_message = SparseRoleCompatibilityError(
            *datasets.train_sparse, *datasets.dev_sparse, "Validation");
    }
    if (result.error_message.empty() && datasets.test_sparse) {
        result.error_message = SparseRoleCompatibilityError(
            *datasets.train_sparse, *datasets.test_sparse, "Test");
    }
    if (!result.error_message.empty()) {
        return result;
    }

    config.train_ratio = partitions.policy.train_ratio;
    config.val_ratio = partitions.policy.dev_ratio;
    config.test_ratio = partitions.policy.test_ratio;
    config.split_seed = partitions.policy.seed;
    config.stratified = partitions.policy.stratified;
    config.has_data_split = true;

    auto assembly_config = config;
    assembly_config.prefetch_factor = 0;
    const bool external_dev = partitions.dev.IsSupplied();
    const bool external_test = partitions.test.IsSupplied();
    if (external_dev && !external_test) {
        assembly_config.val_ratio = partitions.policy.test_ratio;
        assembly_config.test_ratio = 0.0f;
    }
    try {
        result.batchers = BuildSparseTrainingBatchers(
            assembly_config, datasets.train_sparse, batch_size);
        if (external_dev && !external_test) {
            result.batchers.sparse_test =
                std::move(result.batchers.sparse_val);
            result.batchers.val = nullptr;
            result.batchers.test = result.batchers.sparse_test.get();
        }
        if (datasets.dev_sparse) {
            result.batchers.sparse_val = MakeExternalSparseRole(
                config, datasets.dev_sparse, batch_size, BatcherPhase::Val);
            result.batchers.val = result.batchers.sparse_val.get();
        }
        if (datasets.test_sparse) {
            result.batchers.sparse_test = MakeExternalSparseRole(
                config, datasets.test_sparse, batch_size, BatcherPhase::Test);
            result.batchers.test = result.batchers.sparse_test.get();
        }
    } catch (const std::exception& error) {
        result.error_message = error.what();
        return result;
    }

    result.batchers.num_train_samples = result.batchers.train
        ? result.batchers.train->GetNumSamples() : 0;
    result.batchers.num_val_samples = result.batchers.val
        ? result.batchers.val->GetNumSamples() : 0;
    result.batchers.num_test_samples = result.batchers.test
        ? result.batchers.test->GetNumSamples() : 0;
    FinalizePartitionManifest(
        config.dataset_roles,
        static_cast<int64_t>(result.batchers.num_train_samples),
        static_cast<int64_t>(result.batchers.num_val_samples),
        static_cast<int64_t>(result.batchers.num_test_samples));
    AttachTrainingBatcherPrefetchWrappers(
        result.batchers, config, "resolved sparse CSR partitions");
    spdlog::info(
        "Resolved sparse partitions: train={} dev={} test={} samples; "
        "external dev={}, external test={}",
        result.batchers.num_train_samples,
        result.batchers.num_val_samples,
        result.batchers.num_test_samples,
        external_dev,
        external_test);
    return result;
}

} // namespace cyxwiz
