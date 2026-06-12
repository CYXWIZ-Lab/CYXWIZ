#include "training_batcher_setup.h"
#include "prefetch_batcher.h"
#include "worker_defaults.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

namespace {

void ApplyPrefetchWrappers(TrainingBatcherSet& result,
                           const TrainingConfiguration& config,
                           const char* dataset_kind) {
    if (config.prefetch_factor <= 0) {
        return;
    }

    const size_t queue_depth = static_cast<size_t>(config.prefetch_factor);
    if (result.train) {
        result.prefetch_train = std::make_unique<PrefetchBatcher>(
            *result.train, queue_depth, std::string(dataset_kind) + " train");
        result.train = result.prefetch_train.get();
    }
    if (result.val) {
        result.prefetch_val = std::make_unique<PrefetchBatcher>(
            *result.val, queue_depth, std::string(dataset_kind) + " validation");
        result.val = result.prefetch_val.get();
    }
    if (result.test) {
        result.prefetch_test = std::make_unique<PrefetchBatcher>(
            *result.test, queue_depth, std::string(dataset_kind) + " test");
        result.test = result.prefetch_test.get();
    }

    spdlog::info("TrainingExecutor: enabled {} async batch prefetch "
                 "(prefetch_factor={}, train={}, val={}, test={})",
                 dataset_kind,
                 config.prefetch_factor,
                 result.prefetch_train ? "yes" : "no",
                 result.prefetch_val ? "yes" : "no",
                 result.prefetch_test ? "yes" : "no");
}

} // namespace

TrainingInputSizeResolution ResolveTabularTrainingInputSize(
    const TrainingConfiguration& config,
    size_t num_columns) {

    if (config.is_time_series) {
        return {config.input_size, true, true};
    }
    if (num_columns > 1) {
        return {num_columns - 1, false, true};
    }
    return {num_columns, false, false};
}

TrainingBatcherSet BuildArrowTrainingBatchers(
    const TrainingConfiguration& config,
    std::shared_ptr<ArrowDataset> dataset,
    const std::string& label_column,
    int batch_size) {

    TrainingBatcherSet result;
    const int num_workers = ClampNumWorkersToPlatform(config.num_workers);
    if (num_workers != config.num_workers) {
        spdlog::warn("TrainingExecutor: clamping Arrow num_workers from {} to {} based on platform",
                     config.num_workers, num_workers);
    }

    spdlog::info("TrainingExecutor: Using Arrow dataset for training "
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={}, num_workers={}, prefetch_factor={})",
                 batch_size, config.shuffle, config.train_ratio,
                 config.is_time_series, num_workers, config.prefetch_factor);

    const std::string partition_col = config.is_time_series
        ? "__partition__" : "";
    const std::string effective_label =
        config.is_time_series ? "y" : label_column;
    const float effective_val_ratio =
        config.has_data_split ? config.val_ratio : 0.0f;

    result.arrow_train = std::make_unique<ArrowDatasetBatcher>(
        dataset, effective_label, batch_size,
        config.shuffle, config.train_ratio, true,
        partition_col, /*partition_value=*/0, num_workers,
        BatcherPhase::Train, effective_val_ratio);
    result.arrow_val = std::make_unique<ArrowDatasetBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/1, num_workers,
        BatcherPhase::Val, effective_val_ratio);
    result.arrow_test = std::make_unique<ArrowDatasetBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/2, num_workers,
        BatcherPhase::Test, effective_val_ratio);

    if (config.drop_last) {
        spdlog::warn("TrainingExecutor: drop_last=true requested but ArrowDatasetBatcher "
                     "does not yet support it - last partial batch will be kept");
    }
    if (config.preprocessing.has_normalization) {
        result.arrow_train->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
        result.arrow_val->SetNormalization(config.preprocessing.norm_mean,
                                           config.preprocessing.norm_std);
        result.arrow_test->SetNormalization(config.preprocessing.norm_mean,
                                            config.preprocessing.norm_std);
    }

    if (config.is_time_series) {
        result.arrow_train->SetRegressionMode(true);
        result.arrow_val->SetRegressionMode(true);
        result.arrow_test->SetRegressionMode(true);
    } else if (config.preprocessing.has_onehot) {
        result.arrow_train->SetOneHotEncoding(config.preprocessing.num_classes);
        result.arrow_val->SetOneHotEncoding(config.preprocessing.num_classes);
        result.arrow_test->SetOneHotEncoding(config.preprocessing.num_classes);
    } else {
        result.arrow_train->SetOneHotEncoding(config.output_size);
        result.arrow_val->SetOneHotEncoding(config.output_size);
        result.arrow_test->SetOneHotEncoding(config.output_size);
    }

    result.num_train_samples = result.arrow_train->GetNumSamples();
    result.num_val_samples = result.arrow_val->GetNumSamples();
    result.num_test_samples = result.arrow_test->GetNumSamples();
    result.train = result.arrow_train.get();
    result.val = result.arrow_val.get();
    result.test = result.arrow_test.get();
    ApplyPrefetchWrappers(result, config, "Arrow");
    spdlog::info("TrainingExecutor: Arrow split samples train={} val={} test={}",
                 result.num_train_samples, result.num_val_samples, result.num_test_samples);
    return result;
}

TrainingBatcherSet BuildParquetTrainingBatchers(
    const TrainingConfiguration& config,
    std::shared_ptr<ParquetBackedDataset> dataset,
    const std::string& label_column,
    int batch_size) {

    TrainingBatcherSet result;
    const int num_workers = ClampNumWorkersToPlatform(config.num_workers);
    if (num_workers != config.num_workers) {
        spdlog::warn("TrainingExecutor: clamping Parquet num_workers from {} to {} based on platform",
                     config.num_workers, num_workers);
    }

    spdlog::info("TrainingExecutor: Using Parquet-backed dataset for training "
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={}, num_workers={}, prefetch_factor={})",
                 batch_size, config.shuffle, config.train_ratio,
                 config.is_time_series, num_workers, config.prefetch_factor);

    const std::string partition_col = config.is_time_series
        ? "__partition__" : "";
    const std::string effective_label =
        config.is_time_series ? "y" : label_column;
    const float effective_val_ratio =
        config.has_data_split ? config.val_ratio : 0.0f;

    result.parquet_train = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        config.shuffle, config.train_ratio, true,
        partition_col, /*partition_value=*/0, num_workers,
        BatcherPhase::Train, effective_val_ratio);
    result.parquet_val = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/1, num_workers,
        BatcherPhase::Val, effective_val_ratio);
    result.parquet_test = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/2, num_workers,
        BatcherPhase::Test, effective_val_ratio);

    if (config.drop_last) {
        spdlog::warn("TrainingExecutor: drop_last=true requested but ParquetArrowBatcher "
                     "does not yet support it - last partial batch will be kept");
    }
    if (config.preprocessing.has_normalization) {
        result.parquet_train->SetNormalization(config.preprocessing.norm_mean,
                                               config.preprocessing.norm_std);
        result.parquet_val->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
        result.parquet_test->SetNormalization(config.preprocessing.norm_mean,
                                              config.preprocessing.norm_std);
    }

    if (config.is_time_series) {
        result.parquet_train->SetRegressionMode(true);
        result.parquet_val->SetRegressionMode(true);
        result.parquet_test->SetRegressionMode(true);
    } else if (config.preprocessing.has_onehot) {
        result.parquet_train->SetOneHotEncoding(config.preprocessing.num_classes);
        result.parquet_val->SetOneHotEncoding(config.preprocessing.num_classes);
        result.parquet_test->SetOneHotEncoding(config.preprocessing.num_classes);
    } else {
        result.parquet_train->SetOneHotEncoding(config.output_size);
        result.parquet_val->SetOneHotEncoding(config.output_size);
        result.parquet_test->SetOneHotEncoding(config.output_size);
    }

    result.num_train_samples = result.parquet_train->GetNumSamples();
    result.num_val_samples = result.parquet_val->GetNumSamples();
    result.num_test_samples = result.parquet_test->GetNumSamples();
    result.train = result.parquet_train.get();
    result.val = result.parquet_val.get();
    result.test = result.parquet_test.get();
    ApplyPrefetchWrappers(result, config, "Parquet");
    spdlog::info("TrainingExecutor: Parquet split samples train={} val={} test={}",
                 result.num_train_samples, result.num_val_samples, result.num_test_samples);
    return result;
}

} // namespace cyxwiz
