#include "training_batcher_setup.h"
#include "worker_defaults.h"

#include <spdlog/spdlog.h>

namespace cyxwiz {

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
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={}, num_workers={})",
                 batch_size, config.shuffle, config.train_ratio,
                 config.is_time_series, num_workers);

    const std::string partition_col = config.is_time_series
        ? "__partition__" : "";
    const std::string effective_label =
        config.is_time_series ? "y" : label_column;

    result.arrow_train = std::make_unique<ArrowDatasetBatcher>(
        dataset, effective_label, batch_size,
        config.shuffle, config.train_ratio, true,
        partition_col, /*partition_value=*/0, num_workers);
    result.arrow_val = std::make_unique<ArrowDatasetBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/1, num_workers);

    if (config.drop_last) {
        spdlog::warn("TrainingExecutor: drop_last=true requested but ArrowDatasetBatcher "
                     "does not yet support it - last partial batch will be kept");
    }
    if (!config.is_time_series &&
        config.has_data_split && config.test_ratio > 0.01f) {
        spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                     "ArrowDatasetBatcher has no held-out test split - the test portion "
                     "will be merged into validation. train={:.2f}, val+test={:.2f}",
                     config.test_ratio, config.train_ratio, 1.0f - config.train_ratio);
    }

    if (config.preprocessing.has_normalization) {
        result.arrow_train->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
        result.arrow_val->SetNormalization(config.preprocessing.norm_mean,
                                           config.preprocessing.norm_std);
    }

    if (config.is_time_series) {
        result.arrow_train->SetRegressionMode(true);
        result.arrow_val->SetRegressionMode(true);
    } else if (config.preprocessing.has_onehot) {
        result.arrow_train->SetOneHotEncoding(config.preprocessing.num_classes);
        result.arrow_val->SetOneHotEncoding(config.preprocessing.num_classes);
    } else {
        result.arrow_train->SetOneHotEncoding(config.output_size);
        result.arrow_val->SetOneHotEncoding(config.output_size);
    }

    result.num_train_samples = result.arrow_train->GetNumSamples();
    result.train = result.arrow_train.get();
    result.val = result.arrow_val.get();
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
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, num_workers={})",
                 batch_size, config.shuffle, config.train_ratio, num_workers);

    result.parquet_train = std::make_unique<ParquetArrowBatcher>(
        dataset, label_column, batch_size,
        config.shuffle, config.train_ratio, true, num_workers);
    result.parquet_val = std::make_unique<ParquetArrowBatcher>(
        dataset, label_column, batch_size,
        false, config.train_ratio, false, num_workers);

    if (config.drop_last) {
        spdlog::warn("TrainingExecutor: drop_last=true requested but ParquetArrowBatcher "
                     "does not yet support it - last partial batch will be kept");
    }
    if (config.has_data_split && config.test_ratio > 0.01f) {
        spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                     "ParquetArrowBatcher has no held-out test split - the test portion "
                     "will be merged into validation. train={:.2f}, val+test={:.2f}",
                     config.test_ratio, config.train_ratio, 1.0f - config.train_ratio);
    }

    if (config.preprocessing.has_normalization) {
        result.parquet_train->SetNormalization(config.preprocessing.norm_mean,
                                               config.preprocessing.norm_std);
        result.parquet_val->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
    }

    if (config.preprocessing.has_onehot) {
        result.parquet_train->SetOneHotEncoding(config.preprocessing.num_classes);
        result.parquet_val->SetOneHotEncoding(config.preprocessing.num_classes);
    } else {
        result.parquet_train->SetOneHotEncoding(config.output_size);
        result.parquet_val->SetOneHotEncoding(config.output_size);
    }

    result.num_train_samples = result.parquet_train->GetNumSamples();
    result.train = result.parquet_train.get();
    result.val = result.parquet_val.get();
    return result;
}

} // namespace cyxwiz
