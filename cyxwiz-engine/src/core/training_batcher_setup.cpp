#include "training_batcher_setup.h"
#include "training_executor.h"

#include "arrow_dataset.h"
#include "classification_decision.h"
#include "prefetch_batcher.h"
#include "split_partitioning.h"
#include "worker_defaults.h"

#include <spdlog/spdlog.h>

#include <arrow/array.h>
#include <arrow/table.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

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

std::string LowerAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string TrimAscii(const std::string& value) {
    const auto begin = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto end = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    return begin < end ? std::string(begin, end) : std::string{};
}

bool IsBalancedClassWeightRequest(const TrainingConfiguration& config) {
    if (config.loss_type != gui::NodeType::CrossEntropyLoss) {
        return false;
    }
    auto it = config.loss_params.find("class_weight");
    if (it == config.loss_params.end()) {
        return false;
    }
    return LowerAscii(TrimAscii(it->second)) == "balanced";
}

bool RequiresClassIndexTargets(const TrainingConfiguration& config) {
    if (!UsesClassIndexTargets(config.loss_type) ||
        config.preprocessing.has_onehot) {
        return false;
    }
    const auto weights = config.loss_params.find("class_weights");
    const bool has_weights = weights != config.loss_params.end() &&
                             !TrimAscii(weights->second).empty();
    return has_weights || config.loss_params.contains("ignore_index");
}

std::vector<int64_t> TrainRowsFromPartition(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& partition_column) {
    std::vector<int64_t> rows;
    if (!table || partition_column.empty()) {
        return rows;
    }

    auto column = table->GetColumnByName(partition_column);
    if (!column || column->num_chunks() == 0) {
        return rows;
    }
    for (int c = 0; c < column->num_chunks(); ++c) {
        if (column->chunk(c)->type_id() != arrow::Type::INT8) {
            return {};
        }
    }

    int64_t offset = 0;
    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        const int64_t length = chunk->length();
        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
        const int8_t* data = arr->raw_values();
        for (int64_t i = 0; i < length; ++i) {
            if (data[i] == 0) {
                rows.push_back(offset + i);
            }
        }
        offset += length;
    }
    return rows;
}

std::vector<int64_t> TrainRowsFromRatio(const std::shared_ptr<arrow::Table>& table,
                                        float train_ratio) {
    std::vector<int64_t> rows;
    if (!table) {
        return rows;
    }
    const float safe_ratio = std::clamp(train_ratio, 0.0f, 1.0f);
    const int64_t train_count =
        static_cast<int64_t>(table->num_rows() * safe_ratio);
    rows.reserve(static_cast<size_t>(std::max<int64_t>(0, train_count)));
    for (int64_t i = 0; i < train_count && i < table->num_rows(); ++i) {
        rows.push_back(i);
    }
    return rows;
}

std::string FormatFloatVector(const std::vector<float>& values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << ", ";
        out << std::setprecision(8) << values[i];
    }
    out << "]";
    return out.str();
}

} // namespace

void AttachTrainingBatcherPrefetchWrappers(
    TrainingBatcherSet& batchers,
    const TrainingConfiguration& config,
    const char* dataset_kind) {
    ApplyPrefetchWrappers(batchers, config, dataset_kind);
}

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

bool TryApplyBalancedClassWeightsFromArrowTable(
    TrainingConfiguration& config,
    const std::shared_ptr<arrow::Table>& table,
    const std::string& label_column,
    const std::string& partition_column,
    const std::string& context) {

    if (!IsBalancedClassWeightRequest(config)) {
        return false;
    }
    if (config.is_time_series) {
        spdlog::warn("{}: class_weight=balanced is not available for "
                     "time-series/regression training; using unweighted loss",
                     context);
        return false;
    }
    if (!table) {
        spdlog::warn("{}: class_weight=balanced requested but Arrow table is "
                     "null; using unweighted loss", context);
        return false;
    }
    if (label_column.empty()) {
        spdlog::warn("{}: class_weight=balanced requested but label column is "
                     "empty; using unweighted loss", context);
        return false;
    }

    std::shared_ptr<arrow::Table> effective_table = table;
    std::string effective_partition = partition_column;
    if (effective_partition.empty() && config.stratified && !config.is_time_series) {
        auto partitioned = AddSplitPartitionColumn(
            table,
            SplitPartitionOptions{
                label_column,
                config.train_ratio,
                config.val_ratio,
                config.test_ratio,
                config.shuffle,
                static_cast<uint32_t>(std::max(0, config.split_seed)),
                true,
                context + " class weights"});
        if (partitioned.ok()) {
            effective_table = partitioned.ValueOrDie();
            effective_partition = kSplitPartitionColumn;
        } else {
            spdlog::warn("{}: stratified class-weight split could not be "
                         "prepared ({}); using ratio-based train slice",
                         context, partitioned.status().ToString());
        }
    }

    auto labels_result = ReadNumericLabelColumn(effective_table, label_column);
    if (!labels_result.ok()) {
        spdlog::warn("{}: class_weight=balanced requested but labels could "
                     "not be read ({}); using unweighted loss",
                     context, labels_result.status().ToString());
        return false;
    }
    const auto& labels = labels_result.ValueOrDie();

    std::vector<int64_t> train_rows = !effective_partition.empty()
        ? TrainRowsFromPartition(effective_table, effective_partition)
        : TrainRowsFromRatio(effective_table, config.train_ratio);
    if (train_rows.empty()) {
        spdlog::warn("{}: class_weight=balanced requested but the train split "
                     "has no rows; using unweighted loss", context);
        return false;
    }

    std::map<int64_t, size_t> counts;
    for (int64_t row : train_rows) {
        if (row < 0 || static_cast<size_t>(row) >= labels.size()) {
            continue;
        }
        const int64_t label = labels[static_cast<size_t>(row)];
        if (label >= 0) {
            ++counts[label];
        }
    }

    if (counts.size() < 2) {
        spdlog::warn("{}: class_weight=balanced requested but the train split "
                     "has fewer than two non-negative classes; using "
                     "unweighted loss", context);
        return false;
    }

    size_t class_count = config.output_size > 0
        ? static_cast<size_t>(config.output_size)
        : config.preprocessing.num_classes;
    if (class_count == 0) {
        class_count = static_cast<size_t>(counts.rbegin()->first + 1);
    }
    if (class_count == 0) {
        return false;
    }

    std::vector<float> weights(class_count, 0.0f);
    size_t total = 0;
    for (const auto& [label, count] : counts) {
        if (label < 0 || static_cast<size_t>(label) >= class_count) {
            spdlog::warn("{}: class_weight=balanced found label {} outside "
                         "class range 0..{}; using unweighted loss",
                         context, label, class_count - 1);
            return false;
        }
        total += count;
    }

    for (size_t cls = 0; cls < class_count; ++cls) {
        auto it = counts.find(static_cast<int64_t>(cls));
        if (it == counts.end() || it->second == 0) {
            spdlog::warn("{}: class_weight=balanced cannot compute a weight "
                         "for missing train class {}; using unweighted loss",
                         context, cls);
            return false;
        }
        weights[cls] = static_cast<float>(total) /
            static_cast<float>(class_count * it->second);
    }

    config.loss_params["class_weight"] = "manual";
    config.loss_params["class_weights"] = FormatFloatVector(weights);

    std::ostringstream dist;
    bool first = true;
    for (const auto& [label, count] : counts) {
        if (!first) dist << ", ";
        dist << label << ":" << count;
        first = false;
    }
    spdlog::info("{}: computed balanced CrossEntropy class weights from "
                 "train split counts [{}] -> {}",
                 context, dist.str(), config.loss_params["class_weights"]);
    return true;
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
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={}, num_workers={}, prefetch_factor={}, seed={})",
                 batch_size, config.shuffle, config.train_ratio,
                 config.is_time_series, num_workers, config.prefetch_factor,
                 config.dataloader_seed);

    std::string partition_col = config.is_time_series
        ? kSplitPartitionColumn : "";
    auto effective_dataset = dataset;
    if (config.stratified && !config.is_time_series) {
        auto partitioned = AddSplitPartitionColumn(
            dataset ? dataset->GetArrowTable() : nullptr,
            SplitPartitionOptions{
                label_column,
                config.train_ratio,
                config.val_ratio,
                config.test_ratio,
                config.shuffle,
                static_cast<uint32_t>(std::max(0, config.split_seed)),
                true,
                "TrainingExecutor Arrow"});
        if (partitioned.ok()) {
            effective_dataset = std::make_shared<ArrowDataset>(
                partitioned.ValueOrDie(),
                dataset ? dataset->GetName() + "_stratified_split"
                        : "stratified_split");
            partition_col = kSplitPartitionColumn;
        } else {
            spdlog::warn("TrainingExecutor: stratified Arrow DataSplit "
                         "could not be prepared ({}); falling back to "
                         "global train/validation/test slicing",
                         partitioned.status().ToString());
        }
    }
    const bool continuous_target = UsesContinuousTargetMetrics(config);
    const std::string effective_label =
        !config.target.primary_column.empty()
            ? config.target.primary_column
            : (config.is_time_series ? "y" : label_column);
    const float effective_val_ratio =
        !partition_col.empty() ? 0.0f :
        (config.has_data_split ? config.val_ratio : 0.0f);

    result.arrow_train = std::make_unique<ArrowDatasetBatcher>(
        effective_dataset, effective_label, batch_size,
        config.shuffle, config.train_ratio, true,
        partition_col, /*partition_value=*/0, num_workers,
        BatcherPhase::Train, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed),
        config.balance_classes && !continuous_target,
        config.balance_mode,
        config.balance_target,
        static_cast<uint32_t>(std::max(0, config.balance_seed)));
    result.arrow_val = std::make_unique<ArrowDatasetBatcher>(
        effective_dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/1, num_workers,
        BatcherPhase::Val, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed));
    result.arrow_test = std::make_unique<ArrowDatasetBatcher>(
        effective_dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/2, num_workers,
        BatcherPhase::Test, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed));

    if (continuous_target) {
        const size_t target_width =
            config.target.width > 0
                ? config.target.width
                : (config.is_time_series
                       ? config.time_series_target_width
                       : std::max<size_t>(1, config.output_size));
        result.arrow_train->SetRegressionTargetWidth(
            target_width, effective_label);
        result.arrow_val->SetRegressionTargetWidth(
            target_width, effective_label);
        result.arrow_test->SetRegressionTargetWidth(
            target_width, effective_label);
    }

    if (config.drop_last) {
        result.arrow_train->SetDropLast(true);
    }
    if (config.preprocessing.has_normalization) {
        result.arrow_train->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
        result.arrow_val->SetNormalization(config.preprocessing.norm_mean,
                                           config.preprocessing.norm_std);
        result.arrow_test->SetNormalization(config.preprocessing.norm_mean,
                                            config.preprocessing.norm_std);
    }

    if (continuous_target || UsesScalarBinaryTargets(config.loss_type)) {
        result.arrow_train->SetScalarLabelMode(true);
        result.arrow_val->SetScalarLabelMode(true);
        result.arrow_test->SetScalarLabelMode(true);
    } else if (RequiresClassIndexTargets(config)) {
        result.arrow_train->SetClassIndexLabelMode(true);
        result.arrow_val->SetClassIndexLabelMode(true);
        result.arrow_test->SetClassIndexLabelMode(true);
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
    AttachTrainingBatcherPrefetchWrappers(result, config, "Arrow");
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
                 "(batch_size={}, shuffle={}, train_ratio={:.2f}, time_series={}, num_workers={}, prefetch_factor={}, seed={})",
                 batch_size, config.shuffle, config.train_ratio,
                 config.is_time_series, num_workers, config.prefetch_factor,
                 config.dataloader_seed);

    const std::string partition_col = config.is_time_series
        ? kSplitPartitionColumn : "";
    if (config.stratified && !config.is_time_series) {
        spdlog::warn("TrainingExecutor: stratified DataSplit is not available "
                     "for disk-backed Parquet datasets yet; falling back to "
                     "row-group train/validation/test slicing");
    }
    const bool continuous_target = UsesContinuousTargetMetrics(config);
    if (config.balance_classes && !continuous_target) {
        spdlog::warn("TrainingExecutor: DataLoader class balancing is not "
                     "available for disk-backed Parquet datasets yet; training "
                     "uses the row-group train split without resampling");
    }
    const std::string effective_label =
        !config.target.primary_column.empty()
            ? config.target.primary_column
            : (config.is_time_series ? "y" : label_column);
    const float effective_val_ratio =
        config.has_data_split ? config.val_ratio : 0.0f;

    result.parquet_train = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        config.shuffle, config.train_ratio, true,
        partition_col, /*partition_value=*/0, num_workers,
        BatcherPhase::Train, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed));
    result.parquet_val = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/1, num_workers,
        BatcherPhase::Val, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed));
    result.parquet_test = std::make_unique<ParquetArrowBatcher>(
        dataset, effective_label, batch_size,
        false, config.train_ratio, false,
        partition_col, /*partition_value=*/2, num_workers,
        BatcherPhase::Test, effective_val_ratio,
        static_cast<uint32_t>(config.dataloader_seed));

    if (continuous_target) {
        const size_t target_width =
            config.target.width > 0
                ? config.target.width
                : (config.is_time_series
                       ? config.time_series_target_width
                       : std::max<size_t>(1, config.output_size));
        result.parquet_train->SetRegressionTargetWidth(
            target_width, effective_label);
        result.parquet_val->SetRegressionTargetWidth(
            target_width, effective_label);
        result.parquet_test->SetRegressionTargetWidth(
            target_width, effective_label);
    }

    if (config.drop_last) {
        result.parquet_train->SetDropLast(true);
    }
    if (config.preprocessing.has_normalization) {
        result.parquet_train->SetNormalization(config.preprocessing.norm_mean,
                                               config.preprocessing.norm_std);
        result.parquet_val->SetNormalization(config.preprocessing.norm_mean,
                                             config.preprocessing.norm_std);
        result.parquet_test->SetNormalization(config.preprocessing.norm_mean,
                                              config.preprocessing.norm_std);
    }

    if (continuous_target || UsesScalarBinaryTargets(config.loss_type)) {
        result.parquet_train->SetScalarLabelMode(true);
        result.parquet_val->SetScalarLabelMode(true);
        result.parquet_test->SetScalarLabelMode(true);
    } else if (RequiresClassIndexTargets(config)) {
        result.parquet_train->SetClassIndexLabelMode(true);
        result.parquet_val->SetClassIndexLabelMode(true);
        result.parquet_test->SetClassIndexLabelMode(true);
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
    AttachTrainingBatcherPrefetchWrappers(result, config, "Parquet");
    spdlog::info("TrainingExecutor: Parquet split samples train={} val={} test={}",
                 result.num_train_samples, result.num_val_samples, result.num_test_samples);
    return result;
}

ResolvedTabularBatcherBuildResult BuildResolvedTabularTrainingBatchers(
    TrainingConfiguration& config,
    const ResolvedTabularDatasets& datasets,
    int batch_size) {
    ResolvedTabularBatcherBuildResult result;
    const auto& partitions = config.dataset_roles;
    if (!partitions.train.IsSupplied()) {
        result.error_message =
            "resolved tabular partitions do not contain a Training source";
        return result;
    }

    if (!datasets.train_arrow && !datasets.train_parquet) {
        result.error_message = "Training Dataset '" +
            partitions.train.dataset_name +
            "' is not a registered Arrow or Parquet source";
        return result;
    }
    if (partitions.dev.IsSupplied() &&
        !datasets.dev_arrow && !datasets.dev_parquet) {
        result.error_message = "Validation Dataset '" +
            partitions.dev.dataset_name +
            "' is not a registered Arrow or Parquet source";
        return result;
    }
    if (partitions.test.IsSupplied() &&
        !datasets.test_arrow && !datasets.test_parquet) {
        result.error_message = "Test Dataset '" +
            partitions.test.dataset_name +
            "' is not a registered Arrow or Parquet source";
        return result;
    }

    config.train_ratio = partitions.policy.train_ratio;
    config.val_ratio = partitions.policy.dev_ratio;
    config.test_ratio = partitions.policy.test_ratio;
    config.split_seed = partitions.policy.seed;
    config.stratified = partitions.policy.stratified;
    config.has_data_split = true;

    // Construct derived roles from Train without prefetch wrappers. Supplied
    // roles are installed before wrappers are attached, keeping each wrapper
    // and its concrete owner in one final assembly step.
    auto assembly_config = config;
    assembly_config.prefetch_factor = 0;
    const bool external_dev = partitions.dev.IsSupplied();
    const bool external_test = partitions.test.IsSupplied();

    // ArrowDatasetBatcher and ParquetArrowBatcher express derived slices as
    // Train, then Val, then Test. If Dev is supplied but Test is not, build the
    // remaining Train-derived holdout in the Val slot and move that owner to
    // Test before installing the external Dev owner below.
    if (external_dev && !external_test) {
        assembly_config.val_ratio = partitions.policy.test_ratio;
        assembly_config.test_ratio = 0.0f;
    }
    result.batchers = datasets.train_arrow
        ? BuildArrowTrainingBatchers(
              assembly_config,
              datasets.train_arrow,
              partitions.train.label_column,
              batch_size)
        : BuildParquetTrainingBatchers(
              assembly_config,
              datasets.train_parquet,
              partitions.train.label_column,
              batch_size);

    if (external_dev && !external_test) {
        result.batchers.arrow_test = std::move(result.batchers.arrow_val);
        result.batchers.parquet_test = std::move(result.batchers.parquet_val);
        result.batchers.val = nullptr;
        result.batchers.test = result.batchers.arrow_test
            ? static_cast<IBatcher*>(result.batchers.arrow_test.get())
            : static_cast<IBatcher*>(result.batchers.parquet_test.get());
    }

    const int num_workers = ClampNumWorkersToPlatform(config.num_workers);
    const auto make_arrow_role = [&](const std::shared_ptr<ArrowDataset>& dataset,
                                     const std::string& label_column) {
        return std::make_unique<ArrowDatasetBatcher>(
            dataset, label_column, batch_size, false, 1.0f, true, "", 0,
            num_workers, BatcherPhase::Train, 0.0f,
            static_cast<uint32_t>(config.dataloader_seed));
    };
    const auto make_parquet_role = [&, batch_size](
        const std::shared_ptr<ParquetBackedDataset>& dataset,
        const std::string& label_column) {
        return std::make_unique<ParquetArrowBatcher>(
            dataset, label_column, batch_size, false, 1.0f, true, "", 0,
            num_workers, BatcherPhase::Train, 0.0f,
            static_cast<uint32_t>(config.dataloader_seed));
    };

    if (datasets.dev_arrow) {
        result.batchers.parquet_val.reset();
        result.batchers.arrow_val = make_arrow_role(
            datasets.dev_arrow, partitions.dev.label_column);
        result.batchers.val = result.batchers.arrow_val.get();
    } else if (datasets.dev_parquet) {
        result.batchers.arrow_val.reset();
        result.batchers.parquet_val = make_parquet_role(
            datasets.dev_parquet, partitions.dev.label_column);
        result.batchers.val = result.batchers.parquet_val.get();
    }
    if (datasets.test_arrow) {
        result.batchers.parquet_test.reset();
        result.batchers.arrow_test = make_arrow_role(
            datasets.test_arrow, partitions.test.label_column);
        result.batchers.test = result.batchers.arrow_test.get();
    } else if (datasets.test_parquet) {
        result.batchers.arrow_test.reset();
        result.batchers.parquet_test = make_parquet_role(
            datasets.test_parquet, partitions.test.label_column);
        result.batchers.test = result.batchers.parquet_test.get();
    }

    const auto apply_role_transforms = [&](IBatcher* batcher) {
        if (!batcher) return;
        if (config.preprocessing.has_normalization) {
            batcher->SetNormalization(config.preprocessing.norm_mean,
                                      config.preprocessing.norm_std);
        }
        if (config.is_time_series || UsesScalarBinaryTargets(config.loss_type)) {
            if (auto* arrow_batcher =
                    dynamic_cast<ArrowDatasetBatcher*>(batcher)) {
                arrow_batcher->SetScalarLabelMode(true);
            } else if (auto* parquet_batcher =
                           dynamic_cast<ParquetArrowBatcher*>(batcher)) {
                parquet_batcher->SetScalarLabelMode(true);
            }
        } else if (RequiresClassIndexTargets(config)) {
            batcher->SetClassIndexLabelMode(true);
        } else if (config.preprocessing.has_onehot) {
            batcher->SetOneHotEncoding(config.preprocessing.num_classes);
        } else {
            batcher->SetOneHotEncoding(config.output_size);
        }
    };
    if (partitions.dev.IsSupplied()) {
        apply_role_transforms(result.batchers.val);
    }
    if (partitions.test.IsSupplied()) {
        apply_role_transforms(result.batchers.test);
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
        result.batchers, config, "resolved tabular partitions");

    spdlog::info(
        "Resolved tabular partitions: train={} dev={} test={} samples; "
        "external dev={}, external test={}",
        result.batchers.num_train_samples,
        result.batchers.num_val_samples,
        result.batchers.num_test_samples,
        partitions.dev.IsSupplied(),
        partitions.test.IsSupplied());
    return result;
}

ResolvedExternalBatchers TakeResolvedExternalBatchers(TrainingBatcherSet batchers) {
    auto take = [](std::unique_ptr<IBatcher> prefetch,
                   std::unique_ptr<ArrowDatasetBatcher> arrow,
                   std::unique_ptr<ParquetArrowBatcher> parquet) {
        std::shared_ptr<IBatcher> source;
        if (arrow) {
            source = std::shared_ptr<IBatcher>(std::move(arrow));
        } else if (parquet) {
            source = std::shared_ptr<IBatcher>(std::move(parquet));
        }

        if (!prefetch) {
            return source;
        }

        // ApplyPrefetchWrappers initially uses a non-owning reference so the
        // concrete batchers remain inspectable in TrainingBatcherSet. When the
        // set crosses into resolved-role ownership, transfer that concrete
        // source into the wrapper before the local owners are destroyed.
        // Otherwise the wrapper retains a dangling pointer and crashes on its
        // first GetNumSamples/GetNextBatch call.
        auto* wrapper = static_cast<PrefetchBatcher*>(prefetch.get());
        wrapper->AdoptSourceOwnership(std::move(source));
        return std::shared_ptr<IBatcher>(std::move(prefetch));
    };

    ResolvedExternalBatchers resolved;
    resolved.train = take(std::move(batchers.prefetch_train),
                          std::move(batchers.arrow_train),
                          std::move(batchers.parquet_train));
    resolved.dev = take(std::move(batchers.prefetch_val),
                        std::move(batchers.arrow_val),
                        std::move(batchers.parquet_val));
    resolved.test = take(std::move(batchers.prefetch_test),
                         std::move(batchers.arrow_test),
                         std::move(batchers.parquet_test));
    return resolved;
}
} // namespace cyxwiz
