#pragma once

#include "dataset_batcher.h"
#include "graph_compiler.h"
#include "parquet_arrow_batcher.h"

#include <cstddef>
#include <memory>
#include <string>

namespace cyxwiz {

class ArrowDataset;
class ParquetBackedDataset;

struct TrainingBatcherSet {
    std::unique_ptr<ArrowDatasetBatcher> arrow_train;
    std::unique_ptr<ArrowDatasetBatcher> arrow_val;
    std::unique_ptr<ArrowDatasetBatcher> arrow_test;
    std::unique_ptr<ParquetArrowBatcher> parquet_train;
    std::unique_ptr<ParquetArrowBatcher> parquet_val;
    std::unique_ptr<ParquetArrowBatcher> parquet_test;
    std::unique_ptr<IBatcher> prefetch_train;
    std::unique_ptr<IBatcher> prefetch_val;
    std::unique_ptr<IBatcher> prefetch_test;
    IBatcher* train = nullptr;
    IBatcher* val = nullptr;
    IBatcher* test = nullptr;
    size_t num_train_samples = 0;
    size_t num_val_samples = 0;
    size_t num_test_samples = 0;
};

struct TrainingInputSizeResolution {
    size_t input_size = 0;
    bool used_compiled_override = false;
    bool has_separate_label_column = false;
};

TrainingInputSizeResolution ResolveTabularTrainingInputSize(
    const TrainingConfiguration& config,
    size_t num_columns);

TrainingBatcherSet BuildArrowTrainingBatchers(
    const TrainingConfiguration& config,
    std::shared_ptr<ArrowDataset> dataset,
    const std::string& label_column,
    int batch_size);

TrainingBatcherSet BuildParquetTrainingBatchers(
    const TrainingConfiguration& config,
    std::shared_ptr<ParquetBackedDataset> dataset,
    const std::string& label_column,
    int batch_size);

} // namespace cyxwiz
