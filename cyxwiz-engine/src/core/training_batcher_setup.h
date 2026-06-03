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
    std::unique_ptr<ParquetArrowBatcher> parquet_train;
    std::unique_ptr<ParquetArrowBatcher> parquet_val;
    IBatcher* train = nullptr;
    IBatcher* val = nullptr;
    size_t num_train_samples = 0;
};

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
