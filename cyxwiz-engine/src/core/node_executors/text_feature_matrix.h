#pragma once

#include "../materialization_memory_types.h"

#include <arrow/result.h>
#include <arrow/status.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

class SparseFeatureDataset;

struct TextFeatureEntry {
    int32_t column = 0;
    float value = 0.0f;
};

/**
 * Representation-neutral vectorizer output assembled once per input row.
 * Offsets stay uint64 internally so dense output does not inherit ArrayFire's
 * int32 CSR limit. Sparse publication performs explicit checked narrowing.
 */
struct TextFeatureMatrix {
    int64_t num_rows = 0;
    int64_t num_features = 0;
    std::vector<uint64_t> row_offsets;
    std::vector<int32_t> column_indices;
    std::vector<float> values;
    std::vector<std::string> feature_names;
    std::vector<int32_t> labels;
    std::string label_name;
};

struct TextVectorizerMaterialization {
    std::shared_ptr<arrow::Table> dense_table;
    std::shared_ptr<SparseFeatureDataset> sparse_dataset;
};

arrow::Status AppendNormalizedTextFeatureRow(
    TextFeatureMatrix& matrix,
    std::vector<TextFeatureEntry> entries,
    const std::string& norm);

MaterializationMemoryEstimate EstimateSparseTextFeatureMemory(
    uint64_t rows,
    uint64_t nnz,
    bool has_labels);

arrow::Result<std::shared_ptr<arrow::Table>> BuildDenseTextFeatureTable(
    const TextFeatureMatrix& matrix,
    const std::string& feature_prefix);

arrow::Result<std::shared_ptr<SparseFeatureDataset>>
BuildSparseTextFeatureDataset(TextFeatureMatrix matrix,
                              const std::string& dataset_name);

} // namespace cyxwiz
