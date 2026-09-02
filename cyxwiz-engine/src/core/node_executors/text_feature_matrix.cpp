#include "text_feature_matrix.h"

#include "../materialization_memory_guard.h"
#include "../sparse_feature_dataset.h"

#include <arrow/api.h>
#include <arrow/builder.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace cyxwiz {
namespace {

arrow::Status Invalid(const std::string& detail) {
    return arrow::Status::Invalid("TextFeatureMatrix: " + detail);
}

arrow::Status ValidateMatrix(const TextFeatureMatrix& matrix) {
    if (matrix.num_rows < 0 || matrix.num_features <= 0) {
        return Invalid("shape must contain non-negative rows and positive features");
    }
    const uint64_t expected_offsets =
        static_cast<uint64_t>(matrix.num_rows) + 1;
    if (matrix.row_offsets.size() != expected_offsets ||
        matrix.row_offsets.empty() || matrix.row_offsets.front() != 0) {
        return Invalid("row offsets do not match the matrix shape");
    }
    if (matrix.column_indices.size() != matrix.values.size() ||
        matrix.row_offsets.back() != matrix.values.size()) {
        return Invalid("column/value storage does not match row offsets");
    }
    if (matrix.feature_names.size() !=
        static_cast<size_t>(matrix.num_features)) {
        return Invalid("feature names must match the feature width");
    }
    if (!matrix.labels.empty() &&
        matrix.labels.size() != static_cast<size_t>(matrix.num_rows)) {
        return Invalid("labels must be empty or match the row count");
    }
    if (matrix.labels.empty() != matrix.label_name.empty()) {
        return Invalid("label name and label values must be present together");
    }

    for (int64_t row = 0; row < matrix.num_rows; ++row) {
        const uint64_t begin = matrix.row_offsets[static_cast<size_t>(row)];
        const uint64_t end = matrix.row_offsets[static_cast<size_t>(row + 1)];
        if (begin > end || end > matrix.values.size()) {
            return Invalid("row offsets must be monotonic and bounded by nnz");
        }
        int32_t previous_column = -1;
        for (uint64_t index = begin; index < end; ++index) {
            const int32_t column =
                matrix.column_indices[static_cast<size_t>(index)];
            const float value = matrix.values[static_cast<size_t>(index)];
            if (column < 0 ||
                static_cast<int64_t>(column) >= matrix.num_features ||
                column <= previous_column) {
                return Invalid(
                    "column indices must be in range and strictly ordered per row");
            }
            if (!std::isfinite(value) || value == 0.0f) {
                return Invalid("stored values must be finite and nonzero");
            }
            previous_column = column;
        }
    }
    return arrow::Status::OK();
}

} // namespace

arrow::Status AppendNormalizedTextFeatureRow(
    TextFeatureMatrix& matrix,
    std::vector<TextFeatureEntry> entries,
    const std::string& norm) {
    if (matrix.row_offsets.empty()) {
        matrix.row_offsets.push_back(0);
    }

    entries.erase(
        std::remove_if(entries.begin(), entries.end(),
                       [](const TextFeatureEntry& entry) {
                           return entry.value == 0.0f;
                       }),
        entries.end());
    std::sort(entries.begin(), entries.end(),
              [](const TextFeatureEntry& left,
                 const TextFeatureEntry& right) {
                  return left.column < right.column;
              });

    double denominator = 0.0;
    if (norm == "l1") {
        for (const auto& entry : entries) {
            denominator += std::abs(static_cast<double>(entry.value));
        }
    } else if (norm == "l2") {
        for (const auto& entry : entries) {
            denominator += static_cast<double>(entry.value) * entry.value;
        }
        denominator = std::sqrt(denominator);
    } else if (norm != "none") {
        return Invalid("norm must be l1, l2, or none");
    }

    int32_t previous_column = -1;
    for (auto& entry : entries) {
        if (entry.column < 0 ||
            static_cast<int64_t>(entry.column) >= matrix.num_features ||
            entry.column <= previous_column || !std::isfinite(entry.value)) {
            return Invalid("row contains invalid or duplicate feature entries");
        }
        if (denominator > 0.0) {
            entry.value = static_cast<float>(
                static_cast<double>(entry.value) / denominator);
        }
        if (!std::isfinite(entry.value) || entry.value == 0.0f) {
            return Invalid("normalization produced an invalid stored value");
        }
        matrix.column_indices.push_back(entry.column);
        matrix.values.push_back(entry.value);
        previous_column = entry.column;
    }
    matrix.row_offsets.push_back(
        static_cast<uint64_t>(matrix.values.size()));
    return arrow::Status::OK();
}

MaterializationMemoryEstimate EstimateSparseTextFeatureMemory(
    uint64_t rows,
    uint64_t nnz,
    bool has_labels) {
    MaterializationMemoryEstimate estimate;
    estimate.rows = rows;
    estimate.output_features = nnz;
    estimate.bytes_per_value = sizeof(int32_t) + sizeof(float);
    estimate.confidence = nnz == 0 ? "low" : "medium";

    uint64_t offset_count = 0;
    uint64_t offsets = 0;
    uint64_t entries = 0;
    uint64_t labels = 0;
    estimate.overflow =
        !CheckedAddU64(rows, 1, offset_count) ||
        !CheckedMulU64(offset_count, sizeof(int32_t), offsets) ||
        !CheckedMulU64(nnz, sizeof(int32_t) + sizeof(float), entries) ||
        (has_labels &&
         !CheckedMulU64(rows, sizeof(int32_t), labels));
    uint64_t raw = 0;
    if (!CheckedAddU64(offsets, entries, raw) ||
        !CheckedAddU64(raw, labels, raw)) {
        estimate.overflow = true;
    }
    estimate.raw_output_bytes = raw;
    // Account for uint64 intermediate offsets plus bounded publication
    // bookkeeping. Term-count maps are observed by the runtime process-memory
    // guard because their allocator overhead is data-dependent.
    estimate.temporary_bytes = SaturatingScaleBytes(raw, 1.0);
    estimate.arrow_overhead_bytes = has_labels
        ? SaturatingScaleBytes(labels, 0.125)
        : 0;
    uint64_t peak = 0;
    if (!CheckedAddU64(estimate.raw_output_bytes,
                       estimate.temporary_bytes, peak) ||
        !CheckedAddU64(peak, estimate.arrow_overhead_bytes, peak)) {
        estimate.overflow = true;
    }
    estimate.estimated_peak_bytes = estimate.overflow
        ? (std::numeric_limits<uint64_t>::max)()
        : peak;
    return estimate;
}

arrow::Result<std::shared_ptr<arrow::Table>> BuildDenseTextFeatureTable(
    const TextFeatureMatrix& matrix,
    const std::string& feature_prefix) {
    ARROW_RETURN_NOT_OK(ValidateMatrix(matrix));
    if (feature_prefix.empty()) {
        return Invalid("dense feature prefix must not be empty");
    }

    auto* pool = arrow::default_memory_pool();
    std::vector<std::unique_ptr<arrow::FloatBuilder>> builders;
    builders.reserve(static_cast<size_t>(matrix.num_features));
    for (int64_t feature = 0; feature < matrix.num_features; ++feature) {
        auto builder = std::make_unique<arrow::FloatBuilder>(pool);
        ARROW_RETURN_NOT_OK(builder->Reserve(matrix.num_rows));
        builders.push_back(std::move(builder));
    }

    for (int64_t row = 0; row < matrix.num_rows; ++row) {
        size_t cursor = static_cast<size_t>(
            matrix.row_offsets[static_cast<size_t>(row)]);
        const size_t end = static_cast<size_t>(
            matrix.row_offsets[static_cast<size_t>(row + 1)]);
        for (int64_t feature = 0; feature < matrix.num_features; ++feature) {
            float value = 0.0f;
            if (cursor < end && matrix.column_indices[cursor] == feature) {
                value = matrix.values[cursor++];
            }
            ARROW_RETURN_NOT_OK(
                builders[static_cast<size_t>(feature)]->Append(value));
        }
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    arrays.reserve(builders.size() + (matrix.labels.empty() ? 0 : 1));
    fields.reserve(arrays.capacity());
    for (size_t feature = 0; feature < builders.size(); ++feature) {
        ARROW_ASSIGN_OR_RAISE(auto array, builders[feature]->Finish());
        arrays.push_back(std::move(array));
        fields.push_back(arrow::field(
            feature_prefix + std::to_string(feature), arrow::float32()));
    }
    if (!matrix.labels.empty()) {
        arrow::Int32Builder label_builder(pool);
        ARROW_RETURN_NOT_OK(label_builder.AppendValues(matrix.labels));
        ARROW_ASSIGN_OR_RAISE(auto labels, label_builder.Finish());
        arrays.push_back(std::move(labels));
        fields.push_back(arrow::field(matrix.label_name, arrow::int32()));
    }
    return arrow::Table::Make(
        arrow::schema(std::move(fields)), std::move(arrays), matrix.num_rows);
}

arrow::Result<std::shared_ptr<SparseFeatureDataset>>
BuildSparseTextFeatureDataset(TextFeatureMatrix matrix,
                              const std::string& dataset_name) {
    ARROW_RETURN_NOT_OK(ValidateMatrix(matrix));
    if (dataset_name.empty()) {
        return Invalid("sparse dataset name must not be empty");
    }
    constexpr uint64_t kInt32Max = static_cast<uint64_t>(
        (std::numeric_limits<int32_t>::max)());
    if (static_cast<uint64_t>(matrix.num_rows) > kInt32Max ||
        static_cast<uint64_t>(matrix.num_features) > kInt32Max ||
        matrix.values.size() > kInt32Max) {
        return Invalid("matrix exceeds the int32 sparse publication contract");
    }

    SparseFeatureDataset::Contents contents;
    contents.name = dataset_name;
    contents.num_rows = matrix.num_rows;
    contents.num_features = matrix.num_features;
    contents.row_offsets.reserve(matrix.row_offsets.size());
    for (uint64_t offset : matrix.row_offsets) {
        if (offset > kInt32Max) {
            return Invalid("row offset exceeds the int32 sparse contract");
        }
        contents.row_offsets.push_back(static_cast<int32_t>(offset));
    }
    contents.column_indices = std::move(matrix.column_indices);
    contents.values = std::move(matrix.values);
    contents.feature_names = std::move(matrix.feature_names);
    if (!matrix.labels.empty()) {
        arrow::Int32Builder label_builder;
        ARROW_RETURN_NOT_OK(label_builder.AppendValues(matrix.labels));
        ARROW_ASSIGN_OR_RAISE(auto labels, label_builder.Finish());
        contents.labels = std::make_shared<arrow::ChunkedArray>(labels);
        contents.label_name = std::move(matrix.label_name);
    }
    return SparseFeatureDataset::Create(std::move(contents));
}

} // namespace cyxwiz
