#include "sparse_feature_dataset.h"

#include <arrow/status.h>
#include <arrow/util/byte_size.h>

#include <cmath>
#include <limits>
#include <utility>

namespace cyxwiz {
namespace {

arrow::Status Invalid(const std::string& detail) {
    return arrow::Status::Invalid("SparseFeatureDataset: " + detail);
}

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t& result) {
    if (right > (std::numeric_limits<uint64_t>::max)() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t& result) {
    if (left != 0 && right > (std::numeric_limits<uint64_t>::max)() / left) {
        return false;
    }
    result = left * right;
    return true;
}

arrow::Result<uint64_t> SumStringBytes(
    const std::vector<std::string>& values) {
    uint64_t total = 0;
    for (const auto& value : values) {
        uint64_t next = 0;
        if (!CheckedAdd(total, static_cast<uint64_t>(value.size()), next)) {
            return Invalid("feature-name byte count overflows uint64");
        }
        total = next;
    }
    return total;
}

} // namespace

arrow::Result<std::shared_ptr<SparseFeatureDataset>>
SparseFeatureDataset::Create(Contents contents) {
    if (contents.name.empty()) {
        return Invalid("name must not be empty");
    }
    if (contents.num_rows < 0) {
        return Invalid("num_rows must be non-negative");
    }
    if (contents.num_features <= 0) {
        return Invalid("num_features must be positive");
    }

    constexpr int64_t kMaxCsrDimension =
        static_cast<int64_t>((std::numeric_limits<int32_t>::max)());
    if (contents.num_rows > kMaxCsrDimension) {
        return Invalid("num_rows exceeds the int32 CSR contract");
    }
    if (contents.num_features > kMaxCsrDimension) {
        return Invalid("num_features exceeds the int32 CSR contract");
    }

    const uint64_t expected_offsets =
        static_cast<uint64_t>(contents.num_rows) + 1;
    if (contents.row_offsets.size() != expected_offsets) {
        return Invalid("row_offsets length must equal num_rows + 1");
    }
    if (contents.row_offsets.empty() || contents.row_offsets.front() != 0) {
        return Invalid("row_offsets must start at zero");
    }
    if (contents.column_indices.size() != contents.values.size()) {
        return Invalid("column_indices and values must have equal length");
    }
    if (contents.values.size() >
        static_cast<size_t>((std::numeric_limits<int32_t>::max)())) {
        return Invalid("nnz exceeds the int32 CSR offset contract");
    }

    const int32_t expected_nnz = static_cast<int32_t>(contents.values.size());
    int32_t previous_offset = 0;
    for (size_t i = 0; i < contents.row_offsets.size(); ++i) {
        const int32_t offset = contents.row_offsets[i];
        if (offset < 0 || offset < previous_offset || offset > expected_nnz) {
            return Invalid(
                "row_offsets must be non-negative, monotonic, and bounded by nnz");
        }
        previous_offset = offset;
    }
    if (contents.row_offsets.back() != expected_nnz) {
        return Invalid("final row offset must equal nnz");
    }

    for (int64_t row = 0; row < contents.num_rows; ++row) {
        const size_t begin = static_cast<size_t>(
            contents.row_offsets[static_cast<size_t>(row)]);
        const size_t end = static_cast<size_t>(
            contents.row_offsets[static_cast<size_t>(row + 1)]);
        int32_t previous_column = -1;
        for (size_t index = begin; index < end; ++index) {
            const int32_t column = contents.column_indices[index];
            if (column < 0 || column >= contents.num_features) {
                return Invalid("column index is outside [0, num_features)");
            }
            if (column <= previous_column) {
                return Invalid(
                    "column indices must be strictly increasing within each row");
            }
            previous_column = column;
            if (!std::isfinite(contents.values[index]) ||
                contents.values[index] == 0.0f) {
                return Invalid("stored values must be finite and nonzero");
            }
        }
    }

    if (!contents.feature_names.empty() &&
        contents.feature_names.size() !=
            static_cast<size_t>(contents.num_features)) {
        return Invalid(
            "feature_names must be empty or contain exactly num_features entries");
    }
    if (contents.labels) {
        if (contents.label_name.empty()) {
            return Invalid("label_name is required when labels are present");
        }
        if (contents.labels->length() != contents.num_rows) {
            return Invalid("label length must equal num_rows");
        }
    } else if (!contents.label_name.empty()) {
        return Invalid("label_name must be empty when labels are absent");
    }

    uint64_t offset_bytes = 0;
    uint64_t column_bytes = 0;
    uint64_t value_bytes = 0;
    if (!CheckedMultiply(
            static_cast<uint64_t>(contents.row_offsets.size()),
            sizeof(RowOffset), offset_bytes) ||
        !CheckedMultiply(
            static_cast<uint64_t>(contents.column_indices.size()),
            sizeof(ColumnIndex), column_bytes) ||
        !CheckedMultiply(
            static_cast<uint64_t>(contents.values.size()),
            sizeof(float), value_bytes)) {
        return Invalid("CSR byte count overflows uint64");
    }

    uint64_t feature_storage_bytes = 0;
    if (!CheckedAdd(offset_bytes, column_bytes, feature_storage_bytes) ||
        !CheckedAdd(feature_storage_bytes, value_bytes,
                    feature_storage_bytes)) {
        return Invalid("CSR byte count overflows uint64");
    }

    uint64_t label_storage_bytes = 0;
    if (contents.labels) {
        const int64_t arrow_label_bytes =
            arrow::util::TotalBufferSize(*contents.labels);
        if (arrow_label_bytes < 0) {
            return Invalid("Arrow label buffer size is negative");
        }
        label_storage_bytes = static_cast<uint64_t>(arrow_label_bytes);
    }

    ARROW_ASSIGN_OR_RAISE(
        const uint64_t feature_name_bytes,
        SumStringBytes(contents.feature_names));
    uint64_t metadata_bytes = 0;
    if (!CheckedAdd(static_cast<uint64_t>(contents.name.size()),
                    static_cast<uint64_t>(contents.label_name.size()),
                    metadata_bytes) ||
        !CheckedAdd(metadata_bytes, feature_name_bytes, metadata_bytes)) {
        return Invalid("metadata byte count overflows uint64");
    }

    uint64_t estimated_host_memory_bytes = 0;
    if (!CheckedAdd(feature_storage_bytes, label_storage_bytes,
                    estimated_host_memory_bytes) ||
        !CheckedAdd(estimated_host_memory_bytes, metadata_bytes,
                    estimated_host_memory_bytes)) {
        return Invalid("estimated host memory byte count overflows uint64");
    }

    uint64_t dense_elements = 0;
    uint64_t dense_feature_bytes = 0;
    if (!CheckedMultiply(static_cast<uint64_t>(contents.num_rows),
                         static_cast<uint64_t>(contents.num_features),
                         dense_elements) ||
        !CheckedMultiply(dense_elements, sizeof(float), dense_feature_bytes)) {
        return Invalid("dense comparison byte count overflows uint64");
    }

    const long double dense_slots =
        static_cast<long double>(contents.num_rows) *
        static_cast<long double>(contents.num_features);
    const double density = dense_slots == 0.0L
        ? 0.0
        : static_cast<double>(
              static_cast<long double>(contents.values.size()) / dense_slots);

    return std::shared_ptr<SparseFeatureDataset>(new SparseFeatureDataset(
        std::move(contents), density, feature_storage_bytes,
        label_storage_bytes, estimated_host_memory_bytes,
        dense_feature_bytes));
}

SparseFeatureDataset::SparseFeatureDataset(
    Contents contents,
    double density,
    uint64_t feature_storage_bytes,
    uint64_t label_storage_bytes,
    uint64_t estimated_host_memory_bytes,
    uint64_t dense_feature_bytes)
    : name_(std::move(contents.name)),
      num_rows_(contents.num_rows),
      num_features_(contents.num_features),
      row_offsets_(std::move(contents.row_offsets)),
      column_indices_(std::move(contents.column_indices)),
      values_(std::move(contents.values)),
      feature_names_(std::move(contents.feature_names)),
      labels_(std::move(contents.labels)),
      label_name_(std::move(contents.label_name)),
      density_(density),
      feature_storage_bytes_(feature_storage_bytes),
      label_storage_bytes_(label_storage_bytes),
      estimated_host_memory_bytes_(estimated_host_memory_bytes),
      dense_feature_bytes_(dense_feature_bytes) {}

} // namespace cyxwiz
