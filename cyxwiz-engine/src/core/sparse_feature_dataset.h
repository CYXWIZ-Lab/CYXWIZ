#pragma once

#include <arrow/chunked_array.h>
#include <arrow/result.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

/**
 * Immutable host-side CSR feature dataset.
 *
 * This is deliberately narrower than a general sparse Tensor. It owns text
 * feature rows between pipeline materialization and training ingress. Device
 * execution remains owned by the selected ArrayFire backend at the batch/model
 * boundary.
 *
 * CSR offsets and column indices are int32 because that is the direct host
 * contract accepted by ArrayFire sparse construction. Create() validates the
 * complete canonical representation before publishing an instance.
 */
class SparseFeatureDataset {
public:
    using RowOffset = int32_t;
    using ColumnIndex = int32_t;

    struct Contents {
        std::string name;
        int64_t num_rows = 0;
        int64_t num_features = 0;
        std::vector<RowOffset> row_offsets;
        std::vector<ColumnIndex> column_indices;
        std::vector<float> values;
        std::vector<std::string> feature_names;
        std::shared_ptr<arrow::ChunkedArray> labels;
        std::string label_name;
    };

    static arrow::Result<std::shared_ptr<SparseFeatureDataset>> Create(
        Contents contents);

    const std::string& GetName() const noexcept { return name_; }
    int64_t GetNumRows() const noexcept { return num_rows_; }
    int64_t GetNumFeatures() const noexcept { return num_features_; }
    int64_t GetNnz() const noexcept {
        return static_cast<int64_t>(values_.size());
    }
    double GetDensity() const noexcept { return density_; }

    const std::vector<RowOffset>& GetRowOffsets() const noexcept {
        return row_offsets_;
    }
    const std::vector<ColumnIndex>& GetColumnIndices() const noexcept {
        return column_indices_;
    }
    const std::vector<float>& GetValues() const noexcept { return values_; }
    const std::vector<std::string>& GetFeatureNames() const noexcept {
        return feature_names_;
    }
    const std::shared_ptr<const arrow::ChunkedArray>& GetLabels() const noexcept {
        return labels_;
    }
    const std::string& GetLabelName() const noexcept { return label_name_; }

    // Logical owned bytes. Container capacity and allocator bookkeeping are
    // intentionally excluded so this remains deterministic for diagnostics.
    uint64_t GetFeatureStorageBytes() const noexcept {
        return feature_storage_bytes_;
    }
    uint64_t GetLabelStorageBytes() const noexcept {
        return label_storage_bytes_;
    }
    uint64_t GetEstimatedHostMemoryBytes() const noexcept {
        return estimated_host_memory_bytes_;
    }
    uint64_t GetDenseFeatureBytes() const noexcept {
        return dense_feature_bytes_;
    }

private:
    SparseFeatureDataset(Contents contents,
                         double density,
                         uint64_t feature_storage_bytes,
                         uint64_t label_storage_bytes,
                         uint64_t estimated_host_memory_bytes,
                         uint64_t dense_feature_bytes);

    std::string name_;
    int64_t num_rows_ = 0;
    int64_t num_features_ = 0;
    std::vector<RowOffset> row_offsets_;
    std::vector<ColumnIndex> column_indices_;
    std::vector<float> values_;
    std::vector<std::string> feature_names_;
    std::shared_ptr<const arrow::ChunkedArray> labels_;
    std::string label_name_;
    double density_ = 0.0;
    uint64_t feature_storage_bytes_ = 0;
    uint64_t label_storage_bytes_ = 0;
    uint64_t estimated_host_memory_bytes_ = 0;
    uint64_t dense_feature_bytes_ = 0;
};

} // namespace cyxwiz
