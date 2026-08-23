#include "dataset_batcher.h"
#include "label_column_resolver.h"
#include "split_partitioning.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace cyxwiz {

namespace {

constexpr size_t kBatchInspectionColumnPreviewLimit = 32;

bool IsTokenSlotColumnName(const std::string& name) {
    if (name.rfind("tok_", 0) != 0 || name.size() == 4) {
        return false;
    }
    return std::all_of(
        name.begin() + 4,
        name.end(),
        [](unsigned char ch) { return std::isdigit(ch) != 0; });
}

bool TryIsNullAt(const std::shared_ptr<arrow::ChunkedArray>& column,
                 int64_t global_row,
                 bool& is_null) {
    if (!column || global_row < 0 || global_row >= column->length()) {
        return false;
    }
    int64_t offset = 0;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto& chunk = column->chunk(chunk_index);
        if (!chunk) {
            return false;
        }
        if (global_row < offset + chunk->length()) {
            is_null = chunk->IsNull(global_row - offset);
            return true;
        }
        offset += chunk->length();
    }
    return false;
}

void PopulateBatchInspection(
    Batch& batch,
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<int>& feature_columns,
    const std::vector<int>& label_columns,
    const std::vector<int64_t>& row_indices,
    size_t batch_start,
    size_t actual_batch_size) {
    auto& inspection = batch.inspection;
    inspection.available = table && table->schema();
    inspection.row_count = actual_batch_size;
    inspection.feature_column_count = feature_columns.size();
    inspection.label_column_count = label_columns.size();
    if (!inspection.available) {
        return;
    }

    const auto schema = table->schema();
    inspection.feature_columns_truncated =
        feature_columns.size() > kBatchInspectionColumnPreviewLimit;
    inspection.label_columns_truncated =
        label_columns.size() > kBatchInspectionColumnPreviewLimit;
    inspection.token_sequence_columns = !feature_columns.empty();

    const auto append_preview = [&](const std::vector<int>& columns,
                                    std::vector<BatchColumnInspection>& preview,
                                    bool inspect_token_names) {
        const size_t count = std::min(
            columns.size(), kBatchInspectionColumnPreviewLimit);
        preview.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const int column_index = columns[i];
            if (column_index < 0 || column_index >= schema->num_fields()) {
                continue;
            }
            const auto& field = schema->field(column_index);
            preview.push_back({field->name(), field->type()->ToString()});
        }
        if (inspect_token_names) {
            for (int column_index : columns) {
                if (column_index < 0 || column_index >= schema->num_fields() ||
                    !IsTokenSlotColumnName(schema->field(column_index)->name())) {
                    inspection.token_sequence_columns = false;
                    break;
                }
            }
        }
    };
    append_preview(
        feature_columns, inspection.feature_columns_preview, true);
    append_preview(label_columns, inspection.label_columns_preview, false);

    inspection.null_summary_available = true;
    const auto count_nulls = [&](const std::vector<int>& columns,
                                 uint64_t& null_count) {
        for (int column_index : columns) {
            if (column_index < 0 || column_index >= table->num_columns()) {
                inspection.null_summary_available = false;
                return;
            }
            const auto& column = table->column(column_index);
            for (size_t row = 0; row < actual_batch_size; ++row) {
                const size_t position = batch_start + row;
                if (position >= row_indices.size()) {
                    inspection.null_summary_available = false;
                    return;
                }
                bool is_null = false;
                if (!TryIsNullAt(column, row_indices[position], is_null)) {
                    inspection.null_summary_available = false;
                    return;
                }
                ++inspection.inspected_value_count;
                if (is_null) {
                    ++null_count;
                }
            }
        }
    };
    count_nulls(feature_columns, inspection.feature_null_count);
    if (inspection.null_summary_available) {
        count_nulls(label_columns, inspection.label_null_count);
    }
}

float ReadNumericChunkedValue(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    int64_t global_row_idx) {
    if (!column || global_row_idx < 0) return 0.0f;
    int64_t offset = 0;
    for (int chunk_index = 0; chunk_index < column->num_chunks(); ++chunk_index) {
        const auto& chunk = column->chunk(chunk_index);
        const int64_t chunk_length = chunk->length();
        if (global_row_idx >= offset + chunk_length) {
            offset += chunk_length;
            continue;
        }
        const int64_t local = global_row_idx - offset;
        if (chunk->IsNull(local)) return 0.0f;
        switch (chunk->type_id()) {
        case arrow::Type::FLOAT:
            return std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(local);
        case arrow::Type::DOUBLE:
            return static_cast<float>(
                std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(local));
        case arrow::Type::INT64:
            return static_cast<float>(
                std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(local));
        case arrow::Type::INT32:
            return static_cast<float>(
                std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(local));
        case arrow::Type::INT16:
            return static_cast<float>(
                std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(local));
        case arrow::Type::INT8:
            return static_cast<float>(
                std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(local));
        case arrow::Type::UINT64:
            return static_cast<float>(
                std::static_pointer_cast<arrow::UInt64Array>(chunk)->Value(local));
        case arrow::Type::UINT32:
            return static_cast<float>(
                std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(local));
        case arrow::Type::UINT16:
            return static_cast<float>(
                std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(local));
        case arrow::Type::UINT8:
            return static_cast<float>(
                std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(local));
        default:
            return 0.0f;
        }
    }
    return 0.0f;
}

}  // namespace

// ArrowDatasetBatcher Implementation

ArrowDatasetBatcher::ArrowDatasetBatcher(
    std::shared_ptr<ArrowDataset> dataset,
    const std::string& label_column,
    size_t batch_size,
    bool shuffle,
    float train_split,
    bool is_training,
    const std::string& partition_column,
    int partition_value,
    int num_workers,
    BatcherPhase split_phase,
    float val_split,
    uint32_t seed,
    bool balance_classes,
    const std::string& balance_mode,
    const std::string& balance_target,
    uint32_t balance_seed)
    : dataset_(dataset)
    , label_column_(label_column)
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , is_training_(is_training)
    , num_workers_(std::max(0, num_workers))
    , partition_column_(partition_column)
    , partition_value_(partition_value)
    , split_phase_(split_phase)
    , val_split_(val_split)
    , rng_(seed)
    , balance_rng_(balance_seed)
    , balance_seed_(balance_seed)
    , balance_classes_(balance_classes)
    , balance_mode_(balance_mode)
    , balance_target_(balance_target)
{
    if (!dataset_) {
        spdlog::error("ArrowDatasetBatcher: Invalid dataset");
        return;
    }

    int64_t num_rows = dataset_->GetNumRows();

    // Phase 4 Time-Series: if a partition column was specified, walk the
    // column and collect row indices whose int8 partition value matches
    // partition_value_. Bypasses the train_split first-N% slicing so
    // each time-series batcher pulls from a disjoint, contiguous slice
    // of the windowed table that was assigned by TimeSeriesSplitOperator.
    bool partition_filtered = false;
    if (!partition_column_.empty()) {
        auto table = dataset_->GetArrowTable();
        if (table) {
            auto column = table->GetColumnByName(partition_column_);
            if (!column || column->num_chunks() == 0) {
                spdlog::warn("ArrowDatasetBatcher: partition_column '{}' not found in table - "
                             "falling back to train_split slicing",
                             partition_column_);
            } else {
                // Validate all chunks are int8 up front so we don't
                // collect a partial index set and then throw it away on
                // a mid-walk type mismatch.
                bool all_int8 = true;
                for (int c = 0; c < column->num_chunks(); ++c) {
                    if (column->chunk(c)->type_id() != arrow::Type::INT8) {
                        spdlog::warn("ArrowDatasetBatcher: partition_column '{}' chunk {} has type {} "
                                     "(expected int8) - falling back to train_split slicing",
                                     partition_column_, c,
                                     column->chunk(c)->type()->ToString());
                        all_int8 = false;
                        break;
                    }
                }
                if (all_int8) {
                    int64_t offset = 0;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto chunk = column->chunk(c);
                        const int64_t chunk_len = chunk->length();
                        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                        const int8_t* data = arr->raw_values();
                        for (int64_t i = 0; i < chunk_len; ++i) {
                            if (static_cast<int>(data[i]) == partition_value_) {
                                indices_.push_back(offset + i);
                            }
                        }
                        offset += chunk_len;
                    }
                    partition_filtered = true;
                    spdlog::info("ArrowDatasetBatcher: partition '{}'={} selected {} / {} rows",
                                 partition_column_, partition_value_, indices_.size(), num_rows);
                }
            }
        }
    }

    if (!partition_filtered) {
        const float safe_train_split = std::clamp(train_split, 0.0f, 1.0f);
        const float safe_val_split = std::clamp(val_split, 0.0f, 1.0f);
        const int64_t train_count = static_cast<int64_t>(num_rows * safe_train_split);
        const int64_t val_count = safe_val_split > 0.0f
            ? static_cast<int64_t>(num_rows * safe_val_split)
            : (num_rows - train_count);
        const int64_t val_end = std::min(num_rows, train_count + std::max<int64_t>(0, val_count));

        BatcherPhase effective_phase = split_phase_;
        if (val_split_ <= 0.0f && split_phase_ == BatcherPhase::Train && !is_training_) {
            effective_phase = BatcherPhase::Val;
        }
        split_phase_ = effective_phase;

        if (effective_phase == BatcherPhase::Train) {
            indices_.reserve(static_cast<size_t>(train_count));
            for (int64_t i = 0; i < train_count && i < num_rows; ++i) {
                indices_.push_back(i);
            }
        } else if (effective_phase == BatcherPhase::Val) {
            indices_.reserve(static_cast<size_t>(std::max<int64_t>(0, val_end - train_count)));
            for (int64_t i = train_count; i < val_end; ++i) {
                indices_.push_back(i);
            }
        } else {
            indices_.reserve(static_cast<size_t>(std::max<int64_t>(0, num_rows - val_end)));
            for (int64_t i = val_end; i < num_rows; ++i) {
                indices_.push_back(i);
            }
        }
    }

    // Initialize feature/label column indices
    InitializeColumns();
    base_indices_ = indices_;
    RebuildBalancedIndices();

    const char* phase_name = split_phase_ == BatcherPhase::Train ? "train" :
        (split_phase_ == BatcherPhase::Val ? "val" : "test");
    spdlog::info("ArrowDatasetBatcher: {} split, {} samples, {} features, batch_size={}, shuffle={}, num_workers={}",
                 phase_name, indices_.size(), num_features_, batch_size_, shuffle_, num_workers_);

    if (shuffle_) {
        ShuffleIndices();
    }
}

void ArrowDatasetBatcher::InitializeColumns() {
    if (!dataset_) return;

    auto schema = dataset_->GetSchema();
    if (!schema) return;

    feature_cols_.clear();
    label_col_idx_ = -1;
    label_col_indices_.clear();

    // First pass: find explicit label column or auto-detect
    for (int i = 0; i < schema->num_fields(); ++i) {
        auto field = schema->field(i);
        std::string name = field->name();

        // Check for explicit match
        if (!label_column_.empty() && name == label_column_) {
            label_col_idx_ = i;
            spdlog::info("ArrowDatasetBatcher: Found explicit label column '{}' at index {}", name, i);
        }
        // Auto-detect common label names if no explicit label given
        else if (label_column_.empty() && label_col_idx_ < 0) {
            if (IsCommonLabelColumnName(name)) {
                label_col_idx_ = i;
                spdlog::info("ArrowDatasetBatcher: Auto-detected label column '{}' at index {}", name, i);
            }
        }
    }
    if (!label_column_.empty() && label_col_idx_ < 0) {
        spdlog::warn("ArrowDatasetBatcher: explicit label column '{}' not found; "
                     "falling back to common label-name auto-detection",
                     label_column_);
        label_col_idx_ = FindCommonLabelColumnIndex(schema);
        if (label_col_idx_ >= 0) {
            spdlog::info("ArrowDatasetBatcher: Auto-detected fallback label "
                         "column '{}' at index {}",
                         schema->field(label_col_idx_)->name(), label_col_idx_);
        }
    }

    if (label_col_idx_ >= 0) {
        label_col_indices_.push_back(label_col_idx_);
    }

    // Second pass: collect feature columns (all numeric except labels)
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (std::find(label_col_indices_.begin(), label_col_indices_.end(), i) !=
            label_col_indices_.end()) continue;

        auto field = schema->field(i);
        // Skip internal metadata columns (double-underscore prefix) and
        // the partition column specifically. Phase 4's TimeSeriesSplit
        // emits a `__partition__` int8 column which the constructor
        // already uses to build indices_; it must not leak into the
        // feature tensor or Dense layers crash on the dimension mismatch.
        const std::string& fname = field->name();
        if (fname.rfind("__", 0) == 0 ||
            (!partition_column_.empty() && fname == partition_column_)) {
            continue;
        }

        auto type = field->type();
        if (type->id() == arrow::Type::DOUBLE ||
            type->id() == arrow::Type::FLOAT ||
            type->id() == arrow::Type::INT64 ||
            type->id() == arrow::Type::INT32 ||
            type->id() == arrow::Type::INT16 ||
            type->id() == arrow::Type::INT8 ||
            type->id() == arrow::Type::UINT64 ||
            type->id() == arrow::Type::UINT32 ||
            type->id() == arrow::Type::UINT16 ||
            type->id() == arrow::Type::UINT8) {
            feature_cols_.push_back(i);
        }
    }

    num_features_ = feature_cols_.size();
    spdlog::info("ArrowDatasetBatcher: {} feature columns, label_col={}",
                 num_features_, label_col_idx_);
}

void ArrowDatasetBatcher::SetRegressionTargetWidth(
    size_t width, const std::string& target_base) {
    if (width < 1) {
        throw std::invalid_argument(
            "ArrowDatasetBatcher regression target width must be at least 1");
    }
    if (!dataset_ || !dataset_->GetSchema() || label_col_idx_ < 0) {
        throw std::invalid_argument(
            "ArrowDatasetBatcher cannot configure regression targets without a primary target");
    }

    auto schema = dataset_->GetSchema();
    const std::string base =
        target_base.empty() ? label_column_ : target_base;
    label_col_indices_.clear();
    label_col_indices_.push_back(label_col_idx_);
    for (size_t target = 1; target < width; ++target) {
        const std::string name = base + "_" + std::to_string(target);
        const int index = schema->GetFieldIndex(name);
        if (index < 0) {
            throw std::invalid_argument(
                "ArrowDatasetBatcher missing ordered regression target column '" +
                name + "'");
        }
        label_col_indices_.push_back(index);
    }

    feature_cols_.erase(
        std::remove_if(feature_cols_.begin(), feature_cols_.end(),
            [&](int index) {
                return std::find(label_col_indices_.begin(),
                                 label_col_indices_.end(), index) !=
                       label_col_indices_.end();
            }),
        feature_cols_.end());
    num_features_ = feature_cols_.size();
    spdlog::info(
        "ArrowDatasetBatcher: configured {} ordered regression targets and {} features",
        label_col_indices_.size(), num_features_);
}

Batch ArrowDatasetBatcher::GetNextBatch() {
    Batch batch;

    try {
        if (!dataset_ || indices_.empty()) {
            spdlog::warn("ArrowDatasetBatcher::GetNextBatch: No dataset or empty indices");
            return batch;
        }

        // Validate indices to catch potential memory corruption
        int64_t num_total_rows = dataset_->GetNumRows();
        for (size_t i = 0; i < std::min(indices_.size(), size_t(10)); ++i) {
            if (indices_[i] < 0 || indices_[i] >= num_total_rows) {
                spdlog::error("ArrowDatasetBatcher: Invalid index[{}]={} (max={})",
                              i, indices_[i], num_total_rows);
            }
        }

        if (IsEpochComplete()) {
            return batch;
        }

        // Validate we have features
        if (num_features_ == 0) {
            spdlog::error("ArrowDatasetBatcher::GetNextBatch: No feature columns found");
            return batch;
        }

        // Calculate batch bounds
        size_t batch_start = current_index_;
        size_t batch_end = std::min(current_index_ + batch_size_, indices_.size());
        size_t actual_batch_size = batch_end - batch_start;
        batch.size = actual_batch_size;

        // Get Arrow table
        auto table = dataset_->GetArrowTable();
        if (!table) {
            spdlog::error("ArrowDatasetBatcher: No Arrow table");
            return batch;
        }

        int64_t num_rows = table->num_rows();
        int num_cols = table->num_columns();

        if (batch_inspection_enabled_) {
            PopulateBatchInspection(
                batch,
                table,
                feature_cols_,
                label_col_indices_,
                indices_,
                batch_start,
                actual_batch_size);
        }

        // OPTIMIZED: Pre-allocate batch data as [batch_size, num_features] in row-major order
        std::vector<float> batch_data(actual_batch_size * num_features_, 0.0f);
        // Classification: int labels; regression: float labels. Only one
        // of the two is populated based on scalar_label_mode_.
        std::vector<int> batch_labels(actual_batch_size, 0);
        const size_t regression_target_width =
            std::max<size_t>(1, label_col_indices_.size());
        std::vector<float> batch_labels_float(
            scalar_label_mode_
                ? actual_batch_size * regression_target_width
                : 0,
            0.0f);

        // OPTIMIZED: Process column by column (Arrow is columnar, this is much faster)
        auto process_feature_range = [&](size_t feat_begin, size_t feat_end) {
        for (size_t feat_idx = feat_begin; feat_idx < feat_end; ++feat_idx) {
            int col_idx = feature_cols_[feat_idx];

            if (col_idx < 0 || col_idx >= num_cols) {
                spdlog::warn("ArrowDatasetBatcher: Invalid col_idx={} (num_cols={})", col_idx, num_cols);
                continue;
            }

            auto column = table->column(col_idx);
            if (!column || column->num_chunks() == 0) {
                spdlog::warn("ArrowDatasetBatcher: Null or empty column at idx={}", col_idx);
                continue;
            }

            // For single-chunk columns (most common case), use direct pointer access
            if (column->num_chunks() == 1) {
                auto chunk = column->chunk(0);

                // Get raw data pointer based on type
                switch (chunk->type_id()) {
                    case arrow::Type::DOUBLE: {
                        auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        const double* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::FLOAT: {
                        auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                        const float* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = data[row_idx];
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT64: {
                        auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        const int64_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT32: {
                        auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                        const int32_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT16: {
                        auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                        const int16_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::INT8: {
                        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                        const int8_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    case arrow::Type::UINT8: {
                        auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                        const uint8_t* data = arr->raw_values();
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < num_rows) {
                                batch_data[b * num_features_ + feat_idx] = static_cast<float>(data[row_idx]);
                            }
                        }
                        break;
                    }
                    default:
                        // Fallback to Value() method for other types
                        for (size_t b = 0; b < actual_batch_size; ++b) {
                            int64_t row_idx = indices_[batch_start + b];
                            if (row_idx >= 0 && row_idx < chunk->length()) {
                                // Use generic scalar access
                                auto result = chunk->GetScalar(row_idx);
                                if (result.ok()) {
                                    auto scalar = result.ValueOrDie();
                                    if (scalar->is_valid) {
                                        // Try to cast to double
                                        auto cast_result = scalar->CastTo(arrow::float64());
                                        if (cast_result.ok()) {
                                            auto dbl = std::static_pointer_cast<arrow::DoubleScalar>(cast_result.ValueOrDie());
                                            batch_data[b * num_features_ + feat_idx] = static_cast<float>(dbl->value);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                }
            } else {
                // Multi-chunk fallback (slower, but handles chunked data)
                for (size_t b = 0; b < actual_batch_size; ++b) {
                    int64_t row_idx = indices_[batch_start + b];
                    if (row_idx < 0 || row_idx >= num_rows) continue;

                    // Find the right chunk
                    int64_t local_idx = row_idx;
                    std::shared_ptr<arrow::Array> chunk = nullptr;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto ch = column->chunk(c);
                        if (local_idx < ch->length()) {
                            chunk = ch;
                            break;
                        }
                        local_idx -= ch->length();
                    }

                    if (chunk) {
                        float value = 0.0f;
                        switch (chunk->type_id()) {
                            case arrow::Type::DOUBLE:
                                value = static_cast<float>(std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::FLOAT:
                                value = std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(local_idx);
                                break;
                            case arrow::Type::INT64:
                                value = static_cast<float>(std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::INT32:
                                value = static_cast<float>(std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(local_idx));
                                break;
                            case arrow::Type::UINT8:
                                value = static_cast<float>(std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(local_idx));
                                break;
                            default:
                                break;
                        }
                        batch_data[b * num_features_ + feat_idx] = value;
                    }
                }
            }
        }
        };

        if (num_workers_ > 1 && feature_cols_.size() > 1) {
            size_t worker_count = std::min(static_cast<size_t>(num_workers_), feature_cols_.size());
            size_t chunk_size = (feature_cols_.size() + worker_count - 1) / worker_count;
            std::vector<std::thread> workers;
            workers.reserve(worker_count);

            for (size_t worker = 0; worker < worker_count; ++worker) {
                size_t begin = worker * chunk_size;
                size_t end = std::min(feature_cols_.size(), begin + chunk_size);
                if (begin >= end) break;
                workers.emplace_back(process_feature_range, begin, end);
            }

            for (auto& worker : workers) {
                worker.join();
            }
        } else {
            process_feature_range(0, feature_cols_.size());
        }

        // Extract labels - handles both single-chunk (fast path) and multi-chunk
        // (correct path) columns. The previous implementation always read chunk(0)
        // and crashed whenever Arrow's CSV reader split the table into multiple
        // chunks, which it does for any file >~1 MB block size.
        if (label_col_idx_ >= 0 && label_col_idx_ < num_cols) {
            auto column = table->column(label_col_idx_);
            if (column && column->num_chunks() > 0) {
                // Generic per-row read that handles any number of chunks. Slower
                // than a type-specific raw_values() loop on a single chunk, but
                // correct regardless of chunking. Labels are small (1 element per
                // row), so per-batch cost here is trivial.
                auto read_label = [&](int64_t global_row_idx) -> int {
                    // Find which chunk holds this row
                    int64_t offset = 0;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto chk = column->chunk(c);
                        int64_t chk_len = chk->length();
                        if (global_row_idx < offset + chk_len) {
                            int64_t local_idx = global_row_idx - offset;
                            if (chk->IsNull(local_idx)) return 0;
                            switch (chk->type_id()) {
                                case arrow::Type::INT64:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::Int64Array>(chk)->Value(local_idx));
                                case arrow::Type::INT32:
                                    return std::static_pointer_cast<arrow::Int32Array>(chk)->Value(local_idx);
                                case arrow::Type::INT16:
                                    return std::static_pointer_cast<arrow::Int16Array>(chk)->Value(local_idx);
                                case arrow::Type::INT8:
                                    return std::static_pointer_cast<arrow::Int8Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT8:
                                    return std::static_pointer_cast<arrow::UInt8Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT16:
                                    return std::static_pointer_cast<arrow::UInt16Array>(chk)->Value(local_idx);
                                case arrow::Type::UINT32:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::UInt32Array>(chk)->Value(local_idx));
                                case arrow::Type::DOUBLE:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::DoubleArray>(chk)->Value(local_idx));
                                case arrow::Type::FLOAT:
                                    return static_cast<int>(
                                        std::static_pointer_cast<arrow::FloatArray>(chk)->Value(local_idx));
                                default:
                                    return 0;
                            }
                        }
                        offset += chk_len;
                    }
                    return 0;  // row not found across any chunk
                };

                // Warn once per batcher if the label type isn't something we recognise
                if (column->num_chunks() > 0) {
                    auto first_type = column->chunk(0)->type_id();
                    if (first_type != arrow::Type::INT64 &&
                        first_type != arrow::Type::INT32 &&
                        first_type != arrow::Type::INT16 &&
                        first_type != arrow::Type::INT8 &&
                        first_type != arrow::Type::UINT8 &&
                        first_type != arrow::Type::UINT16 &&
                        first_type != arrow::Type::UINT32 &&
                        first_type != arrow::Type::DOUBLE &&
                        first_type != arrow::Type::FLOAT) {
                        spdlog::warn("ArrowDatasetBatcher: Unsupported label type: {}",
                                     column->chunk(0)->type()->ToString());
                    }
                }

                // Scalar-label mode preserves both fractional regression
                // targets and binary 0/1 targets. The int-cast path above
                // loses precision on non-integer regression labels.
                auto read_label_float = [&](int64_t global_row_idx) -> float {
                    int64_t offset = 0;
                    for (int c = 0; c < column->num_chunks(); ++c) {
                        auto chk = column->chunk(c);
                        int64_t chk_len = chk->length();
                        if (global_row_idx < offset + chk_len) {
                            int64_t local_idx = global_row_idx - offset;
                            if (chk->IsNull(local_idx)) return 0.0f;
                            switch (chk->type_id()) {
                                case arrow::Type::FLOAT:
                                    return std::static_pointer_cast<arrow::FloatArray>(chk)->Value(local_idx);
                                case arrow::Type::DOUBLE:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::DoubleArray>(chk)->Value(local_idx));
                                case arrow::Type::INT64:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::Int64Array>(chk)->Value(local_idx));
                                case arrow::Type::INT32:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::Int32Array>(chk)->Value(local_idx));
                                case arrow::Type::INT16:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::Int16Array>(chk)->Value(local_idx));
                                case arrow::Type::INT8:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::Int8Array>(chk)->Value(local_idx));
                                case arrow::Type::UINT8:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::UInt8Array>(chk)->Value(local_idx));
                                case arrow::Type::UINT16:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::UInt16Array>(chk)->Value(local_idx));
                                case arrow::Type::UINT32:
                                    return static_cast<float>(
                                        std::static_pointer_cast<arrow::UInt32Array>(chk)->Value(local_idx));
                                default:
                                    return 0.0f;
                            }
                        }
                        offset += chk_len;
                    }
                    return 0.0f;
                };

                for (size_t b = 0; b < actual_batch_size; ++b) {
                    int64_t row_idx = indices_[batch_start + b];
                    if (row_idx >= 0 && row_idx < num_rows) {
                        if (scalar_label_mode_) {
                            batch_labels_float[b * regression_target_width] =
                                read_label_float(row_idx);
                        } else {
                            batch_labels[b] = read_label(row_idx);
                        }
                    }
                }
            }
        }

        if (scalar_label_mode_ && label_col_indices_.size() > 1) {
            for (size_t target = 1; target < label_col_indices_.size(); ++target) {
                const int column_index = label_col_indices_[target];
                if (column_index < 0 || column_index >= num_cols) continue;
                const auto& target_column = table->column(column_index);
                for (size_t b = 0; b < actual_batch_size; ++b) {
                    const int64_t row_idx = indices_[batch_start + b];
                    if (row_idx >= 0 && row_idx < num_rows) {
                        batch_labels_float[
                            b * regression_target_width + target] =
                            ReadNumericChunkedValue(target_column, row_idx);
                    }
                }
            }
        }

        // Apply normalization if enabled
        if (normalize_ && norm_std_ != 0.0f) {
            for (float& val : batch_data) {
                val = (val - norm_mean_) / norm_std_;
            }
        }

        // Validate data before creating tensor
        if (batch_data.empty() || actual_batch_size == 0 || num_features_ == 0) {
            spdlog::error("ArrowDatasetBatcher: Empty batch data - size={}, features={}",
                          actual_batch_size, num_features_);
            return batch;
        }

        // Create data tensor [batch_size, num_features]
        std::vector<size_t> data_shape = {actual_batch_size, num_features_};
        batch.data = Tensor(data_shape, batch_data.data());

        // Create labels tensor
        // Float labels [batch, target_width] are used for regression and
        // scalar binary classification. This mode takes precedence over one-hot.
        if (scalar_label_mode_ && !batch_labels_float.empty()) {
            std::vector<size_t> label_shape = {
                actual_batch_size, regression_target_width};
            batch.labels = Tensor(label_shape, batch_labels_float.data());
        } else if (class_index_label_mode_ && !batch_labels.empty()) {
            std::vector<size_t> label_shape = {actual_batch_size};
            batch.labels = Tensor(
                label_shape, batch_labels.data(), DataType::Int32);
        } else if (one_hot_ && !batch_labels.empty()) {
            // One-hot encoding
            std::vector<float> onehot_data(actual_batch_size * num_classes_, 0.0f);
            for (size_t i = 0; i < batch_labels.size(); ++i) {
                int label = batch_labels[i];
                if (label >= 0 && static_cast<size_t>(label) < num_classes_) {
                    onehot_data[i * num_classes_ + label] = 1.0f;
                }
            }
            std::vector<size_t> label_shape = {actual_batch_size, num_classes_};
            batch.labels = Tensor(label_shape, onehot_data.data());
        } else if (!batch_labels.empty()) {
            // Integer labels
            std::vector<float> label_data;
            label_data.reserve(batch_labels.size());
            for (int label : batch_labels) {
                label_data.push_back(static_cast<float>(label));
            }
            std::vector<size_t> label_shape = {actual_batch_size};
            batch.labels = Tensor(label_shape, label_data.data());
        }

        current_index_ = batch_end;
        return batch;

    } catch (const std::exception& e) {
        spdlog::error("ArrowDatasetBatcher::GetNextBatch exception: {}", e.what());
        return batch;
    } catch (...) {
        spdlog::error("ArrowDatasetBatcher::GetNextBatch unknown exception");
        return batch;
    }
}

void ArrowDatasetBatcher::Reset() {
    current_index_ = 0;
    indices_ = base_indices_;
    RebuildBalancedIndices();
    if (shuffle_) {
        ShuffleIndices();
    }
}

bool ArrowDatasetBatcher::IsEpochComplete() const {
    if (current_index_ >= indices_.size()) {
        return true;
    }
    return drop_last_ &&
        indices_.size() - current_index_ < batch_size_;
}

size_t ArrowDatasetBatcher::GetNumBatches() const {
    if (batch_size_ == 0 || indices_.empty()) {
        return 0;
    }
    if (drop_last_) {
        return indices_.size() / batch_size_;
    }
    return (indices_.size() + batch_size_ - 1) / batch_size_;
}

void ArrowDatasetBatcher::SetNormalization(float mean, float std) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std;
}

void ArrowDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    one_hot_ = true;
    num_classes_ = num_classes;
}

void ArrowDatasetBatcher::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

void ArrowDatasetBatcher::RebuildBalancedIndices() {
    if (!balance_classes_ ||
        balance_mode_ == "none" ||
        split_phase_ != BatcherPhase::Train ||
        !is_training_ ||
        label_col_idx_ < 0 ||
        !dataset_ ||
        base_indices_.empty()) {
        return;
    }

    auto labels_result = ReadNumericLabelColumn(
        dataset_->GetArrowTable(), label_column_);
    if (!labels_result.ok()) {
        spdlog::warn("ArrowDatasetBatcher: class balancing requested but "
                     "labels could not be read ({}); using unbalanced train split",
                     labels_result.status().ToString());
        return;
    }

    const auto& labels = labels_result.ValueOrDie();
    std::map<int64_t, std::vector<int64_t>> by_label;
    for (int64_t row : base_indices_) {
        if (row >= 0 && static_cast<size_t>(row) < labels.size()) {
            by_label[labels[static_cast<size_t>(row)]].push_back(row);
        }
    }

    if (by_label.size() < 2) {
        spdlog::warn("ArrowDatasetBatcher: class balancing requested but "
                     "the train split has fewer than two labels");
        return;
    }

    std::vector<size_t> class_counts;
    class_counts.reserve(by_label.size());
    for (const auto& entry : by_label) {
        class_counts.push_back(entry.second.size());
    }

    const auto minmax = std::minmax_element(
        class_counts.begin(), class_counts.end());
    size_t target_count = *minmax.second;
    if (balance_target_ == "min") {
        target_count = *minmax.first;
    } else if (balance_target_ == "median") {
        auto sorted = class_counts;
        std::sort(sorted.begin(), sorted.end());
        target_count = sorted[sorted.size() / 2];
    } else if (!balance_target_.empty() &&
               std::all_of(balance_target_.begin(), balance_target_.end(),
                           [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        try {
            target_count = std::max<size_t>(
                1, static_cast<size_t>(std::stoull(balance_target_)));
        } catch (...) {
            target_count = *minmax.second;
        }
    }

    if (target_count == 0) {
        return;
    }

    balance_rng_.seed(balance_seed_ + balance_epoch_++);

    std::ostringstream original_dist;
    bool first = true;
    for (const auto& entry : by_label) {
        if (!first) original_dist << ", ";
        original_dist << entry.first << ":" << entry.second.size();
        first = false;
    }

    std::vector<int64_t> balanced;
    if (balance_mode_ == "undersample") {
        for (auto& entry : by_label) {
            auto rows = entry.second;
            std::shuffle(rows.begin(), rows.end(), balance_rng_);
            const size_t keep = std::min(target_count, rows.size());
            balanced.insert(balanced.end(), rows.begin(), rows.begin() + keep);
        }
    } else if (balance_mode_ == "weighted_sampler") {
        // Match PyTorch WeightedRandomSampler's usual training contract:
        // replacement changes which rows are drawn, not the configured epoch
        // length. This preserves batch and optimizer-update counts.
        const size_t total_samples = base_indices_.size();
        std::vector<int64_t> labels_order;
        labels_order.reserve(by_label.size());
        for (const auto& entry : by_label) {
            labels_order.push_back(entry.first);
        }
        std::uniform_int_distribution<size_t> class_dist(
            0, labels_order.size() - 1);
        balanced.reserve(total_samples);
        for (size_t i = 0; i < total_samples; ++i) {
            auto& rows = by_label[labels_order[class_dist(balance_rng_)]];
            std::uniform_int_distribution<size_t> row_dist(0, rows.size() - 1);
            balanced.push_back(rows[row_dist(balance_rng_)]);
        }
    } else {
        if (balance_mode_ != "oversample") {
            spdlog::warn("ArrowDatasetBatcher: unsupported balance_mode='{}'; "
                         "using oversample", balance_mode_);
        }
        for (auto& entry : by_label) {
            auto rows = entry.second;
            std::shuffle(rows.begin(), rows.end(), balance_rng_);
            for (size_t i = 0; i < target_count; ++i) {
                balanced.push_back(rows[i % rows.size()]);
            }
        }
    }

    if (balanced.empty()) {
        return;
    }

    indices_ = std::move(balanced);

    std::map<int64_t, size_t> effective_counts;
    for (int64_t row : indices_) {
        if (row >= 0 && static_cast<size_t>(row) < labels.size()) {
            ++effective_counts[labels[static_cast<size_t>(row)]];
        }
    }
    std::ostringstream effective_dist;
    first = true;
    for (const auto& entry : effective_counts) {
        if (!first) effective_dist << ", ";
        effective_dist << entry.first << ":" << entry.second;
        first = false;
    }

    spdlog::info("ArrowDatasetBatcher: balanced train split mode='{}', "
                 "target='{}', original=[{}], effective=[{}], samples={}",
                 balance_mode_, balance_target_, original_dist.str(),
                 effective_dist.str(), indices_.size());
}

} // namespace cyxwiz
