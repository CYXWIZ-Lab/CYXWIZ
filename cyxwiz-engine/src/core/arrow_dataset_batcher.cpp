#include "dataset_batcher.h"
#include "label_column_resolver.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory>
#include <thread>
#include <utility>

namespace cyxwiz {

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
    int num_workers)
    : dataset_(dataset)
    , label_column_(label_column)
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , is_training_(is_training)
    , num_workers_(std::max(0, num_workers))
    , partition_column_(partition_column)
    , partition_value_(partition_value)
    , rng_(std::random_device{}())
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
        // Legacy path: chronological first-N% slicing. train_split fraction
        // for the training batcher, remainder for the validation batcher.
        int64_t train_count = static_cast<int64_t>(num_rows * train_split);
        if (is_training_) {
            indices_.reserve(train_count);
            for (int64_t i = 0; i < train_count; ++i) {
                indices_.push_back(i);
            }
        } else {
            indices_.reserve(num_rows - train_count);
            for (int64_t i = train_count; i < num_rows; ++i) {
                indices_.push_back(i);
            }
        }
    }

    // Initialize feature/label column indices
    InitializeColumns();

    spdlog::info("ArrowDatasetBatcher: {} samples, {} features, batch_size={}, shuffle={}, num_workers={}",
                 indices_.size(), num_features_, batch_size_, shuffle_, num_workers_);

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

    // Second pass: collect feature columns (all numeric except label)
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (i == label_col_idx_) continue;  // Skip label column

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

Batch ArrowDatasetBatcher::GetNextBatch() {
    Batch batch;

    try {
        spdlog::debug("ArrowDatasetBatcher::GetNextBatch called, current_index_={}, indices_.size()={}",
                      current_index_, indices_.size());

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

        // OPTIMIZED: Pre-allocate batch data as [batch_size, num_features] in row-major order
        std::vector<float> batch_data(actual_batch_size * num_features_, 0.0f);
        // Classification: int labels; regression: float labels. Only one
        // of the two is populated based on regression_mode_.
        std::vector<int> batch_labels(actual_batch_size, 0);
        std::vector<float> batch_labels_float(
            regression_mode_ ? actual_batch_size : 0, 0.0f);

        spdlog::debug("ArrowDatasetBatcher: Processing {} feature columns, batch_data.size()={}",
                      feature_cols_.size(), batch_data.size());
        spdlog::default_logger()->flush();  // Force flush to ensure logs are written before crash

        // OPTIMIZED: Process column by column (Arrow is columnar, this is much faster)
        auto process_feature_range = [&](size_t feat_begin, size_t feat_end) {
        for (size_t feat_idx = feat_begin; feat_idx < feat_end; ++feat_idx) {
            int col_idx = feature_cols_[feat_idx];

            // Log first column for debugging
            if (feat_idx == 0) {
                spdlog::debug("ArrowDatasetBatcher: First column idx={}, type={}",
                              col_idx, (col_idx < num_cols && table->column(col_idx)) ?
                              table->column(col_idx)->type()->ToString() : "invalid");
                spdlog::default_logger()->flush();
            }

            // Log progress every 100 columns; flush so crash bisection survives.
            if (feat_idx > 0 && feat_idx % 100 == 0) {
                spdlog::debug("ArrowDatasetBatcher: Processing column {}/{}", feat_idx, feature_cols_.size());
                spdlog::default_logger()->flush();
            }

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

        // CHECKPOINT: feature column loop completed successfully
        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT A - feature loop complete, all {} cols processed",
                      feature_cols_.size());
        spdlog::default_logger()->flush();

        // Extract labels - handles both single-chunk (fast path) and multi-chunk
        // (correct path) columns. The previous implementation always read chunk(0)
        // and crashed whenever Arrow's CSV reader split the table into multiple
        // chunks, which it does for any file >~1 MB block size.
        if (label_col_idx_ >= 0 && label_col_idx_ < num_cols) {
            auto column = table->column(label_col_idx_);
            if (column && column->num_chunks() > 0) {
                spdlog::debug("ArrowDatasetBatcher: label column has {} chunks, "
                              "first chunk length={}, table num_rows={}",
                              column->num_chunks(),
                              column->chunk(0) ? column->chunk(0)->length() : -1,
                              num_rows);
                spdlog::default_logger()->flush();

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

                // Phase 4 regression mode: separate float reader that
                // preserves fractional targets (MSELoss needs real-valued
                // labels). The int-cast path above loses precision on
                // anything non-integer and is wrong for TimeSeriesWindow's
                // `y` column.
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
                        if (regression_mode_) {
                            batch_labels_float[b] = read_label_float(row_idx);
                        } else {
                            batch_labels[b] = read_label(row_idx);
                        }
                    }
                }
            }
        }

        // CHECKPOINT: label extraction complete
        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT B - label extraction complete, "
                      "{} features, {} labels",
                      batch_data.size(), batch_labels.size());
        spdlog::default_logger()->flush();

        // Apply normalization if enabled
        if (normalize_ && norm_std_ != 0.0f) {
            for (float& val : batch_data) {
                val = (val - norm_mean_) / norm_std_;
            }
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT C - normalization applied "
                          "(mean={}, std={})", norm_mean_, norm_std_);
            spdlog::default_logger()->flush();
        }

        // Validate data before creating tensor
        if (batch_data.empty() || actual_batch_size == 0 || num_features_ == 0) {
            spdlog::error("ArrowDatasetBatcher: Empty batch data - size={}, features={}",
                          actual_batch_size, num_features_);
            return batch;
        }

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT D - about to construct data tensor [{}, {}]",
                      actual_batch_size, num_features_);
        spdlog::default_logger()->flush();

        // Create data tensor [batch_size, num_features]
        std::vector<size_t> data_shape = {actual_batch_size, num_features_};
        batch.data = Tensor(data_shape, batch_data.data());

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT E - data tensor constructed successfully");
        spdlog::default_logger()->flush();

        // Create labels tensor
        // Phase 4 regression: float labels tensor [batch, 1]. Takes
        // precedence over one_hot_ because SetOneHotEncoding is called
        // unconditionally in TrainingExecutor::Train for classification
        // compatibility; regression_mode_ is the explicit opt-in.
        if (regression_mode_ && !batch_labels_float.empty()) {
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT F - creating float regression labels [{}, 1]",
                          actual_batch_size);
            spdlog::default_logger()->flush();
            std::vector<size_t> label_shape = {actual_batch_size, 1};
            batch.labels = Tensor(label_shape, batch_labels_float.data());
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT G - float labels tensor created");
            spdlog::default_logger()->flush();
        } else if (one_hot_ && !batch_labels.empty()) {
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT F - creating one-hot labels [{}, {}]",
                          actual_batch_size, num_classes_);
            spdlog::default_logger()->flush();
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
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT G - one-hot labels tensor created");
            spdlog::default_logger()->flush();
        } else if (!batch_labels.empty()) {
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT F - creating integer labels [{}]",
                          actual_batch_size);
            spdlog::default_logger()->flush();
            // Integer labels
            std::vector<float> label_data;
            label_data.reserve(batch_labels.size());
            for (int label : batch_labels) {
                label_data.push_back(static_cast<float>(label));
            }
            std::vector<size_t> label_shape = {actual_batch_size};
            batch.labels = Tensor(label_shape, label_data.data());
            spdlog::debug("ArrowDatasetBatcher: CHECKPOINT G - integer labels tensor created");
            spdlog::default_logger()->flush();
        }

        spdlog::debug("ArrowDatasetBatcher: CHECKPOINT H - returning batch to caller");
        spdlog::default_logger()->flush();
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
    if (shuffle_) {
        ShuffleIndices();
    }
}

bool ArrowDatasetBatcher::IsEpochComplete() const {
    return current_index_ >= indices_.size();
}

size_t ArrowDatasetBatcher::GetNumBatches() const {
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

} // namespace cyxwiz
