#include "parquet_arrow_batcher.h"

#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/table.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace cyxwiz {

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

ParquetArrowBatcher::ParquetArrowBatcher(
    std::shared_ptr<ParquetBackedDataset> dataset,
    const std::string& label_column,
    size_t batch_size,
    bool shuffle,
    float train_split,
    bool is_training)
    : dataset_(std::move(dataset)),
      label_column_(label_column),
      batch_size_(batch_size),
      shuffle_(shuffle),
      is_training_(is_training),
      train_split_(train_split),
      rng_(std::random_device{}()) {

    if (!dataset_) {
        spdlog::error("ParquetArrowBatcher: null dataset");
        return;
    }

    InitializeColumns();
    AssignRowGroups();

    // Count total samples in our assigned groups
    num_samples_ = 0;
    for (int g : assigned_row_groups_) {
        num_samples_ += static_cast<size_t>(dataset_->GetRowGroupSize(g));
    }

    spdlog::info("ParquetArrowBatcher: {} ({} row groups assigned, {} samples, "
                 "{} features, batch_size={}, label_col='{}')",
                 is_training_ ? "train" : "val",
                 assigned_row_groups_.size(),
                 num_samples_,
                 num_features_,
                 batch_size_,
                 label_column_);

    Reset();
}

// -----------------------------------------------------------------------------
// InitializeColumns - find feature columns and label column in the schema
// -----------------------------------------------------------------------------

void ParquetArrowBatcher::InitializeColumns() {
    auto schema = dataset_->GetSchema();
    if (!schema) {
        spdlog::error("ParquetArrowBatcher::InitializeColumns: null schema");
        return;
    }

    const int num_cols = schema->num_fields();
    feature_cols_.clear();
    feature_cols_.reserve(num_cols);

    for (int i = 0; i < num_cols; ++i) {
        const auto& field_name = schema->field(i)->name();
        if (!label_column_.empty() && field_name == label_column_) {
            label_col_idx_ = i;
        } else {
            feature_cols_.push_back(i);
        }
    }
    num_features_ = feature_cols_.size();

    // If the caller didn't name a label column, auto-detect: assume the last
    // column is the label (matches ArrowDatasetBatcher behavior for
    // unsupervised vs supervised ambiguity).
    if (label_col_idx_ < 0 && num_cols > 1) {
        label_col_idx_ = num_cols - 1;
        // Remove it from feature_cols_ if it crept in (it will have, since
        // the loop above only excluded it when label_column_ was non-empty).
        feature_cols_.erase(
            std::remove(feature_cols_.begin(), feature_cols_.end(), label_col_idx_),
            feature_cols_.end());
        num_features_ = feature_cols_.size();
        spdlog::info("ParquetArrowBatcher: auto-detected label column at index {} ('{}')",
                     label_col_idx_, schema->field(label_col_idx_)->name());
    }
}

// -----------------------------------------------------------------------------
// AssignRowGroups - split row groups between train and val by index
// -----------------------------------------------------------------------------

void ParquetArrowBatcher::AssignRowGroups() {
    assigned_row_groups_.clear();
    const int total_groups = dataset_->GetNumRowGroups();
    if (total_groups == 0) return;

    // First train_split fraction of row groups go to train; the rest are val.
    // Clamp split so both sides always have at least one group when the file
    // has at least 2 groups.
    int train_cutoff = static_cast<int>(total_groups * train_split_);
    if (total_groups >= 2) {
        train_cutoff = std::max(1, std::min(train_cutoff, total_groups - 1));
    } else {
        train_cutoff = total_groups;  // only one group — give it all to train
    }

    assigned_row_groups_.reserve(total_groups);
    if (is_training_) {
        for (int i = 0; i < train_cutoff; ++i) assigned_row_groups_.push_back(i);
    } else {
        for (int i = train_cutoff; i < total_groups; ++i) assigned_row_groups_.push_back(i);
    }
}

// -----------------------------------------------------------------------------
// ShuffleEpochGroupOrder - permute the row group order for the current epoch
// -----------------------------------------------------------------------------

void ParquetArrowBatcher::ShuffleEpochGroupOrder() {
    epoch_group_order_ = assigned_row_groups_;
    if (shuffle_) {
        std::shuffle(epoch_group_order_.begin(), epoch_group_order_.end(), rng_);
    }
}

// -----------------------------------------------------------------------------
// LoadNextRowGroup - advance to the next row group in epoch order and load it
// -----------------------------------------------------------------------------

bool ParquetArrowBatcher::LoadNextRowGroup() {
    if (current_group_position_ >= epoch_group_order_.size()) {
        current_table_.reset();
        return false;
    }

    const int group_idx = epoch_group_order_[current_group_position_];
    current_table_ = dataset_->ReadRowGroup(group_idx);
    if (!current_table_) {
        spdlog::warn("ParquetArrowBatcher::LoadNextRowGroup: ReadRowGroup({}) returned null", group_idx);
        current_group_position_++;
        return LoadNextRowGroup();  // try the next one
    }

    // Build a row index for this group and shuffle it if shuffling is enabled.
    const int64_t rows_in_group = current_table_->num_rows();
    current_row_indices_.resize(static_cast<size_t>(rows_in_group));
    std::iota(current_row_indices_.begin(), current_row_indices_.end(), size_t{0});
    if (shuffle_) {
        std::shuffle(current_row_indices_.begin(), current_row_indices_.end(), rng_);
    }
    current_row_cursor_ = 0;
    current_group_position_++;
    return true;
}

// -----------------------------------------------------------------------------
// Reset - start a new epoch
// -----------------------------------------------------------------------------

void ParquetArrowBatcher::Reset() {
    ShuffleEpochGroupOrder();
    current_group_position_ = 0;
    current_table_.reset();
    current_row_indices_.clear();
    current_row_cursor_ = 0;
    samples_served_ = 0;

    // Pre-load the first row group so the first GetNextBatch call has data ready
    LoadNextRowGroup();
}

// -----------------------------------------------------------------------------
// IsEpochComplete
// -----------------------------------------------------------------------------

bool ParquetArrowBatcher::IsEpochComplete() const {
    if (samples_served_ >= num_samples_) return true;
    // Also complete if we have no current table and no more groups to load
    if (!current_table_ && current_group_position_ >= epoch_group_order_.size()) return true;
    return false;
}

// -----------------------------------------------------------------------------
// GetNumBatches
// -----------------------------------------------------------------------------

size_t ParquetArrowBatcher::GetNumBatches() const {
    if (batch_size_ == 0 || num_samples_ == 0) return 0;
    return (num_samples_ + batch_size_ - 1) / batch_size_;
}

// -----------------------------------------------------------------------------
// Preprocessing setters (same semantics as ArrowDatasetBatcher)
// -----------------------------------------------------------------------------

void ParquetArrowBatcher::SetNormalization(float mean, float std_dev) {
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = (std_dev != 0.0f) ? std_dev : 1.0f;
}

void ParquetArrowBatcher::SetOneHotEncoding(size_t num_classes) {
    one_hot_ = true;
    num_classes_ = num_classes;
}

// -----------------------------------------------------------------------------
// GetNextBatch - the heavy lift
//
// Fills up a batch of batch_size samples by pulling rows from the current
// row group, advancing to the next group when needed, and converting each
// cell to float32. When the batch buffer is full (or the epoch ends with a
// partial batch), materializes the Tensor and returns.
// -----------------------------------------------------------------------------

Batch ParquetArrowBatcher::GetNextBatch() {
    Batch batch;
    batch.size = 0;

    if (IsEpochComplete()) {
        return batch;
    }

    // We don't yet know the final batch size (the epoch might end mid-batch),
    // so allocate max capacity and truncate at the end if needed.
    const size_t capacity = batch_size_;
    std::vector<float> batch_data(capacity * num_features_, 0.0f);
    std::vector<int> batch_labels(capacity, 0);
    size_t rows_filled = 0;

    while (rows_filled < capacity) {
        // Ensure we have a current row group with rows left.
        if (!current_table_ || current_row_cursor_ >= current_row_indices_.size()) {
            if (!LoadNextRowGroup()) {
                break;  // epoch exhausted
            }
            continue;   // retry with the freshly loaded group
        }

        // How many rows can we take from the current group in one sweep?
        const size_t rows_available = current_row_indices_.size() - current_row_cursor_;
        const size_t rows_needed = capacity - rows_filled;
        const size_t rows_this_sweep = std::min(rows_available, rows_needed);

        // Extract feature columns for the selected rows. Parquet row group
        // tables are usually single-chunked (one chunk per column), so take
        // the fast path when possible and fall back to chunked access only
        // if a column turns out to be chunked.
        const int num_cols_in_table = current_table_->num_columns();

        for (size_t feat_idx = 0; feat_idx < feature_cols_.size(); ++feat_idx) {
            const int col_idx = feature_cols_[feat_idx];
            if (col_idx < 0 || col_idx >= num_cols_in_table) continue;

            auto column = current_table_->column(col_idx);
            if (!column || column->num_chunks() == 0) continue;

            // Single-chunk fast path (overwhelming common case for Parquet row groups)
            if (column->num_chunks() == 1) {
                auto chunk = column->chunk(0);
                switch (chunk->type_id()) {
                    case arrow::Type::DOUBLE: {
                        auto arr = std::static_pointer_cast<arrow::DoubleArray>(chunk);
                        const double* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::FLOAT: {
                        auto arr = std::static_pointer_cast<arrow::FloatArray>(chunk);
                        const float* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                data[local_row];
                        }
                        break;
                    }
                    case arrow::Type::INT64: {
                        auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        const int64_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::INT32: {
                        auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                        const int32_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::INT16: {
                        auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                        const int16_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::INT8: {
                        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                        const int8_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::UINT8: {
                        auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                        const uint8_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::UINT16: {
                        auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
                        const uint16_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::UINT32: {
                        auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
                        const uint32_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_data[(rows_filled + b) * num_features_ + feat_idx] =
                                static_cast<float>(data[local_row]);
                        }
                        break;
                    }
                    default:
                        // Unsupported type - leave as 0 and log once.
                        if (feat_idx == 0) {
                            spdlog::warn("ParquetArrowBatcher: unsupported feature type {}",
                                         chunk->type()->ToString());
                        }
                        break;
                }
            }
            // Multi-chunk fallback omitted in first pass - row-group tables
            // from Parquet are typically single-chunked. If we ever see a
            // chunked column the above branch simply leaves zeros and logs.
        }

        // Extract labels for the same rows
        if (label_col_idx_ >= 0 && label_col_idx_ < num_cols_in_table) {
            auto label_column = current_table_->column(label_col_idx_);
            if (label_column && label_column->num_chunks() > 0) {
                auto chunk = label_column->chunk(0);
                switch (chunk->type_id()) {
                    case arrow::Type::INT64: {
                        auto arr = std::static_pointer_cast<arrow::Int64Array>(chunk);
                        const int64_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = static_cast<int>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::INT32: {
                        auto arr = std::static_pointer_cast<arrow::Int32Array>(chunk);
                        const int32_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = data[local_row];
                        }
                        break;
                    }
                    case arrow::Type::UINT8: {
                        auto arr = std::static_pointer_cast<arrow::UInt8Array>(chunk);
                        const uint8_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = static_cast<int>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::UINT16: {
                        auto arr = std::static_pointer_cast<arrow::UInt16Array>(chunk);
                        const uint16_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = static_cast<int>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::UINT32: {
                        auto arr = std::static_pointer_cast<arrow::UInt32Array>(chunk);
                        const uint32_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = static_cast<int>(data[local_row]);
                        }
                        break;
                    }
                    case arrow::Type::INT16: {
                        auto arr = std::static_pointer_cast<arrow::Int16Array>(chunk);
                        const int16_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = data[local_row];
                        }
                        break;
                    }
                    case arrow::Type::INT8: {
                        auto arr = std::static_pointer_cast<arrow::Int8Array>(chunk);
                        const int8_t* data = arr->raw_values();
                        for (size_t b = 0; b < rows_this_sweep; ++b) {
                            const size_t local_row = current_row_indices_[current_row_cursor_ + b];
                            batch_labels[rows_filled + b] = data[local_row];
                        }
                        break;
                    }
                    default:
                        spdlog::warn("ParquetArrowBatcher: unsupported label type {}",
                                     chunk->type()->ToString());
                        break;
                }
            }
        }

        rows_filled += rows_this_sweep;
        current_row_cursor_ += rows_this_sweep;
        samples_served_ += rows_this_sweep;
    }

    // If we filled zero rows, the epoch is over and there's no partial batch
    // to return.
    if (rows_filled == 0) {
        return batch;
    }

    // Apply normalization in place if enabled
    if (normalize_ && norm_std_ != 0.0f) {
        const size_t n = rows_filled * num_features_;
        for (size_t i = 0; i < n; ++i) {
            batch_data[i] = (batch_data[i] - norm_mean_) / norm_std_;
        }
    }

    // Materialize the feature tensor
    std::vector<size_t> data_shape = {rows_filled, num_features_};
    batch.data = Tensor(data_shape, batch_data.data());

    // Materialize the label tensor (one-hot or int-as-float)
    if (one_hot_ && label_col_idx_ >= 0) {
        std::vector<float> onehot(rows_filled * num_classes_, 0.0f);
        for (size_t i = 0; i < rows_filled; ++i) {
            int lbl = batch_labels[i];
            if (lbl >= 0 && static_cast<size_t>(lbl) < num_classes_) {
                onehot[i * num_classes_ + lbl] = 1.0f;
            }
        }
        std::vector<size_t> label_shape = {rows_filled, num_classes_};
        batch.labels = Tensor(label_shape, onehot.data());
    } else if (label_col_idx_ >= 0) {
        std::vector<float> label_floats(batch_labels.begin(),
                                         batch_labels.begin() + rows_filled);
        std::vector<size_t> label_shape = {rows_filled};
        batch.labels = Tensor(label_shape, label_floats.data());
    }

    batch.size = rows_filled;
    return batch;
}

} // namespace cyxwiz
