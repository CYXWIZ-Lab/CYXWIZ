#include "sparse_feature_dataset_batcher.h"

#include "materialization_memory_guard.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {

constexpr size_t kInspectionPreviewLimit = 32;

const char* PhaseName(BatcherPhase phase) {
    switch (phase) {
    case BatcherPhase::Train:
        return "train";
    case BatcherPhase::Val:
        return "val";
    case BatcherPhase::Test:
        return "test";
    }
    return "unknown";
}

} // namespace

MaterializationMemoryEstimate EstimateSparseBatchDensificationMemory(
    uint64_t rows,
    uint64_t features) {
    MaterializationMemoryEstimate estimate;
    estimate.rows = rows;
    estimate.output_features = features;
    estimate.bytes_per_value = sizeof(float);
    uint64_t cells = 0;
    estimate.overflow = !CheckedMulU64(rows, features, cells) ||
        !CheckedMulU64(cells, sizeof(float), estimate.raw_output_bytes);
    estimate.temporary_bytes = estimate.raw_output_bytes;
    uint64_t peak = 0;
    if (!CheckedAddU64(estimate.raw_output_bytes,
                       estimate.temporary_bytes, peak)) {
        estimate.overflow = true;
    }
    estimate.estimated_peak_bytes = estimate.overflow
        ? (std::numeric_limits<uint64_t>::max)()
        : peak;
    estimate.confidence = "high";
    return estimate;
}

SparseFeatureDatasetBatcher::SparseFeatureDatasetBatcher(
    std::shared_ptr<const SparseFeatureDataset> dataset,
    size_t batch_size,
    bool shuffle,
    float train_split,
    bool is_training,
    BatcherPhase split_phase,
    float val_split,
    uint32_t seed,
    bool balance_classes,
    const std::string& balance_mode,
    const std::string& balance_target,
    uint32_t balance_seed,
    MaterializationMemoryContext memory_context,
    bool stratified,
    bool full_dataset,
    bool sparse_feature_output)
    : dataset_(std::move(dataset))
    , batch_size_(batch_size)
    , shuffle_(shuffle)
    , is_training_(is_training)
    , split_phase_(split_phase)
    , rng_(seed)
    , balance_rng_(balance_seed)
    , balance_seed_(balance_seed)
    , balance_classes_(balance_classes)
    , balance_mode_(balance_mode)
    , balance_target_(balance_target)
    , sparse_feature_output_(sparse_feature_output)
    , memory_context_(std::move(memory_context)) {
    if (!dataset_) {
        throw std::invalid_argument(
            "SparseFeatureDatasetBatcher requires a dataset");
    }
    if (batch_size_ == 0) {
        throw std::invalid_argument(
            "SparseFeatureDatasetBatcher batch_size must be at least 1");
    }

    const int64_t num_rows = dataset_->GetNumRows();
    const float safe_train_split = std::clamp(train_split, 0.0f, 1.0f);
    const float safe_val_split = std::clamp(val_split, 0.0f, 1.0f);
    if (val_split <= 0.0f && split_phase_ == BatcherPhase::Train &&
        !is_training_) {
        split_phase_ = BatcherPhase::Val;
    }

    if (full_dataset) {
        indices_.reserve(static_cast<size_t>(num_rows));
        for (int64_t row = 0; row < num_rows; ++row) {
            indices_.push_back(row);
        }
    } else if (stratified) {
        if (!dataset_->GetLabels()) {
            throw std::invalid_argument(
                "SparseFeatureDatasetBatcher stratified split requires labels");
        }
        float safe_test_split = std::max(
            0.0f, 1.0f - safe_train_split - safe_val_split);
        float ratio_sum =
            safe_train_split + safe_val_split + safe_test_split;
        float train_ratio = safe_train_split;
        float dev_ratio = safe_val_split;
        if (!(ratio_sum > 0.0f)) {
            train_ratio = 0.8f;
            dev_ratio = 0.1f;
            safe_test_split = 0.1f;
            ratio_sum = 1.0f;
        }
        train_ratio /= ratio_sum;
        dev_ratio /= ratio_sum;

        std::map<int64_t, std::vector<int64_t>> by_label;
        for (int64_t row = 0; row < num_rows; ++row) {
            by_label[ReadLabel(row, true).class_index].push_back(row);
        }
        std::mt19937 split_rng(seed);
        for (auto& [label, rows] : by_label) {
            (void)label;
            if (shuffle_) {
                std::shuffle(rows.begin(), rows.end(), split_rng);
            }
            size_t train_count = static_cast<size_t>(rows.size() * train_ratio);
            size_t val_count = static_cast<size_t>(rows.size() * dev_ratio);
            if (!rows.empty() && train_count == 0) {
                train_count = 1;
            }
            train_count = std::min(train_count, rows.size());
            val_count = std::min(val_count, rows.size() - train_count);
            size_t begin = 0;
            size_t end = train_count;
            if (split_phase_ == BatcherPhase::Val) {
                begin = train_count;
                end = train_count + val_count;
            } else if (split_phase_ == BatcherPhase::Test) {
                begin = train_count + val_count;
                end = rows.size();
            }
            indices_.insert(
                indices_.end(), rows.begin() + begin, rows.begin() + end);
        }
    } else {
        const int64_t train_count =
            static_cast<int64_t>(num_rows * safe_train_split);
        const int64_t val_count = safe_val_split > 0.0f
            ? static_cast<int64_t>(num_rows * safe_val_split)
            : num_rows - train_count;
        const int64_t val_end = std::min(
            num_rows, train_count + std::max<int64_t>(0, val_count));
        int64_t begin = 0;
        int64_t end = train_count;
        if (split_phase_ == BatcherPhase::Val) {
            begin = train_count;
            end = val_end;
        } else if (split_phase_ == BatcherPhase::Test) {
            begin = val_end;
            end = num_rows;
        }
        indices_.reserve(
            static_cast<size_t>(std::max<int64_t>(0, end - begin)));
        for (int64_t row = begin; row < end; ++row) {
            indices_.push_back(row);
        }
    }
    base_indices_ = indices_;
    RebuildBalancedIndices();
    if (shuffle_) {
        ShuffleIndices();
    }

    const size_t planned_rows = std::min(batch_size_, indices_.size());
    dense_batch_memory_estimate_ = EstimateSparseBatchDensificationMemory(
        static_cast<uint64_t>(planned_rows),
        static_cast<uint64_t>(dataset_->GetNumFeatures()));
    dense_batch_memory_decision_ = EvaluateMaterializationMemory(
        dense_batch_memory_estimate_, memory_context_);
    if (!sparse_feature_output_ && planned_rows > 0 &&
        dense_batch_memory_decision_.blocked) {
        throw std::length_error(BuildMaterializationMemoryPreflightMessage(
            "SparseFeatureDatasetBatcher",
            "features",
            dense_batch_memory_estimate_,
            dense_batch_memory_decision_,
            "Reduce batch_size or feature width before training."));
    }

    spdlog::info(
        "SparseFeatureDatasetBatcher: {} split, {} samples, {} features, "
        "batch_size={}, shuffle={}, output={}, batch_dense_peak={}",
        PhaseName(split_phase_), indices_.size(), dataset_->GetNumFeatures(),
        batch_size_, shuffle_,
        sparse_feature_output_ ? "sparse CSR" : "dense Tensor",
        FormatMaterializationBytes(
            dense_batch_memory_estimate_.estimated_peak_bytes));
}

SparseFeatureDatasetBatcher::LabelValue
SparseFeatureDatasetBatcher::ReadLabel(
    int64_t row,
    bool require_class_index) const {
    const auto& labels = dataset_->GetLabels();
    if (!labels) {
        throw std::runtime_error(
            "SparseFeatureDatasetBatcher has no label column");
    }
    if (row < 0 || row >= labels->length()) {
        throw std::out_of_range(
            "SparseFeatureDatasetBatcher label row is out of range");
    }

    int64_t local = row;
    std::shared_ptr<arrow::Array> chunk;
    for (int index = 0; index < labels->num_chunks(); ++index) {
        const auto& candidate = labels->chunk(index);
        if (local < candidate->length()) {
            chunk = candidate;
            break;
        }
        local -= candidate->length();
    }
    if (!chunk || chunk->IsNull(local)) {
        throw std::runtime_error(
            "SparseFeatureDatasetBatcher encountered a null label");
    }

    LabelValue value;
    bool exact_integer = false;
    const auto set_signed = [&](int64_t label) {
        value.scalar = static_cast<double>(label);
        value.class_index = label;
        exact_integer = true;
    };
    const auto set_unsigned = [&](uint64_t label) {
        value.scalar = static_cast<double>(label);
        if (label <= static_cast<uint64_t>(
                (std::numeric_limits<int64_t>::max)())) {
            value.class_index = static_cast<int64_t>(label);
            exact_integer = true;
        } else if (require_class_index) {
            throw std::runtime_error(
                "SparseFeatureDatasetBatcher classification label exceeds "
                "int64");
        }
    };
    switch (chunk->type_id()) {
    case arrow::Type::DOUBLE:
        value.scalar =
            std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(local);
        break;
    case arrow::Type::FLOAT:
        value.scalar =
            std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(local);
        break;
    case arrow::Type::INT64:
        set_signed(
            std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(local));
        break;
    case arrow::Type::INT32:
        set_signed(
            std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(local));
        break;
    case arrow::Type::INT16:
        set_signed(
            std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(local));
        break;
    case arrow::Type::INT8:
        set_signed(
            std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(local));
        break;
    case arrow::Type::UINT64:
        set_unsigned(
            std::static_pointer_cast<arrow::UInt64Array>(chunk)->Value(local));
        break;
    case arrow::Type::UINT32:
        set_unsigned(
            std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(local));
        break;
    case arrow::Type::UINT16:
        set_unsigned(
            std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(local));
        break;
    case arrow::Type::UINT8:
        set_unsigned(
            std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(local));
        break;
    default:
        throw std::runtime_error(
            "SparseFeatureDatasetBatcher requires numeric labels, got " +
            chunk->type()->ToString());
    }

    if (!std::isfinite(value.scalar)) {
        throw std::runtime_error(
            "SparseFeatureDatasetBatcher encountered a non-finite label");
    }
    if (require_class_index && !exact_integer) {
        constexpr double kInt64Minimum = -9223372036854775808.0;
        constexpr double kInt64MaximumExclusive = 9223372036854775808.0;
        if (value.scalar < kInt64Minimum ||
            value.scalar >= kInt64MaximumExclusive ||
            std::trunc(value.scalar) != value.scalar) {
            throw std::runtime_error(
                "SparseFeatureDatasetBatcher classification labels must be "
                "integral int64 values");
        }
        value.class_index = static_cast<int64_t>(value.scalar);
    }
    return value;
}

void SparseFeatureDatasetBatcher::ValidateDenseBatchMemory(size_t rows) const {
    const auto estimate = EstimateSparseBatchDensificationMemory(
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(dataset_->GetNumFeatures()));
    const auto decision = EvaluateMaterializationMemory(
        estimate, memory_context_);
    if (decision.blocked) {
        throw std::length_error(BuildMaterializationMemoryPreflightMessage(
            "SparseFeatureDatasetBatcher",
            "features",
            estimate,
            decision,
            "Reduce batch_size or feature width before training."));
    }
    if (estimate.raw_output_bytes >
        static_cast<uint64_t>((std::numeric_limits<size_t>::max)())) {
        throw std::length_error(
            "SparseFeatureDatasetBatcher dense batch exceeds size_t");
    }
}

Batch SparseFeatureDatasetBatcher::GetNextBatch() {
    if (IsEpochComplete()) {
        return {};
    }
    const size_t remaining = indices_.size() - current_index_;
    const size_t actual_batch_size = std::min(batch_size_, remaining);

    const size_t num_features =
        static_cast<size_t>(dataset_->GetNumFeatures());
    const auto& offsets = dataset_->GetRowOffsets();
    const auto& columns = dataset_->GetColumnIndices();
    const auto& values = dataset_->GetValues();

    Batch batch;
    if (sparse_feature_output_) {
        SparseFeatureBatch sparse;
        sparse.rows = actual_batch_size;
        sparse.columns = num_features;
        sparse.row_offsets.reserve(actual_batch_size + 1);
        sparse.row_offsets.push_back(0);

        size_t selected_nnz = 0;
        for (size_t batch_row = 0;
             batch_row < actual_batch_size;
             ++batch_row) {
            const int64_t source_row = indices_[current_index_ + batch_row];
            const size_t begin = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row)]);
            const size_t end = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row + 1)]);
            const size_t row_nnz = end - begin;
            if (row_nnz > static_cast<size_t>(
                    (std::numeric_limits<int32_t>::max)()) - selected_nnz) {
                throw std::length_error(
                    "SparseFeatureDatasetBatcher batch nnz exceeds int32");
            }
            selected_nnz += row_nnz;
            sparse.row_offsets.push_back(
                static_cast<int32_t>(selected_nnz));
        }
        sparse.column_indices.reserve(selected_nnz);
        sparse.values.reserve(selected_nnz);
        for (size_t batch_row = 0;
             batch_row < actual_batch_size;
             ++batch_row) {
            const int64_t source_row = indices_[current_index_ + batch_row];
            const size_t begin = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row)]);
            const size_t end = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row + 1)]);
            sparse.column_indices.insert(
                sparse.column_indices.end(),
                columns.begin() + static_cast<std::ptrdiff_t>(begin),
                columns.begin() + static_cast<std::ptrdiff_t>(end));
            for (size_t index = begin; index < end; ++index) {
                sparse.values.push_back(
                    normalize_ ? values[index] / norm_std_ : values[index]);
            }
        }
        batch.sparse_features = std::move(sparse);
    } else {
        ValidateDenseBatchMemory(actual_batch_size);
        uint64_t dense_cells = 0;
        if (!CheckedMulU64(actual_batch_size, num_features, dense_cells) ||
            dense_cells > static_cast<uint64_t>(
                (std::numeric_limits<size_t>::max)())) {
            throw std::length_error(
                "SparseFeatureDatasetBatcher dense cell count overflow");
        }
        std::vector<float> dense(static_cast<size_t>(dense_cells), 0.0f);
        for (size_t batch_row = 0;
             batch_row < actual_batch_size;
             ++batch_row) {
            const int64_t source_row = indices_[current_index_ + batch_row];
            const size_t begin = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row)]);
            const size_t end = static_cast<size_t>(
                offsets[static_cast<size_t>(source_row + 1)]);
            for (size_t index = begin; index < end; ++index) {
                dense[batch_row * num_features +
                      static_cast<size_t>(columns[index])] = values[index];
            }
        }
        if (normalize_) {
            for (float& value : dense) {
                value = (value - norm_mean_) / norm_std_;
            }
        }
        batch.data = Tensor(
            {actual_batch_size, num_features},
            dense.data(),
            DataType::Float32);
    }
    const auto& labels = dataset_->GetLabels();
    if (labels) {
        if (scalar_label_mode_) {
            std::vector<float> output(actual_batch_size);
            for (size_t row = 0; row < actual_batch_size; ++row) {
                output[row] = static_cast<float>(
                    ReadLabel(indices_[current_index_ + row], false).scalar);
            }
            batch.labels = Tensor(
                {actual_batch_size, 1}, output.data(), DataType::Float32);
        } else if (class_index_label_mode_) {
            std::vector<int32_t> output(actual_batch_size);
            for (size_t row = 0; row < actual_batch_size; ++row) {
                const int64_t label =
                    ReadLabel(indices_[current_index_ + row], true).class_index;
                if (label < (std::numeric_limits<int32_t>::min)() ||
                    label > (std::numeric_limits<int32_t>::max)()) {
                    throw std::runtime_error(
                        "SparseFeatureDatasetBatcher class index exceeds int32");
                }
                output[row] = static_cast<int32_t>(label);
            }
            batch.labels = Tensor(
                {actual_batch_size}, output.data(), DataType::Int32);
        } else if (one_hot_) {
            std::vector<float> output(
                actual_batch_size * num_classes_, 0.0f);
            for (size_t row = 0; row < actual_batch_size; ++row) {
                const int64_t label =
                    ReadLabel(indices_[current_index_ + row], true).class_index;
                if (label < 0 || static_cast<uint64_t>(label) >= num_classes_) {
                    throw std::runtime_error(
                        "SparseFeatureDatasetBatcher class index is outside "
                        "the one-hot width");
                }
                output[row * num_classes_ + static_cast<size_t>(label)] = 1.0f;
            }
            batch.labels = Tensor(
                {actual_batch_size, num_classes_}, output.data(),
                DataType::Float32);
        } else {
            std::vector<float> output(actual_batch_size);
            for (size_t row = 0; row < actual_batch_size; ++row) {
                output[row] = static_cast<float>(
                    ReadLabel(indices_[current_index_ + row], true).class_index);
            }
            batch.labels = Tensor(
                {actual_batch_size}, output.data(), DataType::Float32);
        }
    }

    if (batch_inspection_enabled_) {
        PopulateInspection(batch, current_index_, actual_batch_size);
    }
    current_index_ += actual_batch_size;
    batch.size = actual_batch_size;
    return batch;
}

void SparseFeatureDatasetBatcher::Reset() {
    current_index_ = 0;
    indices_ = base_indices_;
    RebuildBalancedIndices();
    if (shuffle_) {
        ShuffleIndices();
    }
}

bool SparseFeatureDatasetBatcher::IsEpochComplete() const {
    if (current_index_ >= indices_.size()) {
        return true;
    }
    return drop_last_ && indices_.size() - current_index_ < batch_size_;
}

size_t SparseFeatureDatasetBatcher::GetNumBatches() const {
    if (indices_.empty()) {
        return 0;
    }
    if (drop_last_) {
        return indices_.size() / batch_size_;
    }
    return 1 + (indices_.size() - 1) / batch_size_;
}

void SparseFeatureDatasetBatcher::SetNormalization(
    float mean,
    float std_dev) {
    if (!std::isfinite(mean) || !std::isfinite(std_dev) || std_dev == 0.0f) {
        throw std::invalid_argument(
            "SparseFeatureDatasetBatcher normalization requires finite mean "
            "and nonzero finite standard deviation");
    }
    normalize_ = true;
    norm_mean_ = mean;
    norm_std_ = std_dev;
    if (sparse_feature_output_ && norm_mean_ != 0.0f) {
        sparse_feature_output_ = false;
        ValidateDenseBatchMemory(std::min(batch_size_, indices_.size()));
        spdlog::info(
            "SparseFeatureDatasetBatcher: non-zero mean normalization "
            "requires the guarded dense batch path");
    }
}

void SparseFeatureDatasetBatcher::SetSparseFeatureOutput(bool enable) {
    sparse_feature_output_ = enable && (!normalize_ || norm_mean_ == 0.0f);
    if (!sparse_feature_output_) {
        ValidateDenseBatchMemory(std::min(batch_size_, indices_.size()));
    }
}

void SparseFeatureDatasetBatcher::SetOneHotEncoding(size_t num_classes) {
    if (num_classes == 0) {
        throw std::invalid_argument(
            "SparseFeatureDatasetBatcher one-hot width must be positive");
    }
    one_hot_ = true;
    num_classes_ = num_classes;
    scalar_label_mode_ = false;
    class_index_label_mode_ = false;
}

void SparseFeatureDatasetBatcher::SetScalarLabelMode(bool enable) {
    scalar_label_mode_ = enable;
    if (enable) {
        one_hot_ = false;
        class_index_label_mode_ = false;
    }
}

void SparseFeatureDatasetBatcher::SetClassIndexLabelMode(bool enable) {
    class_index_label_mode_ = enable;
    if (enable) {
        one_hot_ = false;
        scalar_label_mode_ = false;
    }
}

void SparseFeatureDatasetBatcher::ShuffleIndices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

void SparseFeatureDatasetBatcher::PopulateInspection(
    Batch& batch,
    size_t batch_start,
    size_t actual_batch_size) const {
    auto& inspection = batch.inspection;
    inspection.available = true;
    inspection.row_count = actual_batch_size;
    inspection.feature_column_count =
        static_cast<size_t>(dataset_->GetNumFeatures());
    inspection.label_column_count = dataset_->GetLabels() ? 1 : 0;
    inspection.feature_columns_truncated =
        inspection.feature_column_count > kInspectionPreviewLimit;
    const size_t preview_count = std::min(
        inspection.feature_column_count, kInspectionPreviewLimit);
    inspection.feature_columns_preview.reserve(preview_count);
    const auto& names = dataset_->GetFeatureNames();
    for (size_t feature = 0; feature < preview_count; ++feature) {
        inspection.feature_columns_preview.push_back({
            names.empty() ? "feature_" + std::to_string(feature)
                          : names[feature],
            "float32 (CSR)"});
    }
    if (const auto& labels = dataset_->GetLabels()) {
        inspection.label_columns_preview.push_back({
            dataset_->GetLabelName(), labels->type()->ToString()});
    }
    for (size_t row = 0; row < actual_batch_size; ++row) {
        const int64_t source_row = indices_[batch_start + row];
        inspection.inspected_value_count += static_cast<uint64_t>(
            dataset_->GetRowOffsets()[static_cast<size_t>(source_row + 1)] -
            dataset_->GetRowOffsets()[static_cast<size_t>(source_row)]);
    }
    inspection.null_summary_available = false;
    inspection.token_sequence_columns = false;
}

} // namespace cyxwiz
