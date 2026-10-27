#pragma once

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kSplitPartitionColumn = "__partition__";

struct SplitPartitionOptions {
    std::string label_column;
    float train_split = 0.8f;
    float val_split = 0.1f;
    float test_split = 0.1f;
    bool shuffle = true;
    uint32_t seed = 42;
    bool stratified = false;
    std::string log_context = "SplitPartitioning";
};

namespace split_partition_detail {

struct NormalizedSplit {
    float train = 0.8f;
    float val = 0.1f;
    float test = 0.1f;
};

struct SplitCounts {
    size_t train = 0;
    size_t val = 0;
    size_t test = 0;
};

inline NormalizedSplit Normalize(float train, float val, float test) {
    float split_sum = train + val + test;
    if (!(split_sum > 0.0f)) {
        return {};
    }
    return {train / split_sum, val / split_sum, test / split_sum};
}

inline SplitCounts CountsFor(size_t total, const NormalizedSplit& split) {
    SplitCounts counts;
    counts.train = static_cast<size_t>(total * split.train);
    counts.val = static_cast<size_t>(total * split.val);
    if (total > 0 && counts.train == 0) counts.train = 1;
    if (counts.train > total) counts.train = total;
    if (counts.train + counts.val > total) {
        counts.val = total - counts.train;
    }
    counts.test = total - counts.train - counts.val;
    return counts;
}

template <typename ArrayType>
arrow::Status AppendNumericLabels(const std::shared_ptr<arrow::Array>& chunk,
                                  std::vector<int64_t>& labels) {
    auto typed = std::static_pointer_cast<ArrayType>(chunk);
    for (int64_t i = 0; i < typed->length(); ++i) {
        if (typed->IsNull(i)) {
            return arrow::Status::Invalid(
                "stratified split requires non-null labels");
        }
        labels.push_back(static_cast<int64_t>(typed->Value(i)));
    }
    return arrow::Status::OK();
}

inline arrow::Result<std::vector<int64_t>> ReadLabels(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& label_column) {

    if (label_column.empty()) {
        return arrow::Status::Invalid(
            "stratified split requires an explicit label column");
    }
    auto column = table->GetColumnByName(label_column);
    if (!column || column->num_chunks() == 0) {
        return arrow::Status::Invalid(
            "stratified split label column '" + label_column + "' was not found");
    }

    std::vector<int64_t> labels;
    labels.reserve(static_cast<size_t>(table->num_rows()));
    for (int c = 0; c < column->num_chunks(); ++c) {
        auto chunk = column->chunk(c);
        switch (chunk->type_id()) {
            case arrow::Type::INT8:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::Int8Array>(chunk, labels));
                break;
            case arrow::Type::INT16:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::Int16Array>(chunk, labels));
                break;
            case arrow::Type::INT32:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::Int32Array>(chunk, labels));
                break;
            case arrow::Type::INT64:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::Int64Array>(chunk, labels));
                break;
            case arrow::Type::UINT8:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::UInt8Array>(chunk, labels));
                break;
            case arrow::Type::UINT16:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::UInt16Array>(chunk, labels));
                break;
            case arrow::Type::UINT32:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::UInt32Array>(chunk, labels));
                break;
            case arrow::Type::UINT64:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::UInt64Array>(chunk, labels));
                break;
            case arrow::Type::FLOAT:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::FloatArray>(chunk, labels));
                break;
            case arrow::Type::DOUBLE:
                ARROW_RETURN_NOT_OK(
                    AppendNumericLabels<arrow::DoubleArray>(chunk, labels));
                break;
            default:
                return arrow::Status::Invalid(
                    "stratified split label column '" + label_column +
                    "' has unsupported type " + chunk->type()->ToString());
        }
    }
    return labels;
}

inline arrow::Result<std::shared_ptr<arrow::Table>> AppendPartitionColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::vector<int8_t>& partitions) {

    arrow::Int8Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(
        static_cast<int64_t>(partitions.size())));
    for (int8_t partition : partitions) {
        ARROW_RETURN_NOT_OK(builder.Append(partition));
    }

    std::shared_ptr<arrow::Array> partition_array;
    ARROW_RETURN_NOT_OK(builder.Finish(&partition_array));

    auto out = table;
    const int existing =
        out->schema()->GetFieldIndex(kSplitPartitionColumn);
    if (existing >= 0) {
        ARROW_ASSIGN_OR_RAISE(out, out->RemoveColumn(existing));
    }

    return out->AddColumn(
        out->num_columns(),
        arrow::field(kSplitPartitionColumn, arrow::int8()),
        std::make_shared<arrow::ChunkedArray>(partition_array));
}

} // namespace split_partition_detail

// The split rule every training path uses (TOFIX178 S0-g1): each class gets
// the same share counts (CountsFor), and the rows, in file order, fill them by
// a draw without replacement - the same as shuffling each class and slicing,
// but one pass with a few counters, so a table on disk splits like one in
// memory. Unshuffled, each class fills Train, then Validation, then Test in
// file order. The draws are portable (a raw 64-bit Mersenne Twister and a
// rejection draw, not std::shuffle or a standard distribution, which differ
// between C++ libraries), so Windows, Linux and macOS split alike.
class SplitAssigner {
public:
    // `class_rows`: the rows of each class (one entry when not stratified).
    SplitAssigner(const std::vector<uint64_t>& class_rows, float train, float val, float test,
                  bool shuffle, uint32_t seed)
        : shuffle_(shuffle), rng_(seed) {
        const auto split = split_partition_detail::Normalize(train, val, test);
        remaining_.reserve(class_rows.size());
        for (const uint64_t rows : class_rows) {
            const auto counts = split_partition_detail::CountsFor(static_cast<size_t>(rows), split);
            remaining_.push_back({counts.train, counts.val, counts.test});
        }
    }

    // The partition (0 train, 1 validation, 2 test) of the next row, which
    // belongs to class `class_id`.
    int8_t Next(size_t class_id) {
        auto& left = remaining_.at(class_id);
        const uint64_t total = left[0] + left[1] + left[2];
        uint64_t draw = shuffle_ && total > 0 ? Below(total) : 0;
        int8_t partition = 0;
        while (partition < 2 && draw >= left[static_cast<size_t>(partition)]) {
            draw -= left[static_cast<size_t>(partition)];
            ++partition;
        }
        if (left[static_cast<size_t>(partition)] > 0) --left[static_cast<size_t>(partition)];
        return partition;
    }

private:
    // Uniform in [0, bound): rejection sampling on the raw 64-bit output.
    uint64_t Below(uint64_t bound) {
        const uint64_t limit = std::numeric_limits<uint64_t>::max() - std::numeric_limits<uint64_t>::max() % bound;
        uint64_t value = rng_();
        while (value >= limit) value = rng_();
        return value % bound;
    }

    bool shuffle_;
    std::mt19937_64 rng_;
    std::vector<std::array<uint64_t, 3>> remaining_;
};

// Each row's partition by SplitAssigner. `class_of_row` gives each row's
// class (class ids numbered from 0 in order of first appearance); empty when
// not stratified.
inline std::vector<int8_t> AssignSplitPartitions(size_t rows, const std::vector<uint32_t>& class_of_row,
                                                 float train, float val, float test, bool shuffle,
                                                 uint32_t seed) {
    std::vector<uint64_t> class_rows;
    if (class_of_row.empty()) {
        class_rows.push_back(rows);
    } else {
        for (const uint32_t c : class_of_row) {
            if (c >= class_rows.size()) class_rows.resize(c + 1, 0);
            ++class_rows[c];
        }
    }
    SplitAssigner assigner(class_rows, train, val, test, shuffle, seed);
    std::vector<int8_t> partitions(rows, 0);
    for (size_t row = 0; row < rows; ++row)
        partitions[row] = assigner.Next(class_of_row.empty() ? 0 : class_of_row[row]);
    return partitions;
}

inline arrow::Result<std::shared_ptr<arrow::Table>> AddSplitPartitionColumn(
    const std::shared_ptr<arrow::Table>& table,
    const SplitPartitionOptions& options) {

    if (!table) {
        return arrow::Status::Invalid(
            options.log_context + ": cannot partition a null table");
    }

    const size_t total = static_cast<size_t>(table->num_rows());
    std::vector<uint32_t> class_of_row;
    if (options.stratified) {
        ARROW_ASSIGN_OR_RAISE(
            auto labels,
            split_partition_detail::ReadLabels(table, options.label_column));
        if (labels.size() != total) {
            return arrow::Status::Invalid(
                "stratified split label count does not match row count");
        }
        std::map<int64_t, uint32_t> ids;
        class_of_row.reserve(total);
        for (const int64_t label : labels) {
            const auto [it, added] = ids.emplace(label, static_cast<uint32_t>(ids.size()));
            (void)added;
            class_of_row.push_back(it->second);
        }
    }
    const auto partitions = AssignSplitPartitions(total, class_of_row, options.train_split, options.val_split,
                                                  options.test_split, options.shuffle, options.seed);
    size_t counts[3] = {0, 0, 0};
    for (const int8_t partition : partitions) ++counts[static_cast<size_t>(partition)];
    spdlog::info("{}: {}split assigned {} train / {} val / {} test rows", options.log_context,
                 options.stratified ? "stratified " : "", counts[0], counts[1], counts[2]);
    return split_partition_detail::AppendPartitionColumn(table, partitions);
}

inline arrow::Result<std::vector<int64_t>> ReadNumericLabelColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& label_column) {
    if (!table) {
        return arrow::Status::Invalid("cannot read labels from a null table");
    }
    return split_partition_detail::ReadLabels(table, label_column);
}

} // namespace cyxwiz
