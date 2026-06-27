#pragma once

#include <arrow/api.h>
#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <algorithm>
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

inline arrow::Result<std::shared_ptr<arrow::Table>> AddSplitPartitionColumn(
    const std::shared_ptr<arrow::Table>& table,
    const SplitPartitionOptions& options) {

    if (!table) {
        return arrow::Status::Invalid(
            options.log_context + ": cannot partition a null table");
    }

    const size_t total = static_cast<size_t>(table->num_rows());
    const auto split = split_partition_detail::Normalize(
        options.train_split, options.val_split, options.test_split);

    std::vector<int8_t> partitions(total, 0);
    split_partition_detail::SplitCounts totals;

    if (options.stratified) {
        ARROW_ASSIGN_OR_RAISE(
            auto labels,
            split_partition_detail::ReadLabels(table, options.label_column));
        if (labels.size() != total) {
            return arrow::Status::Invalid(
                "stratified split label count does not match row count");
        }

        std::map<int64_t, std::vector<size_t>> by_label;
        for (size_t i = 0; i < labels.size(); ++i) {
            by_label[labels[i]].push_back(i);
        }

        std::mt19937 rng(options.seed);
        for (auto& entry : by_label) {
            auto& rows = entry.second;
            if (options.shuffle) {
                std::shuffle(rows.begin(), rows.end(), rng);
            }

            const auto counts =
                split_partition_detail::CountsFor(rows.size(), split);
            totals.train += counts.train;
            totals.val += counts.val;
            totals.test += counts.test;

            for (size_t i = 0; i < rows.size(); ++i) {
                int8_t partition = 0;
                if (i >= counts.train + counts.val) {
                    partition = 2;
                } else if (i >= counts.train) {
                    partition = 1;
                }
                partitions[rows[i]] = partition;
            }
        }

        spdlog::info("{}: stratified split assigned {} labels "
                     "({} train / {} val / {} test rows)",
                     options.log_context, by_label.size(),
                     totals.train, totals.val, totals.test);
    } else {
        const auto counts = split_partition_detail::CountsFor(total, split);
        totals = counts;

        std::vector<size_t> order(total);
        std::iota(order.begin(), order.end(), 0);
        if (options.shuffle) {
            std::mt19937 rng(options.seed);
            std::shuffle(order.begin(), order.end(), rng);
        }

        for (size_t i = 0; i < order.size(); ++i) {
            int8_t partition = 0;
            if (i >= counts.train + counts.val) {
                partition = 2;
            } else if (i >= counts.train) {
                partition = 1;
            }
            partitions[order[i]] = partition;
        }

        spdlog::info("{}: split assigned {} train / {} val / {} test rows",
                     options.log_context, totals.train, totals.val, totals.test);
    }

    return split_partition_detail::AppendPartitionColumn(table, partitions);
}

} // namespace cyxwiz
