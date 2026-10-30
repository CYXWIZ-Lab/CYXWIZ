#pragma once

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/compute/api.h>
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
#include <unordered_map>
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
    // Train gets at least one row when it has a share; a 0 / 0 / 1 split (a
    // dataset that is the whole test part) keeps every row in its part.
    if (total > 0 && counts.train == 0 && split.train > 0.0f) counts.train = 1;
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

// Each row's class, numbered from 0 in order of first appearance: numbers
// by value, text by string (a label not yet encoded, TOFIX178 S0-g2). The
// same classes give the same ids whichever way the label is stored.
inline arrow::Result<std::vector<uint32_t>> ClassIdsOfColumn(const std::shared_ptr<arrow::Table>& table,
                                                            const std::string& label_column) {
    if (label_column.empty()) return arrow::Status::Invalid("stratified split requires an explicit label column");
    auto column = table->GetColumnByName(label_column);
    if (!column) return arrow::Status::Invalid("stratified split label column '" + label_column + "' was not found");
    std::vector<uint32_t> ids;
    ids.reserve(static_cast<size_t>(table->num_rows()));
    const auto type = column->type()->id();
    if (type == arrow::Type::STRING || type == arrow::Type::LARGE_STRING) {
        std::unordered_map<std::string, uint32_t> seen;
        for (const auto& chunk : column->chunks()) {
            for (int64_t i = 0; i < chunk->length(); ++i) {
                if (chunk->IsNull(i)) return arrow::Status::Invalid("stratified split requires non-null labels");
                const std::string value = type == arrow::Type::STRING
                    ? std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(i)
                    : std::static_pointer_cast<arrow::LargeStringArray>(chunk)->GetString(i);
                ids.push_back(seen.emplace(value, static_cast<uint32_t>(seen.size())).first->second);
            }
        }
        return ids;
    }
    ARROW_ASSIGN_OR_RAISE(auto labels, split_partition_detail::ReadLabels(table, label_column));
    std::map<int64_t, uint32_t> seen;
    for (const int64_t label : labels) ids.push_back(seen.emplace(label, static_cast<uint32_t>(seen.size())).first->second);
    return ids;
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
        ARROW_ASSIGN_OR_RAISE(class_of_row, ClassIdsOfColumn(table, options.label_column));
        if (class_of_row.size() != total) {
            return arrow::Status::Invalid(
                "stratified split label count does not match row count");
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

// The rows a step learns from (TOFIX178 S0-g3): the Train rows (partition
// 0) of a table the pipeline run marked with "__partition__", like
// scikit-learn's fit on the training split; the whole table when unmarked.
// The step then transforms every row.
inline arrow::Result<std::shared_ptr<arrow::Table>> TrainRowsOf(const std::shared_ptr<arrow::Table>& table) {
    if (!table) return table;
    const auto column = table->GetColumnByName(kSplitPartitionColumn);
    if (!column) return table;
    arrow::BooleanBuilder keep;
    ARROW_RETURN_NOT_OK(keep.Reserve(table->num_rows()));
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() != arrow::Type::INT8)
            return arrow::Status::Invalid(std::string(kSplitPartitionColumn) + " must be int8");
        const auto values = std::static_pointer_cast<arrow::Int8Array>(chunk);
        for (int64_t i = 0; i < values->length(); ++i) keep.UnsafeAppend(!values->IsNull(i) && values->Value(i) == 0);
    }
    std::shared_ptr<arrow::Array> mask;
    ARROW_RETURN_NOT_OK(keep.Finish(&mask));
    ARROW_ASSIGN_OR_RAISE(auto filtered, arrow::compute::Filter(table, mask));
    return filtered.table();
}

// `marked` (a training input's rows with their partition) followed by the
// rows of a supplied Validation (partition 1) or Test (2) input (TOFIX178
// S0-g4), so the steps after it transform them with what they learned from
// the Train rows. The supplied rows take the training input's columns, by
// name (extra columns are left out); its label column `supplied_label`
// stands for `label` when they differ. Numeric columns of different widths
// (a CSV's inferred uint8 and uint16) both become int64, or float64 when one
// is floating; other types are cast to the training input's. A missing
// column, or a type that cannot be cast, is an error naming `role`.
inline arrow::Result<std::shared_ptr<arrow::Table>> AppendSuppliedRows(
    const std::shared_ptr<arrow::Table>& marked, const std::shared_ptr<arrow::Table>& supplied, int8_t partition,
    const std::string& label, const std::string& supplied_label, const std::string& role) {
    if (!marked || !supplied) return arrow::Status::Invalid(role + " input: no rows");
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> base_columns, added_columns;
    for (int i = 0; i < marked->num_columns(); ++i) {
        const auto field = marked->schema()->field(i);
        auto base = marked->column(i);
        if (field->name() == kSplitPartitionColumn) {
            arrow::Int8Builder parts;
            ARROW_RETURN_NOT_OK(parts.AppendValues(std::vector<int8_t>(static_cast<size_t>(supplied->num_rows()), partition)));
            ARROW_ASSIGN_OR_RAISE(auto array, parts.Finish());
            fields.push_back(field);
            base_columns.push_back(base);
            added_columns.push_back(std::make_shared<arrow::ChunkedArray>(array));
            continue;
        }
        auto column = supplied->GetColumnByName(field->name());
        if (!column && field->name() == label && !supplied_label.empty())
            column = supplied->GetColumnByName(supplied_label);
        if (!column)
            return arrow::Status::Invalid(role + " input has no column '" + field->name() +
                                          "' (it needs the training input's columns)");
        std::shared_ptr<arrow::DataType> type = field->type();
        if (!column->type()->Equals(*type) && arrow::is_numeric(type->id()) &&
            arrow::is_numeric(column->type()->id())) {
            type = arrow::is_floating(type->id()) || arrow::is_floating(column->type()->id())
                ? arrow::float64() : arrow::int64();
        }
        const auto cast_to = [&](std::shared_ptr<arrow::ChunkedArray>& values) -> arrow::Status {
            if (values->type()->Equals(*type)) return arrow::Status::OK();
            auto cast = arrow::compute::Cast(values, type);
            if (!cast.ok())
                return arrow::Status::Invalid(role + " input column '" + field->name() + "' is " +
                                              values->type()->ToString() + ", not " + type->ToString());
            values = cast->chunked_array();
            return arrow::Status::OK();
        };
        ARROW_RETURN_NOT_OK(cast_to(column));
        ARROW_RETURN_NOT_OK(cast_to(base));
        fields.push_back(field->WithType(type));
        base_columns.push_back(base);
        added_columns.push_back(column);
    }
    const auto schema = arrow::schema(fields, marked->schema()->metadata());
    const auto base = arrow::Table::Make(schema, base_columns, marked->num_rows());
    const auto added = arrow::Table::Make(schema, added_columns, supplied->num_rows());
    ARROW_ASSIGN_OR_RAISE(auto joined, arrow::ConcatenateTables({base, added}));
    return joined;
}

// `output` with `input`'s row partition, for a step that builds a new table
// with one row per input row (vectorizers, the tokenizer): training then
// reads the partition the step learned from.
inline arrow::Result<std::shared_ptr<arrow::Table>> CarryPartition(const std::shared_ptr<arrow::Table>& input,
                                                                   const std::shared_ptr<arrow::Table>& output) {
    const auto column = input ? input->GetColumnByName(kSplitPartitionColumn) : nullptr;
    if (!column || !output || output->schema()->GetFieldIndex(kSplitPartitionColumn) >= 0) return output;
    if (output->num_rows() != input->num_rows())
        return arrow::Status::Invalid(std::string(kSplitPartitionColumn) + ": the output has another row count");
    return output->AddColumn(output->num_columns(), arrow::field(kSplitPartitionColumn, arrow::int8()), column);
}

// Each row's partition of a marked table (0 train, 1 validation, 2 test);
// empty when the table is not marked.
inline std::vector<int8_t> PartitionValues(const std::shared_ptr<arrow::Table>& table) {
    std::vector<int8_t> parts;
    const auto column = table ? table->GetColumnByName(kSplitPartitionColumn) : nullptr;
    if (!column) return parts;
    parts.reserve(static_cast<size_t>(table->num_rows()));
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() != arrow::Type::INT8) return {};
        const auto values = std::static_pointer_cast<arrow::Int8Array>(chunk);
        for (int64_t i = 0; i < values->length(); ++i) parts.push_back(values->IsNull(i) ? 0 : values->Value(i));
    }
    return parts;
}

// Which rows are Train rows, for a step that reads its rows itself; empty
// when the table is not marked (every row learns).
inline std::vector<bool> TrainRowMask(const std::shared_ptr<arrow::Table>& table) {
    std::vector<bool> mask;
    const auto column = table ? table->GetColumnByName(kSplitPartitionColumn) : nullptr;
    if (!column) return mask;
    mask.reserve(static_cast<size_t>(table->num_rows()));
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() != arrow::Type::INT8) return {};
        const auto values = std::static_pointer_cast<arrow::Int8Array>(chunk);
        for (int64_t i = 0; i < values->length(); ++i) mask.push_back(!values->IsNull(i) && values->Value(i) == 0);
    }
    return mask;
}

inline arrow::Result<std::shared_ptr<arrow::RecordBatch>> TrainRowsOf(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (!batch || batch->schema()->GetFieldIndex(kSplitPartitionColumn) < 0) return batch;
    ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches({batch}));
    ARROW_ASSIGN_OR_RAISE(auto rows, TrainRowsOf(table));
    ARROW_ASSIGN_OR_RAISE(auto combined, rows->CombineChunksToBatch());
    return combined;
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
