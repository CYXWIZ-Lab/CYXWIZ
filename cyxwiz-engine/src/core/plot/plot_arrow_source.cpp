#include "plot_arrow_source.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz::plot {

namespace {

bool IsNumeric(const arrow::DataType& t) {
    switch (t.id()) {
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::UINT8:
        case arrow::Type::UINT16:
        case arrow::Type::UINT32:
        case arrow::Type::UINT64:
        case arrow::Type::HALF_FLOAT:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::BOOL: return true;
        default: return false;
    }
}

template <typename ArrayType>
void AppendNumbers(const arrow::Array& chunk, size_t take, std::vector<double>& out) {
    const auto& a = static_cast<const ArrayType&>(chunk);
    for (int64_t i = 0; i < a.length() && out.size() < take; ++i)
        out.push_back(a.IsNull(i) ? NAN : static_cast<double>(a.Value(i)));
}

void AppendNumberChunk(const arrow::Array& chunk, size_t take, std::vector<double>& out) {
    switch (chunk.type_id()) {
        case arrow::Type::INT8: AppendNumbers<arrow::Int8Array>(chunk, take, out); break;
        case arrow::Type::INT16: AppendNumbers<arrow::Int16Array>(chunk, take, out); break;
        case arrow::Type::INT32: AppendNumbers<arrow::Int32Array>(chunk, take, out); break;
        case arrow::Type::INT64: AppendNumbers<arrow::Int64Array>(chunk, take, out); break;
        case arrow::Type::UINT8: AppendNumbers<arrow::UInt8Array>(chunk, take, out); break;
        case arrow::Type::UINT16: AppendNumbers<arrow::UInt16Array>(chunk, take, out); break;
        case arrow::Type::UINT32: AppendNumbers<arrow::UInt32Array>(chunk, take, out); break;
        case arrow::Type::UINT64: AppendNumbers<arrow::UInt64Array>(chunk, take, out); break;
        case arrow::Type::FLOAT: AppendNumbers<arrow::FloatArray>(chunk, take, out); break;
        case arrow::Type::DOUBLE: AppendNumbers<arrow::DoubleArray>(chunk, take, out); break;
        case arrow::Type::BOOL: {
            const auto& a = static_cast<const arrow::BooleanArray&>(chunk);
            for (int64_t i = 0; i < a.length() && out.size() < take; ++i)
                out.push_back(a.IsNull(i) ? NAN : (a.Value(i) ? 1.0 : 0.0));
            break;
        }
        default: {  // half floats and anything else numeric-like: via scalars
            for (int64_t i = 0; i < chunk.length() && out.size() < take; ++i) {
                auto s = chunk.GetScalar(i);
                double v = NAN;
                if (s.ok() && (*s)->is_valid) {
                    auto d = (*s)->CastTo(arrow::float64());
                    if (d.ok()) v = std::static_pointer_cast<arrow::DoubleScalar>(*d)->value;
                }
                out.push_back(v);
            }
        }
    }
}

void AppendTextChunk(const arrow::Array& chunk, size_t take, std::vector<std::string>& out) {
    if (chunk.type_id() == arrow::Type::STRING) {
        const auto& a = static_cast<const arrow::StringArray&>(chunk);
        for (int64_t i = 0; i < a.length() && out.size() < take; ++i)
            out.push_back(a.IsNull(i) ? std::string() : a.GetString(i));
        return;
    }
    if (chunk.type_id() == arrow::Type::LARGE_STRING) {
        const auto& a = static_cast<const arrow::LargeStringArray&>(chunk);
        for (int64_t i = 0; i < a.length() && out.size() < take; ++i)
            out.push_back(a.IsNull(i) ? std::string() : a.GetString(i));
        return;
    }
    for (int64_t i = 0; i < chunk.length() && out.size() < take; ++i) {
        auto s = chunk.GetScalar(i);
        out.push_back(s.ok() && (*s)->is_valid ? (*s)->ToString() : std::string());
    }
}

// One value of a chunk (NaN / "" when null).
double NumberAt(const arrow::Array& a, int64_t i) {
    if (a.IsNull(i)) return NAN;
    switch (a.type_id()) {
        case arrow::Type::INT8: return static_cast<const arrow::Int8Array&>(a).Value(i);
        case arrow::Type::INT16: return static_cast<const arrow::Int16Array&>(a).Value(i);
        case arrow::Type::INT32: return static_cast<const arrow::Int32Array&>(a).Value(i);
        case arrow::Type::INT64: return static_cast<double>(static_cast<const arrow::Int64Array&>(a).Value(i));
        case arrow::Type::UINT8: return static_cast<const arrow::UInt8Array&>(a).Value(i);
        case arrow::Type::UINT16: return static_cast<const arrow::UInt16Array&>(a).Value(i);
        case arrow::Type::UINT32: return static_cast<const arrow::UInt32Array&>(a).Value(i);
        case arrow::Type::UINT64: return static_cast<double>(static_cast<const arrow::UInt64Array&>(a).Value(i));
        case arrow::Type::FLOAT: return static_cast<const arrow::FloatArray&>(a).Value(i);
        case arrow::Type::DOUBLE: return static_cast<const arrow::DoubleArray&>(a).Value(i);
        case arrow::Type::BOOL: return static_cast<const arrow::BooleanArray&>(a).Value(i) ? 1.0 : 0.0;
        default: {
            auto sc = a.GetScalar(i);
            if (!sc.ok() || !(*sc)->is_valid) return NAN;
            auto d = (*sc)->CastTo(arrow::float64());
            return d.ok() ? std::static_pointer_cast<arrow::DoubleScalar>(*d)->value : NAN;
        }
    }
}

std::string TextAt(const arrow::Array& a, int64_t i) {
    if (a.IsNull(i)) return std::string();
    if (a.type_id() == arrow::Type::STRING) return static_cast<const arrow::StringArray&>(a).GetString(i);
    if (a.type_id() == arrow::Type::LARGE_STRING) return static_cast<const arrow::LargeStringArray&>(a).GetString(i);
    auto sc = a.GetScalar(i);
    return sc.ok() && (*sc)->is_valid ? (*sc)->ToString() : std::string();
}

}  // namespace

std::vector<ArrowColumnInfo> ArrowColumns(const arrow::Table& table) {
    std::vector<ArrowColumnInfo> out;
    for (const auto& f : table.schema()->fields()) out.push_back({f->name(), IsNumeric(*f->type())});
    return out;
}

Source SourceFromArrow(const arrow::Table& table, const std::vector<std::string>& columns, size_t max_rows) {
    Source src;
    const size_t rows = static_cast<size_t>(table.num_rows());
    const size_t take = max_rows > 0 && max_rows < rows ? max_rows : rows;
    if (take < rows) {
        src.row_limit = take;
        src.total_rows = rows;
    }
    for (const auto& name : columns) {
        if (name.empty()) continue;
        bool seen = false;
        for (const auto& c : src.columns) seen = seen || c.name == name;
        if (seen) continue;
        const int index = table.schema()->GetFieldIndex(name);
        if (index < 0) continue;
        const auto& column = table.column(index);
        SourceColumn col;
        col.name = name;
        col.numeric = IsNumeric(*column->type());
        if (col.numeric) col.numbers.reserve(take);
        else col.text.reserve(take);
        for (const auto& chunk : column->chunks()) {
            if (col.size() >= take) break;
            if (col.numeric) AppendNumberChunk(*chunk, take, col.numbers);
            else AppendTextChunk(*chunk, take, col.text);
        }
        src.columns.push_back(std::move(col));
    }
    return src;
}

Source SourceFromArrowRows(const arrow::Table& table, const std::vector<std::string>& columns, const std::vector<size_t>& rows) {
    Source src;
    for (const auto& name : columns) {
        if (name.empty()) continue;
        bool seen = false;
        for (const auto& c : src.columns) seen = seen || c.name == name;
        if (seen) continue;
        const int index = table.schema()->GetFieldIndex(name);
        if (index < 0) continue;
        const auto& column = table.column(index);
        // Where each chunk starts, to find a row's chunk.
        std::vector<int64_t> starts;
        int64_t at = 0;
        for (const auto& chunk : column->chunks()) {
            starts.push_back(at);
            at += chunk->length();
        }
        SourceColumn col;
        col.name = name;
        col.numeric = IsNumeric(*column->type());
        for (size_t r : rows) {
            const auto it = std::upper_bound(starts.begin(), starts.end(), static_cast<int64_t>(r));
            const size_t k = static_cast<size_t>(it - starts.begin()) - 1;
            const int64_t i = static_cast<int64_t>(r) - starts[k];
            const auto& chunk = *column->chunk(static_cast<int>(k));
            if (i < 0 || i >= chunk.length()) {
                if (col.numeric) col.numbers.push_back(NAN);
                else col.text.emplace_back();
                continue;
            }
            if (col.numeric) col.numbers.push_back(NumberAt(chunk, i));
            else col.text.push_back(TextAt(chunk, i));
        }
        src.columns.push_back(std::move(col));
    }
    return src;
}

}  // namespace cyxwiz::plot
