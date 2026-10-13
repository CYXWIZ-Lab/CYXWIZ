#include "arrow_data_table.h"

#include "../data/data_table.h"

#include <cstdio>

namespace cyxwiz {

namespace {

// The chunk holding `row` and the row inside it.
bool Locate(const arrow::ChunkedArray& column, int64_t row, std::shared_ptr<arrow::Array>& chunk, int64_t& at) {
    int64_t offset = 0;
    for (const auto& c : column.chunks()) {
        if (row < offset + c->length()) {
            chunk = c;
            at = row - offset;
            return true;
        }
        offset += c->length();
    }
    return false;
}

DataTable::CellValue CellOf(const arrow::ChunkedArray& column, int64_t row) {
    std::shared_ptr<arrow::Array> chunk;
    int64_t at = 0;
    if (!Locate(column, row, chunk, at) || chunk->IsNull(at)) return std::monostate{};
    switch (chunk->type_id()) {
        case arrow::Type::INT8: return static_cast<int64_t>(std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(at));
        case arrow::Type::INT16: return static_cast<int64_t>(std::static_pointer_cast<arrow::Int16Array>(chunk)->Value(at));
        case arrow::Type::INT32: return static_cast<int64_t>(std::static_pointer_cast<arrow::Int32Array>(chunk)->Value(at));
        case arrow::Type::INT64: return std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(at);
        case arrow::Type::UINT8: return static_cast<int64_t>(std::static_pointer_cast<arrow::UInt8Array>(chunk)->Value(at));
        case arrow::Type::UINT16: return static_cast<int64_t>(std::static_pointer_cast<arrow::UInt16Array>(chunk)->Value(at));
        case arrow::Type::UINT32: return static_cast<int64_t>(std::static_pointer_cast<arrow::UInt32Array>(chunk)->Value(at));
        case arrow::Type::UINT64: return static_cast<int64_t>(std::static_pointer_cast<arrow::UInt64Array>(chunk)->Value(at));
        case arrow::Type::FLOAT: return static_cast<double>(std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(at));
        case arrow::Type::DOUBLE: return std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(at);
        case arrow::Type::STRING: return std::static_pointer_cast<arrow::StringArray>(chunk)->GetString(at);
        case arrow::Type::LARGE_STRING: return std::static_pointer_cast<arrow::LargeStringArray>(chunk)->GetString(at);
        case arrow::Type::BOOL: return std::string(std::static_pointer_cast<arrow::BooleanArray>(chunk)->Value(at) ? "true" : "false");
        default: break;
    }
    auto scalar = chunk->GetScalar(at);
    return scalar.ok() ? (*scalar)->ToString() : std::string();
}

}  // namespace

std::string ArrowCellText(const arrow::ChunkedArray& column, int64_t row) {
    const DataTable::CellValue v = CellOf(column, row);
    if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
    if (std::holds_alternative<int64_t>(v)) return std::to_string(std::get<int64_t>(v));
    if (std::holds_alternative<double>(v)) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%.6g", std::get<double>(v));
        return buf;
    }
    return "";
}

std::shared_ptr<DataTable> DataTableFromArrow(const arrow::Table& table, const std::string& name, size_t max_rows) {
    auto out = std::make_shared<DataTable>();
    out->SetName(name);
    std::vector<std::string> headers;
    for (const auto& f : table.schema()->fields()) headers.push_back(f->name());
    out->SetHeaders(headers);
    const int64_t rows = max_rows > 0 ? std::min<int64_t>(table.num_rows(), static_cast<int64_t>(max_rows)) : table.num_rows();
    for (int64_t r = 0; r < rows; ++r) {
        DataTable::Row row;
        row.reserve(static_cast<size_t>(table.num_columns()));
        for (int c = 0; c < table.num_columns(); ++c) row.push_back(CellOf(*table.column(c), r));
        out->AddRow(std::move(row));
    }
    return out;
}

}  // namespace cyxwiz
