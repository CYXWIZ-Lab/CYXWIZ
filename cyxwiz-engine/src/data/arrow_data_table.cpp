#include "arrow_data_table.h"

#include "data_table.h"

#include <arrow/api.h>

#include <algorithm>
#include <vector>

namespace cyxwiz {

namespace {
DataTable::CellValue CellOf(const arrow::Array& array, int64_t i) {
    if (array.IsNull(i)) return std::monostate{};
    switch (array.type_id()) {
        case arrow::Type::INT8: return static_cast<int64_t>(static_cast<const arrow::Int8Array&>(array).Value(i));
        case arrow::Type::INT16: return static_cast<int64_t>(static_cast<const arrow::Int16Array&>(array).Value(i));
        case arrow::Type::INT32: return static_cast<int64_t>(static_cast<const arrow::Int32Array&>(array).Value(i));
        case arrow::Type::INT64: return static_cast<const arrow::Int64Array&>(array).Value(i);
        case arrow::Type::UINT8: return static_cast<int64_t>(static_cast<const arrow::UInt8Array&>(array).Value(i));
        case arrow::Type::UINT16: return static_cast<int64_t>(static_cast<const arrow::UInt16Array&>(array).Value(i));
        case arrow::Type::UINT32: return static_cast<int64_t>(static_cast<const arrow::UInt32Array&>(array).Value(i));
        case arrow::Type::FLOAT: return static_cast<double>(static_cast<const arrow::FloatArray&>(array).Value(i));
        case arrow::Type::DOUBLE: return static_cast<const arrow::DoubleArray&>(array).Value(i);
        case arrow::Type::STRING: return std::string(static_cast<const arrow::StringArray&>(array).GetView(i));
        case arrow::Type::LARGE_STRING:
            return std::string(static_cast<const arrow::LargeStringArray&>(array).GetView(i));
        case arrow::Type::BOOL: return std::string(static_cast<const arrow::BooleanArray&>(array).Value(i) ? "true" : "false");
        default: break;
    }
    auto scalar = array.GetScalar(i);
    return scalar.ok() ? (*scalar)->ToString() : std::string();
}
}  // namespace

std::shared_ptr<DataTable> DataTableFromArrow(const arrow::Table& table, const std::string& name, int64_t max_rows) {
    auto out = std::make_shared<DataTable>();
    out->SetName(name);
    std::vector<std::string> headers;
    for (const auto& field : table.schema()->fields()) headers.push_back(field->name());
    out->SetHeaders(headers);
    const int64_t rows = max_rows < 0 ? table.num_rows() : std::min(max_rows, table.num_rows());
    const int columns = table.num_columns();
    // Walk each column's chunks once, filling the rows column by column.
    std::vector<DataTable::Row> data(static_cast<size_t>(rows), DataTable::Row(static_cast<size_t>(columns)));
    for (int c = 0; c < columns; ++c) {
        int64_t row = 0;
        for (const auto& chunk : table.column(c)->chunks()) {
            for (int64_t i = 0; i < chunk->length() && row < rows; ++i, ++row) {
                data[static_cast<size_t>(row)][static_cast<size_t>(c)] = CellOf(*chunk, i);
            }
            if (row >= rows) break;
        }
    }
    for (auto& r : data) out->AddRow(std::move(r));
    return out;
}

ArrowPreview PreviewArrowRows(const arrow::Table& table, int64_t max_rows, size_t max_chars) {
    ArrowPreview preview;
    for (const auto& field : table.schema()->fields()) {
        preview.headers.push_back(field->name());
        preview.types.push_back(field->type()->ToString());
    }
    const int64_t rows = std::max<int64_t>(0, std::min(max_rows, table.num_rows()));
    const int columns = table.num_columns();
    preview.rows.assign(static_cast<size_t>(rows), std::vector<std::string>(static_cast<size_t>(columns)));
    for (int c = 0; c < columns; ++c) {
        int64_t row = 0;
        for (const auto& chunk : table.column(c)->chunks()) {
            for (int64_t i = 0; i < chunk->length() && row < rows; ++i, ++row) {
                const auto cell = CellOf(*chunk, i);
                std::string text;
                if (const auto* s = std::get_if<std::string>(&cell)) text = *s;
                else if (const auto* d = std::get_if<double>(&cell)) text = arrow::DoubleScalar(*d).ToString();
                else if (const auto* n = std::get_if<int64_t>(&cell)) text = std::to_string(*n);
                for (char& ch : text) {
                    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                }
                if (text.size() > max_chars) {
                    size_t cut = max_chars;  // never split a UTF-8 character
                    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
                    text = text.substr(0, cut) + "...";
                }
                preview.rows[static_cast<size_t>(row)][static_cast<size_t>(c)] = std::move(text);
            }
            if (row >= rows) break;
        }
    }
    return preview;
}

}  // namespace cyxwiz
