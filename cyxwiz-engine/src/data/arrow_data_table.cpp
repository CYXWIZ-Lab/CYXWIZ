#include "arrow_data_table.h"

#include "data_table.h"

#include <arrow/api.h>

#include <algorithm>
#include <cstdio>
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
        case arrow::Type::UINT64: return static_cast<int64_t>(static_cast<const arrow::UInt64Array&>(array).Value(i));
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

std::string ArrowCellText(const arrow::ChunkedArray& column, int64_t row) {
    int64_t offset = 0;
    for (const auto& chunk : column.chunks()) {
        if (row < offset + chunk->length()) {
            const DataTable::CellValue v = CellOf(*chunk, row - offset);
            if (const auto* s = std::get_if<std::string>(&v)) return *s;
            if (const auto* n = std::get_if<int64_t>(&v)) return std::to_string(*n);
            if (const auto* d = std::get_if<double>(&v)) {
                char buf[48];
                std::snprintf(buf, sizeof(buf), "%.6g", *d);
                return buf;
            }
            return "";
        }
        offset += chunk->length();
    }
    return "";
}

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

std::shared_ptr<arrow::Table> ArrowFromDataTable(const DataTable& table) {
    const size_t rows = table.GetRowCount();
    const size_t columns = table.GetColumnCount();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::vector<std::string> used;
    for (size_t c = 0; c < columns; ++c) {
        std::string name = c < table.GetHeaders().size() ? table.GetHeaders()[c] : std::string();
        if (name.empty()) name = "column " + std::to_string(c + 1);
        const std::string base = name;
        for (int n = 2; std::find(used.begin(), used.end(), name) != used.end(); ++n) name = base + " " + std::to_string(n);
        used.push_back(name);
        bool all_int = true, all_number = true;
        for (size_t r = 0; r < rows && all_number; ++r) {
            const auto cell = table.GetCell(r, c);
            if (std::holds_alternative<std::monostate>(cell)) continue;
            if (std::holds_alternative<double>(cell)) all_int = false;
            else if (!std::holds_alternative<int64_t>(cell)) all_int = all_number = false;
        }
        std::shared_ptr<arrow::Array> array;
        arrow::Status st;
        if (all_int) {
            arrow::Int64Builder b;
            for (size_t r = 0; r < rows; ++r) {
                const auto cell = table.GetCell(r, c);
                st = std::holds_alternative<int64_t>(cell) ? b.Append(std::get<int64_t>(cell)) : b.AppendNull();
            }
            st = b.Finish(&array);
            fields.push_back(arrow::field(name, arrow::int64()));
        } else if (all_number) {
            arrow::DoubleBuilder b;
            for (size_t r = 0; r < rows; ++r) {
                const auto cell = table.GetCell(r, c);
                if (const auto* d = std::get_if<double>(&cell)) st = b.Append(*d);
                else if (const auto* n = std::get_if<int64_t>(&cell)) st = b.Append(static_cast<double>(*n));
                else st = b.AppendNull();
            }
            st = b.Finish(&array);
            fields.push_back(arrow::field(name, arrow::float64()));
        } else {
            arrow::StringBuilder b;
            for (size_t r = 0; r < rows; ++r) {
                if (std::holds_alternative<std::monostate>(table.GetCell(r, c))) st = b.AppendNull();
                else st = b.Append(table.GetCellAsString(r, c));
            }
            st = b.Finish(&array);
            fields.push_back(arrow::field(name, arrow::utf8()));
        }
        if (!st.ok() || !array) return nullptr;
        arrays.push_back(std::move(array));
    }
    return arrow::Table::Make(arrow::schema(fields), arrays, static_cast<int64_t>(rows));
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
