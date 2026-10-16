#include "dataset_table_slice.h"

#include "../data/data_table.h"

#include <cstdlib>

namespace cyxwiz {

namespace {

enum class CellKind { Integer, Floating, Text };

CellKind KindOf(const std::string& type) {
    if (type.find("int") != std::string::npos) return CellKind::Integer;
    for (const char* name : {"float", "double", "decimal", "halffloat"})
        if (type.find(name) != std::string::npos) return CellKind::Floating;
    return CellKind::Text;
}

DataTable::CellValue Cell(const std::string& text, CellKind kind) {
    if (text.empty()) return std::monostate{};
    const char* begin = text.c_str();
    char* end = nullptr;
    if (kind == CellKind::Integer) {
        const long long value = std::strtoll(begin, &end, 10);
        if (end != begin && *end == '\0') return static_cast<int64_t>(value);
    } else if (kind == CellKind::Floating) {
        const double value = std::strtod(begin, &end);
        if (end != begin && *end == '\0') return value;
    }
    return text;
}

}  // namespace

std::shared_ptr<DataTable> DataTableFromPreview(const DataPreviewPage& page, const std::string& name,
                                                std::shared_ptr<DataTable> table) {
    std::vector<CellKind> kinds;
    for (const auto& column : page.schema) kinds.push_back(KindOf(column.type));
    if (!table) {
        table = std::make_shared<DataTable>();
        table->SetName(name);
        std::vector<std::string> headers;
        for (const auto& column : page.schema) headers.push_back(column.name);
        table->SetHeaders(headers);
    }
    for (const auto& row : page.rows) {
        DataTable::Row cells;
        cells.reserve(kinds.size());
        for (size_t c = 0; c < kinds.size(); ++c)
            cells.push_back(c < row.size() ? Cell(row[c], kinds[c]) : DataTable::CellValue(std::monostate{}));
        table->AddRow(std::move(cells));
    }
    return table;
}

DatasetSlice ReadDatasetSlice(const PreviewPageReader& read_page, const std::string& dataset_name, int64_t max_rows) {
    DatasetSlice slice;
    DataPreviewRequest request;
    request.dataset_name = dataset_name;
    request.offset = 0;
    while (request.offset < max_rows) {
        request.row_limit = max_rows - request.offset;  // the service caps each page
        const DataPreviewPage page = read_page(request);
        if (!page.ok) {
            slice.error = page.reason.empty() ? "The dataset could not be read." : page.reason;
            slice.table.reset();
            return slice;
        }
        slice.total_rows = page.total_rows;
        slice.table = DataTableFromPreview(page, dataset_name, slice.table);
        if (!page.has_next || page.rows_returned == 0) break;
        request.offset = page.next_offset;
    }
    return slice;
}

}  // namespace cyxwiz
