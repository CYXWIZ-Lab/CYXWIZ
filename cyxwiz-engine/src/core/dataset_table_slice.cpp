#include "dataset_table_slice.h"

#include "../data/arrow_data_table.h"
#include "../data/data_table.h"

#include <arrow/api.h>

#include <algorithm>
#include <vector>

namespace cyxwiz {

namespace {
std::string Thousands(int64_t n) {
    std::string digits = std::to_string(n), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return out;
}
}  // namespace

DatasetSlice DatasetSliceFromArrow(const std::shared_ptr<arrow::Table>& source, const std::string& name,
                                   int64_t max_rows, int64_t total_rows, const SliceProgress& progress,
                                   int64_t chunk_rows) {
    DatasetSlice slice;
    slice.total_rows = total_rows;
    if (!source) {
        slice.error = "The dataset could not be read.";
        return slice;
    }
    const int64_t rows = std::min(std::max<int64_t>(0, max_rows), source->num_rows());
    slice.source = rows < source->num_rows() ? source->Slice(0, rows) : source;
    auto table = std::make_shared<DataTable>();
    table->SetName(name);
    std::vector<std::string> headers;
    for (const auto& field : source->schema()->fields()) headers.push_back(field->name());
    table->SetHeaders(headers);
    chunk_rows = std::max<int64_t>(1, chunk_rows);
    for (int64_t offset = 0; offset < rows; offset += chunk_rows) {
        const int64_t take = std::min(chunk_rows, rows - offset);
        AppendArrowRows(*slice.source->Slice(offset, take), *table);
        if (progress && !progress(static_cast<float>(offset + take) / static_cast<float>(rows),
                                  Thousands(offset + take) + " of " + Thousands(rows) + " rows")) {
            slice.stopped = true;
            slice.source.reset();
            return slice;
        }
    }
    slice.table = std::move(table);
    return slice;
}

std::shared_ptr<arrow::Table> FirstRowsOfRowGroups(int groups, const RowGroupReader& read_group, int64_t max_rows,
                                                   const SliceProgress& progress, std::string& error) {
    std::vector<std::shared_ptr<arrow::Table>> parts;
    int64_t rows = 0;
    for (int group = 0; group < groups && rows < max_rows; ++group) {
        if (progress && !progress(static_cast<float>(group) / static_cast<float>(groups),
                                  "Reading row group " + std::to_string(group + 1) + " of " + std::to_string(groups)))
            return nullptr;
        auto part = read_group(group);
        if (!part) {
            error = "Row group " + std::to_string(group + 1) + " could not be read.";
            return nullptr;
        }
        rows += part->num_rows();
        parts.push_back(std::move(part));
    }
    if (parts.empty()) {
        error = "The dataset has no rows to read.";
        return nullptr;
    }
    auto joined = parts.size() == 1 ? arrow::Result<std::shared_ptr<arrow::Table>>(parts.front())
                                    : arrow::ConcatenateTables(parts);
    if (!joined.ok()) {
        error = joined.status().ToString();
        return nullptr;
    }
    return (*joined)->num_rows() > max_rows ? (*joined)->Slice(0, max_rows) : *joined;
}

}  // namespace cyxwiz
