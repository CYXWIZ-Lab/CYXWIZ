#include "data_preview_service.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "parquet_backed_dataset.h"

#include <arrow/api.h>

#include <algorithm>
#include <memory>
#include <unordered_set>

namespace cyxwiz {

namespace {

constexpr int64_t kMaxPreviewRows = 200;

DataPreviewPage FailPreview(std::string dataset_name,
                             std::string reason) {
    DataPreviewPage page;
    page.dataset_name = std::move(dataset_name);
    page.reason = std::move(reason);
    return page;
}

std::string ScalarToPreviewString(const std::shared_ptr<arrow::Scalar>& scalar) {
    if (!scalar || !scalar->is_valid) {
        return "<null>";
    }
    return scalar->ToString();
}

std::vector<int> ResolveColumnIndices(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::string>& selected_columns,
    std::string& error) {
    std::vector<int> indices;
    if (!schema) {
        error = "schema is unavailable";
        return indices;
    }

    if (selected_columns.empty()) {
        indices.reserve(static_cast<size_t>(schema->num_fields()));
        for (int i = 0; i < schema->num_fields(); ++i) {
            indices.push_back(i);
        }
        return indices;
    }

    std::unordered_set<std::string> seen;
    indices.reserve(selected_columns.size());
    for (const auto& name : selected_columns) {
        if (name.empty() || seen.count(name) > 0) {
            continue;
        }
        const int index = schema->GetFieldIndex(name);
        if (index < 0) {
            error = "selected preview column '" + name + "' is not in dataset schema";
            indices.clear();
            return indices;
        }
        seen.insert(name);
        indices.push_back(index);
    }
    return indices;
}

void FillSchema(const std::shared_ptr<arrow::Schema>& schema,
                const std::vector<int>& indices,
                DataPreviewPage& page) {
    page.schema.reserve(indices.size());
    for (int index : indices) {
        auto field = schema->field(index);
        if (!field) continue;
        page.schema.push_back(DataPreviewColumn{
            field->name(),
            field->type() ? field->type()->ToString() : std::string{"unknown"},
            field->nullable(),
        });
    }
}

bool AppendRowsFromTable(const std::shared_ptr<arrow::Table>& table,
                         int64_t row_limit,
                         const std::vector<int>& column_indices,
                         DataPreviewPage& page,
                         std::string& error) {
    if (!table) {
        error = "table is unavailable";
        return false;
    }

    const int64_t capped_rows = std::min<int64_t>(
        table->num_rows(), std::max<int64_t>(0, row_limit));
    page.rows.reserve(page.rows.size() + static_cast<size_t>(capped_rows));

    for (int64_t row = 0; row < capped_rows; ++row) {
        std::vector<std::string> out_row;
        out_row.reserve(column_indices.size());
        for (int column_index : column_indices) {
            if (column_index < 0 || column_index >= table->num_columns()) {
                error = "preview column index is out of range";
                return false;
            }
            const auto column = table->column(column_index);
            if (!column) {
                error = "preview column is unavailable";
                return false;
            }

            int64_t chunk_row = row;
            bool found_chunk = false;
            for (const auto& chunk : column->chunks()) {
                if (!chunk) continue;
                if (chunk_row >= chunk->length()) {
                    chunk_row -= chunk->length();
                    continue;
                }
                auto scalar = chunk->GetScalar(chunk_row);
                if (!scalar.ok()) {
                    error = scalar.status().ToString();
                    return false;
                }
                out_row.push_back(ScalarToPreviewString(scalar.ValueOrDie()));
                found_chunk = true;
                break;
            }
            if (!found_chunk) {
                out_row.emplace_back("<null>");
            }
        }
        page.rows.push_back(std::move(out_row));
    }
    page.rows_returned = static_cast<int64_t>(page.rows.size());
    return true;
}

DataPreviewPage PreviewArrowDataset(const std::shared_ptr<ArrowDataset>& dataset,
                                    const DataPreviewRequest& request) {
    auto table = dataset ? dataset->GetArrowTable() : nullptr;
    if (!table) {
        return FailPreview(request.dataset_name, "registered Arrow dataset has no table");
    }

    const int64_t offset = std::max<int64_t>(0, request.offset);
    const int64_t limit = std::min<int64_t>(
        std::max<int64_t>(0, request.row_limit), kMaxPreviewRows);
    if (offset > table->num_rows()) {
        return FailPreview(request.dataset_name, "preview offset is beyond row count");
    }

    std::string error;
    auto column_indices = ResolveColumnIndices(
        table->schema(), request.selected_columns, error);
    if (!error.empty()) {
        return FailPreview(request.dataset_name, error);
    }

    DataPreviewPage page;
    page.ok = true;
    page.backend = "Arrow";
    page.dataset_name = request.dataset_name;
    page.total_rows = table->num_rows();
    page.total_columns = table->num_columns();
    page.offset = offset;
    FillSchema(table->schema(), column_indices, page);

    const auto sliced = table->Slice(offset, limit);
    if (!AppendRowsFromTable(sliced, limit, column_indices, page, error)) {
        return FailPreview(request.dataset_name, error);
    }

    page.has_next = offset + page.rows_returned < page.total_rows;
    page.next_offset = page.has_next ? offset + page.rows_returned : page.total_rows;
    return page;
}

DataPreviewPage PreviewParquetDataset(
    const std::shared_ptr<ParquetBackedDataset>& dataset,
    const DataPreviewRequest& request) {
    if (!dataset) {
        return FailPreview(request.dataset_name, "registered Parquet dataset is unavailable");
    }

    const int64_t offset = std::max<int64_t>(0, request.offset);
    const int64_t limit = std::min<int64_t>(
        std::max<int64_t>(0, request.row_limit), kMaxPreviewRows);
    if (offset > dataset->GetNumRows()) {
        return FailPreview(request.dataset_name, "preview offset is beyond row count");
    }

    std::string error;
    auto column_indices = ResolveColumnIndices(
        dataset->GetSchema(), request.selected_columns, error);
    if (!error.empty()) {
        return FailPreview(request.dataset_name, error);
    }

    DataPreviewPage page;
    page.ok = true;
    page.backend = "Parquet";
    page.dataset_name = request.dataset_name;
    page.total_rows = dataset->GetNumRows();
    page.total_columns = dataset->GetNumColumns();
    page.offset = offset;
    FillSchema(dataset->GetSchema(), column_indices, page);

    int64_t rows_to_skip = offset;
    int64_t rows_needed = limit;
    for (int group = 0; group < dataset->GetNumRowGroups() && rows_needed > 0; ++group) {
        const int64_t group_rows = dataset->GetRowGroupSize(group);
        if (rows_to_skip >= group_rows) {
            rows_to_skip -= group_rows;
            continue;
        }

        auto group_table = dataset->ReadRowGroup(group);
        if (!group_table) {
            return FailPreview(request.dataset_name, "could not read Parquet row group");
        }

        const int64_t take = std::min<int64_t>(
            rows_needed, group_table->num_rows() - rows_to_skip);
        auto sliced = group_table->Slice(rows_to_skip, take);
        if (!AppendRowsFromTable(sliced, take, column_indices, page, error)) {
            return FailPreview(request.dataset_name, error);
        }
        rows_needed -= take;
        rows_to_skip = 0;
    }

    page.rows_returned = static_cast<int64_t>(page.rows.size());
    page.has_next = offset + page.rows_returned < page.total_rows;
    page.next_offset = page.has_next ? offset + page.rows_returned : page.total_rows;
    return page;
}

} // namespace

DataPreviewPage DataPreviewService::PreviewRegisteredTabular(
    DataRegistry& registry,
    const DataPreviewRequest& request) {
    if (request.dataset_name.empty()) {
        return FailPreview({}, "dataset name is empty");
    }
    if (auto arrow_dataset = registry.GetArrowDataset(request.dataset_name)) {
        return PreviewArrowDataset(arrow_dataset, request);
    }
    if (auto parquet_dataset = registry.GetParquetBackedDataset(request.dataset_name)) {
        return PreviewParquetDataset(parquet_dataset, request);
    }
    return FailPreview(
        request.dataset_name,
        "dataset is not a registered tabular Arrow or Parquet source");
}

} // namespace cyxwiz
