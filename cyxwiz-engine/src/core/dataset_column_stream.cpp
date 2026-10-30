#include "dataset_column_stream.h"

#include "arrow_dataset.h"
#include "data_registry.h"
#include "parquet_backed_dataset.h"

#include <arrow/api.h>

namespace cyxwiz {

namespace {

// Batches of the chosen columns: of one in-memory table, or of each row
// group of a table on disk read one after another (only those columns).
class DatasetColumnsReader final : public arrow::RecordBatchReader {
public:
    DatasetColumnsReader(std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::Table> table,
                         std::shared_ptr<ParquetBackedDataset> disk, std::vector<std::string> columns,
                         ColumnStreamOptions options)
        : schema_(std::move(schema)), table_(std::move(table)), disk_(std::move(disk)),
          columns_(std::move(columns)), options_(std::move(options)) {
        if (table_) part_ = std::make_unique<arrow::TableBatchReader>(table_);
    }

    std::shared_ptr<arrow::Schema> schema() const override { return schema_; }

    arrow::Status ReadNext(std::shared_ptr<arrow::RecordBatch>* out) override {
        *out = nullptr;
        if (options_.should_cancel && options_.should_cancel()) return arrow::Status::Cancelled("Cancelled");
        while (true) {
            if (part_) {
                ARROW_RETURN_NOT_OK(part_->ReadNext(out));
                if (*out) {
                    rows_ += (*out)->num_rows();
                    if (options_.on_rows) options_.on_rows(rows_);
                    return arrow::Status::OK();
                }
                part_.reset();
                table_.reset();
            }
            if (!disk_ || next_group_ >= disk_->GetNumRowGroups()) return arrow::Status::OK();  // the end
            const int group = next_group_++;
            arrow::ChunkedArrayVector parts;
            for (const auto& name : columns_) {
                auto column = disk_->ReadRowGroupColumn(group, name);
                if (!column) return arrow::Status::IOError("Cannot read column '", name, "' of row group ", group);
                parts.push_back(std::move(column));
            }
            table_ = arrow::Table::Make(schema_, std::move(parts));
            part_ = std::make_unique<arrow::TableBatchReader>(table_);
        }
    }

private:
    std::shared_ptr<arrow::Schema> schema_;
    std::shared_ptr<arrow::Table> table_;  // in memory, or the current row group
    std::shared_ptr<ParquetBackedDataset> disk_;
    std::vector<std::string> columns_;
    ColumnStreamOptions options_;
    std::unique_ptr<arrow::TableBatchReader> part_;
    int next_group_ = 0;
    int64_t rows_ = 0;
};

// The chosen fields of `schema`, in the order asked; empty with `error` set
// when one is missing or named twice in the table.
std::vector<int> FieldIndices(const arrow::Schema& schema, const std::vector<std::string>& columns, std::string& error) {
    std::vector<int> indices;
    for (const auto& name : columns) {
        const int i = schema.GetFieldIndex(name);
        if (i < 0) {
            error = schema.GetAllFieldIndices(name).empty() ? "The table has no column '" + name + "'"
                                                             : "The table has more than one column named '" + name + "'";
            return {};
        }
        indices.push_back(i);
    }
    return indices;
}

}  // namespace

std::shared_ptr<arrow::RecordBatchReader> OpenDatasetColumns(const std::string& dataset,
                                                             const std::vector<std::string>& columns,
                                                             const ColumnStreamOptions& options, std::string& error,
                                                             int64_t* total_rows) {
    error.clear();
    if (columns.empty()) {
        error = "No columns chosen";
        return nullptr;
    }
    auto& registry = DataRegistry::Instance();
    if (const auto memory = registry.GetArrowDataset(dataset); memory && memory->GetArrowTable()) {
        const auto table = memory->GetArrowTable();
        const auto indices = FieldIndices(*table->schema(), columns, error);
        if (indices.empty()) return nullptr;
        auto chosen = table->SelectColumns(indices);
        if (!chosen.ok()) {
            error = chosen.status().message();
            return nullptr;
        }
        if (total_rows) *total_rows = table->num_rows();
        return std::make_shared<DatasetColumnsReader>((*chosen)->schema(), *chosen, nullptr, columns, options);
    }
    if (const auto disk = registry.GetParquetBackedDataset(dataset); disk && disk->GetSchema()) {
        const auto indices = FieldIndices(*disk->GetSchema(), columns, error);
        if (indices.empty()) return nullptr;
        arrow::FieldVector fields;
        for (int i : indices) fields.push_back(disk->GetSchema()->field(i));
        if (total_rows) *total_rows = disk->GetNumRows();
        return std::make_shared<DatasetColumnsReader>(arrow::schema(std::move(fields)), nullptr, disk, columns, options);
    }
    error = "'" + dataset + "' is not a table held in memory or on disk";
    return nullptr;
}

}  // namespace cyxwiz
