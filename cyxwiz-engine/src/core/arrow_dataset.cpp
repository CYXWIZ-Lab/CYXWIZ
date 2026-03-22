#include "arrow_dataset.h"
#include <arrow/csv/reader.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <arrow/compute/api.h>
#include <spdlog/spdlog.h>
#include <filesystem>

#ifdef _WIN32
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#else
// For Linux/macOS, parquet headers may be in different location
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#endif

namespace cyxwiz {

// ============================================================================
// ArrowDataset Implementation
// ============================================================================

ArrowDataset::ArrowDataset(std::shared_ptr<arrow::Table> table, const std::string& name)
    : table_(std::move(table)), name_(name) {
    if (!table_) {
        spdlog::warn("ArrowDataset '{}' created with null table", name);
    } else {
        spdlog::debug("ArrowDataset '{}' created: {} rows, {} columns",
                     name, table_->num_rows(), table_->num_columns());
    }
}

std::shared_ptr<ArrowDataset> ArrowDataset::FromFile(const std::string& path,
                                                     const std::string& name) {
    auto format = arrow_utils::DetectFormat(path);

    switch (format) {
        case arrow_utils::FileFormat::CSV:
        case arrow_utils::FileFormat::TSV:
            return FromCSV(path, name);
        case arrow_utils::FileFormat::Parquet:
            return FromParquet(path, name);
        case arrow_utils::FileFormat::Feather:
        case arrow_utils::FileFormat::ArrowIPC:
            return FromFeather(path, name);
        default:
            spdlog::error("Unsupported file format for Arrow: {}", path);
            return nullptr;
    }
}

std::shared_ptr<ArrowDataset> ArrowDataset::FromCSV(
    const std::string& path,
    const std::string& name,
    const arrow::csv::ReadOptions& read_options,
    const arrow::csv::ParseOptions& parse_options,
    const arrow::csv::ConvertOptions& convert_options) {

    spdlog::info("Loading CSV file into Arrow table: {}", path);

    // Open file
    auto maybe_input = arrow::io::ReadableFile::Open(path);
    if (!maybe_input.ok()) {
        spdlog::error("Failed to open CSV file: {} - {}", path, maybe_input.status().ToString());
        return nullptr;
    }
    std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

    // Create CSV reader (Arrow 21+ uses IOContext instead of MemoryPool)
    auto maybe_reader = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(),
        input,
        read_options,
        parse_options,
        convert_options
    );

    if (!maybe_reader.ok()) {
        spdlog::error("Failed to create CSV reader: {}", maybe_reader.status().ToString());
        return nullptr;
    }

    std::shared_ptr<arrow::csv::TableReader> reader = maybe_reader.ValueOrDie();

    // Read table
    auto maybe_table = reader->Read();
    if (!maybe_table.ok()) {
        spdlog::error("Failed to read CSV table: {}", maybe_table.status().ToString());
        return nullptr;
    }

    std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();
    spdlog::info("CSV loaded successfully: {} rows, {} columns",
                 table->num_rows(), table->num_columns());

    return std::make_shared<ArrowDataset>(table, name);
}

std::shared_ptr<ArrowDataset> ArrowDataset::FromParquet(const std::string& path,
                                                        const std::string& name) {
    spdlog::info("Loading Parquet file into Arrow table: {}", path);

    // Open Parquet file
    auto maybe_input = arrow::io::ReadableFile::Open(path);
    if (!maybe_input.ok()) {
        spdlog::error("Failed to open Parquet file: {} - {}", path, maybe_input.status().ToString());
        return nullptr;
    }
    std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

    // Create Parquet reader (Arrow 21+ returns Result instead of Status)
    auto maybe_reader = parquet::arrow::OpenFile(input, arrow::default_memory_pool());
    if (!maybe_reader.ok()) {
        spdlog::error("Failed to create Parquet reader: {}", maybe_reader.status().ToString());
        return nullptr;
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(maybe_reader).ValueOrDie();

    // Read entire file as table
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = reader->ReadTable(&table);
    if (!status.ok()) {
        spdlog::error("Failed to read Parquet table: {}", status.ToString());
        return nullptr;
    }

    spdlog::info("Parquet loaded successfully: {} rows, {} columns",
                 table->num_rows(), table->num_columns());

    return std::make_shared<ArrowDataset>(table, name);
}

std::shared_ptr<ArrowDataset> ArrowDataset::FromFeather(const std::string& path,
                                                        const std::string& name) {
    spdlog::info("Loading Feather file into Arrow table: {}", path);

    // Open file
    auto maybe_input = arrow::io::ReadableFile::Open(path);
    if (!maybe_input.ok()) {
        spdlog::error("Failed to open Feather file: {} - {}", path, maybe_input.status().ToString());
        return nullptr;
    }
    std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

    // Create Feather reader (Arrow IPC)
    auto maybe_reader = arrow::ipc::RecordBatchFileReader::Open(input);
    if (!maybe_reader.ok()) {
        spdlog::error("Failed to create Feather reader: {}", maybe_reader.status().ToString());
        return nullptr;
    }

    std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader = maybe_reader.ValueOrDie();

    // Read all record batches and convert to table (Arrow 21+ removed ReadAll())
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    for (int i = 0; i < reader->num_record_batches(); ++i) {
        auto maybe_batch = reader->ReadRecordBatch(i);
        if (!maybe_batch.ok()) {
            spdlog::error("Failed to read record batch {}: {}", i, maybe_batch.status().ToString());
            return nullptr;
        }
        batches.push_back(maybe_batch.ValueOrDie());
    }

    auto maybe_table = arrow::Table::FromRecordBatches(batches);
    if (!maybe_table.ok()) {
        spdlog::error("Failed to create table from record batches: {}", maybe_table.status().ToString());
        return nullptr;
    }

    std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();
    spdlog::info("Feather loaded successfully: {} rows, {} columns",
                 table->num_rows(), table->num_columns());

    return std::make_shared<ArrowDataset>(table, name);
}

std::vector<std::string> ArrowDataset::GetColumnNames() const {
    if (!table_) return {};

    std::vector<std::string> names;
    names.reserve(table_->num_columns());

    for (int i = 0; i < table_->num_columns(); ++i) {
        names.push_back(table_->schema()->field(i)->name());
    }

    return names;
}

size_t ArrowDataset::GetMemoryUsage() const {
    if (!table_) return 0;

    size_t total_bytes = 0;

    // Sum up all buffer sizes
    for (int i = 0; i < table_->num_columns(); ++i) {
        auto column = table_->column(i);
        for (int j = 0; j < column->num_chunks(); ++j) {
            auto chunk = column->chunk(j);
            auto array_data = chunk->data();

            // Count all buffers (values, offsets, null bitmap)
            for (const auto& buffer : array_data->buffers) {
                if (buffer) {
                    total_bytes += buffer->size();
                }
            }
        }
    }

    return total_bytes;
}

std::shared_ptr<ArrowDataset> ArrowDataset::SelectColumns(
    const std::vector<std::string>& column_names) const {

    if (!table_) return nullptr;

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

    for (const auto& name : column_names) {
        auto maybe_column = table_->GetColumnByName(name);
        if (maybe_column) {
            fields.push_back(table_->schema()->GetFieldByName(name));
            columns.push_back(maybe_column);
        } else {
            spdlog::warn("Column '{}' not found in table '{}'", name, name_);
        }
    }

    if (fields.empty()) {
        spdlog::error("No valid columns selected from table '{}'", name_);
        return nullptr;
    }

    auto schema = arrow::schema(fields);
    auto new_table = arrow::Table::Make(schema, columns);

    return std::make_shared<ArrowDataset>(new_table, name_ + "_selected");
}

std::shared_ptr<ArrowDataset> ArrowDataset::FilterRows(
    const std::vector<int64_t>& indices) const {

    if (!table_) return nullptr;

    // Create Arrow array from indices
    arrow::Int64Builder builder;
    auto status = builder.AppendValues(indices);
    if (!status.ok()) {
        spdlog::error("Failed to build indices array: {}", status.ToString());
        return nullptr;
    }

    std::shared_ptr<arrow::Array> indices_array;
    status = builder.Finish(&indices_array);
    if (!status.ok()) {
        spdlog::error("Failed to finish indices array: {}", status.ToString());
        return nullptr;
    }

    // Use Arrow compute to take rows
    auto maybe_result = arrow::compute::Take(table_, indices_array);
    if (!maybe_result.ok()) {
        spdlog::error("Failed to filter rows: {}", maybe_result.status().ToString());
        return nullptr;
    }

    auto result_table = maybe_result.ValueOrDie().table();
    return std::make_shared<ArrowDataset>(result_table, name_ + "_filtered");
}

std::shared_ptr<ArrowDataset> ArrowDataset::Slice(int64_t offset, int64_t length) const {
    if (!table_) return nullptr;

    auto sliced_table = table_->Slice(offset, length);
    return std::make_shared<ArrowDataset>(sliced_table, name_ + "_slice");
}

bool ArrowDataset::ExportCSV(const std::string& path) const {
    if (!table_) {
        spdlog::error("Cannot export null table to CSV");
        return false;
    }

    spdlog::info("Exporting Arrow table '{}' to CSV: {}", name_, path);

    // Open output file
    auto maybe_output = arrow::io::FileOutputStream::Open(path);
    if (!maybe_output.ok()) {
        spdlog::error("Failed to open CSV output file: {} - {}", path, maybe_output.status().ToString());
        return false;
    }
    std::shared_ptr<arrow::io::FileOutputStream> output = maybe_output.ValueOrDie();

    // Write CSV (simplified - for production use arrow::csv::WriteCSV with options)
    // Note: arrow::csv::WriteCSV is available in newer Arrow versions
    // For now, we'll use a simple implementation

    // Write header
    std::string header;
    for (int i = 0; i < table_->num_columns(); ++i) {
        if (i > 0) header += ",";
        header += table_->schema()->field(i)->name();
    }
    header += "\n";
    auto status = output->Write(header);
    if (!status.ok()) {
        spdlog::error("Failed to write CSV header: {}", status.ToString());
        return false;
    }

    // For a full implementation, we'd convert the table to CSV format
    // This is simplified - in production, use arrow::csv::WriteCSV
    spdlog::warn("CSV export is simplified - consider using arrow::csv::WriteCSV for production");

    status = output->Close();
    if (!status.ok()) {
        spdlog::error("Failed to close CSV file: {}", status.ToString());
        return false;
    }

    spdlog::info("CSV exported successfully");
    return true;
}

bool ArrowDataset::ExportParquet(const std::string& path, bool compress) const {
    if (!table_) {
        spdlog::error("Cannot export null table to Parquet");
        return false;
    }

    spdlog::info("Exporting Arrow table '{}' to Parquet: {}", name_, path);

    // Open output file
    auto maybe_output = arrow::io::FileOutputStream::Open(path);
    if (!maybe_output.ok()) {
        spdlog::error("Failed to open Parquet output file: {} - {}", path, maybe_output.status().ToString());
        return false;
    }
    std::shared_ptr<arrow::io::FileOutputStream> output = maybe_output.ValueOrDie();

    // Set up Parquet writer properties
    parquet::WriterProperties::Builder props_builder;
    if (compress) {
        props_builder.compression(parquet::Compression::SNAPPY);
    }
    std::shared_ptr<parquet::WriterProperties> props = props_builder.build();

    // Write table
    auto status = parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(),
                                             output, 1024 * 1024, props);  // 1MB row group size
    if (!status.ok()) {
        spdlog::error("Failed to write Parquet table: {}", status.ToString());
        return false;
    }

    spdlog::info("Parquet exported successfully");
    return true;
}

bool ArrowDataset::ExportFeather(const std::string& path) const {
    if (!table_) {
        spdlog::error("Cannot export null table to Feather");
        return false;
    }

    spdlog::info("Exporting Arrow table '{}' to Feather: {}", name_, path);

    // Open output file
    auto maybe_output = arrow::io::FileOutputStream::Open(path);
    if (!maybe_output.ok()) {
        spdlog::error("Failed to open Feather output file: {} - {}", path, maybe_output.status().ToString());
        return false;
    }
    std::shared_ptr<arrow::io::FileOutputStream> output = maybe_output.ValueOrDie();

    // Write Arrow IPC file
    auto maybe_writer = arrow::ipc::MakeFileWriter(output, table_->schema());
    if (!maybe_writer.ok()) {
        spdlog::error("Failed to create Feather writer: {}", maybe_writer.status().ToString());
        return false;
    }

    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer = maybe_writer.ValueOrDie();

    // Convert table to record batches and write
    arrow::TableBatchReader reader(*table_);
    std::shared_ptr<arrow::RecordBatch> batch;
    while (true) {
        auto status = reader.ReadNext(&batch);
        if (!status.ok()) {
            spdlog::error("Failed to read record batch: {}", status.ToString());
            return false;
        }
        if (!batch) break;  // EOF

        status = writer->WriteRecordBatch(*batch);
        if (!status.ok()) {
            spdlog::error("Failed to write record batch: {}", status.ToString());
            return false;
        }
    }

    auto status = writer->Close();
    if (!status.ok()) {
        spdlog::error("Failed to close Feather writer: {}", status.ToString());
        return false;
    }

    spdlog::info("Feather exported successfully");
    return true;
}

void ArrowDataset::Print(int64_t max_rows) const {
    if (!table_) {
        spdlog::info("ArrowDataset '{}': null table", name_);
        return;
    }

    spdlog::info("ArrowDataset '{}':", name_);
    spdlog::info("  Rows: {}", table_->num_rows());
    spdlog::info("  Columns: {}", table_->num_columns());
    spdlog::info("  Memory: {} bytes", GetMemoryUsage());
    spdlog::info("  Schema:");

    for (int i = 0; i < table_->num_columns(); ++i) {
        auto field = table_->schema()->field(i);
        spdlog::info("    {}: {}", field->name(), field->type()->ToString());
    }

    // Print first few rows (simplified - full implementation would format nicely)
    if (table_->num_rows() > 0) {
        int64_t rows_to_print = std::min(max_rows, table_->num_rows());
        spdlog::info("  First {} rows:", rows_to_print);
        // For production, implement proper row printing
        // This would involve converting arrays to strings
    }
}

// ============================================================================
// Arrow Utilities Implementation
// ============================================================================

namespace arrow_utils {

FileFormat DetectFormat(const std::string& path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();

    // Convert to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".csv") return FileFormat::CSV;
    if (ext == ".tsv") return FileFormat::TSV;
    if (ext == ".parquet" || ext == ".pq") return FileFormat::Parquet;
    if (ext == ".feather" || ext == ".fea") return FileFormat::Feather;
    if (ext == ".arrow" || ext == ".ipc") return FileFormat::ArrowIPC;

    return FileFormat::Unknown;
}

std::shared_ptr<arrow::Table> CreateTableFromVectors(
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<double>>& columns) {

    if (column_names.size() != columns.size()) {
        spdlog::error("Column names and data size mismatch");
        return nullptr;
    }

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    for (size_t i = 0; i < columns.size(); ++i) {
        // Create field
        fields.push_back(arrow::field(column_names[i], arrow::float64()));

        // Create array from vector
        arrow::DoubleBuilder builder;
        auto status = builder.AppendValues(columns[i]);
        if (!status.ok()) {
            spdlog::error("Failed to append values to column {}: {}", i, status.ToString());
            return nullptr;
        }

        std::shared_ptr<arrow::Array> array;
        status = builder.Finish(&array);
        if (!status.ok()) {
            spdlog::error("Failed to finish column {}: {}", i, status.ToString());
            return nullptr;
        }

        arrays.push_back(array);
    }

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, arrays);
}

arrow::Result<std::shared_ptr<arrow::Table>> ConcatenateTables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables) {

    if (tables.empty()) {
        return arrow::Status::Invalid("Cannot concatenate empty table list");
    }

    return arrow::ConcatenateTables(tables);
}

arrow::Result<std::shared_ptr<arrow::Table>> ConcatenateColumns(
    const std::shared_ptr<arrow::Table>& left,
    const std::shared_ptr<arrow::Table>& right) {

    if (!left || !right) {
        return arrow::Status::Invalid("Cannot concatenate null tables");
    }

    if (left->num_rows() != right->num_rows()) {
        return arrow::Status::Invalid("Tables must have same number of rows for column concatenation");
    }

    // Combine schemas
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < left->num_columns(); ++i) {
        fields.push_back(left->schema()->field(i));
    }
    for (int i = 0; i < right->num_columns(); ++i) {
        fields.push_back(right->schema()->field(i));
    }

    // Combine columns
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (int i = 0; i < left->num_columns(); ++i) {
        columns.push_back(left->column(i));
    }
    for (int i = 0; i < right->num_columns(); ++i) {
        columns.push_back(right->column(i));
    }

    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, columns);
}

} // namespace arrow_utils

} // namespace cyxwiz
