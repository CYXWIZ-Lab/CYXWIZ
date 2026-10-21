#include "asset_preview.h"

#include "excel_table_adapter.h"
#include "../utils/hdf5_browser.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cyxwiz::assets {

namespace {

std::string Extension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return ext;
}

bool OneOf(const std::string& ext, std::initializer_list<const char*> list) {
    for (const char* item : list)
        if (ext == item) return true;
    return false;
}

// One line of a delimited file; quoted fields keep their delimiters and "" is a quote.
std::vector<std::string> SplitDelimited(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                field += c;
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == delimiter) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

bool IsNumber(const std::string& text) {
    if (text.empty()) return true;  // a missing value does not make a column text
    const char* begin = text.c_str();
    char* end = nullptr;
    std::strtod(begin, &end);
    while (end && *end && std::isspace(static_cast<unsigned char>(*end))) ++end;
    return end != begin && end && *end == '\0';
}

void StripLineEnd(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

// Newlines in the whole file (small files only).
int64_t CountLines(const std::string& path, bool& ends_with_newline) {
    std::ifstream in(path, std::ios::binary);
    char buffer[1 << 16];
    int64_t lines = 0;
    char last = '\n';
    while (in) {
        in.read(buffer, sizeof(buffer));
        const std::streamsize n = in.gcount();
        lines += std::count(buffer, buffer + n, '\n');
        if (n > 0) last = buffer[n - 1];
    }
    ends_with_newline = last == '\n';
    return lines;
}

FilePeek PeekDelimited(const std::string& path, char delimiter, const PeekLimits& limits) {
    FilePeek peek;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        peek.error = "Could not open the file.";
        return peek;
    }
    std::uintmax_t bytes_read = 0;
    std::string line;
    int data_lines = 0;
    while (std::getline(in, line) && bytes_read < limits.head_bytes) {
        bytes_read += line.size() + 1;
        StripLineEnd(line);
        if (peek.columns.empty()) {
            if (line.empty()) continue;
            peek.columns = SplitDelimited(line, delimiter);
            continue;
        }
        ++data_lines;
        if (static_cast<int>(peek.rows.size()) < limits.max_rows) {
            auto fields = SplitDelimited(line, delimiter);
            fields.resize(peek.columns.size());
            peek.rows.push_back(std::move(fields));
        }
    }
    if (peek.columns.empty()) {
        peek.error = "The file is empty.";
        return peek;
    }
    peek.table = true;
    peek.numeric.assign(peek.columns.size(), true);
    for (const auto& row : peek.rows)
        for (size_t c = 0; c < row.size(); ++c)
            if (!IsNumber(row[c])) peek.numeric[c] = false;

    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (!ec && size <= limits.exact_count_bytes) {
        bool ends_with_newline = true;
        const int64_t newlines = CountLines(path, ends_with_newline);
        peek.total_rows = std::max<int64_t>(0, newlines + (ends_with_newline ? 0 : 1) - 1);
    } else if (!ec && bytes_read > 0 && data_lines > 0) {
        // A large file: estimate from the head's average line length.
        const double per_line = static_cast<double>(bytes_read) / (data_lines + 1);
        peek.total_rows = static_cast<int64_t>(static_cast<double>(size) / per_line) - 1;
        peek.rows_estimated = true;
    }
    peek.ok = true;
    return peek;
}

// Columns, kinds and the first rows of an Arrow table.
void FillTablePeek(const arrow::Table& table, const PeekLimits& limits, FilePeek& peek) {
    for (const auto& field : table.schema()->fields()) {
        peek.columns.push_back(field->name());
        peek.numeric.push_back(arrow::is_integer(field->type()->id()) || arrow::is_floating(field->type()->id()));
    }
    const int64_t n = std::min<int64_t>(table.num_rows(), limits.max_rows);
    for (int64_t r = 0; r < n; ++r) {
        std::vector<std::string> row;
        for (int c = 0; c < table.num_columns(); ++c) {
            auto cell = table.column(c)->GetScalar(r);
            row.push_back(!cell.ok() || !(*cell)->is_valid ? std::string() : (*cell)->ToString());
        }
        peek.rows.push_back(std::move(row));
    }
    peek.table = true;
    peek.ok = true;
}

FilePeek PeekParquet(const std::string& path, const PeekLimits& limits) {
    FilePeek peek;
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        peek.error = "Could not open the file.";
        return peek;
    }
    try {
        parquet::ArrowReaderProperties properties;
        properties.set_batch_size(std::max(1, limits.max_rows));
        parquet::arrow::FileReaderBuilder builder;
        auto status = builder.Open(*input);
        if (!status.ok()) {
            peek.error = "Not a readable Parquet file: " + status.message();
            return peek;
        }
        std::unique_ptr<parquet::arrow::FileReader> reader;
        status = builder.properties(properties)->Build(&reader);
        if (!status.ok()) {
            peek.error = "Not a readable Parquet file: " + status.message();
            return peek;
        }
        peek.total_rows = reader->parquet_reader()->metadata()->num_rows();
        std::shared_ptr<arrow::Schema> schema;
        if (!reader->GetSchema(&schema).ok() || !schema) {
            peek.error = "Not a readable Parquet file: no schema";
            return peek;
        }
        std::shared_ptr<arrow::Table> head = arrow::Table::MakeEmpty(schema).ValueOr(nullptr);
        // One small batch from the first row group.
        if (reader->num_row_groups() > 0 && peek.total_rows > 0) {
            const std::vector<int> first_group{0};
            auto batches = reader->GetRecordBatchReader(first_group);
            if (batches.ok()) {
                std::shared_ptr<arrow::RecordBatch> batch;
                if ((*batches)->ReadNext(&batch).ok() && batch) {
                    auto table = arrow::Table::FromRecordBatches({batch});
                    if (table.ok()) head = *table;
                }
            }
        }
        if (!head) {
            peek.error = "Not a readable Parquet file";
            return peek;
        }
        FillTablePeek(*head, limits, peek);
    } catch (const std::exception& e) {
        peek.error = std::string("Not a readable Parquet file: ") + e.what();
        peek.ok = false;
    }
    return peek;
}

// Arrow IPC / Feather v2: the footer gives the row count, the first record
// batch the rows; an IPC stream gives its first batch only.
FilePeek PeekArrowIpc(const std::string& path, const PeekLimits& limits) {
    FilePeek peek;
    auto input = arrow::io::ReadableFile::Open(path);
    if (!input.ok()) {
        peek.error = "Could not open the file.";
        return peek;
    }
    std::shared_ptr<arrow::RecordBatch> first;
    std::shared_ptr<arrow::Schema> schema;
    auto file_reader = arrow::ipc::RecordBatchFileReader::Open(*input);
    if (file_reader.ok()) {
        schema = (*file_reader)->schema();
        auto rows = (*file_reader)->CountRows();
        if (rows.ok()) peek.total_rows = *rows;
        if ((*file_reader)->num_record_batches() > 0) {
            auto batch = (*file_reader)->ReadRecordBatch(0);
            if (batch.ok()) first = *batch;
        }
    } else {
        auto reopened = arrow::io::ReadableFile::Open(path);
        auto stream = reopened.ok() ? arrow::ipc::RecordBatchStreamReader::Open(*reopened)
                                    : arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchStreamReader>>(
                                          reopened.status());
        if (!stream.ok()) {
            peek.error = "Not a readable Arrow / Feather file: " + file_reader.status().message();
            return peek;
        }
        schema = (*stream)->schema();
        (void)(*stream)->ReadNext(&first);
    }
    auto head = first ? arrow::Table::FromRecordBatches({first}) : arrow::Table::MakeEmpty(schema);
    if (!head.ok()) {
        peek.error = "Not a readable Arrow / Feather file: " + head.status().message();
        return peek;
    }
    FillTablePeek(**head, limits, peek);
    return peek;
}

// The first sheet of a small workbook (the adapter reads a sheet whole).
FilePeek PeekWorkbook(const std::string& path, const PeekLimits& limits) {
    FilePeek peek;
    peek.ok = true;
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (!ec && size > limits.max_workbook_bytes) {
        peek.note = "Large workbook: rows are read when a Data Input or Data Convert opens it.";
        return peek;
    }
    ExcelTableReadOptions options;
    std::string error;
    auto table = ReadExcelTable(path, options, error);
    if (!table) {
        peek.note = "No preview: " + (error.empty() ? std::string("the workbook could not be read") : error) + ".";
        return peek;
    }
    peek.total_rows = table->num_rows();
    FillTablePeek(*table, limits, peek);
    return peek;
}

// An HDF5 file's top-level groups and datasets (metadata only).
FilePeek PeekHdf5(const std::string& path, const PeekLimits& limits) {
    FilePeek peek;
    Hdf5BrowseRequest request;
    request.group_path = "/";
    request.limit = static_cast<uint32_t>(std::max(1, limits.max_lines));
    const Hdf5BrowsePage page = HDF5Browser::BrowsePage(path, request);
    if (page.status != Hdf5TableStatus::Ok) {
        peek.ok = true;
        peek.note = "No preview: " + (page.error.empty() ? std::string("the file could not be listed") : page.error) + ".";
        return peek;
    }
    for (const auto& entry : page.entries) {
        std::string line = entry.kind == Hdf5ObjectKind::Group ? entry.name + "/" : entry.name;
        if (entry.kind == Hdf5ObjectKind::Dataset) {
            std::string shape;
            for (const auto dim : entry.dataset.shape) shape += (shape.empty() ? "" : " x ") + std::to_string(dim);
            line += "   [" + shape + "]";
            if (!entry.dataset.source_type.empty()) line += " " + entry.dataset.source_type;
        }
        peek.lines.push_back(std::move(line));
    }
    peek.more_lines = page.total_entries > page.entries.size();
    peek.line_noun = "object";
    peek.ok = true;
    return peek;
}

FilePeek PeekText(const std::string& path, const PeekLimits& limits) {
    FilePeek peek;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        peek.error = "Could not open the file.";
        return peek;
    }
    std::string line;
    std::uintmax_t bytes_read = 0;
    while (std::getline(in, line) && bytes_read < limits.head_bytes) {
        bytes_read += line.size() + 1;
        if (static_cast<int>(peek.lines.size()) == limits.max_lines) {
            peek.more_lines = true;
            break;
        }
        StripLineEnd(line);
        if (line.size() > 400) line = line.substr(0, 400) + "...";
        peek.lines.push_back(std::move(line));
    }
    peek.ok = true;
    return peek;
}

std::string WithThousands(int64_t value) {
    std::string digits = std::to_string(value);
    for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<size_t>(i), ",");
    return digits;
}

}  // namespace

FileKind KindOfPath(const std::string& path, bool is_directory) {
    if (is_directory) return FileKind::Folder;
    const std::string ext = Extension(path);
    if (OneOf(ext, {".csv", ".tsv"})) return FileKind::Table;
    if (OneOf(ext, {".xlsx", ".xls"})) return FileKind::Spreadsheet;
    if (OneOf(ext, {".parquet", ".arrow", ".feather", ".h5", ".hdf5", ".hdf"})) return FileKind::Columnar;
    if (OneOf(ext, {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"})) return FileKind::Config;
    if (OneOf(ext, {".py", ".pyw", ".cyx"})) return FileKind::Python;
    if (ext == ".ipynb") return FileKind::Notebook;
    if (ext == ".cyxgraph") return FileKind::Graph;
    if (OneOf(ext, {".onnx", ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".checkpoint", ".gguf", ".lora"}))
        return FileKind::Model;
    if (OneOf(ext, {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"})) return FileKind::Image;
    if (OneOf(ext, {".wav", ".mp3", ".flac", ".ogg", ".m4a"})) return FileKind::Audio;
    if (OneOf(ext, {".mp4", ".avi", ".mov", ".mkv", ".webm"})) return FileKind::Video;
    if (OneOf(ext, {".txt", ".md", ".log", ".rst"})) return FileKind::Text;
    if (OneOf(ext, {".dll", ".so", ".dylib"})) return FileKind::Plugin;
    return FileKind::Other;
}

std::string TypeLabel(const std::string& path, bool is_directory) {
    if (is_directory) return "Folder";
    std::string ext = std::filesystem::path(path).extension().string();
    if (ext.size() < 2) return "File";
    ext.erase(0, 1);
    for (auto& ch : ext) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return ext;
}

bool PeeksAsTable(const std::string& path) {
    return OneOf(Extension(path), {".csv", ".tsv", ".parquet", ".arrow", ".feather", ".ipc", ".xlsx"});
}

bool PeeksAsText(FileKind kind) {
    return kind == FileKind::Config || kind == FileKind::Python || kind == FileKind::Text;
}

FilePeek PeekFile(const std::string& path, const PeekLimits& limits) {
    const std::string ext = Extension(path);
    if (ext == ".csv") return PeekDelimited(path, ',', limits);
    if (ext == ".tsv") return PeekDelimited(path, '\t', limits);
    if (ext == ".parquet") return PeekParquet(path, limits);
    if (OneOf(ext, {".arrow", ".feather", ".ipc"})) return PeekArrowIpc(path, limits);
    if (ext == ".xlsx") return PeekWorkbook(path, limits);
    if (OneOf(ext, {".h5", ".hdf5", ".hdf"})) return PeekHdf5(path, limits);
    if (ext == ".xls") {
        FilePeek old;
        old.ok = true;
        old.note = "Only .xlsx workbooks can be previewed.";
        return old;
    }
    if (PeeksAsText(KindOfPath(path, false))) return PeekText(path, limits);
    FilePeek none;
    none.ok = true;  // nothing to show beyond the file's details
    return none;
}

std::string SummaryLine(const std::string& type_label, const std::string& size_text, const FilePeek& peek) {
    std::string line = type_label;
    if (!size_text.empty()) line += " \xC2\xB7 " + size_text;
    if (peek.ok && peek.table) {
        if (peek.total_rows >= 0)
            line += " \xC2\xB7 " + std::string(peek.rows_estimated ? "about " : "") + WithThousands(peek.total_rows) +
                    (peek.total_rows == 1 ? " row" : " rows");
        line += " \xC2\xB7 " + std::to_string(peek.columns.size()) + (peek.columns.size() == 1 ? " column" : " columns");
    } else if (peek.ok && !peek.lines.empty() && !peek.more_lines) {
        line += " \xC2\xB7 " + std::to_string(peek.lines.size()) + " " + peek.line_noun +
                (peek.lines.size() == 1 ? "" : "s");
    }
    return line;
}

}  // namespace cyxwiz::assets
