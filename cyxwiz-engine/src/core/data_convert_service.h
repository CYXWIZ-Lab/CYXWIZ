#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace arrow {
class Table;
}

namespace cyxwiz {

struct DataConvertColumnPreview {
    std::string name;
    std::string type;
    bool nullable = false;
};

struct DataConvertPreview {
    bool ok = false;
    std::string error;
    int64_t rows = 0;
    int64_t columns = 0;
    char detected_delimiter = ',';
    std::vector<DataConvertColumnPreview> schema;
    std::vector<std::vector<std::string>> sample_rows;
};

struct DataConvertOptions {
    std::string input_path;
    std::string output_path;
    std::string input_format = "auto";
    std::string output_format = "parquet";
    char delimiter = ',';
    bool auto_detect_delimiter = false;
    bool has_header = true;
    bool allow_newlines_in_values = true;
    int skip_rows = 0;
    // Kept for backward compatibility with older JSON parameter naming.
    std::string parquet_compression = "snappy";
    int64_t row_group_size = 1024 * 1024;
    bool overwrite = false;
    bool create_parent_dirs = true;
    bool write_manifest = true;
    int preview_rows = 20;
    std::shared_ptr<arrow::Table> input_table;
};

struct DataConvertResult {
    bool ok = false;
    bool skipped_fresh_output = false;
    std::string error;
    std::string output_path;
    std::string manifest_path;
    int64_t rows_read = 0;
    int64_t rows_written = 0;
    int64_t columns = 0;
    int64_t bytes_written = 0;
    char detected_delimiter = ',';
};

class DataConvertService {
public:
    static std::shared_ptr<arrow::Table> LoadTable(
        const DataConvertOptions& options,
        std::string& error);
    static DataConvertPreview Preview(const DataConvertOptions& options);
    static DataConvertResult Convert(const DataConvertOptions& options);

    // Backward-compatible phase-1 entry points.
    static DataConvertPreview PreviewCsv(const DataConvertOptions& options);
    static DataConvertResult ConvertCsvToParquet(const DataConvertOptions& options);
};

} // namespace cyxwiz
