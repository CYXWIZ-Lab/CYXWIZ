#pragma once

#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <memory>
#include <map>

namespace cyxwiz {

/**
 * DataTable - In-memory representation of tabular data
 * Supports CSV, Excel, HDF5 file formats
 */
class DataTable {
public:
    // Cell value can be string, double, int, or null
    using CellValue = std::variant<std::string, double, int64_t, std::monostate>;

    // Row is a vector of cells
    using Row = std::vector<CellValue>;

    DataTable() = default;
    ~DataTable() = default;

    // Set column headers
    void SetHeaders(const std::vector<std::string>& headers);
    const std::vector<std::string>& GetHeaders() const { return headers_; }

    // Add data rows
    void AddRow(const Row& row);
    void AddRow(Row&& row);
    void Clear();

    // Access data
    size_t GetRowCount() const { return rows_.size(); }
    size_t GetColumnCount() const { return headers_.size(); }
    const Row& GetRow(size_t index) const;
    CellValue GetCell(size_t row, size_t col) const;

    // Modify data
    bool SetCell(size_t row, size_t col, const CellValue& value);
    bool SetCellFromString(size_t row, size_t col, const std::string& value);

    // Get cell as string (for display)
    std::string GetCellAsString(size_t row, size_t col) const;

    // File I/O
    bool LoadFromCSV(const std::string& filepath);
    bool SaveToCSV(const std::string& filepath) const;

    bool LoadFromTXT(const std::string& filepath, char delimiter = '\t');
    bool SaveToTXT(const std::string& filepath, char delimiter = '\t') const;

    bool LoadFromHDF5(const std::string& filepath, const std::string& dataset_name = "data");
    bool SaveToHDF5(const std::string& filepath, const std::string& dataset_name = "data") const;

    // Python-based loaders (uses ScriptingEngine)
    bool LoadFromExcel(const std::string& filepath, const std::string& sheet_name = "");

    // Metadata
    void SetName(const std::string& name) { name_ = name; }
    const std::string& GetName() const { return name_; }

private:
    std::string name_;
    std::vector<std::string> headers_;
    std::vector<Row> rows_;
};

} // namespace cyxwiz
