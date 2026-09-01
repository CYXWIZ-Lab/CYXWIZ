#include "data_table.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

// HDF5 support (optional - only if HighFive is available)
#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#endif

namespace cyxwiz {

// ============================================================================
// DataTable Implementation
// ============================================================================

void DataTable::SetHeaders(const std::vector<std::string>& headers) {
    headers_ = headers;
}

void DataTable::AddRow(const Row& row) {
    if (!headers_.empty() && row.size() != headers_.size()) {
        spdlog::warn("Row size ({}) does not match header count ({})", row.size(), headers_.size());
    }
    rows_.push_back(row);
}

void DataTable::AddRow(Row&& row) {
    if (!headers_.empty() && row.size() != headers_.size()) {
        spdlog::warn("Row size ({}) does not match header count ({})", row.size(), headers_.size());
    }
    rows_.push_back(std::move(row));
}

void DataTable::Clear() {
    headers_.clear();
    rows_.clear();
}

const DataTable::Row& DataTable::GetRow(size_t index) const {
    if (index >= rows_.size()) {
        throw std::out_of_range("Row index out of range");
    }
    return rows_[index];
}

DataTable::CellValue DataTable::GetCell(size_t row, size_t col) const {
    if (row >= rows_.size()) {
        return std::monostate{};
    }
    if (col >= rows_[row].size()) {
        return std::monostate{};
    }
    return rows_[row][col];
}

std::string DataTable::GetCellAsString(size_t row, size_t col) const {
    auto cell = GetCell(row, col);

    return std::visit([](auto&& arg) -> std::string {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
            return arg;
        } else if constexpr (std::is_same_v<T, double>) {
            return std::to_string(arg);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return std::to_string(arg);
        } else {
            return "";
        }
    }, cell);
}

bool DataTable::SetCell(size_t row, size_t col, const CellValue& value) {
    if (row >= rows_.size()) {
        spdlog::warn("SetCell: Row index {} out of range ({})", row, rows_.size());
        return false;
    }
    if (col >= rows_[row].size()) {
        spdlog::warn("SetCell: Col index {} out of range ({})", col, rows_[row].size());
        return false;
    }
    rows_[row][col] = value;
    return true;
}

bool DataTable::SetCellFromString(size_t row, size_t col, const std::string& value) {
    if (row >= rows_.size() || col >= rows_[row].size()) {
        return false;
    }

    // Try to parse as number, otherwise store as string
    if (value.empty()) {
        return SetCell(row, col, std::monostate{});
    }

    // Try integer first
    try {
        if (value.find('.') == std::string::npos &&
            value.find('e') == std::string::npos &&
            value.find('E') == std::string::npos) {
            int64_t int_val = std::stoll(value);
            return SetCell(row, col, int_val);
        }
    } catch (...) {}

    // Try double
    try {
        double dbl_val = std::stod(value);
        return SetCell(row, col, dbl_val);
    } catch (...) {}

    // Store as string
    return SetCell(row, col, value);
}

// ============================================================================
// CSV Support
// ============================================================================

bool DataTable::LoadFromCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open CSV file: {}", filepath);
        return false;
    }

    Clear();

    std::string line;
    bool first_row = true;

    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        // Simple CSV parsing (doesn't handle quoted commas)
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            tokens.push_back(token);
        }

        if (first_row) {
            // First row is headers
            SetHeaders(tokens);
            first_row = false;
        } else {
            // Data rows - try to parse as numbers
            Row row;
            for (const auto& str : tokens) {
                if (str.empty()) {
                    row.push_back(std::monostate{});
                } else {
                    // Try to parse as number
                    try {
                        // Check if it's an integer
                        if (str.find('.') == std::string::npos && str.find('e') == std::string::npos && str.find('E') == std::string::npos) {
                            int64_t val = std::stoll(str);
                            row.push_back(val);
                        } else {
                            double val = std::stod(str);
                            row.push_back(val);
                        }
                    } catch (...) {
                        // Not a number, store as string
                        row.push_back(str);
                    }
                }
            }
            AddRow(std::move(row));
        }
    }

    spdlog::info("Loaded CSV: {} rows, {} columns", GetRowCount(), GetColumnCount());
    return true;
}

bool DataTable::SaveToCSV(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open CSV file for writing: {}", filepath);
        return false;
    }

    // Write headers
    for (size_t i = 0; i < headers_.size(); i++) {
        file << headers_[i];
        if (i < headers_.size() - 1) file << ",";
    }
    file << "\n";

    // Write data rows
    for (const auto& row : rows_) {
        for (size_t i = 0; i < row.size(); i++) {
            file << GetCellAsString(std::distance(rows_.data(), &row), i);
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    spdlog::info("Saved CSV: {} rows, {} columns", GetRowCount(), GetColumnCount());
    return true;
}

// ============================================================================
// TXT Support (Tab/Space delimited)
// ============================================================================

bool DataTable::LoadFromTXT(const std::string& filepath, char delimiter) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open TXT file: {}", filepath);
        return false;
    }

    Clear();

    std::string line;
    bool first_row = true;

    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        // Parse by delimiter
        while (std::getline(ss, token, delimiter)) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            tokens.push_back(token);
        }

        // Skip empty rows
        if (tokens.empty()) continue;

        if (first_row) {
            // First row is headers
            SetHeaders(tokens);
            first_row = false;
        } else {
            // Data rows - try to parse as numbers
            Row row;
            for (const auto& str : tokens) {
                if (str.empty()) {
                    row.push_back(std::monostate{});
                } else {
                    // Try to parse as number
                    try {
                        // Check if it's an integer
                        if (str.find('.') == std::string::npos && str.find('e') == std::string::npos && str.find('E') == std::string::npos) {
                            int64_t val = std::stoll(str);
                            row.push_back(val);
                        } else {
                            double val = std::stod(str);
                            row.push_back(val);
                        }
                    } catch (...) {
                        // Not a number, store as string
                        row.push_back(str);
                    }
                }
            }
            AddRow(std::move(row));
        }
    }

    std::string delim_name = (delimiter == '\t') ? "tab" : "space";
    spdlog::info("Loaded TXT ({}): {} rows, {} columns", delim_name, GetRowCount(), GetColumnCount());
    return true;
}

bool DataTable::SaveToTXT(const std::string& filepath, char delimiter) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("Failed to open TXT file for writing: {}", filepath);
        return false;
    }

    // Write headers
    for (size_t i = 0; i < headers_.size(); i++) {
        file << headers_[i];
        if (i < headers_.size() - 1) file << delimiter;
    }
    file << "\n";

    // Write data rows
    for (const auto& row : rows_) {
        for (size_t i = 0; i < row.size(); i++) {
            file << GetCellAsString(std::distance(rows_.data(), &row), i);
            if (i < row.size() - 1) file << delimiter;
        }
        file << "\n";
    }

    std::string delim_name = (delimiter == '\t') ? "tab" : "space";
    spdlog::info("Saved TXT ({}): {} rows, {} columns", delim_name, GetRowCount(), GetColumnCount());
    return true;
}

// ============================================================================
// HDF5 Support
// ============================================================================

bool DataTable::LoadFromHDF5(const std::string& filepath, const std::string& dataset_name) {
#ifdef CYXWIZ_HAS_HDF5
    try {
        HighFive::File file(filepath, HighFive::File::ReadOnly);
        
        // Try to find the dataset - first the specified name, then common alternatives
        std::string actual_name = dataset_name;
        if (!file.exist(dataset_name)) {
            // Try common dataset names
            const std::vector<std::string> common_names = {
                "data", "images", "features", "X", "x", "inputs",
                "labels", "targets", "y", "Y", "outputs"
            };
            bool found = false;
            for (const auto& name : common_names) {
                if (file.exist(name)) {
                    actual_name = name;
                    found = true;
                    spdlog::info("HDF5: Using dataset '{}' (requested '{}' not found)", name, dataset_name);
                    break;
                }
            }
            if (!found) {
                spdlog::error("HDF5: No suitable dataset found in file");
                return false;
            }
        }
        
        HighFive::DataSet dataset = file.getDataSet(actual_name);

        // Get dimensions
        auto dims = dataset.getDimensions();
        if (dims.size() < 1) {
            spdlog::error("HDF5 dataset is empty");
            return false;
        }
        
        // Handle multi-dimensional data by flattening inner dimensions
        size_t rows = dims[0];
        size_t cols = 1;
        for (size_t i = 1; i < dims.size(); i++) {
            cols *= dims[i];
        }
        
        if (dims.size() != 2) {
            spdlog::info("HDF5: Flattening {}D dataset to 2D ({} x {})", dims.size(), rows, cols);
        }

        Clear();

        // Create default headers
        std::vector<std::string> headers;
        for (size_t i = 0; i < cols; i++) {
            headers.push_back("Col_" + std::to_string(i));
        }
        SetHeaders(headers);

        // Read data row by row - container must match selection dimensionality
        std::vector<size_t> offset(dims.size(), 0);
        std::vector<size_t> count(dims.size(), 1);
        for (size_t d = 1; d < dims.size(); d++) {
            count[d] = dims[d];
        }

        for (size_t r = 0; r < rows; r++) {
            offset[0] = r;
            auto selection = dataset.select(offset, count);

            Row row;
            row.reserve(cols);

            if (dims.size() == 1) {
                // 1D: single value per row
                std::vector<double> val(1);
                selection.read(val);
                row.push_back(val[0]);
            } else if (dims.size() == 2) {
                // 2D: [N, features] - read into 1D vector
                std::vector<double> row_data(cols);
                selection.read(row_data);
                for (double val : row_data) {
                    row.push_back(val);
                }
            } else if (dims.size() == 3) {
                // 3D: [N, H, W] - selection [1, H, W] needs 3D container
                std::vector<std::vector<std::vector<double>>> data_3d;
                selection.read(data_3d);
                for (const auto& plane : data_3d) {
                    for (const auto& row_values : plane) {
                        for (double val : row_values) {
                            row.push_back(val);
                        }
                    }
                }
            } else if (dims.size() == 4) {
                // 4D: [N, H, W, C] - selection [1, H, W, C] needs 4D container
                std::vector<std::vector<std::vector<std::vector<double>>>> data_4d;
                selection.read(data_4d);
                for (const auto& vol : data_4d) {
                    for (const auto& plane : vol) {
                        for (const auto& row_values : plane) {
                            for (double val : row_values) {
                                row.push_back(val);
                            }
                        }
                    }
                }
            } else {
                spdlog::error("HDF5: Unsupported dimensionality {} for table view", dims.size());
                return false;
            }

            AddRow(std::move(row));
        }

        spdlog::info("Loaded HDF5: {} rows, {} columns", GetRowCount(), GetColumnCount());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("HDF5 load error: {}", e.what());
        return false;
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return false;
#endif
}

bool DataTable::SaveToHDF5(const std::string& filepath, const std::string& dataset_name) const {
#ifdef CYXWIZ_HAS_HDF5
    try {
        HighFive::File file(filepath, HighFive::File::Truncate);

        // Convert table to 2D vector of doubles (best effort)
        std::vector<std::vector<double>> data;
        for (size_t r = 0; r < GetRowCount(); r++) {
            std::vector<double> row_data;
            for (size_t c = 0; c < GetColumnCount(); c++) {
                auto cell = GetCell(r, c);
                std::visit([&row_data](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, double>) {
                        row_data.push_back(arg);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        row_data.push_back(static_cast<double>(arg));
                    } else {
                        row_data.push_back(0.0); // Default for strings/null
                    }
                }, cell);
            }
            data.push_back(row_data);
        }

        // Create dataset
        HighFive::DataSet dataset = file.createDataSet<double>(
            dataset_name,
            HighFive::DataSpace::From(data)
        );
        dataset.write(data);

        spdlog::info("Saved HDF5: {} rows, {} columns", GetRowCount(), GetColumnCount());
        return true;

    } catch (const std::exception& e) {
        spdlog::error("HDF5 save error: {}", e.what());
        return false;
    }
#else
    spdlog::error("HDF5 support not compiled (HighFive library missing)");
    return false;
#endif
}

// ============================================================================
// Excel Support (Python-based)
// ============================================================================

bool DataTable::LoadFromExcel(const std::string& filepath, const std::string& sheet_name) {
    // TODO: Implement Excel loading via Python openpyxl
    // This will be implemented later when we integrate with ScriptingEngine
    spdlog::error(
        "Excel support is not implemented; cannot load file '{}' (sheet='{}')",
        filepath,
        sheet_name.empty() ? "<default>" : sheet_name);
    return false;
}

// ============================================================================
// DataTableRegistry Implementation
// ============================================================================

DataTableRegistry& DataTableRegistry::Instance() {
    static DataTableRegistry instance;
    return instance;
}

void DataTableRegistry::AddTable(const std::string& name, std::shared_ptr<DataTable> table) {
    tables_[name] = table;
    spdlog::info("Added table to registry: {}", name);
}

void DataTableRegistry::RemoveTable(const std::string& name) {
    tables_.erase(name);
    spdlog::info("Removed table from registry: {}", name);
}

void DataTableRegistry::Clear() {
    tables_.clear();
    spdlog::info("Cleared table registry");
}

std::shared_ptr<DataTable> DataTableRegistry::GetTable(const std::string& name) {
    auto it = tables_.find(name);
    if (it != tables_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::string> DataTableRegistry::GetTableNames() const {
    std::vector<std::string> names;
    for (const auto& [name, table] : tables_) {
        names.push_back(name);
    }
    return names;
}

} // namespace cyxwiz
