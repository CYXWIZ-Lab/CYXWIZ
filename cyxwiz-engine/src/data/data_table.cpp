#include "data_table.h"
#include "../core/csv_records.h"
#include "../core/gcc_diagnostics.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

// HDF5 support (optional - only if HighFive is available)
#ifdef CYXWIZ_HAS_HDF5
// HighFive's own path constructor: GCC -Wuninitialized false positive (TOFIX187).
CYXWIZ_GCC_IGNORE_BEGIN("-Wuninitialized")
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
CYXWIZ_GCC_IGNORE_END
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

// RFC 4180 records (core/csv_records): quoted fields may hold commas, line
// breaks and quotes, as pandas writes text columns. A cell is a number only
// when the whole field is one ("3 rows" stays text).
bool DataTable::LoadFromCSV(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Failed to open CSV file: {}", filepath);
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();

    Clear();
    auto records = csv::ReadRecords(buffer.str());
    bool first_row = true;
    for (auto& tokens : records) {
        if (first_row) {
            SetHeaders(tokens);
            first_row = false;
            continue;
        }
        Row row;
        row.reserve(tokens.size());
        for (auto& str : tokens) {
            long long i = 0;
            double d = 0.0;
            if (str.empty()) row.push_back(std::monostate{});
            else if (csv::ParseInt(str, i)) row.push_back(static_cast<int64_t>(i));
            else if (csv::ParseDouble(str, d)) row.push_back(d);
            else row.push_back(std::move(str));
        }
        AddRow(std::move(row));
    }

    spdlog::info("Loaded CSV: {} rows, {} columns", GetRowCount(), GetColumnCount());
    return true;
}

bool DataTable::SaveToCSV(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::error("Failed to open CSV file for writing: {}", filepath);
        return false;
    }

    // Fields with commas, quotes or line breaks are quoted, so the file reads back.
    for (size_t i = 0; i < headers_.size(); i++) {
        file << csv::Field(headers_[i]);
        if (i < headers_.size() - 1) file << ",";
    }
    file << "\n";
    for (size_t r = 0; r < rows_.size(); ++r) {
        const auto& row = rows_[r];
        for (size_t i = 0; i < row.size(); i++) {
            file << csv::Field(GetCellAsString(r, i));
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

bool DataTable::SaveToHDF5([[maybe_unused]] const std::string& filepath,
                            [[maybe_unused]] const std::string& dataset_name) const {
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

} // namespace cyxwiz
