#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace arrow { class Table; }

namespace cyxwiz {

struct ExcelTableReadOptions {
    std::string sheet_name; // Empty selects the first worksheet, not a chart sheet.
    bool has_header = true;
    int skip_rows = 0;
    uint64_t max_cells = 1000000;
};

// Host file ingress. Reads values only: dates retain their stored numeric serials;
// formulas and errors reject. Never evaluates formulas or saves the workbook.
// ZIP/XML budgets are checked in a staged snapshot before OpenXLSX parses it.
// Max one million cells, 1 MiB headers and 64 MiB Arrow string payload per read.
std::shared_ptr<arrow::Table> ReadExcelTable(
    const std::string& path, const ExcelTableReadOptions& options, std::string& error);

} // namespace cyxwiz
