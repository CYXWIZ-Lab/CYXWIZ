#pragma once

// A DataTable written out as CSV, TSV or JSON (the Table Viewer's Export and
// Copy, TOFIX148). Numbers are written exactly (their shortest exact text,
// as numbers in JSON), empty cells as nothing in CSV / TSV and null in JSON.

#include "data_table.h"

#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

enum class TableTextFormat { Csv, Tsv, Json };

// `rows`: the rows to write, in this order. CSV and TSV quote a value that
// holds the separator, a quote or a line break; JSON is an array of objects
// keyed by the column names.
std::string TableText(const DataTable& table, const std::vector<size_t>& rows, TableTextFormat format,
                      bool headers = true);

// The same columns with only `rows`, in this order (for writing a subset).
std::shared_ptr<DataTable> TableRows(const DataTable& table, const std::vector<size_t>& rows);

}  // namespace cyxwiz
