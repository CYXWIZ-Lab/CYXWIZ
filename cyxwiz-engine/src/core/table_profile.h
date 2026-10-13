#pragma once

// Column profile of a result table (TOFIX146 SQL Workbench, "Column
// profile"): per column, how many values are missing, empty or distinct;
// numbers get min, median, mean and max; any column gets its most common
// values and how many rows repeat an earlier value. No ImGui.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

struct ArrowColumnProfile {
    std::string name;
    std::string type;       // Arrow type name, e.g. "string", "int64"
    int64_t rows = 0;
    int64_t nulls = 0;
    int64_t empty = 0;      // text columns: "" values
    int64_t distinct = 0;   // distinct non-null values
    int64_t repeated = 0;   // non-null rows whose value appeared in an earlier row
    bool numeric = false;
    double min = 0.0;
    double median = 0.0;
    double mean = 0.0;
    double max = 0.0;
    // Most common non-null values, most frequent first (ties: first seen).
    std::vector<std::pair<std::string, int64_t>> top;
};

std::vector<ArrowColumnProfile> ProfileArrowTable(const arrow::Table& table, size_t top_values = 5);

}  // namespace cyxwiz
