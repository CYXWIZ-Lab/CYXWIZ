#pragma once

// Class balancing rules shared by the batcher of a table in memory
// (ArrowDatasetBatcher) and of a table on disk (ParquetArrowBatcher,
// TOFIX157 A3), so both give each class the same number of rows.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

// Rows each class gets from the Data Loader's balance target: "min" the
// smallest class, "median" the median class, a number that many (at least
// 1), anything else the largest class. 0 when there are no classes.
inline size_t BalanceTargetCount(std::vector<size_t> class_counts, const std::string& target) {
    if (class_counts.empty()) return 0;
    std::sort(class_counts.begin(), class_counts.end());
    if (target == "min") return class_counts.front();
    if (target == "median") return class_counts[class_counts.size() / 2];
    if (!target.empty() &&
        std::all_of(target.begin(), target.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        try {
            return std::max<size_t>(1, static_cast<size_t>(std::stoull(target)));
        } catch (...) {
        }
    }
    return class_counts.back();
}

}  // namespace cyxwiz
