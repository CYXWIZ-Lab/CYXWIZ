#pragma once

// Class index = position in the sorted class names, so the same data gives
// the same indices whatever the CSV row order, the directory listing order or
// the platform. Names that are all integers sort as numbers ("2" before "10");
// otherwise they sort as text.

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace cyxwiz {

inline bool IsIntegerClassName(const std::string& name) {
    if (name.empty()) return false;
    size_t i = (name[0] == '-' || name[0] == '+') ? 1 : 0;
    if (i == name.size()) return false;
    for (; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    return true;
}

inline void SortClassNames(std::vector<std::string>& names) {
    const bool numeric = !names.empty() && std::all_of(names.begin(), names.end(), IsIntegerClassName);
    if (numeric) {
        std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
            return std::strtoll(a.c_str(), nullptr, 10) < std::strtoll(b.c_str(), nullptr, 10);
        });
    } else {
        std::sort(names.begin(), names.end());
    }
}

}  // namespace cyxwiz
