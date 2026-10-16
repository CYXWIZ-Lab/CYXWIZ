#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class DataTable;

// Class labels read from one or more table columns that share one class set
// (true and predicted labels, or a single target column).
//
// When every non-empty cell is a number, the numbers are the ids (so DBSCAN's
// -1 for noise stays -1) and `names` is empty. Otherwise each distinct text
// gets an id 0..k-1 in order of first appearance across the columns, and
// `names[id]` is that text. Empty cells are std::nullopt.
struct TableClassLabels {
    std::vector<std::vector<std::optional<int>>> columns;  // [column][row]
    std::vector<std::string> names;

    bool IsText() const { return !names.empty(); }
    // Display name of a class id: its text, or the number itself.
    std::string Name(int id) const;
};

TableClassLabels ReadTableClassLabels(const DataTable& table, const std::vector<int>& columns);

}  // namespace cyxwiz
