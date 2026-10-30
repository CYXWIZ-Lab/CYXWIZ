#pragma once

// The searchable column picker of the Plot window (TOFIX134 P2, approved
// board 6): a field that opens a list with a search box, each column's type,
// range and share not 0, and an option to hide columns with one value. Made
// for wide tables (MNIST: 785 columns). Up/Down move, Enter picks.

#include "../../core/plot/plot_prepare.h"

#include <string>
#include <vector>

namespace cyxwiz::plot {

class ColumnPicker {
public:
    // One column. `none` names the empty choice ("(row number)", "(none)");
    // nullptr when a column is required. True when the choice changed.
    bool Pick(const char* id, std::string& value, const std::vector<ColumnSummary>& columns, bool numeric_only,
              const char* none, float width);
    // Several columns (one series each), with "add all matches", a range
    // and chips to remove one. True when the choice changed.
    bool PickMany(const char* id, std::vector<std::string>& values, const std::vector<ColumnSummary>& columns,
                  bool numeric_only, float width);
    // Several columns, where `refused` (one per column, in order) says why
    // the field cannot take a column: those stay in the list, dimmed, with
    // the reason. Empty: the column can be picked.
    bool PickMany(const char* id, std::vector<std::string>& values, const std::vector<ColumnSummary>& columns,
                  const std::vector<std::string>& refused, float width);

private:
    // The list inside an open picker; returns the index of the picked
    // column in `columns`, or -1.
    int DrawList(const std::vector<ColumnSummary>& columns, bool numeric_only, const std::vector<std::string>* refused,
                 const std::vector<std::string>& chosen, const std::string& current, bool many,
                 std::vector<int>* matches);
    bool PickManyIn(const char* id, std::vector<std::string>& values, const std::vector<ColumnSummary>& columns,
                    bool numeric_only, const std::vector<std::string>* refused, float width);
    void Opened(const char* id);

    std::string open_id_;   // the picker whose list is open
    char search_[128] = {};
    char range_from_[128] = {};
    char range_to_[128] = {};
    bool hide_one_value_ = true;
    int cursor_ = 0;        // highlighted row among the matches
    bool focus_search_ = false;
};

}  // namespace cyxwiz::plot
