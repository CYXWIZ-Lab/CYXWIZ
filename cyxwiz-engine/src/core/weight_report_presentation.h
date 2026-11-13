#pragma once

// What the Data Loader dialog shows about a weights file: the summary card
// and the match report's rows. Pure data in, words out, so the dialog only
// draws.

#include "weight_match.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct WeightReportRow {
    enum class Kind { Loaded, Fresh, FileOnly };
    Kind kind = Kind::Loaded;
    std::string node;     // node name; empty for an entry no node took
    std::string layer;    // "Conv2D"
    std::string entries;  // "conv1.weight [64, 3, 7, 7]", several joined with ", "
    std::string shapes;   // the graph's shapes
    std::string result;   // "Loaded", "Starts fresh: shape differs ...", "Not used: ..."
};

struct WeightReportView {
    std::string file_line;                 // "resnet18.pth: PyTorch state-dict, 102 entries"
    std::string loaded_line;               // "40 of 41 layers take their weights"
    std::vector<std::string> fresh_lines;  // one per fresh layer, at most three, then "and N more"
    std::string unused_line;               // "2 entries in the file are not used"; empty when none
    std::vector<WeightReportRow> rows;     // the graph's layers in model order, then file-only entries
    size_t layers = 0;
    size_t loaded = 0;
    size_t fresh = 0;
    size_t file_only = 0;
};

WeightReportView BuildWeightReportView(const weight_match::Report& report, const std::string& file_name);

// The rows a filter shows: 0 all layers, 1 loaded, 2 starts fresh, 3 in the file only.
bool WeightReportRowShown(const WeightReportRow& row, int filter);

}  // namespace cyxwiz
