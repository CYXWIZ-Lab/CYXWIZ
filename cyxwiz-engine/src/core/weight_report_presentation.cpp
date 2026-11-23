#include "weight_report_presentation.h"

namespace cyxwiz {

WeightReportView BuildWeightReportView(const weight_match::Report& report, const std::string& file_name) {
    WeightReportView view;
    view.loaded = report.loaded;
    view.fresh = report.fresh;
    view.layers = report.loaded + report.fresh;
    view.file_only = report.unused.size();
    view.file_line = file_name + ": " + report.format + ", " + std::to_string(report.file_entries) +
                     (report.file_entries == 1 ? " entry" : " entries");
    view.loaded_line = std::to_string(view.loaded) + " of " + std::to_string(view.layers) +
                       (view.layers == 1 ? " layer takes its weights" : " layers take their weights");
    size_t shown = 0;
    for (const auto& layer : report.layers) {
        WeightReportRow row;
        row.kind = layer.loaded ? WeightReportRow::Kind::Loaded : WeightReportRow::Kind::Fresh;
        row.node = layer.node_name;
        row.layer = layer.type_label;
        for (const auto& entry : layer.file_entries) row.entries += (row.entries.empty() ? "" : ", ") + entry;
        row.shapes = layer.graph_shapes;
        row.result = layer.loaded ? "Loaded" : "Starts fresh: " + layer.reason;
        view.rows.push_back(std::move(row));
        if (layer.loaded) continue;
        if (shown < 3) view.fresh_lines.push_back(layer.node_name + " starts fresh: " + layer.reason);
        ++shown;
    }
    if (shown > 3) view.fresh_lines.push_back("and " + std::to_string(shown - 3) + " more (see the match report)");
    for (const auto& unused : report.unused) {
        WeightReportRow row;
        row.kind = WeightReportRow::Kind::FileOnly;
        row.entries = unused.name;
        row.result = "Not used: " + unused.reason;
        view.rows.push_back(std::move(row));
    }
    if (view.file_only) {
        view.unused_line = std::to_string(view.file_only) +
                           (view.file_only == 1 ? " entry in the file is not used" : " entries in the file are not used");
    }
    return view;
}

bool WeightReportRowShown(const WeightReportRow& row, int filter) {
    switch (filter) {
        case 1: return row.kind == WeightReportRow::Kind::Loaded;
        case 2: return row.kind == WeightReportRow::Kind::Fresh;
        case 3: return row.kind == WeightReportRow::Kind::FileOnly;
        default: return row.kind != WeightReportRow::Kind::FileOnly;
    }
}

}  // namespace cyxwiz
