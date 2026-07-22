#pragma once

#include "../node_config_dialog.h"
#include <string>
#include <utility>
#include <vector>

namespace gui::visualization {

// Configuration + rendering dialog for the BarChart node (Cat 2
// inspection). Opens when the user double-clicks the node.
//
// For v1 the dataset is selected from a dropdown of DataRegistry
// entries (user-driven). Once TrainingExecutor walks pins the dialog
// can auto-populate dataset + label column from the upstream
// DataInput/DatasetInput source the user wired in. See tofix.md under
// "TrainingExecutor should walk pin connections".
//
// Data source: reads the selected column from an ArrowDataset in
// DataRegistry, stringifies values (String/Int32/Int64 supported;
// Float/Double bucketed via std::to_string with limited precision),
// counts occurrences, and renders a horizontal ImPlot bar chart
// sorted by count descending. Capped at max_bars categories plus
// 1M rows scanned so free-text columns don't explode the map.
class BarChartDialog : public NodeConfigDialog {
public:
    BarChartDialog(MLNode* node);
    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(720, 580); }

protected:
    void RenderContent() override;

private:
    void RenderConfigPanel();
    void RenderChart();

    // Stream the selected dataset's `column_` and fill counts_. Cache
    // key skips repeat work when neither dataset nor column changed.
    void ComputeDistribution();

    // Walk upstream pins from this BarChart node to find a DataInput or
    // DatasetInput ancestor; if found, return its dataset_name parameter.
    // Empty string if no upstream dataset source. Uses node_editor_ from the base
    // class; no-op when the editor pointer is null.
    std::string FindUpstreamDatasetName() const;

    // Dialog state (mirrors node params; written back on Apply).
    std::string dataset_name_;
    std::string column_;
    std::string chart_type_ = "bar";   // bar | horizontal_bar
    std::string title_     = "Bar Chart";
    int         max_bars_  = 20;

    // Chart cache (dataset|column key).
    std::vector<std::pair<std::string, int>> counts_;
    std::string cache_key_;
    int64_t     total_rows_    = 0;
    bool        truncated_     = false;
    std::string last_error_;
};

} // namespace gui::visualization
