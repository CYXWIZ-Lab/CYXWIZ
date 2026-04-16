#include "visualization_nodes.h"

namespace gui::visualization {

void PopulateBarChartNode(MLNode& node, int& next_pin_id) {
    // Cat 2 inspection node — no output pin, reads a dataset or label
    // stream and renders a bar chart on double-click via
    // BarChartDialog. Does NOT transform data; training skips it.
    // First node of the visualization framework; Histogram / LinePlot
    // / ScatterPlot / PieChart follow the same pattern as they land.
    NodePin data_in;
    data_in.id = next_pin_id++;
    data_in.type = PinType::Tensor;
    data_in.name = "Data";
    data_in.is_input = true;
    data_in.description =
        "Tabular stream to chart. Usually wired from DataInput.Data "
        "(or a preprocessing node's output). The `column` parameter "
        "picks which column's values to plot.";
    node.inputs.push_back(data_in);

    NodePin labels_in;
    labels_in.id = next_pin_id++;
    labels_in.type = PinType::Labels;
    labels_in.name = "Labels";
    labels_in.is_input = true;
    labels_in.is_required = false;  // Optional — class-distribution mode only.
    labels_in.description =
        "Optional. Connect DataInput.Labels to switch to "
        "class-distribution mode: the chart counts values in the "
        "label stream instead of a column of Data.";
    node.inputs.push_back(labels_in);

    // No output pin — the chart is a terminal sink for the data path.
    // The node reads its upstream data when the user opens its dialog,
    // not during training.
    node.parameters["chart_type"] = "bar";   // bar | horizontal_bar
    node.parameters["column"]     = "";      // empty = auto (label col for class-dist mode)
    node.parameters["title"]      = "Bar Chart";
    node.parameters["max_bars"]   = "20";    // truncate to top-N when many categories
}

} // namespace gui::visualization
