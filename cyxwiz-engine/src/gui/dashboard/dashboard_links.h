#pragma once

// Add to Dashboard (TOFIX134 P3.11, approved board 15): Data Studio's
// Visualize and Query tabs put a widget into a Dashboard node of the open
// graph. The node editor fills these hooks (MainWindow wires them); the
// screens draw one shared menu. UI thread only.

#include "../../core/dashboard/dashboard_model.h"

#include <arrow/type_fwd.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::dashboard {

struct DashboardTarget {
    int node_id = 0;
    std::string name;          // the node's name
    std::string feeder;        // what it shows ("Spotify")
    int widgets = 0;
    bool same_data = false;    // it shows the dataset the widget reads
    std::string reason;        // why it cannot take it (other data, nothing connected)
};

struct DashboardLinkHooks {
    // The Dashboard nodes of the open graph, and whether each shows `dataset`.
    std::function<std::vector<DashboardTarget>(const std::string& dataset)> list;
    // The Data Input a new Dashboard node would read `dataset` from (its name), or empty.
    std::function<std::string(const std::string& dataset)> new_source;
    // Adds the widget to that node (0: a new Dashboard node on the dataset's
    // Data Input) and opens the dashboard with it selected.
    std::function<void(int node_id, const std::string& dataset, const WidgetSpec& widget)> add;
};

DashboardLinkHooks& DashboardLinks();

// The menu's items (inside an open popup); true when the widget was handed over.
bool DrawAddToDashboardItems(const std::string& dataset, const WidgetSpec& widget);

// A query result as a dashboard widget: Bar of the first number by the first
// text column, else Scatter of two numbers, else Histogram of one; the query's
// table name stands for the dashboard's rows. Empty `reason` when it can be one.
WidgetSpec QueryResultWidget(const std::string& sql, const std::string& table_name, const std::shared_ptr<arrow::Schema>& schema,
                             std::string* reason);

}  // namespace cyxwiz::dashboard
