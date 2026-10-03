#pragma once

// Dashboard runtime, pure parts (TOFIX134 P3.7, dashboard_architecture.md L5):
//   - CheckBindings: does each widget's field still exist and fit its role?
//     A missing field gets a rename candidate when one new column of the same
//     type appeared (addendum A.5: rebind, never stale numbers).
//   - Widget queries: the SQL a widget runs (only the columns it needs, the
//     other widgets' filters, bound values; KPIs filtered and unfiltered in
//     one query).
//   - AutomaticWidgets: the automatic layout from a profile and its contract
//     (the target, a card per category and number column, correlations,
//     the target by its strongest feature).

#include "../dataset_profiler.h"
#include "dashboard_model.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz::dashboard {

struct Binding {
    enum class State { Ok, FieldMissing, RoleMismatch };
    State state = State::Ok;
    std::string field;              // the field at fault
    std::string rename_candidate;   // FieldMissing: a new column of the same type
    std::string message;            // in words, for the widget
};

// `known_types`: column -> type when the dashboard last bound (DashboardSpec keeps it).
Binding CheckBinding(const WidgetSpec& w, const DatasetContract& contract, const std::map<std::string, std::string>& known_types);

// The SQL for a plot or table widget: its columns, the filters of the other
// widgets; `row_cap` 0: all rows (else a reproducible sample of that many).
QueryRequest WidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap = 0);
// A KPI: value (filtered) and all (unfiltered) in one row.
QueryRequest KpiQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters);
// The dashboard's own KPI strip: rows (filtered and all), missing cells
// (filtered), and the target's mean or most frequent value (filtered and all).
QueryRequest StripQuery(const std::string& table, const FilterState& filters, const DatasetProfile& profile, const std::string& target,
                        bool target_numeric);

// The automatic widgets (marked automatic), laid out three across.
std::vector<WidgetSpec> AutomaticWidgets(const DatasetProfile& profile, const DatasetContract& contract, DashboardSpec& spec);

}  // namespace cyxwiz::dashboard
