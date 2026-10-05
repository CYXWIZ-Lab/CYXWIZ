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
//     the target by its strongest feature; for a text column its KPIs, length,
//     top words and phrases, words by class and sample texts).

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
QueryRequest WidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap = 0,
                         const std::string& words_table = {});
// A text widget's query (TOFIX134 P3 text): lengths, top words or phrases
// (common words left out unless kept), or each top word's share of each
// class's texts, over the rows under the other widgets' filters.
// `words_table`: the text column's words split once (text_words.h; empty:
// split here).
QueryRequest TextWidgetQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, size_t row_cap = 0,
                             const std::string& words_table = {});
// Missing values per column (nulls plus the texts marked as missing), under
// the other widgets' filters: a row with "rows" and one count per column.
QueryRequest MissingQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, const std::vector<std::string>& columns,
                          const std::map<std::string, std::vector<std::string>>& missing_text);
// A KPI: value (filtered) and all (unfiltered) in one row.
QueryRequest KpiQuery(const WidgetSpec& w, const std::string& table, const FilterState& filters, const std::string& words_table = {});
// The dashboard's own KPI strip: rows (filtered and all), missing cells
// (filtered), and the target's mean or most frequent value (filtered and all).
QueryRequest StripQuery(const std::string& table, const FilterState& filters, const DatasetProfile& profile, const std::string& target,
                        bool target_numeric, const std::map<std::string, std::vector<std::string>>& missing_text = {});

// The query as text to read or run elsewhere (View SQL, the Query tab): the
// bound values written in as literals (texts quoted, numbers exact).
std::string InlineParams(const std::string& sql, const std::vector<QueryParam>& params);
// The dashboard's rows under all its filters, as SQL (Open in Data Studio).
std::string FilteredRowsSql(const std::string& table, const FilterState& filters);

// The automatic widgets (marked automatic), laid out three across.
std::vector<WidgetSpec> AutomaticWidgets(const DatasetProfile& profile, const DatasetContract& contract, DashboardSpec& spec);

}  // namespace cyxwiz::dashboard
