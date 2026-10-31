#pragma once

// Data Studio's Explore > Plugins tab (TOFIX181 step 4, approved mockup
// tofix181/step4_plugins_tab_mockup.html): the analytics installed plugins
// add, which columns each can read, when Run can start, and how an answer
// reads. Pure data in and out; the view (gui/data_studio/plugins_view.cpp)
// draws it and the test checks it.

#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz::explore_plugins {

enum class ColumnKind { Any, Numeric, Text };

// One analytic as its plugin declares it.
struct Analytic {
    std::string id;
    std::string name;
    std::string description;
    std::string category;
    std::string plugin_name;     // as the Plugin Manager shows it
    std::string plugin_version;
    ColumnKind kind = ColumnKind::Any;
    int min_columns = 1;
    int max_columns = 0;         // 0: no limit
};

// One column of the table, as the profile reads it.
struct Column {
    std::string name;
    bool numeric = false;
    bool text = false;
};

std::string Label(const Analytic& a);     // "Coefficient of variation (Distribution)"
std::string FromLine(const Analytic& a);  // "from Example Data Formats 1.0.0"

// Per column (table order), why the analytic cannot read it; empty when it can.
std::vector<std::string> RefusedReasons(const Analytic& a, const std::vector<Column>& columns);
// Every column the analytic reads, in table order, up to its maximum.
std::vector<std::string> DefaultColumns(const Analytic& a, const std::vector<Column>& columns);
// Drops picks the table no longer has or the analytic cannot read, and any
// beyond its maximum.
void KeepReadable(const Analytic& a, const std::vector<Column>& columns, std::vector<std::string>& picked);
// "numbers only: 2 of 5 columns"; empty for an analytic that reads any column.
std::string ColumnsHint(const Analytic& a, const std::vector<Column>& columns);

struct RunState {
    bool enabled = false;
    std::string reason;   // why Run is off
    std::string summary;  // "reads 14 rows of 2 columns"
};
RunState CanRun(const Analytic& a, const std::vector<std::string>& picked, int64_t rows, bool plugin_loaded);

// The question an answer belongs to: the dataset, the analytic, the columns.
std::string Key(const std::string& dataset, const std::string& analytic_id, const std::vector<std::string>& picked);

// What came back from the plugin, plus how it was computed.
struct Result {
    bool ok = false;
    std::string error;
    std::string name;
    std::string category;
    bool has_scalar = false;
    double scalar = 0.0;
    std::vector<std::string> labels;
    std::vector<double> values;
    std::string text;
    std::string plugin_name;
    std::string plugin_version;
    int64_t rows = 0;
    double milliseconds = 0.0;
    std::string finished_at;  // "11:24:05"
};

struct Bar {
    std::string label;
    std::string value;   // printed at the end of the bar
    float fraction = 0;  // of the largest magnitude, 0..1
};

struct AnswerView {
    std::string headline;  // the single number, when the plugin gave one
    std::vector<Bar> bars;
    std::string note;
    std::string by_line;   // "Computed by Example Data Formats 1.0.0 on 14 rows in 3 ms, 11:24:05"
    std::string value_header;  // the table's value column: the analytic's name, lower case
};
AnswerView Present(const Result& r);

// The values as text: tab-separated for Copy, CSV for Export CSV.
std::string CopyText(const Result& r);
std::string Csv(const Result& r);

std::string NumberText(double v);  // 0.0893, 0.1260, 1234.5, 1.235e+07

}  // namespace cyxwiz::explore_plugins
