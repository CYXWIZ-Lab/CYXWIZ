#pragma once

// The dashboard model (TOFIX134 P3.6, dashboard_architecture.md L4, approved
// board 13). Pure data, saved as versioned JSON in the Dashboard node; never
// holds results (widgets query their data when shown).
//
// A widget is a KPI, a table, or a plot. A plot widget holds a PlotSpec: the
// same settings as the Plot window, so every one of the plot types is a
// widget and "a plot is a one-widget dashboard". Widgets name fields
// (columns) by name; the runtime checks them against the dataset contract
// (roles) and says when a field is gone or no longer fits.
//
// One filter is shared by every widget (cross-filtering): clicking a bar
// adds "album_type = single"; each widget is drawn with the filters of the
// other widgets (its own filter does not filter itself).

#include "../dataset_contract.h"
#include "../plot/plot_model.h"
#include "../session_query_engine.h"

#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::dashboard {

enum class WidgetType { Kpi, Table, Plot };

// KPI measures (the fixed set of addendum A.9).
enum class Measure { Count, Sum, Mean, Median, Min, Max, Distinct, MissingPct };
const char* MeasureId(Measure m);       // "count", ...
const char* MeasureLabel(Measure m);    // "Rows", "Sum", "Mean", ...
std::optional<Measure> MeasureFromId(const std::string& id);

struct Placement {
    int x = 0, y = 0, w = 4, h = 3;      // grid cells (12 columns)
};

struct WidgetSpec {
    std::string id;                      // stable ("w3")
    WidgetType type = WidgetType::Plot;
    std::string title;                   // empty: from the content
    Placement at;
    bool automatic = false;              // made by the automatic layout (Regenerate replaces these only)
    // Plot widgets.
    plot::PlotSpec plot;
    // KPI widgets: a measure of a field (Count needs no field).
    Measure measure = Measure::Count;
    std::string field;
    // Table widgets: columns (empty: all) and how many rows.
    std::vector<std::string> columns;
    int rows = 20;
    // The fields this widget names (for checks, rebinding and cross-filters).
    std::vector<std::string> Fields() const;
    // Renames a field everywhere in the widget; true when it named it.
    bool RenameField(const std::string& from, const std::string& to);
};

// One condition of the shared filter.
struct FilterPredicate {
    enum class Op { In, Range, IsNull, NotNull };
    std::string field;
    Op op = Op::In;
    std::vector<std::string> values;     // In: the values (as text)
    double lo = 0, hi = 0;               // Range: lo <= field <= hi
    std::string source_widget;           // the widget that set it (its own query ignores it)
    std::string Text() const;            // "album_type = single", "age 20 to 40"
};

struct FilterState {
    std::vector<FilterPredicate> predicates;
    bool Empty() const { return predicates.empty(); }
    // Sets the widget's filter on a field (replacing its previous one on that field).
    void Set(FilterPredicate p);
    void ClearWidget(const std::string& widget_id);
    void Clear() { predicates.clear(); }
    // The SQL condition for a widget (the other widgets' filters), with the
    // values as bound parameters; empty when nothing applies. Identifiers quoted.
    std::string WhereFor(const std::string& widget_id, std::vector<QueryParam>& params) const;
    std::string Text() const;            // all conditions in words, " and "-joined
};

struct DashboardSpec {
    static constexpr int kVersion = 1;
    static constexpr int kColumns = 12;
    std::string title;
    std::vector<WidgetSpec> widgets;
    FilterState filters;
    int next_id = 1;
    std::string NewId() { return "w" + std::to_string(next_id++); }
    WidgetSpec* Find(const std::string& id);
    const WidgetSpec* Find(const std::string& id) const;
    // Removes the automatic widgets (Regenerate puts new ones in).
    void RemoveAutomatic();
};

std::string DashboardToJson(const DashboardSpec& spec);
// false + problem when the text is not a dashboard this version reads.
bool DashboardFromJson(const std::string& json, DashboardSpec& spec, std::string* problem = nullptr);

// ---- Widget kinds: one registry for the palette, checks and drawing ----
enum class FieldNeed { Any, Number, Category };  // what a field slot accepts

struct WidgetKind {
    std::string id;           // "kpi", "table", or "plot.<kind id>" ("plot.histogram")
    std::string label;        // "KPI", "Table", "Histogram"
    std::string group;        // "Summary" or the plot group ("Basic", "Model results", ...)
    WidgetType type = WidgetType::Plot;
    plot::Kind plot_kind = plot::Kind::Line;
    FieldNeed x_need = FieldNeed::Any, y_need = FieldNeed::Any;
};
const std::vector<WidgetKind>& WidgetKinds();
const WidgetKind* FindWidgetKind(const std::string& id);
const WidgetKind& KindOf(const WidgetSpec& w);

// Whether a column with this role suits a field slot.
bool RoleFits(ColumnRole role, FieldNeed need);

}  // namespace cyxwiz::dashboard
