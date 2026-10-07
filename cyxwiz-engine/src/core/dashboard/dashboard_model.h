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

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::dashboard {

enum class WidgetType { Kpi, Table, Plot, Missing };

// KPI measures (the fixed set of addendum A.9; the last three are of a text
// column, TOFIX134 P3 text).
enum class Measure { Count, Sum, Mean, Median, Min, Max, Distinct, MissingPct, MedianWords, Vocabulary, EmptyTexts };
const char* MeasureId(Measure m);       // "count", ...
const char* MeasureLabel(Measure m);    // "Rows", "Sum", "Mean", ...
std::optional<Measure> MeasureFromId(const std::string& id);

// What a text widget shows of a text column (TOFIX134 P3, board 19). Its
// data is a query the runtime writes (words, phrases, lengths), so its plot
// names the query's columns: Length "words"; Words "word", "count"; Phrases
// "phrase", "count"; WordsByClass "class", "word", "share".
enum class TextView { None, Length, Words, Phrases, WordsByClass };
const char* TextViewId(TextView v);    // "length", "words", "phrases", "words_by_class"
const char* TextViewLabel(TextView v); // "Text length", "Top words", ...
std::optional<TextView> TextViewFromId(const std::string& id);

// A text column's words as a DuckDB list: lower case, runs of letters, digits
// and apostrophes (the Dashboard's one tokenizer). `quoted_column` is quoted.
std::string TokensSql(const std::string& quoted_column);
// The saved words (TokenColumns.words, a list column) as the same list.
std::string SavedTokensSql(const std::string& quoted_words_column);

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
    // A date X field counted per year ("year"; empty: as it is).
    std::string bucket;
    // A query widget (Add to Dashboard from Data Studio's Query tab): its plot
    // reads this SQL, where `query_table` stands for the dashboard's rows under
    // the other widgets' filters. Its columns are the query's, not the data's.
    std::string query;
    std::string query_table;
    bool IsQuery() const { return !query.empty(); }
    // A text widget: the text column, the class column (WordsByClass), and
    // whether common words count (off: they are left out).
    TextView text_view = TextView::None;
    std::string text_field;
    std::string label_field;
    bool keep_stop_words = false;
    bool IsText() const { return text_view != TextView::None; }
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
    // Contains: the text field has values[0] as a whole word or phrase.
    enum class Op { In, Range, IsNull, NotNull, Contains };
    std::string field;
    Op op = Op::In;
    std::vector<std::string> values;     // In: the values (as text)
    double lo = 0, hi = 0;               // Range: lo <= field <= hi
    std::string source_widget;           // the widget that set it (its own query ignores it)
    std::string bucket;                  // "year": the field's year (a date widget's range); "words": its word count
    std::string Text() const;            // "album_type = single", "age 20 to 40"
};

// A text column whose words were split once (text_words.h): filters on it
// read the saved words instead of splitting the text again.
struct TokenColumns {
    std::string field;   // the text column
    std::string words;   // its words (a list column)
    std::string count;   // its word count
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
    std::string WhereFor(const std::string& widget_id, std::vector<QueryParam>& params, const TokenColumns* tokens = nullptr) const;
    std::string Text() const;            // all conditions in words, " and "-joined
};

struct DashboardSpec {
    static constexpr int kVersion = 1;
    static constexpr int kColumns = 12;
    std::string title;
    std::vector<WidgetSpec> widgets;
    FilterState filters;
    // Column -> type when every widget last bound (a renamed column is then
    // offered as the rebind for a field that is gone).
    std::map<std::string, std::string> known_types;
    // The automatic layout was built (a dashboard made with a widget from
    // Data Studio still gets it on its first profile).
    bool automatic_done = false;
    int next_id = 1;
    std::string NewId() { return "w" + std::to_string(next_id++); }
    WidgetSpec* Find(const std::string& id);
    const WidgetSpec* Find(const std::string& id) const;
    // Removes the automatic widgets (Regenerate puts new ones in).
    void RemoveAutomatic();
};

std::string DashboardToJson(const DashboardSpec& spec);
// The saved layout in words, for the node's Properties ("Widgets", "Filters", "Title").
std::vector<std::pair<std::string, std::string>> DashboardSummary(const DashboardSpec& spec);
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
