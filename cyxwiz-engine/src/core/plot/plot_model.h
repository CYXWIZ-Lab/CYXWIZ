#pragma once

// The plot model shared by every screen that draws a plot (TOFIX134 P1 step
// 1.2): what to draw (PlotSpec), which plot kinds exist (the kind registry)
// and what the drawn data is (Prepared, with a label that says whether all
// values are shown). Pure: no ImGui, no ImPlot; the renderer is gui/plot.

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz::plot {

enum class Kind {
    Line,
    Scatter,
    Bar,
    Histogram,
    Area,
    Step,
    Stem,
    Pie,
    Box,
    Violin,
    ErrorBars,
    Heatmap,
    Histogram2D,
};

enum class Group { Basic, Distribution, GridDensity };

// Which columns a kind uses. X and Y are column names; Color splits the
// rows into one series per value of a column.
enum Encoding : unsigned {
    kEncX = 1u << 0,
    kEncY = 1u << 1,
    kEncColor = 1u << 2,
    kEncValue = 1u << 3,  // heatmap: a column summed per cell (else rows are counted)
};

struct KindInfo {
    Kind kind;
    const char* id;     // saved in specs ("histogram")
    const char* label;  // shown ("Histogram")
    Group group;
    unsigned required;  // Encoding bits that must be set
    unsigned optional;  // Encoding bits that may be set
    bool multi_y;       // more than one Y column (one series each)
    const char* x_hint; // what the X picker asks for
    const char* y_hint; // what the Y picker asks for ("" when unused)
    const char* value_hint = "";  // what the value picker asks for (kEncValue)
};

// Every kind, in the order the plot-type list shows them.
const std::vector<KindInfo>& Kinds();
const KindInfo& Info(Kind kind);
// nullptr when the id is unknown.
const KindInfo* FindKind(const std::string& id);
const char* GroupLabel(Group group);

// Which rows are plotted (TOFIX134 P2, board 6).
enum class RowMode { All, First, Range, Filter };

// One filter condition: `column op value`. Numbers compare as numbers when
// the column is numeric and the value is a number; otherwise as text.
struct RowCondition {
    std::string column;
    std::string op = "=";
    std::string value;
};

// The condition operators, in the order the window lists them.
const std::vector<std::string>& ConditionOps();

// How Colour by uses a number column: Auto makes a scale for a scatter when
// the column has more than 12 values, groups otherwise.
enum class ColourMode { Auto, Groups, Scale };

// What the user chose. Saved as versioned JSON (Plot node params in P2).
struct PlotSpec {
    static constexpr int kVersion = 1;
    Kind kind = Kind::Line;
    std::string x_column;               // empty: the row number
    std::vector<std::string> y_columns; // one series each
    std::string color_column;           // empty: no split
    std::string value_column;           // heatmap cells: sum of this column (empty: count rows)
    std::string title;
    std::string x_label;                // empty: the column name
    std::string y_label;
    int bins = 30;                      // histogram, 2D histogram
    int smooth = 0;                     // moving-average window, 0 = off
    bool density = false;               // histogram: density, not count
    bool show_mean = false;
    bool show_median = false;
    bool log_x = false;
    bool log_y = false;
    bool legend = true;
    bool show_diagonal = false;         // line, scatter: a y = x reference line (ROC chance)
    RowMode rows = RowMode::All;
    size_t first_rows = 1000;              // RowMode::First
    size_t row_from = 1, row_to = 1000;    // RowMode::Range: 1-based, inclusive
    std::vector<RowCondition> conditions;  // RowMode::Filter: all must match
    ColourMode color_mode = ColourMode::Auto;
};

// "class = 7 and pixel407 > 0" (the filter in words).
std::string ConditionsText(const std::vector<RowCondition>& conditions);

std::string SpecToJson(const PlotSpec& spec);
// false (and `problem` set) when the text is not a plot spec this version
// reads; unknown keys are ignored, missing keys keep their defaults.
bool SpecFromJson(const std::string& json, PlotSpec& spec, std::string* problem = nullptr);
// What is missing for this kind ("" when the spec can be drawn).
std::string MissingEncoding(const PlotSpec& spec);

// Whether the drawn data is all of it (TOFIX134: nothing is cut silently).
struct DataLabel {
    enum class State {
        Exact,      // all values drawn
        Reduced,    // long series drawn with the min and max of each step
        Sampled,    // an even sample of the rows
        Truncated,  // the source was read with a row limit
    };
    State state = State::Exact;
    size_t shown = 0;  // values, points or rows drawn
    size_t total = 0;  // all of them; 0 when unknown (truncated)
    // The rows chosen in the spec, in words ("filtered · 7,293 of 70,000
    // rows", "first 1,000 of 70,000 rows"); empty for all rows.
    std::string selection;
    // "exact · all 2,000 values", "reduced · 4,000 of 120,000 points",
    // "sampled · 50,000 of 1,200,000 rows", "first 100,000 rows"; the
    // selection comes first and an exact label then adds nothing.
    std::string Text() const;
};

std::string Thousands(long long n);

// Summary of one numeric column (shown beside a plot).
struct ColumnStats {
    size_t count = 0;    // finite values
    size_t missing = 0;  // NaN or infinite
    double min = 0, max = 0, mean = 0, median = 0, q1 = 0, q3 = 0;
};
ColumnStats Summarize(const std::vector<double>& values);

// One drawn series.
struct Series {
    std::string label;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> low;   // error bars: y - low; violin: left edge
    std::vector<double> high;  // error bars: y + high; violin: right edge
    std::vector<double> smooth_x, smooth_y;  // moving average, when asked
    // All values when x/y were reduced or sampled for drawing: hover values
    // and exports use these (empty when x/y are already all of them).
    std::vector<double> all_x, all_y;
    // Colour by a number column as a scale: the value per point (NaN when
    // missing), with all_c beside all_x / all_y.
    std::vector<double> c, all_c;
    bool x_sorted = false;  // x ascends (hover finds the nearest x by search)
    int colour = -1;        // theme series colour index; -1: by position
    bool markers = false;   // lines: also a marker at every point
};

// Data ready to draw: built off the UI thread from a source, then only drawn.
struct Prepared {
    PlotSpec spec;
    std::vector<Series> series;
    // Histogram: bin edges (bins + 1) and counts per series (in Series::y,
    // with Series::x the bin centres). Pie and Bar: category names.
    std::vector<double> edges;
    std::vector<std::string> categories;
    // Heatmap / 2D histogram: row-major values, rows x cols, with the
    // axis ranges or category names.
    std::vector<double> grid;
    int grid_rows = 0, grid_cols = 0;
    double x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    std::vector<std::string> row_names, col_names;
    // Box: per series q1, median, q3, whisker low/high, mean.
    struct Box { double low, q1, median, q3, high, mean; };
    std::vector<Box> boxes;
    ColumnStats stats;     // of the X (histogram) or first Y column
    // Colour by a number column as a scale (scatter): the theme's sequential
    // scale over colour_min..colour_max, or the two-sided scale (symmetric
    // around 0) when the values lie on both sides of 0.
    bool colour_scale = false;
    bool colour_diverging = false;
    double colour_min = 0, colour_max = 0;
    std::string colour_label;
    // Rows chosen by the spec's row selection, of all rows of the source.
    size_t rows_selected = 0, rows_total = 0;
    DataLabel label;
    std::string problem;   // why nothing can be drawn ("" when fine)
};

}  // namespace cyxwiz::plot
