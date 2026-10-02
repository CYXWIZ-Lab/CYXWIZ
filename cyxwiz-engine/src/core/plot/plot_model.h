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
};

// Every kind, in the order the plot-type list shows them.
const std::vector<KindInfo>& Kinds();
const KindInfo& Info(Kind kind);
// nullptr when the id is unknown.
const KindInfo* FindKind(const std::string& id);
const char* GroupLabel(Group group);

// What the user chose. Saved as versioned JSON (Plot node params in P2).
struct PlotSpec {
    static constexpr int kVersion = 1;
    Kind kind = Kind::Line;
    std::string x_column;               // empty: the row number
    std::vector<std::string> y_columns; // one series each
    std::string color_column;           // empty: no split
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
};

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
    // "exact · all 2,000 values", "reduced · 4,000 of 120,000 points",
    // "sampled · 50,000 of 1,200,000 rows", "first 100,000 rows".
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
    DataLabel label;
    std::string problem;   // why nothing can be drawn ("" when fine)
};

}  // namespace cyxwiz::plot
