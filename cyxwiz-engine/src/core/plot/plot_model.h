#pragma once

// The plot model shared by every screen that draws a plot (TOFIX134 P1 step
// 1.2): what to draw (PlotSpec), which plot kinds exist (the kind registry)
// and what the drawn data is (Prepared, with a label that says whether all
// values are shown). Pure: no ImGui, no ImPlot; the renderer is gui/plot.

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
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
    // P2b group 1 (approved board 8).
    Kde,
    Matrix,
    Hexbin,
    Contour,
    FilledContour,
    // P2b group 2 (approved board 9).
    Polar,
    Quiver,
    Stream,
    // P2b group 3 (approved board 10).
    Image,
    PairPlot,
    Parallel,
    // P2b group 4 (approved board 11): model results from plain columns.
    Confusion,
    Roc,
    PrCurve,
    Calibration,
    Residuals,
    LearningCurve,
    Importance,
    // P2b group 5 (approved board 12): flows, hierarchies and maps.
    Sankey,
    Treemap,
    MapPoints,
    MapRegions,
};

enum class Group { Basic, Distribution, GridDensity, VectorFields, Images, ModelResults, FlowsHierarchies, Maps };

// Which columns a kind uses. X and Y are column names; Color splits the
// rows into one series per value of a column.
enum Encoding : unsigned {
    kEncX = 1u << 0,
    kEncY = 1u << 1,
    kEncColor = 1u << 2,
    kEncValue = 1u << 3,  // heatmap: a column summed per cell (else rows are counted)
    kEncVector = 1u << 4, // quiver, stream: the arrow columns (u, v or direction, length)
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
    // P2b group 1 (approved board 8).
    enum class BarLayout { Grouped, Stacked, Percent };
    BarLayout bar_layout = BarLayout::Grouped;  // bar with Colour by
    bool donut = false;                          // pie: a hole with the total
    double kde_bandwidth = 1.0;                  // KDE: factor on Silverman's bandwidth
    enum class MatrixValues { Pearson, Spearman, Values };
    MatrixValues matrix_values = MatrixValues::Pearson;
    int levels = 7;                              // contour levels
    bool log_colour = false;                     // hexbin: colour by the log of the count
    // P2b group 2 (approved board 9).
    std::string u_column, v_column;              // quiver, stream: the arrow columns
    enum class VectorFrom { UV, DirectionLength };
    VectorFrom vector_from = VectorFrom::UV;     // u, v; or degrees clockwise from north and a length
    bool wind_from = false;                      // the direction says where it comes from (wind)
    int arrow_every = 0;                         // quiver: show every Nth arrow (0: as many as fit)
    double stream_density = 1.0;                 // stream: lines per area
    enum class AngleUnit { Auto, Degrees, Radians, Categories };
    AngleUnit angle_unit = AngleUnit::Auto;      // polar: Auto = categories for text, degrees for numbers
    bool polar_points = false;                   // polar: points instead of lines
    // P2b group 3 (approved board 10). Image: the Y columns are the pixels
    // (any table: square when the count is a square, else image_width
    // wide; 3 channels when asked or when count / 3 is a square).
    enum class ImageMode { OneRow, Gallery, MeanPerClass };
    ImageMode image_mode = ImageMode::Gallery;
    int image_row = 1;                           // one row: 1-based, among the chosen rows
    int image_width = 0;                         // 0: square
    int image_channels = 0;                      // 0: auto (1, or 3 when count / 3 is a square)
    bool image_planar = false;                   // 3 channels as three planes (R..., G..., B...) like CIFAR
    enum class ImageRange { Auto, Byte, Unit };
    ImageRange image_range = ImageRange::Auto;   // the values' min..max, 0..255 or 0..1
    bool image_grey = false;                     // grey instead of the theme scale (1 channel)
    bool image_invert = false;
    int gallery_max = 40;                        // gallery: pictures at most
    bool pair_histogram = false;                 // pair plot diagonal: histogram instead of KDE
    // Histogram: a fixed x range (NaN: the data's min..max). Values outside
    // are not counted. A dashboard uses it so filtered bins match all rows.
    double range_lo = NAN, range_hi = NAN;
    // P2b group 4 (approved board 11). Model results read x_column as the
    // actual (or x / feature) and y_columns as the predicted / score /
    // probability / curves / importance.
    enum class ConfusionShow { Counts, ByActual, ByPredicted, All };
    ConfusionShow confusion_show = ConfusionShow::ByActual;  // the colour and the second number
    std::string positive_class;                  // ROC, PR, calibration: "" = auto
    int calibration_bins = 10;
    std::vector<std::string> spread_columns;     // learning curve (one per curve), importance (one)
    enum class Best { Auto, Highest, Lowest };
    Best best = Best::Auto;                      // learning curve: Auto = lowest for a loss / error
    int top_n = 20;                              // feature importance
    // P2b group 5 (approved board 12). Sankey: y_columns are the steps (left
    // to right), value_column the summed value (empty: rows). Treemap:
    // y_columns are the groups (outer to inner), value_column the size,
    // color_column a number for the colour scale (empty: by the top group).
    // Map points: x = longitude, y = latitude, value = size, colour.
    // Map regions: x = country (name or ISO code), y = the value.
    int sankey_top = 8;                          // per step: the largest categories, the rest as "other"
    enum class RegionAgg { Sum, Mean };
    RegionAgg region_agg = RegionAgg::Sum;       // several rows for one country
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
    // Map points: the size value per point (NaN when missing), with all_z.
    std::vector<double> z, all_z;
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
    // The colour scale of a grid (heatmap, matrix, 2D histogram, contours,
    // hexbin): grid_lo..grid_hi, two-sided around 0 when grid_diverging.
    double grid_lo = 0, grid_hi = 0;
    bool grid_diverging = false;
    // Contour / filled contour: the level values, and per level its line
    // segments as x0, y0, x1, y1 (data units). Filled contour also keeps
    // band_grid: the grid upsampled (band_rows x band_cols) and set to the
    // middle of its band, drawn like a heatmap over x_min..x_max, y_min..y_max.
    std::vector<double> contour_levels;
    std::vector<std::vector<double>> contour_segments;
    std::vector<double> band_grid;
    int band_rows = 0, band_cols = 0;
    // Hexbin: hexagon centres and values (count or mean), and the lattice
    // steps (matplotlib's layout: vertices (+-sx/2, +-sy/6), (0, +-sy/3)).
    std::vector<double> hex_x, hex_y, hex_v;
    double hex_sx = 0, hex_sy = 0;
    // Polar: Series x = angle in radians (0 at the top, clockwise), y =
    // radius; the angle names of categories, the largest radius, and
    // whether each line closes (categories go round the whole turn).
    std::vector<std::string> polar_names;
    double polar_rmax = 0;
    bool polar_closed = false;
    // Quiver: every row's arrow (u, v in data units), the rows drawn (a
    // dense grid shows every Nth), how much to stretch u, v so arrows fit
    // their spacing, and the step.
    std::vector<double> qx, qy, qu, qv;
    std::vector<size_t> q_drawn;
    double q_scale = 1.0;
    int q_every = 1;
    // Stream: the lines (x0, y0, x1, y1, ... per line) with the speed at
    // each line's middle, and the field on a grid (grid_rows x grid_cols,
    // row 0 at the top) for hover.
    std::vector<std::vector<double>> stream_lines;
    std::vector<double> stream_speed;
    std::vector<double> field_u, field_v;
    // Image: each picture's values scaled to 0..1 (img_w x img_h x
    // img_channels, row by row, channels last), its label, its row in the
    // chosen rows (1-based; 0 for a mean) and how many rows it averages.
    struct Picture {
        std::vector<float> pix;
        std::string label;
        size_t row = 0;
        size_t count = 1;
    };
    std::vector<Picture> pictures;
    int img_w = 0, img_h = 0, img_channels = 1;
    double img_lo = 0, img_hi = 1;
    size_t img_rows = 0;                         // the chosen rows (one row: Previous / Next stop here)
    // Pair plot and parallel coordinates: the columns, each column's range,
    // the sampled rows' values per column and group, and the group names.
    std::vector<std::string> multi_cols;
    std::vector<double> multi_lo, multi_hi;
    std::vector<std::vector<double>> multi_values;  // per column, per sampled row
    std::vector<int> multi_group;                   // per sampled row
    std::vector<std::string> multi_groups;
    // Pair plot diagonal: per column, per group, the density (KDE or
    // histogram) over pair_steps points across the column's range.
    std::vector<std::vector<std::vector<double>>> pair_diag;
    int pair_steps = 0;
    // Model results: the figures shown with the plot ("AUC" 0.871, "RMSE"
    // 20.8, ...), the confusion counts behind shares (grid layout), the
    // positive class used, the PR baseline, and a learning curve's best point.
    std::vector<std::pair<std::string, double>> metrics;
    std::vector<double> grid_counts;
    std::string positive_label;
    double baseline = NAN;
    int best_series = -1;
    size_t best_index = 0;
    // Sankey: the step columns, the nodes (per step, largest first, "other"
    // last) and the bands between neighbouring steps, laid out on 0..1 with
    // y from the top; the total value of the rows drawn.
    struct SankeyNode {
        int step = 0;
        std::string name;
        double value = 0, y0 = 0, y1 = 0;
    };
    struct SankeyLink {
        int from = 0, to = 0;  // node indices
        double value = 0, y_from = 0, y_to = 0, thickness = 0;
    };
    std::vector<std::string> sankey_steps;
    std::vector<SankeyNode> sankey_nodes;
    std::vector<SankeyLink> sankey_links;
    double sankey_total = 0;
    // Treemap: the leaves (their path, outer group first), each leaf's size
    // and colour value (NaN without a colour column), the group columns and
    // the top-level names (a leaf's `top` indexes them; colour without a
    // colour column). Laid out by TreemapLayout (plot_layout.h).
    struct TreeLeaf {
        std::vector<std::string> path;
        double size = 0;
        double colour = NAN;
        int top = 0;
    };
    std::vector<TreeLeaf> tree_leaves;
    std::vector<std::string> tree_levels, tree_tops;
    // Map points: the size scale (Series::z) over size_min..size_max.
    double size_min = 0, size_max = 0;
    std::string size_label;
    // Map regions: the value per country (WorldCountries() order; NaN when no
    // row names it) and its rows, coloured over grid_lo..grid_hi (log when
    // spec.log_colour); the region texts that matched no country, with rows.
    std::vector<double> region_value;
    std::vector<size_t> region_rows;
    std::vector<std::pair<std::string, size_t>> unmatched;
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
