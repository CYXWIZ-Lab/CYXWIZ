#include "plot_model.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>

namespace cyxwiz::plot {

const std::vector<KindInfo>& Kinds() {
    static const std::vector<KindInfo> kinds = {
        {Kind::Line, "line", "Line", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Scatter, "scatter", "Scatter", Group::Basic, kEncX | kEncY, kEncColor, true, "X values", "Y values"},
        {Kind::Bar, "bar", "Bar", Group::Basic, kEncX, kEncY | kEncColor, false, "Categories", "Values (optional: count rows)"},
        {Kind::Histogram, "histogram", "Histogram", Group::Basic, kEncX, kEncColor, false, "Values", ""},
        {Kind::Area, "area", "Area", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Step, "step", "Step", Group::Basic, kEncY, kEncX | kEncColor, true, "X (optional: row number)", "Y values"},
        {Kind::Stem, "stem", "Stem", Group::Basic, kEncY, kEncX, true, "X (optional: row number)", "Y values"},
        {Kind::Pie, "pie", "Pie", Group::Basic, kEncX, kEncY, false, "Categories", "Values (optional: count rows)"},
        {Kind::Polar, "polar", "Polar", Group::Basic, kEncX | kEncY, kEncColor, false, "Angle", "Radius"},
        {Kind::Box, "box", "Box", Group::Distribution, kEncY, kEncColor, true, "", "Values"},
        {Kind::Violin, "violin", "Violin", Group::Distribution, kEncY, kEncColor, true, "", "Values"},
        {Kind::Kde, "kde", "KDE", Group::Distribution, kEncY, kEncColor, true, "", "Values"},
        {Kind::ErrorBars, "error_bars", "Error bars", Group::Distribution, kEncX | kEncY, 0, false, "Groups", "Values (mean and spread)"},
        {Kind::Heatmap, "heatmap", "Heatmap", Group::GridDensity, kEncX | kEncY, kEncValue, false, "Columns (categories)", "Rows (categories)",
         "Cell values (optional: count rows)"},
        {Kind::Matrix, "matrix", "Matrix", Group::GridDensity, kEncY, 0, true, "", "Columns (two or more numbers)"},
        {Kind::Histogram2D, "histogram_2d", "2D histogram", Group::GridDensity, kEncX | kEncY, 0, false, "X values", "Y values"},
        {Kind::Hexbin, "hexbin", "Hexbin", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values", "Y values",
         "Colour (optional: count rows)"},
        {Kind::Contour, "contour", "Contour", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values", "Y values",
         "Z (optional: density of rows)"},
        {Kind::FilledContour, "filled_contour", "Filled contour", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values",
         "Y values", "Z (optional: density of rows)"},
        {Kind::Quiver, "quiver", "Quiver", Group::VectorFields, kEncX | kEncY | kEncVector, 0, false, "X values", "Y values"},
        {Kind::Stream, "stream", "Stream", Group::VectorFields, kEncX | kEncY | kEncVector, 0, false, "X values", "Y values"},
    };
    return kinds;
}

const KindInfo& Info(Kind kind) {
    for (const auto& k : Kinds())
        if (k.kind == kind) return k;
    return Kinds().front();
}

const KindInfo* FindKind(const std::string& id) {
    for (const auto& k : Kinds())
        if (id == k.id) return &k;
    return nullptr;
}

const char* GroupLabel(Group group) {
    switch (group) {
        case Group::Basic: return "Basic";
        case Group::Distribution: return "Distribution";
        case Group::GridDensity: return "Grid and density";
        case Group::VectorFields: return "Vector fields";
    }
    return "";
}

const std::vector<std::string>& ConditionOps() {
    static const std::vector<std::string> ops = {"=", "!=", "<", "<=", ">", ">=", "contains"};
    return ops;
}

std::string ConditionsText(const std::vector<RowCondition>& conditions) {
    std::string out;
    for (const auto& c : conditions) {
        if (c.column.empty()) continue;
        if (!out.empty()) out += " and ";
        out += c.column + " " + c.op + " " + c.value;
    }
    return out;
}

namespace {
const char* RowModeId(RowMode m) {
    switch (m) {
        case RowMode::All: return "all";
        case RowMode::First: return "first";
        case RowMode::Range: return "range";
        case RowMode::Filter: return "filter";
    }
    return "all";
}
const char* ColourModeId(ColourMode m) {
    switch (m) {
        case ColourMode::Auto: return "auto";
        case ColourMode::Groups: return "groups";
        case ColourMode::Scale: return "scale";
    }
    return "auto";
}
}  // namespace

std::string SpecToJson(const PlotSpec& s) {
    nlohmann::json j;
    j["version"] = PlotSpec::kVersion;
    j["kind"] = Info(s.kind).id;
    j["x"] = s.x_column;
    j["y"] = s.y_columns;
    j["color"] = s.color_column;
    j["value"] = s.value_column;
    j["diagonal"] = s.show_diagonal;
    j["bar_layout"] = s.bar_layout == PlotSpec::BarLayout::Stacked   ? "stacked"
                      : s.bar_layout == PlotSpec::BarLayout::Percent ? "percent"
                                                                     : "grouped";
    j["donut"] = s.donut;
    j["kde_bandwidth"] = s.kde_bandwidth;
    j["matrix"] = s.matrix_values == PlotSpec::MatrixValues::Spearman ? "spearman"
                  : s.matrix_values == PlotSpec::MatrixValues::Values ? "values"
                                                                      : "pearson";
    j["levels"] = s.levels;
    j["log_colour"] = s.log_colour;
    j["u"] = s.u_column;
    j["v"] = s.v_column;
    j["vector_from"] = s.vector_from == PlotSpec::VectorFrom::DirectionLength ? "direction" : "uv";
    j["wind_from"] = s.wind_from;
    j["arrow_every"] = s.arrow_every;
    j["stream_density"] = s.stream_density;
    j["angle_unit"] = s.angle_unit == PlotSpec::AngleUnit::Degrees      ? "degrees"
                      : s.angle_unit == PlotSpec::AngleUnit::Radians    ? "radians"
                      : s.angle_unit == PlotSpec::AngleUnit::Categories ? "categories"
                                                                        : "auto";
    j["polar_points"] = s.polar_points;
    j["title"] = s.title;
    j["x_label"] = s.x_label;
    j["y_label"] = s.y_label;
    j["bins"] = s.bins;
    j["smooth"] = s.smooth;
    j["density"] = s.density;
    j["show_mean"] = s.show_mean;
    j["show_median"] = s.show_median;
    j["log_x"] = s.log_x;
    j["log_y"] = s.log_y;
    j["legend"] = s.legend;
    nlohmann::json rows;
    rows["mode"] = RowModeId(s.rows);
    rows["first"] = s.first_rows;
    rows["from"] = s.row_from;
    rows["to"] = s.row_to;
    rows["conditions"] = nlohmann::json::array();
    for (const auto& c : s.conditions) rows["conditions"].push_back({{"column", c.column}, {"op", c.op}, {"value", c.value}});
    j["rows"] = rows;
    j["color_mode"] = ColourModeId(s.color_mode);
    return j.dump();
}

bool SpecFromJson(const std::string& text, PlotSpec& s, std::string* problem) {
    const auto j = nlohmann::json::parse(text, nullptr, false);
    const auto fail = [&](const std::string& why) {
        if (problem) *problem = why;
        return false;
    };
    if (j.is_discarded() || !j.is_object()) return fail("not a plot spec (JSON object expected)");
    const int version = j.value("version", 0);
    if (version < 1 || version > PlotSpec::kVersion)
        return fail("plot spec version " + std::to_string(version) + " is not supported");
    const KindInfo* kind = FindKind(j.value("kind", std::string()));
    if (!kind) return fail("unknown plot kind '" + j.value("kind", std::string()) + "'");
    PlotSpec out;
    out.kind = kind->kind;
    out.x_column = j.value("x", std::string());
    if (j.contains("y") && j["y"].is_array())
        for (const auto& y : j["y"])
            if (y.is_string()) out.y_columns.push_back(y.get<std::string>());
    out.color_column = j.value("color", std::string());
    out.value_column = j.value("value", std::string());
    out.show_diagonal = j.value("diagonal", false);
    const std::string layout = j.value("bar_layout", std::string("grouped"));
    if (layout == "grouped") out.bar_layout = PlotSpec::BarLayout::Grouped;
    else if (layout == "stacked") out.bar_layout = PlotSpec::BarLayout::Stacked;
    else if (layout == "percent") out.bar_layout = PlotSpec::BarLayout::Percent;
    else return fail("unknown bar layout '" + layout + "'");
    out.donut = j.value("donut", false);
    out.kde_bandwidth = std::clamp(j.value("kde_bandwidth", 1.0), 0.05, 20.0);
    const std::string matrix = j.value("matrix", std::string("pearson"));
    if (matrix == "pearson") out.matrix_values = PlotSpec::MatrixValues::Pearson;
    else if (matrix == "spearman") out.matrix_values = PlotSpec::MatrixValues::Spearman;
    else if (matrix == "values") out.matrix_values = PlotSpec::MatrixValues::Values;
    else return fail("unknown matrix values '" + matrix + "'");
    out.levels = std::clamp(j.value("levels", 7), 1, 50);
    out.log_colour = j.value("log_colour", false);
    out.u_column = j.value("u", std::string());
    out.v_column = j.value("v", std::string());
    const std::string from = j.value("vector_from", std::string("uv"));
    if (from == "uv") out.vector_from = PlotSpec::VectorFrom::UV;
    else if (from == "direction") out.vector_from = PlotSpec::VectorFrom::DirectionLength;
    else return fail("unknown vector columns '" + from + "'");
    out.wind_from = j.value("wind_from", false);
    out.arrow_every = std::clamp(j.value("arrow_every", 0), 0, 1000);
    out.stream_density = std::clamp(j.value("stream_density", 1.0), 0.2, 5.0);
    const std::string unit = j.value("angle_unit", std::string("auto"));
    if (unit == "auto") out.angle_unit = PlotSpec::AngleUnit::Auto;
    else if (unit == "degrees") out.angle_unit = PlotSpec::AngleUnit::Degrees;
    else if (unit == "radians") out.angle_unit = PlotSpec::AngleUnit::Radians;
    else if (unit == "categories") out.angle_unit = PlotSpec::AngleUnit::Categories;
    else return fail("unknown angle unit '" + unit + "'");
    out.polar_points = j.value("polar_points", false);
    out.title = j.value("title", std::string());
    out.x_label = j.value("x_label", std::string());
    out.y_label = j.value("y_label", std::string());
    out.bins = std::clamp(j.value("bins", 30), 1, 1000);
    out.smooth = std::clamp(j.value("smooth", 0), 0, 100000);
    out.density = j.value("density", false);
    out.show_mean = j.value("show_mean", false);
    out.show_median = j.value("show_median", false);
    out.log_x = j.value("log_x", false);
    out.log_y = j.value("log_y", false);
    out.legend = j.value("legend", true);
    if (j.contains("rows") && j["rows"].is_object()) {
        const auto& r = j["rows"];
        const std::string mode = r.value("mode", std::string("all"));
        if (mode == "all") out.rows = RowMode::All;
        else if (mode == "first") out.rows = RowMode::First;
        else if (mode == "range") out.rows = RowMode::Range;
        else if (mode == "filter") out.rows = RowMode::Filter;
        else return fail("unknown row selection '" + mode + "'");
        out.first_rows = std::max<size_t>(1, r.value("first", size_t{1000}));
        out.row_from = std::max<size_t>(1, r.value("from", size_t{1}));
        out.row_to = std::max(out.row_from, r.value("to", size_t{1000}));
        if (r.contains("conditions") && r["conditions"].is_array()) {
            for (const auto& c : r["conditions"]) {
                if (!c.is_object()) continue;
                RowCondition rc{c.value("column", std::string()), c.value("op", std::string("=")), c.value("value", std::string())};
                const auto& ops = ConditionOps();
                if (std::find(ops.begin(), ops.end(), rc.op) == ops.end())
                    return fail("unknown filter condition '" + rc.op + "'");
                out.conditions.push_back(std::move(rc));
            }
        }
    }
    const std::string colour_mode = j.value("color_mode", std::string("auto"));
    if (colour_mode == "auto") out.color_mode = ColourMode::Auto;
    else if (colour_mode == "groups") out.color_mode = ColourMode::Groups;
    else if (colour_mode == "scale") out.color_mode = ColourMode::Scale;
    else return fail("unknown colour mode '" + colour_mode + "'");
    s = std::move(out);
    return true;
}

std::string MissingEncoding(const PlotSpec& s) {
    const KindInfo& k = Info(s.kind);
    if ((k.required & kEncX) && s.x_column.empty()) return std::string("Choose ") + k.x_hint + ".";
    if ((k.required & kEncY) && s.y_columns.empty()) return std::string("Choose ") + k.y_hint + ".";
    if ((k.required & kEncVector) && (s.u_column.empty() || s.v_column.empty()))
        return s.vector_from == PlotSpec::VectorFrom::UV ? "Choose the arrow columns (u and v)."
                                                         : "Choose the direction and length columns.";
    return "";
}

std::string Thousands(long long n) {
    std::string digits = std::to_string(n < 0 ? -n : n);
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return n < 0 ? "-" + out : out;
}

std::string DataLabel::Text() const {
    if (!selection.empty()) {
        if (state == State::Exact) return selection;
        DataLabel base = *this;
        base.selection.clear();
        return selection + " \xC2\xB7 " + base.Text();
    }
    const std::string n = Thousands(static_cast<long long>(shown));
    const std::string all = Thousands(static_cast<long long>(total));
    switch (state) {
        case State::Exact: return "exact \xC2\xB7 all " + n + " values";
        case State::Reduced: return "reduced \xC2\xB7 " + n + " of " + all + " points";
        case State::Sampled: return "sampled \xC2\xB7 " + n + " of " + all + " rows";
        case State::Truncated:
            return total > shown ? "first " + n + " of " + all + " rows" : "first " + n + " rows";
    }
    return "";
}

namespace {
double Quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
}
}  // namespace

ColumnStats Summarize(const std::vector<double>& values) {
    ColumnStats st;
    std::vector<double> finite;
    finite.reserve(values.size());
    double sum = 0.0;
    for (double v : values) {
        if (std::isfinite(v)) {
            finite.push_back(v);
            sum += v;
        } else {
            ++st.missing;
        }
    }
    st.count = finite.size();
    if (finite.empty()) return st;
    std::sort(finite.begin(), finite.end());
    st.min = finite.front();
    st.max = finite.back();
    st.mean = sum / static_cast<double>(finite.size());
    st.median = Quantile(finite, 0.5);
    st.q1 = Quantile(finite, 0.25);
    st.q3 = Quantile(finite, 0.75);
    return st;
}

}  // namespace cyxwiz::plot
