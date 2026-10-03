#include "plot_model.h"

#include "plot_scales.h"

#include <cstdio>

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
        {Kind::Parallel, "parallel", "Parallel coordinates", Group::Distribution, kEncY, kEncColor, true, "", "Columns (two or more numbers)"},
        {Kind::ErrorBars, "error_bars", "Error bars", Group::Distribution, kEncX | kEncY, 0, false, "Groups", "Values (mean and spread)"},
        {Kind::Heatmap, "heatmap", "Heatmap", Group::GridDensity, kEncX | kEncY, kEncValue, false, "Columns (categories)", "Rows (categories)",
         "Cell values (optional: count rows)"},
        {Kind::Matrix, "matrix", "Matrix", Group::GridDensity, kEncY, 0, true, "", "Columns (two or more numbers)"},
        {Kind::Histogram2D, "histogram_2d", "2D histogram", Group::GridDensity, kEncX | kEncY, 0, false, "X values", "Y values"},
        {Kind::PairPlot, "pair_plot", "Pair plot", Group::GridDensity, kEncY, kEncColor, true, "", "Columns (2 to 6 numbers)"},
        {Kind::Hexbin, "hexbin", "Hexbin", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values", "Y values",
         "Colour (optional: count rows)"},
        {Kind::Contour, "contour", "Contour", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values", "Y values",
         "Z (optional: density of rows)"},
        {Kind::FilledContour, "filled_contour", "Filled contour", Group::GridDensity, kEncX | kEncY, kEncValue, false, "X values",
         "Y values", "Z (optional: density of rows)"},
        {Kind::Quiver, "quiver", "Quiver", Group::VectorFields, kEncX | kEncY | kEncVector, 0, false, "X values", "Y values"},
        {Kind::Stream, "stream", "Stream", Group::VectorFields, kEncX | kEncY | kEncVector, 0, false, "X values", "Y values"},
        {Kind::Image, "image", "Image", Group::Images, kEncY, kEncColor, true, "", "Pixel columns"},
        {Kind::Confusion, "confusion_matrix", "Confusion matrix", Group::ModelResults, kEncX | kEncY, 0, false, "Actual", "Predicted"},
        {Kind::Roc, "roc_curve", "ROC curve", Group::ModelResults, kEncX | kEncY, 0, false, "Actual", "Score (of the positive class)"},
        {Kind::PrCurve, "pr_curve", "Precision-recall curve", Group::ModelResults, kEncX | kEncY, 0, false, "Actual",
         "Score (of the positive class)"},
        {Kind::Calibration, "calibration", "Calibration", Group::ModelResults, kEncX | kEncY, 0, false, "Actual",
         "Probability (of the positive class)"},
        {Kind::Residuals, "residuals", "Residuals", Group::ModelResults, kEncX | kEncY, 0, false, "Actual", "Predicted"},
        {Kind::LearningCurve, "learning_curve", "Learning curve", Group::ModelResults, kEncX | kEncY, 0, true,
         "X (training rows or epoch)", "Curves (train, validation)"},
        {Kind::Importance, "feature_importance", "Feature importance", Group::ModelResults, kEncX | kEncY, 0, false, "Feature",
         "Importance"},
        {Kind::Sankey, "sankey", "Sankey", Group::FlowsHierarchies, kEncY, kEncValue, true, "", "Steps (two or more, left to right)",
         "Value (summed; empty: count rows)"},
        {Kind::Treemap, "treemap", "Treemap", Group::FlowsHierarchies, kEncY, kEncValue | kEncColor, true, "",
         "Groups (outer to inner)", "Size (summed; empty: count rows)"},
        {Kind::MapPoints, "map_points", "Map: points", Group::Maps, kEncX | kEncY, kEncValue | kEncColor, false, "Longitude",
         "Latitude", "Size"},
        {Kind::MapRegions, "map_regions", "Map: regions", Group::Maps, kEncX | kEncY, 0, false, "Country (name or ISO code)",
         "Value"},
        {Kind::Scatter3D, "scatter3d", "Scatter 3D", Group::ThreeD, kEncX | kEncY | kEncZ, kEncColor | kEncValue, false, "X", "Y",
         "Size"},
        {Kind::Line3D, "line3d", "Line 3D", Group::ThreeD, kEncX | kEncY | kEncZ, kEncColor, false, "X", "Y"},
        // Surface: X, Y, Z columns or grid columns (Y); MissingEncoding checks by the source.
        {Kind::Surface, "surface", "Surface", Group::ThreeD, 0, kEncX | kEncY | kEncZ, true, "X", "Y"},
        {Kind::Mesh, "mesh", "Mesh", Group::ThreeD, kEncX | kEncY | kEncZ, 0, false, "X", "Y"},
        {Kind::Network, "network", "Network", Group::Graphs, kEncX | kEncY, kEncValue | kEncColor, false, "Source", "Target",
         "Weight (optional)"},
        {Kind::Tree, "tree", "Tree", Group::Graphs, kEncX | kEncY, kEncValue | kEncColor, false, "Node", "Parent (empty: a root)",
         "Value (optional)"},
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
        case Group::Images: return "Images";
        case Group::ModelResults: return "Model results";
        case Group::FlowsHierarchies: return "Flows and hierarchies";
        case Group::Maps: return "Maps";
        case Group::ThreeD: return "3D";
        case Group::Graphs: return "Graphs";
    }
    return "";
}

const std::vector<std::string>& ConditionOps() {
    static const std::vector<std::string> ops = {"=", "!=", "<", "<=", ">", ">=", "contains"};
    return ops;
}

std::vector<std::pair<std::string, std::string>> SpecSummary(const PlotSpec& s) {
    std::vector<std::pair<std::string, std::string>> rows;
    const KindInfo& k = Info(s.kind);
    rows.push_back({"Type", std::string(k.label) + " (" + GroupLabel(k.group) + ")"});
    // Columns: up to four by name, a long list as first .. last (count).
    const auto list = [](const std::vector<std::string>& cols) {
        if (cols.size() > 4) return cols.front() + " .. " + cols.back() + " (" + std::to_string(cols.size()) + ")";
        std::string out;
        for (const auto& c : cols) out += (out.empty() ? "" : ", ") + c;
        return out;
    };
    std::string data;
    const auto add = [&](const std::string& what, const std::string& col) {
        if (!col.empty()) data += (data.empty() ? "" : " \xC2\xB7 ") + what + " " + col;
    };
    if (s.kind == Kind::Surface && s.surface_from == PlotSpec::SurfaceFrom::Grid) {
        add("grid columns", list(s.y_columns));
    } else {
        add("X", s.x_column);
        add(k.multi_y && s.y_columns.size() > 1 ? "Y" : "Y", list(s.y_columns));
        add("Z", s.z_column);
        add("colour by", s.color_column);
        add(k.value_hint[0] ? "value" : "value", s.value_column);
    }
    rows.push_back({"Data", data.empty() ? std::string("no columns chosen yet") : data});
    switch (s.rows) {
        case RowMode::All: break;
        case RowMode::First: rows.push_back({"Rows", "first " + Thousands(static_cast<long long>(s.first_rows))}); break;
        case RowMode::Range:
            rows.push_back({"Rows", Thousands(static_cast<long long>(s.row_from)) + " to " + Thousands(static_cast<long long>(s.row_to))});
            break;
        case RowMode::Filter: rows.push_back({"Rows", ConditionsText(s.conditions)}); break;
    }
    std::string colour;
    if (const ScaleInfo* scale = FindScale(s.colour_scale)) colour = scale->label;
    if (s.colour_reverse) colour += colour.empty() ? "theme scale, reversed" : ", reversed";
    if (std::isfinite(s.scale_lo) && std::isfinite(s.scale_hi)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%g to %g", s.scale_lo, s.scale_hi);
        colour += (colour.empty() ? "theme scale, " : ", ") + std::string(buf);
    }
    size_t picked = 0;
    for (const auto& c : s.series_colours) picked += c.empty() ? 0 : 1;
    if (picked > 0) colour += (colour.empty() ? "" : "; ") + std::to_string(picked) + (picked == 1 ? " series colour" : " series colours");
    if (!colour.empty()) rows.push_back({"Colour", colour});
    if (std::isfinite(s.view_elevation) && std::isfinite(s.view_azimuth)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "turned (elevation %.0f\xC2\xB0, azimuth %.0f\xC2\xB0)", s.view_elevation, s.view_azimuth);
        rows.push_back({"View", buf});
    }
    if (!s.title.empty()) rows.push_back({"Title", s.title});
    return rows;
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
    j["image_mode"] = s.image_mode == PlotSpec::ImageMode::OneRow      ? "one_row"
                      : s.image_mode == PlotSpec::ImageMode::MeanPerClass ? "mean_per_class"
                                                                          : "gallery";
    j["image_row"] = s.image_row;
    j["image_width"] = s.image_width;
    j["image_channels"] = s.image_channels;
    j["image_planar"] = s.image_planar;
    j["image_range"] = s.image_range == PlotSpec::ImageRange::Byte ? "0-255" : s.image_range == PlotSpec::ImageRange::Unit ? "0-1" : "auto";
    j["image_grey"] = s.image_grey;
    j["image_invert"] = s.image_invert;
    j["gallery_max"] = s.gallery_max;
    j["pair_histogram"] = s.pair_histogram;
    if (std::isfinite(s.range_lo) && std::isfinite(s.range_hi)) j["range"] = {s.range_lo, s.range_hi};
    j["confusion_show"] = s.confusion_show == PlotSpec::ConfusionShow::Counts       ? "counts"
                          : s.confusion_show == PlotSpec::ConfusionShow::ByPredicted ? "by_predicted"
                          : s.confusion_show == PlotSpec::ConfusionShow::All         ? "all"
                                                                                     : "by_actual";
    j["positive_class"] = s.positive_class;
    j["calibration_bins"] = s.calibration_bins;
    j["spread"] = s.spread_columns;
    j["best"] = s.best == PlotSpec::Best::Highest ? "highest" : s.best == PlotSpec::Best::Lowest ? "lowest" : "auto";
    j["top_n"] = s.top_n;
    j["sankey_top"] = s.sankey_top;
    j["region_agg"] = s.region_agg == PlotSpec::RegionAgg::Mean ? "mean" : "sum";
    j["z"] = s.z_column;
    j["surface_from"] = s.surface_from == PlotSpec::SurfaceFrom::Grid ? "grid" : "xyz";
    j["surface_draw"] = s.surface_draw == PlotSpec::SurfaceDraw::Lines ? "lines" : s.surface_draw == PlotSpec::SurfaceDraw::Both ? "both" : "fill";
    j["shade"] = s.shade;
    j["floor_contours"] = s.floor_contours;
    j["min_cell_rows"] = s.min_cell_rows;
    if (std::isfinite(s.view_elevation) && std::isfinite(s.view_azimuth)) j["view"] = {s.view_elevation, s.view_azimuth};
    if (!s.colour_scale.empty()) j["scale"] = s.colour_scale;
    if (s.colour_reverse) j["scale_reverse"] = true;
    if (std::isfinite(s.scale_lo) && std::isfinite(s.scale_hi)) j["scale_range"] = {s.scale_lo, s.scale_hi};
    if (!s.series_colours.empty()) j["series_colours"] = s.series_colours;
    if (s.kind == Kind::Network || s.kind == Kind::Tree) {
        j["graph_layout"] = s.graph_layout == PlotSpec::GraphLayout::Layered ? "layered" : s.graph_layout == PlotSpec::GraphLayout::Circle ? "circle" : "force";
        j["node_colour"] = s.node_colour == PlotSpec::NodeColour::Column ? "column" : s.node_colour == PlotSpec::NodeColour::One ? "one" : "groups";
        j["node_size"] = s.node_size == PlotSpec::NodeSize::Weight ? "weight" : s.node_size == PlotSpec::NodeSize::Same ? "same" : "links";
        j["node_labels"] = s.node_labels == PlotSpec::NodeLabels::All ? "all" : s.node_labels == PlotSpec::NodeLabels::None ? "none" : "top";
        j["label_top"] = s.label_top;
        j["directed"] = s.directed;
        j["tree_left_right"] = s.tree_left_right;
    }
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
    const std::string image_mode_id = j.value("image_mode", std::string("gallery"));
    if (image_mode_id == "gallery") out.image_mode = PlotSpec::ImageMode::Gallery;
    else if (image_mode_id == "one_row") out.image_mode = PlotSpec::ImageMode::OneRow;
    else if (image_mode_id == "mean_per_class") out.image_mode = PlotSpec::ImageMode::MeanPerClass;
    else return fail("unknown image image_mode_id '" + image_mode_id + "'");
    out.image_row = std::max(1, j.value("image_row", 1));
    out.image_width = std::clamp(j.value("image_width", 0), 0, 4096);
    out.image_channels = j.value("image_channels", 0);
    if (out.image_channels != 0 && out.image_channels != 1 && out.image_channels != 3)
        return fail("image channels must be 1 or 3 (or 0 for auto)");
    out.image_planar = j.value("image_planar", false);
    const std::string range = j.value("image_range", std::string("auto"));
    if (range == "auto") out.image_range = PlotSpec::ImageRange::Auto;
    else if (range == "0-255") out.image_range = PlotSpec::ImageRange::Byte;
    else if (range == "0-1") out.image_range = PlotSpec::ImageRange::Unit;
    else return fail("unknown image range '" + range + "'");
    out.image_grey = j.value("image_grey", false);
    out.image_invert = j.value("image_invert", false);
    out.gallery_max = std::clamp(j.value("gallery_max", 40), 1, 400);
    out.pair_histogram = j.value("pair_histogram", false);
    if (j.contains("range") && j["range"].is_array() && j["range"].size() == 2 && j["range"][0].is_number() && j["range"][1].is_number() &&
        j["range"][1].get<double>() > j["range"][0].get<double>()) {
        out.range_lo = j["range"][0].get<double>();
        out.range_hi = j["range"][1].get<double>();
    }
    const std::string show = j.value("confusion_show", std::string("by_actual"));
    if (show == "by_actual") out.confusion_show = PlotSpec::ConfusionShow::ByActual;
    else if (show == "counts") out.confusion_show = PlotSpec::ConfusionShow::Counts;
    else if (show == "by_predicted") out.confusion_show = PlotSpec::ConfusionShow::ByPredicted;
    else if (show == "all") out.confusion_show = PlotSpec::ConfusionShow::All;
    else return fail("unknown confusion display '" + show + "'");
    out.positive_class = j.value("positive_class", std::string());
    out.calibration_bins = std::clamp(j.value("calibration_bins", 10), 2, 100);
    if (j.contains("spread") && j["spread"].is_array())
        for (const auto& c : j["spread"])
            if (c.is_string()) out.spread_columns.push_back(c.get<std::string>());
    const std::string best = j.value("best", std::string("auto"));
    if (best == "auto") out.best = PlotSpec::Best::Auto;
    else if (best == "highest") out.best = PlotSpec::Best::Highest;
    else if (best == "lowest") out.best = PlotSpec::Best::Lowest;
    else return fail("unknown best '" + best + "'");
    out.top_n = std::clamp(j.value("top_n", 20), 1, 500);
    out.sankey_top = std::clamp(j.value("sankey_top", 8), 1, 50);
    const std::string agg = j.value("region_agg", std::string("sum"));
    if (agg == "sum") out.region_agg = PlotSpec::RegionAgg::Sum;
    else if (agg == "mean") out.region_agg = PlotSpec::RegionAgg::Mean;
    else return fail("unknown region aggregation '" + agg + "'");
    out.z_column = j.value("z", std::string());
    const std::string surface_from = j.value("surface_from", std::string("xyz"));
    if (surface_from == "xyz") out.surface_from = PlotSpec::SurfaceFrom::XYZ;
    else if (surface_from == "grid") out.surface_from = PlotSpec::SurfaceFrom::Grid;
    else return fail("unknown surface source '" + surface_from + "'");
    const std::string draw = j.value("surface_draw", std::string("fill"));
    if (draw == "fill") out.surface_draw = PlotSpec::SurfaceDraw::Fill;
    else if (draw == "lines") out.surface_draw = PlotSpec::SurfaceDraw::Lines;
    else if (draw == "both") out.surface_draw = PlotSpec::SurfaceDraw::Both;
    else return fail("unknown surface drawing '" + draw + "'");
    out.shade = j.value("shade", true);
    out.floor_contours = j.value("floor_contours", true);
    out.min_cell_rows = std::clamp(j.value("min_cell_rows", 5), 1, 100000);
    if (j.contains("view") && j["view"].is_array() && j["view"].size() == 2) {
        out.view_elevation = j["view"][0].get<double>();
        out.view_azimuth = j["view"][1].get<double>();
    }
    out.colour_scale = j.value("scale", std::string());
    if (!out.colour_scale.empty() && !FindScale(out.colour_scale)) return fail("unknown colour scale '" + out.colour_scale + "'");
    out.colour_reverse = j.value("scale_reverse", false);
    if (j.contains("scale_range") && j["scale_range"].is_array() && j["scale_range"].size() == 2) {
        out.scale_lo = j["scale_range"][0].get<double>();
        out.scale_hi = j["scale_range"][1].get<double>();
    }
    if (j.contains("series_colours") && j["series_colours"].is_array()) out.series_colours = j["series_colours"].get<std::vector<std::string>>();
    const auto pick = [&](const char* key, const char* fallback, std::initializer_list<const char*> ids, int& out_index) {
        const std::string v = j.value(key, std::string(fallback));
        int i = 0;
        for (const char* id : ids) {
            if (v == id) {
                out_index = i;
                return true;
            }
            ++i;
        }
        return false;
    };
    int choice = 0;
    if (!pick("graph_layout", "force", {"force", "layered", "circle"}, choice)) return fail("unknown graph layout");
    out.graph_layout = static_cast<PlotSpec::GraphLayout>(choice);
    if (!pick("node_colour", "groups", {"groups", "column", "one"}, choice)) return fail("unknown node colour");
    out.node_colour = static_cast<PlotSpec::NodeColour>(choice);
    if (!pick("node_size", "links", {"links", "weight", "same"}, choice)) return fail("unknown node size");
    out.node_size = static_cast<PlotSpec::NodeSize>(choice);
    if (!pick("node_labels", "top", {"top", "all", "none"}, choice)) return fail("unknown node labels");
    out.node_labels = static_cast<PlotSpec::NodeLabels>(choice);
    out.label_top = std::clamp(j.value("label_top", 12), 1, 1000);
    out.directed = j.value("directed", false);
    out.tree_left_right = j.value("tree_left_right", false);
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
    if ((k.required & kEncZ) && s.z_column.empty()) return "Choose Z.";
    if (s.kind == Kind::Surface) {
        if (s.surface_from == PlotSpec::SurfaceFrom::Grid)
            return s.y_columns.size() < 2 ? "Choose the grid columns (two or more)." : "";
        if (s.x_column.empty()) return "Choose X.";
        if (s.y_columns.empty()) return "Choose Y.";
        if (s.z_column.empty()) return "Choose Z.";
    }
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
