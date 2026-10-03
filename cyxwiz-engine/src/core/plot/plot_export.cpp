#include "plot_export.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "plot_layout.h"
#include "world_map.h"

namespace cyxwiz::plot {

namespace {

std::string Num(double v) {
    if (!std::isfinite(v)) return "";
    std::ostringstream out;
    out << std::setprecision(10) << v;
    return out.str();
}

std::string CsvCell(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string q = "\"";
    for (char c : s) q += c == '"' ? std::string("\"\"") : std::string(1, c);
    return q + "\"";
}

std::string Xml(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

std::string Category(const Prepared& p, size_t i) {
    return i < p.categories.size() ? p.categories[i] : std::to_string(i);
}

}  // namespace

std::string HexColour(float r, float g, float b) {
    const auto c = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", c(r), c(g), c(b));
    return buf;
}

int PictureColumns(const Prepared& p) {
    const size_t n = p.pictures.size();
    if (n <= 1) return 1;
    if (p.spec.image_mode == PlotSpec::ImageMode::MeanPerClass && n <= 12) return static_cast<int>(n);
    return static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n) * 1.6)));
}

std::string ToCsv(const Prepared& p) {
    std::ostringstream out;
    switch (p.spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
        case Kind::Kde:
        case Kind::Scatter:
            out << "series,x,y";
            if (p.colour_scale) out << ',' << CsvCell(p.colour_label);
            out << '\n';
            for (const auto& s : p.series) {
                const auto& xs = s.all_x.empty() ? s.x : s.all_x;
                const auto& ys = s.all_y.empty() ? s.y : s.all_y;
                const auto& cs = s.all_x.empty() ? s.c : s.all_c;
                for (size_t i = 0; i < std::min(xs.size(), ys.size()); ++i) {
                    out << CsvCell(s.label) << ',' << Num(xs[i]) << ',' << Num(ys[i]);
                    if (p.colour_scale) out << ',' << (i < cs.size() ? Num(cs[i]) : std::string());
                    out << '\n';
                }
            }
            break;
        case Kind::Histogram:
            out << "series,bin_start,bin_end," << (p.spec.density ? "density" : "count") << '\n';
            for (const auto& s : p.series)
                for (size_t i = 0; i < s.y.size() && i + 1 < p.edges.size(); ++i)
                    out << CsvCell(s.label) << ',' << Num(p.edges[i]) << ',' << Num(p.edges[i + 1]) << ',' << Num(s.y[i]) << '\n';
            break;
        case Kind::Bar:
        case Kind::Pie:
            if (p.series.size() > 1) {  // bars with Colour by
                out << "category,series," << (p.spec.bar_layout == PlotSpec::BarLayout::Percent ? "percent" : "value") << '\n';
                for (const auto& s : p.series)
                    for (size_t i = 0; i < s.y.size(); ++i)
                        out << CsvCell(Category(p, i)) << ',' << CsvCell(s.label) << ',' << Num(s.y[i]) << '\n';
                break;
            }
            out << "category," << CsvCell(p.series.empty() ? "value" : p.series[0].label) << '\n';
            if (!p.series.empty())
                for (size_t i = 0; i < p.series[0].y.size(); ++i)
                    out << CsvCell(Category(p, i)) << ',' << Num(p.series[0].y[i]) << '\n';
            break;
        case Kind::ErrorBars:
            out << "group,mean,sd\n";
            if (!p.series.empty())
                for (size_t i = 0; i < p.series[0].y.size(); ++i)
                    out << CsvCell(Category(p, i)) << ',' << Num(p.series[0].y[i]) << ','
                        << Num(i < p.series[0].low.size() ? p.series[0].low[i] : NAN) << '\n';
            break;
        case Kind::Box:
        case Kind::Violin:
            out << "series,low_whisker,q1,median,q3,high_whisker,mean\n";
            for (size_t i = 0; i < p.boxes.size() && i < p.series.size(); ++i) {
                const auto& b = p.boxes[i];
                out << CsvCell(p.series[i].label) << ',' << Num(b.low) << ',' << Num(b.q1) << ',' << Num(b.median) << ','
                    << Num(b.q3) << ',' << Num(b.high) << ',' << Num(b.mean) << '\n';
            }
            break;
        case Kind::Heatmap:
        case Kind::Matrix:
            out << "row,column,"
                << (p.spec.kind == Kind::Matrix ? (p.spec.matrix_values == PlotSpec::MatrixValues::Values ? "value" : "correlation")
                                                : (p.spec.value_column.empty() ? "count" : p.spec.value_column.c_str()))
                << '\n';
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c)
                    out << CsvCell(r < static_cast<int>(p.row_names.size()) ? p.row_names[static_cast<size_t>(r)] : "") << ','
                        << CsvCell(c < static_cast<int>(p.col_names.size()) ? p.col_names[static_cast<size_t>(c)] : "") << ','
                        << Num(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) << '\n';
            break;
        case Kind::Polar: {
            // The angle as it was given: the category, degrees or radians.
            const bool names = !p.polar_names.empty();
            const bool radians = p.spec.angle_unit == PlotSpec::AngleUnit::Radians;
            out << "series,angle" << (names ? "" : radians ? "_radians" : "_degrees") << ",radius\n";
            for (const auto& s : p.series)
                for (size_t i = 0; i < std::min(s.x.size(), s.y.size()); ++i) {
                    std::string a;
                    if (names) {
                        const size_t k = static_cast<size_t>(std::llround(s.x[i] / 6.283185307179586 * static_cast<double>(p.polar_names.size())));
                        a = CsvCell(k < p.polar_names.size() ? p.polar_names[k] : "");
                    } else {
                        a = Num(radians ? s.x[i] : s.x[i] * 360.0 / 6.283185307179586);
                    }
                    out << CsvCell(s.label) << ',' << a << ',' << Num(s.y[i]) << '\n';
                }
            break;
        }
        case Kind::Image: {
            // One line per pixel value, in the table's units.
            out << "picture,label,row,x,y,channel,value\n";
            for (size_t i = 0; i < p.pictures.size(); ++i) {
                const auto& pic = p.pictures[i];
                for (size_t k = 0; k < pic.pix.size(); ++k) {
                    const size_t pixel = k / static_cast<size_t>(p.img_channels), channel = k % static_cast<size_t>(p.img_channels);
                    const double v = std::isfinite(pic.pix[k]) ? p.img_lo + pic.pix[k] * (p.img_hi - p.img_lo) : NAN;
                    out << i + 1 << ',' << CsvCell(pic.label) << ',' << pic.row << ',' << pixel % static_cast<size_t>(p.img_w) << ','
                        << pixel / static_cast<size_t>(p.img_w) << ',' << channel << ',' << Num(v) << '\n';
                }
            }
            break;
        }
        case Kind::PairPlot:
        case Kind::Parallel: {
            out << "group";
            for (const auto& c : p.multi_cols) out << ',' << CsvCell(c);
            out << '\n';
            for (size_t r = 0; r < p.multi_group.size(); ++r) {
                const int g = p.multi_group[r];
                out << CsvCell(g >= 0 && static_cast<size_t>(g) < p.multi_groups.size() ? p.multi_groups[static_cast<size_t>(g)] : "");
                for (const auto& col : p.multi_values) out << ',' << Num(col[r]);
                out << '\n';
            }
            break;
        }
        case Kind::Quiver:
            out << "x,y,u,v,length\n";
            for (size_t i = 0; i < p.qx.size(); ++i)
                out << Num(p.qx[i]) << ',' << Num(p.qy[i]) << ',' << Num(p.qu[i]) << ',' << Num(p.qv[i]) << ','
                    << Num(std::hypot(p.qu[i], p.qv[i])) << '\n';
            break;
        case Kind::Stream:
            out << "line,x,y\n";
            for (size_t l = 0; l < p.stream_lines.size(); ++l)
                for (size_t k = 0; k + 1 < p.stream_lines[l].size(); k += 2)
                    out << l + 1 << ',' << Num(p.stream_lines[l][k]) << ',' << Num(p.stream_lines[l][k + 1]) << '\n';
            break;
        case Kind::Hexbin:
            out << "x,y," << (p.spec.value_column.empty() ? "count" : "mean of " + p.spec.value_column) << '\n';
            for (size_t i = 0; i < p.hex_x.size(); ++i) out << Num(p.hex_x[i]) << ',' << Num(p.hex_y[i]) << ',' << Num(p.hex_v[i]) << '\n';
            break;
        case Kind::Contour:
        case Kind::FilledContour: {
            out << "x,y," << (p.spec.value_column.empty() ? "density" : "mean of " + p.spec.value_column) << '\n';
            const double dx = (p.x_max - p.x_min) / std::max(1, p.grid_cols), dy = (p.y_max - p.y_min) / std::max(1, p.grid_rows);
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c)
                    out << Num(p.x_min + dx * (c + 0.5)) << ',' << Num(p.y_max - dy * (r + 0.5)) << ','
                        << Num(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) << '\n';
            break;
        }
        case Kind::Histogram2D: {
            out << "x_start,x_end,y_start,y_end,count\n";
            const double dx = (p.x_max - p.x_min) / std::max(1, p.grid_cols);
            const double dy = (p.y_max - p.y_min) / std::max(1, p.grid_rows);
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const double y_top = p.y_max - dy * r;
                    out << Num(p.x_min + dx * c) << ',' << Num(p.x_min + dx * (c + 1)) << ',' << Num(y_top - dy) << ','
                        << Num(y_top) << ',' << Num(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) << '\n';
                }
            break;
        }
        case Kind::Confusion:
            out << "actual,predicted,count,share\n";
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const size_t k = static_cast<size_t>(r * p.grid_cols + c);
                    out << CsvCell(p.row_names[static_cast<size_t>(r)]) << ',' << CsvCell(p.col_names[static_cast<size_t>(c)]) << ','
                        << Num(k < p.grid_counts.size() ? p.grid_counts[k] : NAN) << ','
                        << (p.spec.confusion_show == PlotSpec::ConfusionShow::Counts ? std::string() : Num(p.grid[k])) << '\n';
                }
            break;
        case Kind::Roc:
        case Kind::PrCurve:
            out << (p.spec.kind == Kind::Roc ? "threshold,false_positive_rate,true_positive_rate\n" : "threshold,recall,precision\n");
            if (!p.series.empty())
                for (size_t i = 0; i < p.series[0].x.size(); ++i)
                    out << Num(p.series[0].c[i]) << ',' << Num(p.series[0].x[i]) << ',' << Num(p.series[0].y[i]) << '\n';
            break;
        case Kind::Calibration:
            out << "mean_predicted,share_positive,rows\n";
            if (!p.series.empty())
                for (size_t i = 0; i < p.series[0].x.size(); ++i)
                    out << Num(p.series[0].x[i]) << ',' << Num(p.series[0].y[i]) << ',' << Num(p.series[0].low[i]) << '\n';
            break;
        case Kind::Residuals:
            out << "predicted,residual\n";
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                const auto& xs = s.all_x.empty() ? s.x : s.all_x;
                const auto& ys = s.all_y.empty() ? s.y : s.all_y;
                for (size_t i = 0; i < std::min(xs.size(), ys.size()); ++i) out << Num(xs[i]) << ',' << Num(ys[i]) << '\n';
            }
            break;
        case Kind::LearningCurve:
            out << "series,x,y,low,high\n";
            for (const auto& s : p.series)
                for (size_t i = 0; i < std::min(s.x.size(), s.y.size()); ++i)
                    out << CsvCell(s.label) << ',' << Num(s.x[i]) << ',' << Num(s.y[i]) << ',' << (i < s.low.size() ? Num(s.low[i]) : std::string())
                        << ',' << (i < s.high.size() ? Num(s.high[i]) : std::string()) << '\n';
            break;
        case Kind::Importance:
            out << "feature," << CsvCell(p.series.empty() ? "importance" : p.series[0].label) << ",spread\n";
            if (!p.series.empty())
                for (size_t i = 0; i < p.series[0].y.size(); ++i)
                    out << CsvCell(Category(p, i)) << ',' << Num(p.series[0].y[i]) << ','
                        << (i < p.series[0].low.size() ? Num(p.series[0].low[i]) : std::string()) << '\n';
            break;
        case Kind::Sankey:
            out << "from_step,from,to_step,to,value\n";
            for (const auto& l : p.sankey_links) {
                const auto& a = p.sankey_nodes[static_cast<size_t>(l.from)];
                const auto& b = p.sankey_nodes[static_cast<size_t>(l.to)];
                out << CsvCell(p.sankey_steps[static_cast<size_t>(a.step)]) << ',' << CsvCell(a.name) << ','
                    << CsvCell(p.sankey_steps[static_cast<size_t>(b.step)]) << ',' << CsvCell(b.name) << ',' << Num(l.value) << '\n';
            }
            break;
        case Kind::Treemap:
            for (const auto& level : p.tree_levels) out << CsvCell(level) << ',';
            out << CsvCell(p.spec.value_column.empty() ? "rows" : p.spec.value_column);
            if (p.colour_scale) out << ',' << CsvCell("mean of " + p.colour_label);
            out << '\n';
            for (const auto& l : p.tree_leaves) {
                for (const auto& name : l.path) out << CsvCell(name) << ',';
                out << Num(l.size);
                if (p.colour_scale) out << ',' << Num(l.colour);
                out << '\n';
            }
            break;
        case Kind::MapPoints:
            out << "series,longitude,latitude";
            if (!p.size_label.empty()) out << ',' << CsvCell(p.size_label);
            if (p.colour_scale) out << ',' << CsvCell(p.colour_label);
            out << '\n';
            for (const auto& s : p.series) {
                const bool all = !s.all_x.empty();
                const auto& xs = all ? s.all_x : s.x;
                const auto& ys = all ? s.all_y : s.y;
                const auto& zs = all ? s.all_z : s.z;
                const auto& cs = all ? s.all_c : s.c;
                for (size_t i = 0; i < std::min(xs.size(), ys.size()); ++i) {
                    out << CsvCell(s.label) << ',' << Num(xs[i]) << ',' << Num(ys[i]);
                    if (!p.size_label.empty()) out << ',' << (i < zs.size() ? Num(zs[i]) : std::string());
                    if (p.colour_scale) out << ',' << (i < cs.size() ? Num(cs[i]) : std::string());
                    out << '\n';
                }
            }
            break;
        case Kind::MapRegions: {
            out << "country,iso_a3," << CsvCell(p.spec.y_columns.empty() ? "value" : p.spec.y_columns.front()) << ",rows,matched\n";
            const auto& countries = WorldCountries();
            for (size_t i = 0; i < countries.size() && i < p.region_value.size(); ++i)
                if (std::isfinite(p.region_value[i]))
                    out << CsvCell(countries[i].name) << ',' << countries[i].iso_a3 << ',' << Num(p.region_value[i]) << ','
                        << p.region_rows[i] << ",yes\n";
            for (const auto& [text, rows] : p.unmatched) out << CsvCell(text) << ",,," << rows << ",no\n";
            break;
        }
    }
    return out.str();
}

namespace {

struct Frame {
    double left, top, width, height;
    AxisRange r;
    double X(double x) const { return left + (x - r.x0) / (r.x1 - r.x0) * width; }
    double Fy(double y) const { return r.log_y ? std::log10(std::max(y, 1e-300)) : y; }
    double Y(double y) const { return top + height - (Fy(y) - Fy(r.y0)) / (Fy(r.y1) - Fy(r.y0)) * height; }
};

std::vector<double> Ticks(double a, double b) {
    std::vector<double> t;
    if (!(b > a)) return t;
    const double raw = (b - a) / 5.0;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double norm = raw / mag;
    const double step = (norm < 1.5 ? 1.0 : norm < 3.0 ? 2.0 : norm < 7.0 ? 5.0 : 10.0) * mag;
    for (double v = std::ceil(a / step) * step; v <= b + step * 1e-9; v += step) t.push_back(std::fabs(v) < step * 1e-9 ? 0.0 : v);
    return t;
}

std::string Points(const Frame& f, const std::vector<double>& xs, const std::vector<double>& ys) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < std::min(xs.size(), ys.size()); ++i) o << f.X(xs[i]) << ',' << f.Y(ys[i]) << ' ';
    return o.str();
}

std::string Mix(const std::string& lo, const std::string& hi, double t) {
    const auto channel = [](const std::string& hex, int i) { return std::stoi(hex.substr(1 + i * 2, 2), nullptr, 16); };
    char buf[8];
    int c[3];
    for (int i = 0; i < 3; ++i) c[i] = static_cast<int>(std::lround(channel(lo, i) + (channel(hi, i) - channel(lo, i)) * t));
    std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", c[0], c[1], c[2]);
    return buf;
}

// A grid cell's colour on its range (two-sided around 0 when asked); a
// missing cell in the background.
std::string RangeColour(double v, double lo, double hi, bool diverging, const SvgStyle& st) {
    if (!std::isfinite(v)) return st.background;
    const double t = hi > lo ? std::clamp((v - lo) / (hi - lo), 0.0, 1.0) : 0.0;
    if (!diverging) return Mix(st.scale_low, st.scale_high, t);
    return t < 0.5 ? Mix(st.diverging_low, st.diverging_mid, t * 2.0) : Mix(st.diverging_mid, st.diverging_high, (t - 0.5) * 2.0);
}

// A colour-scale point: the sequential scale, or the two-sided scale around
// 0; a missing value in the dim text colour.
std::string ScaleColour(const Prepared& p, const SvgStyle& st, double v) {
    if (!std::isfinite(v)) return st.text_dim;
    const double span = p.colour_max - p.colour_min;
    const double t = span > 0 ? std::clamp((v - p.colour_min) / span, 0.0, 1.0) : 0.0;
    if (!p.colour_diverging) return Mix(st.scale_low, st.scale_high, t);
    return t < 0.5 ? Mix(st.diverging_low, st.diverging_mid, t * 2.0) : Mix(st.diverging_mid, st.diverging_high, (t - 0.5) * 2.0);
}

}  // namespace

std::string ToSvg(const Prepared& p, const AxisRange& range, const SvgStyle& st) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(1);
    const bool title = !p.spec.title.empty();
    const bool pie = p.spec.kind == Kind::Pie || p.spec.kind == Kind::Polar || p.spec.kind == Kind::Image ||
                     p.spec.kind == Kind::PairPlot || p.spec.kind == Kind::Parallel || p.spec.kind == Kind::Sankey ||
                     p.spec.kind == Kind::Treemap;
    Frame f{64, title ? 40.0 : 16.0, st.width - 64.0 - 18.0, st.height - (title ? 40.0 : 16.0) - 46.0, range};
    if (!(f.r.x1 > f.r.x0)) f.r.x1 = f.r.x0 + 1;
    if (!(f.r.y1 > f.r.y0)) f.r.y1 = f.r.y0 + 1;
    const auto colour = [&](size_t i) { return st.series[i % 6]; };
    o << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << st.width << "\" height=\"" << st.height
      << "\" viewBox=\"0 0 " << st.width << ' ' << st.height << "\" font-family=\"Inter, Segoe UI, sans-serif\" font-size=\"12\">\n";
    o << "<rect width=\"100%\" height=\"100%\" fill=\"" << st.background << "\"/>\n";
    if (title) o << "<text x=\"" << f.left << "\" y=\"24\" fill=\"" << st.text << "\" font-size=\"15\" font-weight=\"600\">" << Xml(p.spec.title) << "</text>\n";
    o << "<defs><clipPath id=\"area\"><rect x=\"" << f.left << "\" y=\"" << f.top << "\" width=\"" << f.width << "\" height=\"" << f.height << "\"/></clipPath></defs>\n";

    // Axes: grid, ticks and labels (categories under bars).
    const bool grid_names = p.spec.kind == Kind::Heatmap || p.spec.kind == Kind::Matrix || p.spec.kind == Kind::Confusion;
    const bool importance = p.spec.kind == Kind::Importance;
    const bool categorical_x = p.spec.kind == Kind::Bar || p.spec.kind == Kind::ErrorBars || grid_names;
    if (!pie) {
        if (categorical_x) {
            const auto& names = grid_names ? p.col_names : p.categories;
            const size_t every = names.size() > 30 ? names.size() / 30 + 1 : 1;
            for (size_t i = 0; i < names.size(); i += every) {
                const double x = grid_names ? f.r.x0 + (static_cast<double>(i) + 0.5) : static_cast<double>(i);
                o << "<text x=\"" << f.X(x) << "\" y=\"" << f.top + f.height + 16 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(names[i]) << "</text>\n";
            }
        } else {
            for (double t : Ticks(f.r.x0, f.r.x1)) {
                o << "<line x1=\"" << f.X(t) << "\" y1=\"" << f.top << "\" x2=\"" << f.X(t) << "\" y2=\"" << f.top + f.height << "\" stroke=\"" << st.grid << "\"/>\n";
                o << "<text x=\"" << f.X(t) << "\" y=\"" << f.top + f.height + 16 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Num(t) << "</text>\n";
            }
        }
        if (importance) {
            // Features down the side, the largest on top.
            const size_t n = p.categories.size();
            for (size_t i = 0; i < n; ++i)
                o << "<text x=\"" << f.left - 8 << "\" y=\"" << f.Y(static_cast<double>(n - 1 - i)) + 4 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"end\">" << Xml(p.categories[i]) << "</text>\n";
        } else if (grid_names) {
            for (size_t i = 0; i < p.row_names.size() && i < 60; ++i)
                o << "<text x=\"" << f.left - 8 << "\" y=\"" << f.Y(f.r.y1 - (static_cast<double>(i) + 0.5)) + 4 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"end\">" << Xml(p.row_names[i]) << "</text>\n";
        } else {
            const double a = f.r.log_y ? std::log10(std::max(f.r.y0, 1e-300)) : f.r.y0;
            const double b = f.r.log_y ? std::log10(std::max(f.r.y1, 1e-300)) : f.r.y1;
            for (double t : Ticks(a, b)) {
                const double v = f.r.log_y ? std::pow(10.0, t) : t;
                o << "<line x1=\"" << f.left << "\" y1=\"" << f.Y(v) << "\" x2=\"" << f.left + f.width << "\" y2=\"" << f.Y(v) << "\" stroke=\"" << st.grid << "\"/>\n";
                o << "<text x=\"" << f.left - 8 << "\" y=\"" << f.Y(v) + 4 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"end\">" << Num(v) << "</text>\n";
            }
        }
        const std::string xl = !p.spec.x_label.empty() ? p.spec.x_label : p.spec.x_column;
        std::string yl = importance ? std::string() : p.spec.y_label;
        if (yl.empty()) yl = p.spec.kind == Kind::Histogram ? (p.spec.density ? "density" : "count")
                             : !p.spec.y_columns.empty() ? p.spec.y_columns.front() : "";
        o << "<text x=\"" << f.left + f.width / 2 << "\" y=\"" << st.height - 10 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(xl) << "</text>\n";
        o << "<text transform=\"translate(16," << f.top + f.height / 2 << ") rotate(-90)\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(yl) << "</text>\n";
    }

    o << "<g clip-path=\"url(#area)\">\n";
    if (p.spec.show_diagonal && (p.spec.kind == Kind::Line || p.spec.kind == Kind::Scatter || p.spec.kind == Kind::Roc ||
                                 p.spec.kind == Kind::Calibration)) {
        const double lo = std::max(range.x0, range.y0), hi = std::min(range.x1, range.y1);
        if (hi > lo)
            o << "<line x1=\"" << f.X(lo) << "\" y1=\"" << f.Y(lo) << "\" x2=\"" << f.X(hi) << "\" y2=\"" << f.Y(hi)
              << "\" stroke=\"" << st.text_dim << "\" stroke-width=\"1.2\"/>\n";
    }
    switch (p.spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
        case Kind::Kde:
        case Kind::Scatter:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const std::string c = colour(i);
                if (p.spec.kind == Kind::Scatter) {
                    for (size_t k = 0; k < std::min(s.x.size(), s.y.size()); ++k) {
                        const std::string fill = p.colour_scale ? ScaleColour(p, st, k < s.c.size() ? s.c[k] : NAN) : c;
                        o << "<circle cx=\"" << f.X(s.x[k]) << "\" cy=\"" << f.Y(s.y[k]) << "\" r=\"2\" fill=\"" << fill << "\" fill-opacity=\"0.7\"/>\n";
                    }
                } else if (p.spec.kind == Kind::Stem) {
                    for (size_t k = 0; k < std::min(s.x.size(), s.y.size()); ++k)
                        o << "<line x1=\"" << f.X(s.x[k]) << "\" y1=\"" << f.Y(0) << "\" x2=\"" << f.X(s.x[k]) << "\" y2=\"" << f.Y(s.y[k]) << "\" stroke=\"" << c << "\"/><circle cx=\"" << f.X(s.x[k]) << "\" cy=\"" << f.Y(s.y[k]) << "\" r=\"2.5\" fill=\"" << c << "\"/>\n";
                } else if (p.spec.kind == Kind::Step) {
                    o << "<path fill=\"none\" stroke=\"" << c << "\" stroke-width=\"1.6\" d=\"";
                    for (size_t k = 0; k < std::min(s.x.size(), s.y.size()); ++k) {
                        if (k == 0) o << 'M' << f.X(s.x[k]) << ' ' << f.Y(s.y[k]);
                        else o << " H" << f.X(s.x[k]) << " V" << f.Y(s.y[k]);
                    }
                    o << "\"/>\n";
                } else {
                    if ((p.spec.kind == Kind::Area || p.spec.kind == Kind::Kde) && !s.x.empty())
                        o << "<polygon fill=\"" << c << "\" fill-opacity=\"0.25\" points=\"" << f.X(s.x.front()) << ',' << f.Y(0) << ' '
                          << Points(f, s.x, s.y) << f.X(s.x.back()) << ',' << f.Y(0) << "\"/>\n";
                    const bool smoothed = !s.smooth_y.empty();
                    o << "<polyline fill=\"none\" stroke=\"" << c << "\" stroke-width=\"1.6\"" << (smoothed ? " stroke-opacity=\"0.55\"" : "") << " points=\"" << Points(f, s.x, s.y) << "\"/>\n";
                    if (smoothed) o << "<polyline fill=\"none\" stroke=\"" << c << "\" stroke-width=\"2.6\" points=\"" << Points(f, s.smooth_x, s.smooth_y) << "\"/>\n";
                }
            }
            break;
        case Kind::Histogram:
            for (size_t i = 0; i < p.series.size(); ++i)
                for (size_t k = 0; k < p.series[i].y.size() && k + 1 < p.edges.size(); ++k) {
                    const double x0 = f.X(p.edges[k]), x1 = f.X(p.edges[k + 1]), y = f.Y(p.series[i].y[k]);
                    o << "<rect x=\"" << x0 + 0.5 << "\" y=\"" << y << "\" width=\"" << std::max(0.5, x1 - x0 - 1) << "\" height=\"" << std::max(0.0, f.Y(f.r.log_y ? f.r.y0 : 0) - y) << "\" fill=\"" << colour(i) << "\"" << (p.series.size() > 1 ? " fill-opacity=\"0.6\"" : "") << "/>\n";
                }
            break;
        case Kind::Bar:
            if (p.series.size() > 1) {
                // Colour by: side by side, or stacked (stacked and 100%).
                const bool stacked = p.spec.bar_layout != PlotSpec::BarLayout::Grouped;
                const double n = static_cast<double>(p.series.size());
                for (size_t k = 0; k < p.series[0].y.size(); ++k) {
                    double base = 0;
                    for (size_t i = 0; i < p.series.size(); ++i) {
                        const double v = p.series[i].y[k];
                        const double x0 = stacked ? k - 0.33 : k - 0.33 + 0.66 * i / n;
                        const double x1 = stacked ? k + 0.33 : x0 + 0.66 / n;
                        const double y0 = stacked ? base : 0.0, y1 = y0 + v;
                        o << "<rect x=\"" << f.X(x0) << "\" y=\"" << f.Y(y1) << "\" width=\"" << f.X(x1) - f.X(x0) << "\" height=\""
                          << std::max(0.0, f.Y(y0) - f.Y(y1)) << "\" fill=\"" << colour(i) << "\"/>\n";
                        base = y1;
                    }
                }
                break;
            }
            if (!p.series.empty())
                for (size_t k = 0; k < p.series[0].y.size(); ++k) {
                    const double x = static_cast<double>(k), y = f.Y(p.series[0].y[k]);
                    o << "<rect x=\"" << f.X(x - 0.33) << "\" y=\"" << y << "\" width=\"" << f.X(x + 0.33) - f.X(x - 0.33) << "\" height=\"" << std::max(0.0, f.Y(0) - y) << "\" fill=\"" << colour(0) << "\"/>\n";
                }
            break;
        case Kind::ErrorBars:
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                for (size_t k = 0; k < s.y.size(); ++k) {
                    const double x = f.X(static_cast<double>(k)), lo = f.Y(s.y[k] - s.low[k]), hi = f.Y(s.y[k] + s.high[k]);
                    o << "<line x1=\"" << x << "\" y1=\"" << lo << "\" x2=\"" << x << "\" y2=\"" << hi << "\" stroke=\"" << colour(0) << "\" stroke-width=\"1.5\"/>"
                      << "<line x1=\"" << x - 6 << "\" y1=\"" << lo << "\" x2=\"" << x + 6 << "\" y2=\"" << lo << "\" stroke=\"" << colour(0) << "\"/>"
                      << "<line x1=\"" << x - 6 << "\" y1=\"" << hi << "\" x2=\"" << x + 6 << "\" y2=\"" << hi << "\" stroke=\"" << colour(0) << "\"/>"
                      << "<circle cx=\"" << x << "\" cy=\"" << f.Y(s.y[k]) << "\" r=\"3.5\" fill=\"" << colour(0) << "\"/>\n";
                }
            }
            break;
        case Kind::Box:
        case Kind::Violin:
            for (size_t i = 0; i < p.boxes.size(); ++i) {
                const auto& b = p.boxes[i];
                const double x = static_cast<double>(i);
                const std::string c = colour(i);
                if (p.spec.kind == Kind::Violin && i < p.series.size()) {
                    const auto& s = p.series[i];
                    o << "<polygon fill=\"" << c << "\" fill-opacity=\"0.45\" points=\"" << Points(f, s.high, s.y);
                    std::vector<double> lx(s.low.rbegin(), s.low.rend()), ly(s.y.rbegin(), s.y.rend());
                    o << Points(f, lx, ly) << "\"/>\n";
                }
                const double w = p.spec.kind == Kind::Violin ? 0.08 : 0.3;
                o << "<rect x=\"" << f.X(x - w) << "\" y=\"" << f.Y(b.q3) << "\" width=\"" << f.X(x + w) - f.X(x - w) << "\" height=\"" << f.Y(b.q1) - f.Y(b.q3) << "\" fill=\"" << c << "\" fill-opacity=\"0.35\" stroke=\"" << c << "\"/>"
                  << "<line x1=\"" << f.X(x - w) << "\" y1=\"" << f.Y(b.median) << "\" x2=\"" << f.X(x + w) << "\" y2=\"" << f.Y(b.median) << "\" stroke=\"" << st.text << "\" stroke-width=\"2\"/>"
                  << "<line x1=\"" << f.X(x) << "\" y1=\"" << f.Y(b.q3) << "\" x2=\"" << f.X(x) << "\" y2=\"" << f.Y(b.high) << "\" stroke=\"" << c << "\"/>"
                  << "<line x1=\"" << f.X(x) << "\" y1=\"" << f.Y(b.q1) << "\" x2=\"" << f.X(x) << "\" y2=\"" << f.Y(b.low) << "\" stroke=\"" << c << "\"/>\n";
            }
            break;
        case Kind::Pie: {
            if (p.series.empty()) break;
            const auto& v = p.series[0].y;
            double total = 0;
            for (double x : v) total += std::max(0.0, x);
            const double cx = f.left + f.width / 2, cy = f.top + f.height / 2, rad = std::min(f.width, f.height) / 2.4;
            double a = -1.5707963267948966;
            for (size_t k = 0; k < v.size() && total > 0; ++k) {
                const double sweep = std::max(0.0, v[k]) / total * 6.283185307179586;
                const double a2 = a + sweep;
                o << "<path fill=\"" << colour(k) << "\" stroke=\"" << st.background << "\" d=\"M" << cx << ' ' << cy << " L" << cx + rad * std::cos(a) << ' ' << cy + rad * std::sin(a)
                  << " A" << rad << ' ' << rad << " 0 " << (sweep > 3.141592653589793 ? 1 : 0) << " 1 " << cx + rad * std::cos(a2) << ' ' << cy + rad * std::sin(a2) << " Z\"/>\n";
                const double mid = (a + a2) / 2;
                o << "<text x=\"" << cx + rad * 1.12 * std::cos(mid) << "\" y=\"" << cy + rad * 1.12 * std::sin(mid) << "\" fill=\"" << st.text << "\" text-anchor=\"middle\">" << Xml(Category(p, k)) << "</text>\n";
                a = a2;
            }
            if (p.spec.donut && total > 0) {
                // The hole, with the total in it.
                o << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << rad * 0.58 << "\" fill=\"" << st.background << "\"/>\n";
                o << "<text x=\"" << cx << "\" y=\"" << cy + 6 << "\" fill=\"" << st.text << "\" font-size=\"18\" font-weight=\"600\" text-anchor=\"middle\">"
                  << Num(total) << "</text>\n";
            }
            break;
        }
        case Kind::Polar: {
            // Rings at round radii, spokes (the category names or every 30
            // degrees), then the lines or points; 0 at the top, clockwise.
            const double cx = f.left + f.width / 2, cy = f.top + f.height / 2, rad = std::min(f.width, f.height) / 2.3;
            const double rmax = p.polar_rmax > 0 ? p.polar_rmax : 1.0;
            const auto px = [&](double a, double r) { return cx + r / rmax * rad * std::sin(a); };
            const auto py = [&](double a, double r) { return cy - r / rmax * rad * std::cos(a); };
            for (double t : Ticks(0, rmax))
                if (t > 0 && t <= rmax)
                    o << "<circle cx=\"" << cx << "\" cy=\"" << cy << "\" r=\"" << t / rmax * rad << "\" fill=\"none\" stroke=\"" << st.grid << "\"/>"
                      << "<text x=\"" << cx + 3 << "\" y=\"" << cy - t / rmax * rad - 2 << "\" fill=\"" << st.text_dim << "\" font-size=\"10\">" << Num(t) << "</text>\n";
            const size_t spokes = !p.polar_names.empty() ? std::min<size_t>(p.polar_names.size(), 36) : 12;
            for (size_t k = 0; k < spokes; ++k) {
                const double a = static_cast<double>(k) / static_cast<double>(spokes) * 6.283185307179586;
                o << "<line x1=\"" << cx << "\" y1=\"" << cy << "\" x2=\"" << px(a, rmax) << "\" y2=\"" << py(a, rmax) << "\" stroke=\"" << st.grid << "\"/>";
                const std::string name = !p.polar_names.empty() ? p.polar_names[k] : Num(static_cast<double>(k) * 30.0);
                o << "<text x=\"" << px(a, rmax * 1.1) << "\" y=\"" << py(a, rmax * 1.1) + 4 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(name) << "</text>\n";
            }
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& sr = p.series[i];
                const std::string c = colour(sr.colour >= 0 ? static_cast<size_t>(sr.colour) : i);
                if (p.spec.polar_points) {
                    for (size_t k = 0; k < sr.x.size(); ++k)
                        o << "<circle cx=\"" << px(sr.x[k], sr.y[k]) << "\" cy=\"" << py(sr.x[k], sr.y[k]) << "\" r=\"2.5\" fill=\"" << c << "\"/>\n";
                } else {
                    o << "<polyline fill=\"none\" stroke=\"" << c << "\" stroke-width=\"1.8\" points=\"";
                    for (size_t k = 0; k < sr.x.size(); ++k) o << px(sr.x[k], sr.y[k]) << ',' << py(sr.x[k], sr.y[k]) << ' ';
                    if (p.polar_closed && !sr.x.empty()) o << px(sr.x[0], sr.y[0]) << ',' << py(sr.x[0], sr.y[0]);
                    o << "\"/>\n";
                }
            }
            break;
        }
        case Kind::Image: {
            const int cols = PictureColumns(p);
            const int rows = static_cast<int>((p.pictures.size() + static_cast<size_t>(cols) - 1) / static_cast<size_t>(cols));
            const double cell = std::min(f.width / cols, f.height / std::max(1, rows) - 14.0);
            const double px = cell * 0.94 / std::max(p.img_w, p.img_h);
            for (size_t i = 0; i < p.pictures.size(); ++i) {
                const auto& pic = p.pictures[i];
                const double ox = f.left + static_cast<double>(static_cast<int>(i) % cols) * cell;
                const double oy = f.top + static_cast<double>(static_cast<int>(i) / cols) * (cell + 14.0);
                for (int y = 0; y < p.img_h; ++y)
                    for (int x = 0; x < p.img_w; ++x) {
                        const size_t k = (static_cast<size_t>(y) * static_cast<size_t>(p.img_w) + static_cast<size_t>(x)) * static_cast<size_t>(p.img_channels);
                        if (k >= pic.pix.size() || !std::isfinite(pic.pix[k])) continue;
                        std::string fill;
                        if (p.img_channels == 3) {
                            const auto ch = [&](size_t c) {
                                const float v = std::isfinite(pic.pix[k + c]) ? pic.pix[k + c] : 0.0f;
                                return p.spec.image_invert ? 1.0f - v : v;
                            };
                            fill = HexColour(ch(0), ch(1), ch(2));
                        } else {
                            const double v = p.spec.image_invert ? 1.0 - pic.pix[k] : pic.pix[k];
                            fill = p.spec.image_grey ? HexColour(static_cast<float>(v), static_cast<float>(v), static_cast<float>(v))
                                                     : Mix(st.background, st.scale_high, v);
                        }
                        o << "<rect x=\"" << ox + x * px << "\" y=\"" << oy + y * px << "\" width=\"" << px + 0.05 << "\" height=\"" << px + 0.05
                          << "\" fill=\"" << fill << "\"/>";
                    }
                std::string caption = pic.label;
                if (pic.row > 0) caption = "row " + std::to_string(pic.row) + (caption.empty() ? "" : " \xC2\xB7 " + caption);
                else caption += " (" + Num(static_cast<double>(pic.count)) + ")";
                o << "\n<text x=\"" << ox << "\" y=\"" << oy + cell + 6 << "\" fill=\"" << st.text_dim << "\" font-size=\"10\">" << Xml(caption) << "</text>\n";
            }
            break;
        }
        case Kind::PairPlot: {
            const size_t n = p.multi_cols.size();
            const double cell = std::min(f.width, f.height) / static_cast<double>(std::max<size_t>(1, n));
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const double ox = f.left + static_cast<double>(j) * cell, oy = f.top + static_cast<double>(i) * cell, in = cell - 6;
                    o << "<rect x=\"" << ox << "\" y=\"" << oy << "\" width=\"" << in << "\" height=\"" << in << "\" fill=\"none\" stroke=\"" << st.grid << "\"/>\n";
                    const auto sx = [&](double v) { return ox + (v - p.multi_lo[j]) / (p.multi_hi[j] - p.multi_lo[j]) * in; };
                    if (i == j) {
                        double peak = 0;
                        for (const auto& g : p.pair_diag[i]) for (double v : g) peak = std::max(peak, v);
                        for (size_t g = 0; g < p.pair_diag[i].size(); ++g) {
                            o << "<polyline fill=\"none\" stroke=\"" << colour(g) << "\" points=\"";
                            for (int k = 0; k < p.pair_steps; ++k) {
                                const double at = p.multi_lo[j] + (p.multi_hi[j] - p.multi_lo[j]) * k / std::max(1, p.pair_steps - 1);
                                o << sx(at) << ',' << oy + in - (peak > 0 ? p.pair_diag[i][g][static_cast<size_t>(k)] / peak : 0) * (in - 4) << ' ';
                            }
                            o << "\"/>\n";
                        }
                        o << "<text x=\"" << ox + 3 << "\" y=\"" << oy + 12 << "\" fill=\"" << st.text_dim << "\" font-size=\"10\">" << Xml(p.multi_cols[i]) << "</text>\n";
                    } else {
                        for (size_t r = 0; r < p.multi_group.size(); ++r) {
                            const double vy = oy + in - (p.multi_values[i][r] - p.multi_lo[i]) / (p.multi_hi[i] - p.multi_lo[i]) * in;
                            o << "<circle cx=\"" << sx(p.multi_values[j][r]) << "\" cy=\"" << vy << "\" r=\"1.2\" fill=\"" << colour(static_cast<size_t>(std::max(0, p.multi_group[r]))) << "\" fill-opacity=\"0.7\"/>";
                        }
                        o << "\n";
                    }
                }
            break;
        }
        case Kind::Parallel: {
            const size_t n = p.multi_cols.size();
            const auto ax = [&](size_t c) { return f.left + 20 + (f.width - 40) * static_cast<double>(c) / static_cast<double>(std::max<size_t>(1, n - 1)); };
            const auto ay = [&](size_t c, double v) { return f.top + 14 + (f.height - 34) * (1.0 - (v - p.multi_lo[c]) / (p.multi_hi[c] - p.multi_lo[c])); };
            for (size_t r = 0; r < p.multi_group.size(); ++r) {
                o << "<polyline fill=\"none\" stroke=\"" << colour(static_cast<size_t>(std::max(0, p.multi_group[r]))) << "\" stroke-opacity=\"0.45\" stroke-width=\"0.8\" points=\"";
                for (size_t c = 0; c < n; ++c) o << ax(c) << ',' << ay(c, p.multi_values[c][r]) << ' ';
                o << "\"/>\n";
            }
            for (size_t c = 0; c < n; ++c) {
                o << "<line x1=\"" << ax(c) << "\" y1=\"" << f.top + 14 << "\" x2=\"" << ax(c) << "\" y2=\"" << f.top + f.height - 20 << "\" stroke=\"" << st.text_dim << "\"/>"
                  << "<text x=\"" << ax(c) << "\" y=\"" << f.top + 8 << "\" fill=\"" << st.text_dim << "\" font-size=\"10\" text-anchor=\"middle\">" << Num(p.multi_hi[c]) << "</text>"
                  << "<text x=\"" << ax(c) << "\" y=\"" << f.top + f.height - 6 << "\" fill=\"" << st.text << "\" text-anchor=\"middle\">" << Xml(p.multi_cols[c]) << "</text>\n";
            }
            break;
        }
        case Kind::Quiver:
            for (size_t i : p.q_drawn) {
                const double x0 = f.X(p.qx[i]), y0 = f.Y(p.qy[i]);
                const double x1 = f.X(p.qx[i] + p.qu[i] * p.q_scale), y1 = f.Y(p.qy[i] + p.qv[i] * p.q_scale);
                const double len = std::hypot(x1 - x0, y1 - y0);
                if (len < 0.5) continue;
                const double ux = (x1 - x0) / len, uy = (y1 - y0) / len, hs = std::min(5.0, len * 0.45);
                const std::string c = RangeColour(std::hypot(p.qu[i], p.qv[i]), p.grid_lo, p.grid_hi, false, st);
                o << "<line x1=\"" << x0 << "\" y1=\"" << y0 << "\" x2=\"" << x1 << "\" y2=\"" << y1 << "\" stroke=\"" << c << "\" stroke-width=\"1.3\"/>"
                  << "<polygon fill=\"" << c << "\" points=\"" << x1 << ',' << y1 << ' ' << x1 - ux * hs - uy * hs * 0.6 << ',' << y1 - uy * hs + ux * hs * 0.6 << ' '
                  << x1 - ux * hs + uy * hs * 0.6 << ',' << y1 - uy * hs - ux * hs * 0.6 << "\"/>\n";
            }
            break;
        case Kind::Stream:
            for (size_t l = 0; l < p.stream_lines.size(); ++l) {
                const auto& ln = p.stream_lines[l];
                o << "<polyline fill=\"none\" stroke=\"" << RangeColour(l < p.stream_speed.size() ? p.stream_speed[l] : 0.0, p.grid_lo, p.grid_hi, false, st)
                  << "\" stroke-width=\"1.3\" points=\"";
                for (size_t k = 0; k + 1 < ln.size(); k += 2) o << f.X(ln[k]) << ',' << f.Y(ln[k + 1]) << ' ';
                o << "\"/>\n";
            }
            break;
        case Kind::Hexbin:
            for (size_t i = 0; i < p.hex_x.size(); ++i) {
                const double v = p.spec.log_colour ? std::log1p(p.hex_v[i]) : p.hex_v[i];
                const double hi = p.spec.log_colour ? std::log1p(p.grid_hi) : p.grid_hi;
                const double lo = p.spec.log_colour ? std::log1p(p.grid_lo) : p.grid_lo;
                const double vx[6] = {0.5, 0.5, 0.0, -0.5, -0.5, 0.0}, vy[6] = {-1.0 / 6, 1.0 / 6, 1.0 / 3, 1.0 / 6, -1.0 / 6, -1.0 / 3};
                o << "<polygon fill=\"" << RangeColour(v, lo, hi, false, st) << "\" points=\"";
                for (int k = 0; k < 6; ++k) o << f.X(p.hex_x[i] + vx[k] * p.hex_sx) << ',' << f.Y(p.hex_y[i] + vy[k] * p.hex_sy) << ' ';
                o << "\"/>\n";
            }
            break;
        case Kind::Contour:
        case Kind::FilledContour: {
            if (p.spec.kind == Kind::FilledContour && p.band_rows > 0) {
                const double dx = (p.x_max - p.x_min) / p.band_cols, dy = (p.y_max - p.y_min) / p.band_rows;
                for (int r = 0; r < p.band_rows; ++r)
                    for (int c = 0; c < p.band_cols; ++c) {
                        const double v = p.band_grid[static_cast<size_t>(r * p.band_cols + c)];
                        if (!std::isfinite(v)) continue;
                        const double top = p.y_max - dy * r;
                        o << "<rect x=\"" << f.X(p.x_min + dx * c) << "\" y=\"" << f.Y(top) << "\" width=\"" << f.X(p.x_min + dx * (c + 1)) - f.X(p.x_min + dx * c) + 0.5
                          << "\" height=\"" << f.Y(top - dy) - f.Y(top) + 0.5 << "\" fill=\"" << RangeColour(v, p.grid_lo, p.grid_hi, false, st) << "\"/>\n";
                    }
            }
            for (size_t l = 0; l < p.contour_segments.size(); ++l) {
                const std::string c = p.spec.kind == Kind::FilledContour ? st.background
                                                                         : RangeColour(p.contour_levels[l], p.grid_lo, p.grid_hi, false, st);
                const auto& seg = p.contour_segments[l];
                o << "<path fill=\"none\" stroke=\"" << c << "\" stroke-width=\"" << (p.spec.kind == Kind::FilledContour ? 0.8 : 1.6) << "\" d=\"";
                for (size_t k = 0; k + 3 < seg.size(); k += 4)
                    o << 'M' << f.X(seg[k]) << ' ' << f.Y(seg[k + 1]) << 'L' << f.X(seg[k + 2]) << ' ' << f.Y(seg[k + 3]);
                o << "\"/>\n";
            }
            break;
        }
        case Kind::Heatmap:
        case Kind::Matrix:
        case Kind::Histogram2D: {
            const bool names = p.spec.kind != Kind::Histogram2D;
            const double x0 = names ? f.r.x0 : p.x_min, x1 = names ? f.r.x0 + p.grid_cols : p.x_max;
            const double y1 = names ? f.r.y1 : p.y_max, y0 = names ? f.r.y1 - p.grid_rows : p.y_min;
            const double dx = (x1 - x0) / std::max(1, p.grid_cols), dy = (y1 - y0) / std::max(1, p.grid_rows);
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const double v = p.grid[static_cast<size_t>(r * p.grid_cols + c)];
                    const double top = y1 - dy * r;
                    o << "<rect x=\"" << f.X(x0 + dx * c) << "\" y=\"" << f.Y(top) << "\" width=\"" << f.X(x0 + dx * (c + 1)) - f.X(x0 + dx * c) << "\" height=\"" << f.Y(top - dy) - f.Y(top) << "\" fill=\""
                      << RangeColour(v, p.grid_lo, p.grid_hi, p.grid_diverging, st) << "\"/>\n";
                }
            break;
        }
        case Kind::Confusion: {
            // Cells: the count, and the share under it unless counts are shown.
            const double x0 = f.r.x0, y1 = f.r.y1;
            const bool share = p.spec.confusion_show != PlotSpec::ConfusionShow::Counts;
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const size_t k = static_cast<size_t>(r * p.grid_cols + c);
                    const double top = y1 - r;
                    const double t = p.grid_hi > p.grid_lo ? (p.grid[k] - p.grid_lo) / (p.grid_hi - p.grid_lo) : 0.0;
                    o << "<rect x=\"" << f.X(x0 + c) << "\" y=\"" << f.Y(top) << "\" width=\"" << f.X(x0 + c + 1) - f.X(x0 + c) << "\" height=\""
                      << f.Y(top - 1) - f.Y(top) << "\" fill=\"" << RangeColour(p.grid[k], p.grid_lo, p.grid_hi, false, st) << "\"/>\n";
                    if (p.grid_rows <= 20) {
                        const double cx = f.X(x0 + c + 0.5), cy = f.Y(top - 0.5);
                        const std::string ink = t > 0.55 ? st.background : st.text;
                        o << "<text x=\"" << cx << "\" y=\"" << cy + (share ? -2 : 4) << "\" fill=\"" << ink << "\" text-anchor=\"middle\" font-weight=\"600\">"
                          << Num(p.grid_counts[k]) << "</text>\n";
                        if (share)
                            o << "<text x=\"" << cx << "\" y=\"" << cy + 13 << "\" fill=\"" << ink << "\" text-anchor=\"middle\" font-size=\"10\">"
                              << Num(std::round(p.grid[k] * 1000.0) / 10.0) << "%</text>\n";
                    }
                }
            break;
        }
        case Kind::Roc:
        case Kind::PrCurve:
        case Kind::Calibration:
            if (p.spec.kind == Kind::PrCurve && std::isfinite(p.baseline))
                o << "<line x1=\"" << f.left << "\" y1=\"" << f.Y(p.baseline) << "\" x2=\"" << f.left + f.width << "\" y2=\"" << f.Y(p.baseline)
                  << "\" stroke=\"" << st.text_dim << "\" stroke-dasharray=\"4 4\"/>\n";
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                if (p.spec.kind == Kind::PrCurve) {  // a step down at each threshold
                    o << "<path fill=\"none\" stroke=\"" << colour(0) << "\" stroke-width=\"1.8\" d=\"";
                    for (size_t k = 0; k < s.x.size(); ++k) o << (k == 0 ? "M" : " H") << f.X(s.x[k]) << (k == 0 ? " " : " V") << f.Y(s.y[k]);
                    o << "\"/>\n";
                } else {
                    o << "<polyline fill=\"none\" stroke=\"" << colour(0) << "\" stroke-width=\"1.8\" points=\"" << Points(f, s.x, s.y) << "\"/>\n";
                }
                if (p.spec.kind == Kind::Calibration)
                    for (size_t k = 0; k < s.x.size(); ++k)
                        o << "<circle cx=\"" << f.X(s.x[k]) << "\" cy=\"" << f.Y(s.y[k]) << "\" r=\"3\" fill=\"" << colour(0) << "\"/>\n";
            }
            break;
        case Kind::Residuals:
            o << "<line x1=\"" << f.left << "\" y1=\"" << f.Y(0) << "\" x2=\"" << f.left + f.width << "\" y2=\"" << f.Y(0) << "\" stroke=\"" << st.text_dim << "\"/>\n";
            if (!p.series.empty())
                for (size_t k = 0; k < std::min(p.series[0].x.size(), p.series[0].y.size()); ++k)
                    o << "<circle cx=\"" << f.X(p.series[0].x[k]) << "\" cy=\"" << f.Y(p.series[0].y[k]) << "\" r=\"2\" fill=\"" << colour(0) << "\" fill-opacity=\"0.6\"/>\n";
            break;
        case Kind::LearningCurve:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                if (s.low.size() == s.x.size() && !s.x.empty()) {
                    std::vector<double> bx(s.x.rbegin(), s.x.rend()), by(s.low.rbegin(), s.low.rend());
                    o << "<polygon fill=\"" << colour(i) << "\" fill-opacity=\"0.2\" points=\"" << Points(f, s.x, s.high) << Points(f, bx, by) << "\"/>\n";
                }
                o << "<polyline fill=\"none\" stroke=\"" << colour(i) << "\" stroke-width=\"1.8\" points=\"" << Points(f, s.x, s.y) << "\"/>\n";
                if (static_cast<int>(i) == p.best_series && p.best_index < s.x.size())
                    o << "<circle cx=\"" << f.X(s.x[p.best_index]) << "\" cy=\"" << f.Y(s.y[p.best_index]) << "\" r=\"5\" fill=\"none\" stroke=\"" << st.text << "\" stroke-width=\"1.5\"/>\n";
            }
            break;
        case Kind::Importance:
            if (!p.series.empty()) {
                const auto& s = p.series[0];
                const size_t n = s.y.size();
                for (size_t k = 0; k < n; ++k) {
                    const double y = static_cast<double>(n - 1 - k);
                    const double a = f.X(std::min(0.0, s.y[k])), b = f.X(std::max(0.0, s.y[k]));
                    o << "<rect x=\"" << a << "\" y=\"" << f.Y(y + 0.33) << "\" width=\"" << b - a << "\" height=\"" << f.Y(y - 0.33) - f.Y(y + 0.33)
                      << "\" fill=\"" << colour(0) << "\"/>\n";
                    if (k < s.low.size() && s.low[k] > 0)
                        o << "<line x1=\"" << f.X(s.y[k] - s.low[k]) << "\" y1=\"" << f.Y(y) << "\" x2=\"" << f.X(s.y[k] + s.high[k]) << "\" y2=\"" << f.Y(y)
                          << "\" stroke=\"" << st.text << "\" stroke-width=\"1.2\"/>\n";
                }
            }
            break;
    }
    o << "</g>\n";

    // P2b group 5.
    const auto map_outline = [&](bool regions) {
        const auto& countries = WorldCountries();
        o << "<g clip-path=\"url(#area)\">\n";
        for (size_t i = 0; i < countries.size(); ++i) {
            std::string fill = st.grid;
            if (regions && i < p.region_value.size() && std::isfinite(p.region_value[i])) {
                double v = p.region_value[i], lo = p.grid_lo, hi = p.grid_hi;
                if (p.spec.log_colour) {
                    v = std::log10(std::max(v, lo));
                    lo = std::log10(lo);
                    hi = std::log10(hi);
                }
                fill = RangeColour(v, lo, hi, false, st);
            }
            for (const auto& ring : countries[i].rings) {
                o << "<path fill=\"" << fill << "\" stroke=\"" << st.background << "\" stroke-width=\"0.5\" d=\"";
                for (size_t k = 0; k + 1 < ring.size(); k += 2) o << (k == 0 ? 'M' : 'L') << f.X(ring[k]) << ',' << f.Y(ring[k + 1]);
                o << "Z\"/>\n";
            }
        }
        o << "</g>\n";
    };
    switch (p.spec.kind) {
        case Kind::MapRegions: map_outline(true); break;
        case Kind::MapPoints: {
            map_outline(false);
            o << "<g clip-path=\"url(#area)\">\n";
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& sr = p.series[i];
                for (size_t k = 0; k < std::min(sr.x.size(), sr.y.size()); ++k) {
                    double r = 2.5;
                    if (k < sr.z.size() && std::isfinite(sr.z[k]))
                        r = 2.0 + 8.0 * std::sqrt(std::clamp((sr.z[k] - p.size_min) / (p.size_max - p.size_min), 0.0, 1.0));
                    const std::string fill = p.colour_scale ? ScaleColour(p, st, k < sr.c.size() ? sr.c[k] : NAN) : colour(i);
                    o << "<circle cx=\"" << f.X(sr.x[k]) << "\" cy=\"" << f.Y(sr.y[k]) << "\" r=\"" << r << "\" fill=\"" << fill
                      << "\" fill-opacity=\"0.75\"/>\n";
                }
            }
            o << "</g>\n";
            break;
        }
        case Kind::Sankey: {
            const size_t steps = p.sankey_steps.size();
            const double node_w = 12, label_w = 140;
            const auto sx = [&](int step) {
                return f.left + (steps > 1 ? static_cast<double>(step) / static_cast<double>(steps - 1) : 0.0) * (f.width - node_w - label_w);
            };
            const auto sy = [&](double y) { return f.top + y * f.height; };
            std::vector<int> first_of_step(steps, -1);
            for (size_t i = 0; i < p.sankey_nodes.size(); ++i)
                if (first_of_step[static_cast<size_t>(p.sankey_nodes[i].step)] < 0) first_of_step[static_cast<size_t>(p.sankey_nodes[i].step)] = static_cast<int>(i);
            for (const auto& l : p.sankey_links) {
                const auto& a = p.sankey_nodes[static_cast<size_t>(l.from)];
                const auto& b = p.sankey_nodes[static_cast<size_t>(l.to)];
                const double x0 = sx(a.step) + node_w, x1 = sx(b.step), mx = (x0 + x1) / 2;
                const double y0 = sy(l.y_from), y1 = sy(l.y_to), t = l.thickness * f.height;
                const size_t c = a.step == 0 ? static_cast<size_t>(l.from) : static_cast<size_t>(l.from - first_of_step[static_cast<size_t>(a.step)]) + 3;
                o << "<path fill=\"" << colour(c) << "\" fill-opacity=\"0.35\" d=\"M" << x0 << ',' << y0 << " C" << mx << ',' << y0 << ' ' << mx << ','
                  << y1 << ' ' << x1 << ',' << y1 << " L" << x1 << ',' << y1 + t << " C" << mx << ',' << y1 + t << ' ' << mx << ',' << y0 + t << ' ' << x0
                  << ',' << y0 + t << " Z\"/>\n";
            }
            for (const auto& nd : p.sankey_nodes) {
                const double x = sx(nd.step), y0 = sy(nd.y0), y1 = sy(nd.y1);
                o << "<rect x=\"" << x << "\" y=\"" << y0 << "\" width=\"" << node_w << "\" height=\"" << std::max(1.0, y1 - y0) << "\" fill=\"" << st.text << "\"/>"
                  << "<text x=\"" << x + node_w + 5 << "\" y=\"" << (y0 + y1) / 2 + 4 << "\" fill=\"" << st.text << "\">" << Xml(nd.name) << " <tspan fill=\""
                  << st.text_dim << "\">" << Num(nd.value) << "</tspan></text>\n";
            }
            for (size_t k = 0; k < steps; ++k)
                o << "<text x=\"" << sx(static_cast<int>(k)) << "\" y=\"" << f.top + f.height + 18 << "\" fill=\"" << st.text_dim << "\">" << Xml(p.sankey_steps[k]) << "</text>\n";
            break;
        }
        case Kind::Treemap: {
            const auto rects = TreemapLayout(p, {}, f.width, f.height, 16.0);
            for (const auto& r : rects) {
                const double x = f.left + r.x, y = f.top + r.y;
                if (!r.leaf) {
                    o << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << r.w << "\" height=\"" << r.h << "\" fill=\"" << st.grid << "\"/>\n";
                    if (r.h > 16 * 2.6 && r.w > 48)
                        o << "<text x=\"" << x + 4 << "\" y=\"" << y + 12 << "\" fill=\"" << st.text << "\" font-size=\"11\" font-weight=\"600\">"
                          << Xml(r.path.back()) << "</text>\n";
                    continue;
                }
                const std::string fill = p.colour_scale ? ScaleColour(p, st, r.colour) : colour(static_cast<size_t>(r.top));
                o << "<rect x=\"" << x + 0.5 << "\" y=\"" << y + 0.5 << "\" width=\"" << std::max(0.0, r.w - 1) << "\" height=\"" << std::max(0.0, r.h - 1)
                  << "\" fill=\"" << fill << "\"/>\n";
                if (r.w > 50 && r.h > 18)
                    o << "<text x=\"" << x + 4 << "\" y=\"" << y + 13 << "\" fill=\"" << st.background << "\" font-size=\"10.5\">" << Xml(r.path.back()) << "</text>\n";
            }
            break;
        }
        default: break;
    }

    // Model results: their figures top left (AUC, RMSE, ...).
    if (!p.metrics.empty()) {
        double y = f.top + 18;
        for (const auto& [name, v] : p.metrics) {
            o << "<text x=\"" << f.left + 10 << "\" y=\"" << y << "\" fill=\"" << st.text << "\">" << Xml(name) << " " << Num(std::round(v * 1000.0) / 1000.0) << "</text>\n";
            y += 16;
        }
    }

    // Legend: the series names, top right.
    if (p.spec.legend && (p.spec.kind == Kind::PairPlot || p.spec.kind == Kind::Parallel) && p.multi_groups.size() > 1) {
        double y = f.top + 16;
        for (size_t i = 0; i < p.multi_groups.size() && i < 12; ++i, y += 18)
            o << "<rect x=\"" << st.width - 150 << "\" y=\"" << y - 9 << "\" width=\"10\" height=\"10\" fill=\"" << colour(i) << "\"/>"
              << "<text x=\"" << st.width - 134 << "\" y=\"" << y << "\" fill=\"" << st.text << "\">" << Xml(p.multi_groups[i]) << "</text>\n";
    }
    if (p.spec.legend && !pie && p.series.size() > 1) {
        double y = f.top + 16;
        for (size_t i = 0; i < p.series.size() && i < 12; ++i, y += 18) {
            o << "<rect x=\"" << f.left + f.width - 150 << "\" y=\"" << y - 9 << "\" width=\"10\" height=\"10\" fill=\"" << colour(i) << "\"/>"
              << "<text x=\"" << f.left + f.width - 134 << "\" y=\"" << y << "\" fill=\"" << st.text << "\">" << Xml(p.series[i].label) << "</text>\n";
        }
    }
    o << "</svg>\n";
    return o.str();
}

}  // namespace cyxwiz::plot
