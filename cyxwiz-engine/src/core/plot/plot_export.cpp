#include "plot_export.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

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

std::string ToCsv(const Prepared& p) {
    std::ostringstream out;
    switch (p.spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
        case Kind::Scatter:
            out << "series,x,y\n";
            for (const auto& s : p.series) {
                const auto& xs = s.all_x.empty() ? s.x : s.all_x;
                const auto& ys = s.all_y.empty() ? s.y : s.all_y;
                for (size_t i = 0; i < std::min(xs.size(), ys.size()); ++i)
                    out << CsvCell(s.label) << ',' << Num(xs[i]) << ',' << Num(ys[i]) << '\n';
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
            out << "row,column,count\n";
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c)
                    out << CsvCell(r < static_cast<int>(p.row_names.size()) ? p.row_names[static_cast<size_t>(r)] : "") << ','
                        << CsvCell(c < static_cast<int>(p.col_names.size()) ? p.col_names[static_cast<size_t>(c)] : "") << ','
                        << Num(p.grid[static_cast<size_t>(r * p.grid_cols + c)]) << '\n';
            break;
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

}  // namespace

std::string ToSvg(const Prepared& p, const AxisRange& range, const SvgStyle& st) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(1);
    const bool title = !p.spec.title.empty();
    const bool pie = p.spec.kind == Kind::Pie;
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
    const bool categorical_x = p.spec.kind == Kind::Bar || p.spec.kind == Kind::ErrorBars || p.spec.kind == Kind::Heatmap;
    if (!pie) {
        if (categorical_x) {
            const auto& names = p.spec.kind == Kind::Heatmap ? p.col_names : p.categories;
            const size_t every = names.size() > 30 ? names.size() / 30 + 1 : 1;
            for (size_t i = 0; i < names.size(); i += every) {
                const double x = p.spec.kind == Kind::Heatmap ? f.r.x0 + (static_cast<double>(i) + 0.5) : static_cast<double>(i);
                o << "<text x=\"" << f.X(x) << "\" y=\"" << f.top + f.height + 16 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(names[i]) << "</text>\n";
            }
        } else {
            for (double t : Ticks(f.r.x0, f.r.x1)) {
                o << "<line x1=\"" << f.X(t) << "\" y1=\"" << f.top << "\" x2=\"" << f.X(t) << "\" y2=\"" << f.top + f.height << "\" stroke=\"" << st.grid << "\"/>\n";
                o << "<text x=\"" << f.X(t) << "\" y=\"" << f.top + f.height + 16 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Num(t) << "</text>\n";
            }
        }
        if (p.spec.kind == Kind::Heatmap) {
            for (size_t i = 0; i < p.row_names.size(); ++i)
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
        std::string yl = p.spec.y_label;
        if (yl.empty()) yl = p.spec.kind == Kind::Histogram ? (p.spec.density ? "density" : "count")
                             : !p.spec.y_columns.empty() ? p.spec.y_columns.front() : "";
        o << "<text x=\"" << f.left + f.width / 2 << "\" y=\"" << st.height - 10 << "\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(xl) << "</text>\n";
        o << "<text transform=\"translate(16," << f.top + f.height / 2 << ") rotate(-90)\" fill=\"" << st.text_dim << "\" text-anchor=\"middle\">" << Xml(yl) << "</text>\n";
    }

    o << "<g clip-path=\"url(#area)\">\n";
    switch (p.spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem:
        case Kind::Scatter:
            for (size_t i = 0; i < p.series.size(); ++i) {
                const auto& s = p.series[i];
                const std::string c = colour(i);
                if (p.spec.kind == Kind::Scatter) {
                    for (size_t k = 0; k < std::min(s.x.size(), s.y.size()); ++k)
                        o << "<circle cx=\"" << f.X(s.x[k]) << "\" cy=\"" << f.Y(s.y[k]) << "\" r=\"2\" fill=\"" << c << "\" fill-opacity=\"0.7\"/>\n";
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
                    if (p.spec.kind == Kind::Area && !s.x.empty())
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
            break;
        }
        case Kind::Heatmap:
        case Kind::Histogram2D: {
            double peak = 0;
            for (double v : p.grid) peak = std::max(peak, v);
            const double x0 = p.spec.kind == Kind::Heatmap ? f.r.x0 : p.x_min, x1 = p.spec.kind == Kind::Heatmap ? f.r.x0 + p.grid_cols : p.x_max;
            const double y1 = p.spec.kind == Kind::Heatmap ? f.r.y1 : p.y_max, y0 = p.spec.kind == Kind::Heatmap ? f.r.y1 - p.grid_rows : p.y_min;
            const double dx = (x1 - x0) / std::max(1, p.grid_cols), dy = (y1 - y0) / std::max(1, p.grid_rows);
            for (int r = 0; r < p.grid_rows; ++r)
                for (int c = 0; c < p.grid_cols; ++c) {
                    const double v = p.grid[static_cast<size_t>(r * p.grid_cols + c)];
                    const double top = y1 - dy * r;
                    o << "<rect x=\"" << f.X(x0 + dx * c) << "\" y=\"" << f.Y(top) << "\" width=\"" << f.X(x0 + dx * (c + 1)) - f.X(x0 + dx * c) << "\" height=\"" << f.Y(top - dy) - f.Y(top) << "\" fill=\""
                      << Mix(st.scale_low, st.scale_high, peak > 0 ? v / peak : 0) << "\"/>\n";
                }
            break;
        }
    }
    o << "</g>\n";

    // Legend: the series names, top right.
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
