#include "plot_prepare.h"

#include "../series_decimation.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>

namespace cyxwiz::plot {

const SourceColumn* Source::Find(const std::string& name) const {
    for (const auto& c : columns)
        if (c.name == name) return &c;
    return nullptr;
}

namespace {

std::string CellText(const SourceColumn& c, size_t row) {
    if (!c.numeric) return row < c.text.size() ? c.text[row] : std::string();
    if (row >= c.numbers.size() || !std::isfinite(c.numbers[row])) return "missing";
    std::ostringstream out;
    out << c.numbers[row];
    return out.str();
}

double Number(const SourceColumn* c, size_t row) {
    if (!c) return static_cast<double>(row);  // no X column: the row number
    return row < c->numbers.size() ? c->numbers[row] : NAN;
}

// Rows grouped by the colour column's value: the largest groups keep their
// own series, the rest share "other". One group ("") without a column.
struct Groups {
    std::vector<std::string> names;
    std::vector<int> of_row;  // group index per row
};

Groups GroupRows(const SourceColumn* colour, size_t rows) {
    Groups g;
    g.of_row.assign(rows, 0);
    if (!colour) {
        g.names = {""};
        return g;
    }
    std::vector<std::string> order;
    std::unordered_map<std::string, size_t> count;
    std::vector<std::string> key(rows);
    for (size_t r = 0; r < rows; ++r) {
        key[r] = CellText(*colour, r);
        if (count[key[r]]++ == 0) order.push_back(key[r]);
    }
    std::vector<std::string> kept = order;
    if (kept.size() > kMaxColorGroups) {
        std::stable_sort(kept.begin(), kept.end(), [&](const auto& a, const auto& b) { return count[a] > count[b]; });
        kept.resize(kMaxColorGroups - 1);
        // Keep first-appearance order among the kept groups.
        std::vector<std::string> ordered;
        for (const auto& k : order)
            if (std::find(kept.begin(), kept.end(), k) != kept.end()) ordered.push_back(k);
        kept = ordered;
        kept.push_back("other");
    }
    std::unordered_map<std::string, int> index;
    for (size_t i = 0; i < kept.size(); ++i) index[kept[i]] = static_cast<int>(i);
    const int other = index.count("other") ? index["other"] : -1;
    for (size_t r = 0; r < rows; ++r) {
        auto it = index.find(key[r]);
        g.of_row[r] = it != index.end() ? it->second : other;
    }
    g.names = kept;
    return g;
}

std::string SeriesName(const std::string& column, const std::string& group, bool many_columns) {
    if (group.empty()) return column;
    return many_columns ? column + " \xC2\xB7 " + group : group;
}

// Line-like kinds: one series per Y column and colour group, reduced when long.
void PrepareLines(Prepared& p, const Source& src, const SourceColumn* xcol, const std::vector<const SourceColumn*>& ys,
                  const Groups& groups) {
    size_t shown = 0, total = 0;
    for (const auto* y : ys) {
        for (size_t g = 0; g < groups.names.size(); ++g) {
            std::vector<double> xs, vs;
            for (size_t r = 0; r < y->numbers.size(); ++r) {
                if (groups.of_row[r] != static_cast<int>(g)) continue;
                const double x = Number(xcol, r), v = y->numbers[r];
                if (!std::isfinite(x) || !std::isfinite(v)) continue;
                xs.push_back(x);
                vs.push_back(v);
            }
            if (xs.empty()) continue;
            Series s;
            s.label = SeriesName(y->name, groups.names[g], ys.size() > 1);
            total += xs.size();
            if (p.spec.smooth > 1 && static_cast<int>(vs.size()) >= p.spec.smooth) {
                auto d = series::MinMaxDecimate(xs, series::MovingAverage(vs, p.spec.smooth), kMaxLinePoints);
                s.smooth_x = std::move(d.x);
                s.smooth_y = std::move(d.y);
            }
            s.x_sorted = std::is_sorted(xs.begin(), xs.end());
            auto d = series::MinMaxDecimate(xs, vs, kMaxLinePoints);
            shown += d.x.size();
            s.x = std::move(d.x);
            s.y = std::move(d.y);
            if (s.x.size() < xs.size()) {
                s.all_x = std::move(xs);
                s.all_y = std::move(vs);
            }
            p.series.push_back(std::move(s));
        }
    }
    (void)src;
    if (shown < total) p.label = {DataLabel::State::Reduced, shown, total};
    else p.label = {DataLabel::State::Exact, total, total};
}

void PrepareScatter(Prepared& p, const SourceColumn* xcol, const std::vector<const SourceColumn*>& ys,
                    const Groups& groups) {
    // Rows with a finite x and y, sampled evenly and the same way each time.
    std::vector<size_t> rows;
    const size_t n = xcol->numbers.size();
    for (size_t r = 0; r < n; ++r)
        if (std::isfinite(xcol->numbers[r])) rows.push_back(r);
    const size_t total = rows.size();
    const std::vector<size_t> all_rows = rows;
    if (rows.size() > kMaxScatterPoints) {
        std::mt19937_64 rng(0x5eed);
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(kMaxScatterPoints);
        std::sort(rows.begin(), rows.end());
        p.label = {DataLabel::State::Sampled, rows.size(), total};
    } else {
        p.label = {DataLabel::State::Exact, total, total};
    }
    for (const auto* y : ys) {
        for (size_t g = 0; g < groups.names.size(); ++g) {
            Series s;
            s.label = SeriesName(y->name, groups.names[g], ys.size() > 1);
            for (size_t r : rows) {
                if (groups.of_row[r] != static_cast<int>(g) || r >= y->numbers.size()) continue;
                if (!std::isfinite(y->numbers[r])) continue;
                s.x.push_back(xcol->numbers[r]);
                s.y.push_back(y->numbers[r]);
            }
            if (rows.size() < all_rows.size()) {
                for (size_t r : all_rows) {
                    if (groups.of_row[r] != static_cast<int>(g) || r >= y->numbers.size()) continue;
                    if (!std::isfinite(y->numbers[r])) continue;
                    s.all_x.push_back(xcol->numbers[r]);
                    s.all_y.push_back(y->numbers[r]);
                }
            }
            if (!s.x.empty()) p.series.push_back(std::move(s));
        }
    }
}

void PrepareHistogram(Prepared& p, const SourceColumn* xcol, const Groups& groups) {
    p.stats = Summarize(xcol->numbers);
    const int bins = std::max(1, p.spec.bins);
    double lo = p.stats.min, hi = p.stats.max;
    if (p.stats.count == 0) {
        p.problem = xcol->name + " has no numeric values.";
        return;
    }
    if (hi <= lo) {
        lo -= 0.5;
        hi += 0.5;
    }
    const double width = (hi - lo) / bins;
    p.edges.resize(static_cast<size_t>(bins) + 1);
    for (int i = 0; i <= bins; ++i) p.edges[static_cast<size_t>(i)] = lo + width * i;
    for (size_t g = 0; g < groups.names.size(); ++g) {
        Series s;
        s.label = groups.names[g].empty() ? xcol->name : groups.names[g];
        s.y.assign(static_cast<size_t>(bins), 0.0);
        size_t n = 0;
        for (size_t r = 0; r < xcol->numbers.size(); ++r) {
            const double v = xcol->numbers[r];
            if (groups.of_row[r] != static_cast<int>(g) || !std::isfinite(v)) continue;
            int b = static_cast<int>((v - lo) / width);
            b = std::clamp(b, 0, bins - 1);  // the maximum lands in the last bin
            s.y[static_cast<size_t>(b)] += 1.0;
            ++n;
        }
        if (n == 0) continue;
        if (p.spec.density)
            for (double& c : s.y) c /= static_cast<double>(n) * width;
        for (int i = 0; i < bins; ++i) s.x.push_back(lo + width * (i + 0.5));
        p.series.push_back(std::move(s));
    }
    p.label = {DataLabel::State::Exact, p.stats.count, p.stats.count};
}

// Bar and pie: rows per category (no Y) or the mean (bar) / sum (pie) of Y.
void PrepareCategories(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol) {
    std::vector<std::string> order;
    std::unordered_map<std::string, std::pair<double, size_t>> acc;  // sum, rows
    const size_t rows = xcol->size();
    for (size_t r = 0; r < rows; ++r) {
        const std::string k = CellText(*xcol, r);
        double v = 1.0;
        if (ycol) {
            v = r < ycol->numbers.size() ? ycol->numbers[r] : NAN;
            if (!std::isfinite(v)) continue;
        }
        auto& a = acc[k];
        if (a.second == 0) order.push_back(k);
        a.first += v;
        a.second += 1;
    }
    const bool pie = p.spec.kind == Kind::Pie;
    const size_t cap = pie ? kMaxPieSlices : kMaxCategories;
    std::vector<std::string> kept = order;
    if (kept.size() > cap) {
        std::stable_sort(kept.begin(), kept.end(), [&](const auto& a, const auto& b) { return acc[a].second > acc[b].second; });
        kept.resize(cap - 1);
        std::vector<std::string> ordered;
        for (const auto& k : order)
            if (std::find(kept.begin(), kept.end(), k) != kept.end()) ordered.push_back(k);
        kept = ordered;
    }
    Series s;
    s.label = ycol ? (pie ? "sum of " : "mean of ") + ycol->name : "rows";
    double other_sum = 0;
    size_t other_rows = 0;
    for (const auto& k : order) {
        if (std::find(kept.begin(), kept.end(), k) == kept.end()) {
            other_sum += acc[k].first;
            other_rows += acc[k].second;
        }
    }
    for (size_t i = 0; i < kept.size(); ++i) {
        const auto& a = acc[kept[i]];
        p.categories.push_back(kept[i]);
        s.x.push_back(static_cast<double>(i));
        s.y.push_back(!ycol ? static_cast<double>(a.second) : pie ? a.first : a.first / static_cast<double>(a.second));
    }
    if (other_rows > 0) {
        p.categories.push_back("other");
        s.x.push_back(static_cast<double>(kept.size()));
        s.y.push_back(!ycol ? static_cast<double>(other_rows) : pie ? other_sum : other_sum / static_cast<double>(other_rows));
    }
    p.series.push_back(std::move(s));
    size_t counted = 0;
    for (const auto& [k, a] : acc) counted += a.second;
    p.label = {DataLabel::State::Exact, counted, counted};
}

std::vector<std::pair<std::string, std::vector<double>>> ValueSets(const std::vector<const SourceColumn*>& ys,
                                                                   const Groups& groups) {
    std::vector<std::pair<std::string, std::vector<double>>> sets;
    for (const auto* y : ys) {
        for (size_t g = 0; g < groups.names.size(); ++g) {
            std::vector<double> v;
            for (size_t r = 0; r < y->numbers.size(); ++r)
                if (groups.of_row[r] == static_cast<int>(g) && std::isfinite(y->numbers[r])) v.push_back(y->numbers[r]);
            if (!v.empty()) sets.emplace_back(SeriesName(y->name, groups.names[g], ys.size() > 1), std::move(v));
        }
    }
    return sets;
}

Prepared::Box BoxOf(const std::vector<double>& values) {
    const ColumnStats st = Summarize(values);
    const double iqr = st.q3 - st.q1;
    const double lo_fence = st.q1 - 1.5 * iqr, hi_fence = st.q3 + 1.5 * iqr;
    double low = st.max, high = st.min;
    for (double v : values) {
        if (v >= lo_fence) low = std::min(low, v);
        if (v <= hi_fence) high = std::max(high, v);
    }
    return {low, st.q1, st.median, st.q3, high, st.mean};
}

void PrepareBoxes(Prepared& p, const std::vector<const SourceColumn*>& ys, const Groups& groups) {
    size_t total = 0;
    const auto sets = ValueSets(ys, groups);
    for (size_t i = 0; i < sets.size(); ++i) {
        const auto& [name, values] = sets[i];
        total += values.size();
        p.boxes.push_back(BoxOf(values));
        Series s;
        s.label = name;
        if (p.spec.kind == Kind::Violin) {
            // Gaussian kernel density, Silverman's bandwidth, drawn as
            // half-widths (0..0.4) around the set's position.
            const ColumnStats st = Summarize(values);
            double sd = 0;
            for (double v : values) sd += (v - st.mean) * (v - st.mean);
            sd = std::sqrt(sd / std::max<size_t>(1, values.size() - 1));
            const double spread = std::min(sd, (st.q3 - st.q1) / 1.34);
            double bw = 0.9 * (spread > 0 ? spread : (sd > 0 ? sd : 1.0)) * std::pow(static_cast<double>(values.size()), -0.2);
            if (bw <= 0) bw = 1.0;
            std::vector<double> dens(kViolinSteps);
            double peak = 0;
            for (int k = 0; k < kViolinSteps; ++k) {
                const double at = st.min + (st.max - st.min) * k / (kViolinSteps - 1);
                double sum = 0;
                for (double v : values) sum += std::exp(-0.5 * ((at - v) / bw) * ((at - v) / bw));
                dens[static_cast<size_t>(k)] = sum;
                peak = std::max(peak, sum);
                s.y.push_back(at);
            }
            for (int k = 0; k < kViolinSteps; ++k) {
                const double half = peak > 0 ? 0.4 * dens[static_cast<size_t>(k)] / peak : 0.0;
                s.low.push_back(static_cast<double>(i) - half);
                s.high.push_back(static_cast<double>(i) + half);
            }
        }
        p.series.push_back(std::move(s));
    }
    if (!ys.empty()) p.stats = Summarize(ys.front()->numbers);
    p.label = {DataLabel::State::Exact, total, total};
}

void PrepareErrorBars(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol) {
    std::vector<std::string> order;
    std::unordered_map<std::string, std::vector<double>> by;
    for (size_t r = 0; r < xcol->size(); ++r) {
        const double v = r < ycol->numbers.size() ? ycol->numbers[r] : NAN;
        if (!std::isfinite(v)) continue;
        const std::string k = CellText(*xcol, r);
        auto& vec = by[k];
        if (vec.empty()) order.push_back(k);
        vec.push_back(v);
    }
    if (order.size() > kMaxCategories) order.resize(kMaxCategories);
    Series s;
    s.label = "mean of " + ycol->name + " \xC2\xB1 1 sd";
    size_t total = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const auto& v = by[order[i]];
        total += v.size();
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
        double var = 0;
        for (double x : v) var += (x - mean) * (x - mean);
        const double sd = v.size() > 1 ? std::sqrt(var / static_cast<double>(v.size() - 1)) : 0.0;
        p.categories.push_back(order[i]);
        s.x.push_back(static_cast<double>(i));
        s.y.push_back(mean);
        s.low.push_back(sd);
        s.high.push_back(sd);
    }
    p.series.push_back(std::move(s));
    p.label = {DataLabel::State::Exact, total, total};
}

void PrepareHeatmap(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol) {
    std::vector<std::string> cols, rows;
    std::unordered_map<std::string, int> ci, ri;
    std::map<std::pair<int, int>, double> counts;
    const size_t n = std::min(xcol->size(), ycol->size());
    for (size_t r = 0; r < n; ++r) {
        const std::string cx = CellText(*xcol, r), cy = CellText(*ycol, r);
        if (!ci.count(cx)) {
            if (cols.size() >= 50) continue;
            ci[cx] = static_cast<int>(cols.size());
            cols.push_back(cx);
        }
        if (!ri.count(cy)) {
            if (rows.size() >= 50) continue;
            ri[cy] = static_cast<int>(rows.size());
            rows.push_back(cy);
        }
        counts[{ri[cy], ci[cx]}] += 1.0;
    }
    p.grid_rows = static_cast<int>(rows.size());
    p.grid_cols = static_cast<int>(cols.size());
    p.grid.assign(static_cast<size_t>(p.grid_rows) * static_cast<size_t>(p.grid_cols), 0.0);
    size_t total = 0;
    for (const auto& [rc, c] : counts) {
        p.grid[static_cast<size_t>(rc.first) * static_cast<size_t>(p.grid_cols) + static_cast<size_t>(rc.second)] = c;
        total += static_cast<size_t>(c);
    }
    p.row_names = rows;
    p.col_names = cols;
    p.label = {DataLabel::State::Exact, total, total};
}

void PrepareHistogram2D(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol) {
    const int bins = std::clamp(p.spec.bins, 1, 400);
    std::vector<std::pair<double, double>> pts;
    const size_t n = std::min(xcol->numbers.size(), ycol->numbers.size());
    for (size_t r = 0; r < n; ++r)
        if (std::isfinite(xcol->numbers[r]) && std::isfinite(ycol->numbers[r])) pts.emplace_back(xcol->numbers[r], ycol->numbers[r]);
    if (pts.empty()) {
        p.problem = "No rows have numbers in both " + xcol->name + " and " + ycol->name + ".";
        return;
    }
    double x0 = pts[0].first, x1 = x0, y0 = pts[0].second, y1 = y0;
    for (const auto& [x, y] : pts) {
        x0 = std::min(x0, x);
        x1 = std::max(x1, x);
        y0 = std::min(y0, y);
        y1 = std::max(y1, y);
    }
    if (x1 <= x0) { x0 -= 0.5; x1 += 0.5; }
    if (y1 <= y0) { y0 -= 0.5; y1 += 0.5; }
    p.grid_rows = p.grid_cols = bins;
    p.grid.assign(static_cast<size_t>(bins) * static_cast<size_t>(bins), 0.0);
    for (const auto& [x, y] : pts) {
        const int c = std::clamp(static_cast<int>((x - x0) / (x1 - x0) * bins), 0, bins - 1);
        // Row 0 is the top of the picture: the highest y.
        const int r = std::clamp(bins - 1 - static_cast<int>((y - y0) / (y1 - y0) * bins), 0, bins - 1);
        p.grid[static_cast<size_t>(r) * static_cast<size_t>(bins) + static_cast<size_t>(c)] += 1.0;
    }
    p.x_min = x0;
    p.x_max = x1;
    p.y_min = y0;
    p.y_max = y1;
    p.label = {DataLabel::State::Exact, pts.size(), pts.size()};
}

}  // namespace

Prepared Prepare(const PlotSpec& spec, const Source& src) {
    Prepared p;
    p.spec = spec;
    const KindInfo& kind = Info(spec.kind);
    p.problem = MissingEncoding(spec);
    if (!p.problem.empty()) return p;

    const auto column = [&](const std::string& name, bool must_be_numeric) -> const SourceColumn* {
        const SourceColumn* c = src.Find(name);
        if (!c) {
            p.problem = "Column '" + name + "' is not in the table.";
            return nullptr;
        }
        if (must_be_numeric && !c->numeric) {
            p.problem = name + " is not numeric.";
            return nullptr;
        }
        return c;
    };
    const bool numeric_x = spec.kind != Kind::Bar && spec.kind != Kind::Pie && spec.kind != Kind::ErrorBars &&
                           spec.kind != Kind::Heatmap;
    const SourceColumn* xcol = nullptr;
    if (!spec.x_column.empty() && (kind.required | kind.optional) & kEncX) {
        xcol = column(spec.x_column, numeric_x);
        if (!xcol) return p;
    }
    std::vector<const SourceColumn*> ys;
    if ((kind.required | kind.optional) & kEncY) {
        const size_t take = kind.multi_y ? spec.y_columns.size() : std::min<size_t>(1, spec.y_columns.size());
        for (size_t i = 0; i < take; ++i) {
            const SourceColumn* y = column(spec.y_columns[i], spec.kind != Kind::Heatmap);
            if (!y) return p;
            ys.push_back(y);
        }
    }
    const SourceColumn* colour = nullptr;
    if (!spec.color_column.empty() && (kind.optional & kEncColor)) {
        colour = column(spec.color_column, false);
        if (!colour) return p;
    }
    size_t rows = 0;
    if (xcol) rows = xcol->size();
    for (const auto* y : ys) rows = std::max(rows, y->size());
    const Groups groups = GroupRows(colour, rows);

    switch (spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem: PrepareLines(p, src, xcol, ys, groups); break;
        case Kind::Scatter: PrepareScatter(p, xcol, ys, groups); break;
        case Kind::Histogram: PrepareHistogram(p, xcol, groups); break;
        case Kind::Bar:
        case Kind::Pie: PrepareCategories(p, xcol, ys.empty() ? nullptr : ys.front()); break;
        case Kind::Box:
        case Kind::Violin: PrepareBoxes(p, ys, groups); break;
        case Kind::ErrorBars: PrepareErrorBars(p, xcol, ys.front()); break;
        case Kind::Heatmap: PrepareHeatmap(p, xcol, ys.front()); break;
        case Kind::Histogram2D: PrepareHistogram2D(p, xcol, ys.front()); break;
    }
    if (spec.kind != Kind::Histogram && spec.kind != Kind::Box && spec.kind != Kind::Violin && !ys.empty())
        p.stats = Summarize(ys.front()->numbers);
    // A source read with a row limit says so, whatever the kind did.
    if (src.row_limit > 0) p.label = {DataLabel::State::Truncated, src.row_limit, src.total_rows};
    if (p.problem.empty() && p.series.empty() && p.grid.empty()) p.problem = "No values to draw.";
    return p;
}

}  // namespace cyxwiz::plot
