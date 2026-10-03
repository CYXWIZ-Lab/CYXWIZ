#include "plot_prepare.h"

#include "../series_decimation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

size_t Source::Rows() const {
    size_t rows = 0;
    for (const auto& c : columns) rows = std::max(rows, c.size());
    return rows;
}

namespace {

std::string CellText(const SourceColumn& c, size_t row) {
    if (!c.numeric) return row < c.text.size() ? c.text[row] : std::string();
    if (row >= c.numbers.size() || !std::isfinite(c.numbers[row])) return "missing";
    std::ostringstream out;
    out << c.numbers[row];
    return out.str();
}

std::string Trim(const std::string& v) {
    const size_t a = v.find_first_not_of(" \t");
    if (a == std::string::npos) return "";
    return v.substr(a, v.find_last_not_of(" \t") - a + 1);
}

bool ParseNumber(const std::string& text, double& out) {
    if (text.empty()) return false;
    char* end = nullptr;
    out = std::strtod(text.c_str(), &end);
    return end && *end == '\0' && std::isfinite(out);
}

double Number(const Source& src, const SourceColumn* c, size_t row) {
    if (!c) {  // no X column: the row's place in the table
        return row < src.row_index.size() ? src.row_index[row] : static_cast<double>(row);
    }
    return row < c->numbers.size() ? c->numbers[row] : NAN;
}

std::string Short(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", v);
    return buf;
}

// Distinct values of a column, counted up to cap + 1.
size_t CountDistinct(const SourceColumn& c, size_t cap) {
    if (c.numeric) {
        std::vector<double> seen;
        for (double v : c.numbers) {
            if (!std::isfinite(v) || std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
            seen.push_back(v);
            if (seen.size() > cap) break;
        }
        return seen.size();
    }
    std::vector<std::string> seen;
    for (const auto& v : c.text) {
        if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
        seen.push_back(v);
        if (seen.size() > cap) break;
    }
    return seen.size();
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
    if (colour->numeric && CountDistinct(*colour, kMaxColorGroups) > kMaxColorGroups) {
        // Many numbers: equal ranges from the lowest to the highest value.
        const ColumnStats st = Summarize(colour->numbers);
        const double lo = st.min, hi = st.max > st.min ? st.max : st.min + 1.0;
        const double step = (hi - lo) / kColourRanges;
        for (int i = 0; i < kColourRanges; ++i) g.names.push_back(Short(lo + step * i) + " to " + Short(lo + step * (i + 1)));
        bool missing = false;
        for (size_t r = 0; r < rows; ++r) {
            const double v = r < colour->numbers.size() ? colour->numbers[r] : NAN;
            if (!std::isfinite(v)) {
                g.of_row[r] = kColourRanges;
                missing = true;
                continue;
            }
            g.of_row[r] = std::clamp(static_cast<int>((v - lo) / step), 0, kColourRanges - 1);
        }
        if (missing) g.names.push_back("missing");
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
                const double x = Number(src, xcol, r), v = y->numbers[r];
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
    if (shown < total) p.label = {DataLabel::State::Reduced, shown, total};
    else p.label = {DataLabel::State::Exact, total, total};
}

void PrepareScatter(Prepared& p, const SourceColumn* xcol, const std::vector<const SourceColumn*>& ys,
                    const Groups& groups, const SourceColumn* scale) {
    const auto colour_at = [&](size_t r) { return scale && r < scale->numbers.size() ? scale->numbers[r] : NAN; };
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
                if (scale) s.c.push_back(colour_at(r));
            }
            if (rows.size() < all_rows.size()) {
                for (size_t r : all_rows) {
                    if (groups.of_row[r] != static_cast<int>(g) || r >= y->numbers.size()) continue;
                    if (!std::isfinite(y->numbers[r])) continue;
                    s.all_x.push_back(xcol->numbers[r]);
                    s.all_y.push_back(y->numbers[r]);
                    if (scale) s.all_c.push_back(colour_at(r));
                }
            }
            if (!s.x.empty()) p.series.push_back(std::move(s));
        }
    }
    if (scale) {
        // The scale spans the colour values of the plotted rows; values on
        // both sides of 0 get the two-sided scale, centred on 0.
        const ColumnStats st = Summarize(scale->numbers);
        p.colour_scale = true;
        p.colour_label = scale->name;
        p.colour_min = st.min;
        p.colour_max = st.max > st.min ? st.max : st.min + 1.0;
        if (st.count > 0 && st.min < 0.0 && st.max > 0.0) {
            const double m = std::max(-st.min, st.max);
            p.colour_diverging = true;
            p.colour_min = -m;
            p.colour_max = m;
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

void PrepareHeatmap(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const SourceColumn* vcol) {
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
        // A value column is summed per cell (a confusion matrix's counts);
        // otherwise rows are counted.
        double add = 1.0;
        if (vcol) {
            add = r < vcol->numbers.size() ? vcol->numbers[r] : NAN;
            if (!std::isfinite(add)) continue;
        }
        counts[{ri[cy], ci[cx]}] += add;
    }
    // Labels that are all numbers (class labels 0..9) read in numeric order
    // on both axes; others keep the order they first appear in.
    const auto numeric_order = [](const std::vector<std::string>& names) {
        std::vector<int> order(names.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<double> values(names.size());
        for (size_t i = 0; i < names.size(); ++i)
            if (!ParseNumber(Trim(names[i]), values[i])) return order;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return values[static_cast<size_t>(a)] < values[static_cast<size_t>(b)]; });
        return order;
    };
    const std::vector<int> row_order = numeric_order(rows), col_order = numeric_order(cols);
    std::vector<int> row_at(rows.size()), col_at(cols.size());  // old index -> drawn index
    for (size_t i = 0; i < row_order.size(); ++i) row_at[static_cast<size_t>(row_order[i])] = static_cast<int>(i);
    for (size_t i = 0; i < col_order.size(); ++i) col_at[static_cast<size_t>(col_order[i])] = static_cast<int>(i);
    p.grid_rows = static_cast<int>(rows.size());
    p.grid_cols = static_cast<int>(cols.size());
    p.grid.assign(static_cast<size_t>(p.grid_rows) * static_cast<size_t>(p.grid_cols), 0.0);
    for (const auto& [rc, c] : counts)
        p.grid[static_cast<size_t>(row_at[static_cast<size_t>(rc.first)]) * static_cast<size_t>(p.grid_cols) +
               static_cast<size_t>(col_at[static_cast<size_t>(rc.second)])] = c;
    std::vector<std::string> sorted_rows, sorted_cols;
    for (int i : row_order) sorted_rows.push_back(rows[static_cast<size_t>(i)]);
    for (int i : col_order) sorted_cols.push_back(cols[static_cast<size_t>(i)]);
    rows = std::move(sorted_rows);
    cols = std::move(sorted_cols);
    const size_t total = n;
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

template <typename T>
bool Compare(const T& a, const std::string& op, const T& b) {
    if (op == "=") return a == b;
    if (op == "!=") return a != b;
    if (op == "<") return a < b;
    if (op == "<=") return a <= b;
    if (op == ">") return a > b;
    if (op == ">=") return a >= b;
    return false;
}

struct Condition {
    const SourceColumn* column = nullptr;
    std::string op, value;
    bool as_number = false;
    double number = 0;
};

bool Matches(const Condition& c, size_t r) {
    const SourceColumn& col = *c.column;
    if (col.numeric && (r >= col.numbers.size() || !std::isfinite(col.numbers[r]))) return false;  // missing
    if (col.numeric && c.as_number && c.op != "contains") return Compare(col.numbers[r], c.op, c.number);
    const std::string text = CellText(col, r);
    if (c.op == "contains") return text.find(c.value) != std::string::npos;
    return Compare(text, c.op, c.value);
}


// ---- P2b group 1 (approved board 8) ----

// Bars with Colour by: one series per group, a value per category (rows,
// or the mean of Y); 100% layout gives each category's shares.
void PrepareGroupedBars(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const Groups& groups) {
    std::vector<std::string> order;
    std::unordered_map<std::string, size_t> index;
    std::unordered_map<std::string, size_t> rows_of;
    const size_t rows = xcol->size();
    for (size_t r = 0; r < rows; ++r) {
        const std::string k = CellText(*xcol, r);
        if (!index.count(k)) {
            index[k] = order.size();
            order.push_back(k);
        }
        ++rows_of[k];
    }
    // The largest categories keep their own bar (the rest are left out and
    // said in the label).
    std::vector<std::string> kept = order;
    if (kept.size() > kMaxCategories) {
        std::stable_sort(kept.begin(), kept.end(), [&](const auto& a, const auto& b) { return rows_of[a] > rows_of[b]; });
        kept.resize(kMaxCategories);
        std::vector<std::string> ordered;
        for (const auto& k : order)
            if (std::find(kept.begin(), kept.end(), k) != kept.end()) ordered.push_back(k);
        kept = ordered;
    }
    std::unordered_map<std::string, size_t> at;
    for (size_t i = 0; i < kept.size(); ++i) at[kept[i]] = i;
    const size_t ng = groups.names.size(), nc = kept.size();
    std::vector<double> sum(ng * nc, 0.0), cnt(ng * nc, 0.0);
    size_t counted = 0;
    for (size_t r = 0; r < rows; ++r) {
        const int g = groups.of_row[r];
        if (g < 0) continue;
        auto it = at.find(CellText(*xcol, r));
        if (it == at.end()) continue;
        double v = 1.0;
        if (ycol) {
            v = r < ycol->numbers.size() ? ycol->numbers[r] : NAN;
            if (!std::isfinite(v)) continue;
        }
        sum[static_cast<size_t>(g) * nc + it->second] += v;
        cnt[static_cast<size_t>(g) * nc + it->second] += 1.0;
        ++counted;
    }
    p.categories = kept;
    for (size_t g = 0; g < ng; ++g) {
        Series s;
        s.label = groups.names[g].empty() ? (ycol ? "mean of " + ycol->name : std::string("rows")) : groups.names[g];
        for (size_t c = 0; c < nc; ++c) {
            const double v = ycol ? (cnt[g * nc + c] > 0 ? sum[g * nc + c] / cnt[g * nc + c] : 0.0) : sum[g * nc + c];
            s.x.push_back(static_cast<double>(c));
            s.y.push_back(v);
        }
        s.colour = static_cast<int>(g);
        p.series.push_back(std::move(s));
    }
    if (p.spec.bar_layout == PlotSpec::BarLayout::Percent) {
        for (size_t c = 0; c < nc; ++c) {
            double total = 0;
            for (const auto& s : p.series) total += s.y[c];
            for (auto& s : p.series) s.y[c] = total > 0 ? 100.0 * s.y[c] / total : 0.0;
        }
    }
    p.label = {DataLabel::State::Exact, counted, counted};
}

// Kernel density: one curve per Y column and group, Gaussian kernel with
// Silverman's bandwidth times the spec's factor, on one shared grid.
void PrepareKde(Prepared& p, const std::vector<const SourceColumn*>& ys, const Groups& groups) {
    const auto sets = ValueSets(ys, groups);
    constexpr int kSteps = 128;
    std::vector<double> bws;
    double lo = 0, hi = 0;
    bool first = true;
    size_t total = 0;
    for (const auto& [name, values] : sets) {
        const ColumnStats st = Summarize(values);
        double sd = 0;
        for (double v : values) sd += (v - st.mean) * (v - st.mean);
        sd = std::sqrt(sd / std::max<size_t>(1, values.size() - 1));
        const double iqr = (st.q3 - st.q1) / 1.34;
        const double spread = iqr > 0 ? std::min(sd, iqr) : sd;
        double bw = 0.9 * (spread > 0 ? spread : 1.0) * std::pow(static_cast<double>(values.size()), -0.2) * p.spec.kde_bandwidth;
        if (!(bw > 0)) bw = 1.0;
        bws.push_back(bw);
        lo = first ? st.min - 3 * bw : std::min(lo, st.min - 3 * bw);
        hi = first ? st.max + 3 * bw : std::max(hi, st.max + 3 * bw);
        first = false;
        total += values.size();
    }
    if (first) return;
    for (size_t i = 0; i < sets.size(); ++i) {
        const auto& [name, values] = sets[i];
        const double bw = bws[i];
        const double norm = 1.0 / (static_cast<double>(values.size()) * bw * std::sqrt(2.0 * 3.141592653589793));
        Series s;
        s.label = name;
        s.x_sorted = true;
        for (int k = 0; k < kSteps; ++k) {
            const double at = lo + (hi - lo) * k / (kSteps - 1);
            double sum = 0;
            for (double v : values) {
                const double u = (at - v) / bw;
                if (std::fabs(u) < 8.0) sum += std::exp(-0.5 * u * u);
            }
            s.x.push_back(at);
            s.y.push_back(sum * norm);
        }
        p.series.push_back(std::move(s));
    }
    if (!ys.empty()) p.stats = Summarize(ys.front()->numbers);
    p.label = {DataLabel::State::Exact, total, total};
}

// Average ranks (ties share the mean rank); NaN stays NaN.
std::vector<double> Ranks(const std::vector<double>& v) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < v.size(); ++i)
        if (std::isfinite(v[i])) idx.push_back(i);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size(), NAN);
    for (size_t i = 0; i < idx.size();) {
        size_t j = i;
        while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double rank = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) r[idx[k]] = rank;
        i = j + 1;
    }
    return r;
}

// Pearson correlation over the rows where both are numbers.
double Correlation(const std::vector<double>& a, const std::vector<double>& b) {
    double sa = 0, sb = 0;
    size_t n = 0;
    const size_t m = std::min(a.size(), b.size());
    for (size_t i = 0; i < m; ++i)
        if (std::isfinite(a[i]) && std::isfinite(b[i])) {
            sa += a[i];
            sb += b[i];
            ++n;
        }
    if (n < 2) return NAN;
    const double ma = sa / static_cast<double>(n), mb = sb / static_cast<double>(n);
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < m; ++i)
        if (std::isfinite(a[i]) && std::isfinite(b[i])) {
            cov += (a[i] - ma) * (b[i] - mb);
            va += (a[i] - ma) * (a[i] - ma);
            vb += (b[i] - mb) * (b[i] - mb);
        }
    return va > 0 && vb > 0 ? cov / std::sqrt(va * vb) : NAN;
}

constexpr size_t kMaxMatrixColumns = 60;
constexpr size_t kMaxMatrixRows = 500;

void PrepareMatrix(Prepared& p, const std::vector<const SourceColumn*>& ys) {
    if (ys.size() < 2 && p.spec.matrix_values != PlotSpec::MatrixValues::Values) {
        p.problem = "Choose two or more number columns.";
        return;
    }
    if (ys.size() > kMaxMatrixColumns) {
        p.problem = "Choose up to " + std::to_string(kMaxMatrixColumns) + " columns (" + std::to_string(ys.size()) + " chosen).";
        return;
    }
    const size_t n = ys.size();
    for (const auto* y : ys) p.col_names.push_back(y->name);
    if (p.spec.matrix_values == PlotSpec::MatrixValues::Values) {
        // The columns themselves: one grid row per table row.
        size_t rows = 0;
        for (const auto* y : ys) rows = std::max(rows, y->numbers.size());
        const size_t shown = std::min(rows, kMaxMatrixRows);
        p.grid_rows = static_cast<int>(shown);
        p.grid_cols = static_cast<int>(n);
        double lo = 0, hi = 0;
        bool first = true;
        for (size_t r = 0; r < shown; ++r) {
            p.row_names.push_back(std::to_string(r + 1));
            for (const auto* y : ys) {
                const double v = r < y->numbers.size() ? y->numbers[r] : NAN;
                p.grid.push_back(v);
                if (!std::isfinite(v)) continue;
                lo = first ? v : std::min(lo, v);
                hi = first ? v : std::max(hi, v);
                first = false;
            }
        }
        p.grid_lo = lo;
        p.grid_hi = hi > lo ? hi : lo + 1.0;
        if (lo < 0 && hi > 0) {
            const double m = std::max(-lo, hi);
            p.grid_lo = -m;
            p.grid_hi = m;
            p.grid_diverging = true;
        }
        p.label = shown < rows ? DataLabel{DataLabel::State::Truncated, shown, rows} : DataLabel{DataLabel::State::Exact, rows, rows};
        return;
    }
    // Correlation: Pearson, or Spearman (Pearson of the ranks).
    std::vector<std::vector<double>> cols;
    for (const auto* y : ys)
        cols.push_back(p.spec.matrix_values == PlotSpec::MatrixValues::Spearman ? Ranks(y->numbers) : y->numbers);
    p.row_names = p.col_names;
    p.grid_rows = p.grid_cols = static_cast<int>(n);
    p.grid.assign(n * n, NAN);
    for (size_t a = 0; a < n; ++a)
        for (size_t b = a; b < n; ++b) {
            const double c = a == b ? 1.0 : Correlation(cols[a], cols[b]);
            p.grid[a * n + b] = p.grid[b * n + a] = c;
        }
    p.grid_lo = -1.0;
    p.grid_hi = 1.0;
    p.grid_diverging = true;
    size_t rows = 0;
    for (const auto* y : ys) rows = std::max(rows, y->numbers.size());
    p.label = {DataLabel::State::Exact, rows, rows};
}

// Hexagonal bins, laid out as matplotlib does: two lattices, each point in
// the nearer centre.
void PrepareHexbin(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const SourceColumn* vcol) {
    std::vector<size_t> rows;
    const size_t n = std::min(xcol->numbers.size(), ycol->numbers.size());
    double x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    for (size_t r = 0; r < n; ++r) {
        const double x = xcol->numbers[r], y = ycol->numbers[r];
        if (!std::isfinite(x) || !std::isfinite(y)) continue;
        if (vcol && (r >= vcol->numbers.size() || !std::isfinite(vcol->numbers[r]))) continue;
        if (rows.empty()) {
            x0 = x1 = x;
            y0 = y1 = y;
        }
        x0 = std::min(x0, x);
        x1 = std::max(x1, x);
        y0 = std::min(y0, y);
        y1 = std::max(y1, y);
        rows.push_back(r);
    }
    if (rows.empty()) {
        p.problem = "No rows have numbers in both " + xcol->name + " and " + ycol->name + ".";
        return;
    }
    if (x1 <= x0) { x0 -= 0.5; x1 += 0.5; }
    if (y1 <= y0) { y0 -= 0.5; y1 += 0.5; }
    const int nx = std::clamp(p.spec.bins, 2, 200);
    const int ny = std::max(1, static_cast<int>(nx / std::sqrt(3.0)));
    const double sx = (x1 - x0) / nx, sy = (y1 - y0) / ny;
    // Lattice 1: (nx+1) x (ny+1) centres at (i, j); lattice 2: nx x ny at (i+.5, j+.5).
    const size_t n1 = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny + 1), n2 = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    std::vector<double> count(n1 + n2, 0.0), sum(n1 + n2, 0.0);
    for (size_t r : rows) {
        const double ix = (xcol->numbers[r] - x0) / sx, iy = (ycol->numbers[r] - y0) / sy;
        const double ix1 = std::round(ix), iy1 = std::round(iy);
        const double ix2 = std::floor(ix), iy2 = std::floor(iy);
        const double d1 = (ix - ix1) * (ix - ix1) + 3.0 * (iy - iy1) * (iy - iy1);
        const double d2 = (ix - ix2 - 0.5) * (ix - ix2 - 0.5) + 3.0 * (iy - iy2 - 0.5) * (iy - iy2 - 0.5);
        size_t cell;
        if (d1 < d2) {
            const int i = std::clamp(static_cast<int>(ix1), 0, nx), j = std::clamp(static_cast<int>(iy1), 0, ny);
            cell = static_cast<size_t>(i) * static_cast<size_t>(ny + 1) + static_cast<size_t>(j);
        } else {
            const int i = std::clamp(static_cast<int>(ix2), 0, nx - 1), j = std::clamp(static_cast<int>(iy2), 0, ny - 1);
            cell = n1 + static_cast<size_t>(i) * static_cast<size_t>(ny) + static_cast<size_t>(j);
        }
        count[cell] += 1.0;
        if (vcol) sum[cell] += vcol->numbers[r];
    }
    double lo = 0, hi = 0;
    bool first = true;
    for (size_t c = 0; c < count.size(); ++c) {
        if (count[c] <= 0) continue;
        double cx, cy;
        if (c < n1) {
            cx = x0 + sx * static_cast<double>(c / static_cast<size_t>(ny + 1));
            cy = y0 + sy * static_cast<double>(c % static_cast<size_t>(ny + 1));
        } else {
            const size_t k = c - n1;
            cx = x0 + sx * (static_cast<double>(k / static_cast<size_t>(ny)) + 0.5);
            cy = y0 + sy * (static_cast<double>(k % static_cast<size_t>(ny)) + 0.5);
        }
        const double v = vcol ? sum[c] / count[c] : count[c];
        p.hex_x.push_back(cx);
        p.hex_y.push_back(cy);
        p.hex_v.push_back(v);
        lo = first ? v : std::min(lo, v);
        hi = first ? v : std::max(hi, v);
        first = false;
    }
    p.hex_sx = sx;
    p.hex_sy = sy;
    p.grid_lo = vcol ? lo : 0.0;
    p.grid_hi = hi > p.grid_lo ? hi : p.grid_lo + 1.0;
    p.x_min = x0;
    p.x_max = x1;
    p.y_min = y0;
    p.y_max = y1;
    p.label = {DataLabel::State::Exact, rows.size(), rows.size()};
}

// Marching squares over the grid's cell centres: the segments where the
// grid crosses `level` (a saddle is split by the centre's average).
std::vector<double> ContourSegments(const Prepared& p, double level) {
    std::vector<double> out;
    const int R = p.grid_rows, C = p.grid_cols;
    const double dx = (p.x_max - p.x_min) / C, dy = (p.y_max - p.y_min) / R;
    const auto X = [&](double c) { return p.x_min + (c + 0.5) * dx; };
    const auto Y = [&](double r) { return p.y_max - (r + 0.5) * dy; };
    const auto at = [&](int r, int c) { return p.grid[static_cast<size_t>(r) * static_cast<size_t>(C) + static_cast<size_t>(c)]; };
    for (int r = 0; r + 1 < R; ++r)
        for (int c = 0; c + 1 < C; ++c) {
            // Corners: a top-left, b top-right, d bottom-right, e bottom-left.
            const double a = at(r, c), b = at(r, c + 1), d = at(r + 1, c + 1), e = at(r + 1, c);
            if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(d) || !std::isfinite(e)) continue;
            const int code = (a > level ? 8 : 0) | (b > level ? 4 : 0) | (d > level ? 2 : 0) | (e > level ? 1 : 0);
            if (code == 0 || code == 15) continue;
            const auto lerp = [&](double v0, double v1) { return v1 != v0 ? (level - v0) / (v1 - v0) : 0.5; };
            // Edge points: top (a-b), right (b-d), bottom (e-d), left (a-e).
            const double tx = X(c + lerp(a, b)), ty = Y(r);
            const double rx = X(c + 1), ry = Y(r + lerp(b, d));
            const double bx = X(c + lerp(e, d)), by = Y(r + 1);
            const double lx = X(c), ly = Y(r + lerp(a, e));
            const auto seg = [&](double x0, double y0, double x1, double y1) {
                out.insert(out.end(), {x0, y0, x1, y1});
            };
            const bool centre_high = (a + b + d + e) / 4.0 > level;
            switch (code) {
                case 1: case 14: seg(lx, ly, bx, by); break;
                case 2: case 13: seg(bx, by, rx, ry); break;
                case 3: case 12: seg(lx, ly, rx, ry); break;
                case 4: case 11: seg(tx, ty, rx, ry); break;
                case 6: case 9: seg(tx, ty, bx, by); break;
                case 7: case 8: seg(lx, ly, tx, ty); break;
                case 5:  // b and e high
                    if (centre_high) { seg(lx, ly, tx, ty); seg(bx, by, rx, ry); }
                    else { seg(tx, ty, rx, ry); seg(lx, ly, bx, by); }
                    break;
                case 10:  // a and d high
                    if (centre_high) { seg(tx, ty, rx, ry); seg(lx, ly, bx, by); }
                    else { seg(lx, ly, tx, ty); seg(bx, by, rx, ry); }
                    break;
            }
        }
    return out;
}

// Contour / filled contour: a grid of Z (the mean of a column per cell, or
// the density of rows lightly smoothed), its level lines, and for filled
// contours the grid upsampled 4x and set to the middle of its band.
void PrepareContour(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const SourceColumn* vcol) {
    const int bins = std::clamp(p.spec.bins, 4, 200);
    std::vector<std::array<double, 3>> pts;
    const size_t n = std::min(xcol->numbers.size(), ycol->numbers.size());
    for (size_t r = 0; r < n; ++r) {
        const double x = xcol->numbers[r], y = ycol->numbers[r];
        const double z = vcol ? (r < vcol->numbers.size() ? vcol->numbers[r] : NAN) : 1.0;
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) pts.push_back({x, y, z});
    }
    if (pts.empty()) {
        p.problem = "No rows have numbers in both " + xcol->name + " and " + ycol->name + ".";
        return;
    }
    double x0 = pts[0][0], x1 = x0, y0 = pts[0][1], y1 = y0;
    for (const auto& q : pts) {
        x0 = std::min(x0, q[0]);
        x1 = std::max(x1, q[0]);
        y0 = std::min(y0, q[1]);
        y1 = std::max(y1, q[1]);
    }
    if (x1 <= x0) { x0 -= 0.5; x1 += 0.5; }
    if (y1 <= y0) { y0 -= 0.5; y1 += 0.5; }
    const size_t cells = static_cast<size_t>(bins) * static_cast<size_t>(bins);
    std::vector<double> sum(cells, 0.0), cnt(cells, 0.0);
    for (const auto& q : pts) {
        const int c = std::clamp(static_cast<int>((q[0] - x0) / (x1 - x0) * bins), 0, bins - 1);
        const int r = std::clamp(bins - 1 - static_cast<int>((q[1] - y0) / (y1 - y0) * bins), 0, bins - 1);
        sum[static_cast<size_t>(r) * static_cast<size_t>(bins) + static_cast<size_t>(c)] += q[2];
        cnt[static_cast<size_t>(r) * static_cast<size_t>(bins) + static_cast<size_t>(c)] += 1.0;
    }
    p.grid.assign(cells, NAN);
    if (vcol) {
        for (size_t i = 0; i < cells; ++i)
            if (cnt[i] > 0) p.grid[i] = sum[i] / cnt[i];
    } else {
        // Density: counts smoothed over 3 x 3 cells.
        for (int r = 0; r < bins; ++r)
            for (int c = 0; c < bins; ++c) {
                double total = 0;
                int k = 0;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        const int rr = r + dr, cc = c + dc;
                        if (rr < 0 || cc < 0 || rr >= bins || cc >= bins) continue;
                        total += cnt[static_cast<size_t>(rr) * static_cast<size_t>(bins) + static_cast<size_t>(cc)];
                        ++k;
                    }
                p.grid[static_cast<size_t>(r) * static_cast<size_t>(bins) + static_cast<size_t>(c)] = total / k;
            }
    }
    p.grid_rows = p.grid_cols = bins;
    p.x_min = x0;
    p.x_max = x1;
    p.y_min = y0;
    p.y_max = y1;
    double lo = 0, hi = 0;
    bool first = true;
    for (double v : p.grid) {
        if (!std::isfinite(v)) continue;
        lo = first ? v : std::min(lo, v);
        hi = first ? v : std::max(hi, v);
        first = false;
    }
    if (hi <= lo) hi = lo + 1.0;
    p.grid_lo = lo;
    p.grid_hi = hi;
    const int levels = std::clamp(p.spec.levels, 1, 50);
    for (int k = 1; k <= levels; ++k) {
        const double level = lo + (hi - lo) * k / (levels + 1);
        p.contour_levels.push_back(level);
        p.contour_segments.push_back(ContourSegments(p, level));
    }
    if (p.spec.kind == Kind::FilledContour) {
        // Upsample 4x (bilinear over the cell centres), then set each cell to
        // the middle of its band so the fill steps at the level lines.
        constexpr int kUp = 4;
        p.band_rows = p.band_cols = bins * kUp;
        p.band_grid.assign(static_cast<size_t>(p.band_rows) * static_cast<size_t>(p.band_cols), NAN);
        const auto at = [&](int r, int c) {
            r = std::clamp(r, 0, bins - 1);
            c = std::clamp(c, 0, bins - 1);
            return p.grid[static_cast<size_t>(r) * static_cast<size_t>(bins) + static_cast<size_t>(c)];
        };
        for (int R = 0; R < p.band_rows; ++R)
            for (int C = 0; C < p.band_cols; ++C) {
                const double fr = (R + 0.5) / kUp - 0.5, fc = (C + 0.5) / kUp - 0.5;
                const int r0 = static_cast<int>(std::floor(fr)), c0 = static_cast<int>(std::floor(fc));
                const double ur = fr - r0, uc = fc - c0;
                const double v = (1 - ur) * ((1 - uc) * at(r0, c0) + uc * at(r0, c0 + 1)) + ur * ((1 - uc) * at(r0 + 1, c0) + uc * at(r0 + 1, c0 + 1));
                if (!std::isfinite(v)) continue;
                int band = 0;
                while (band < levels && v > p.contour_levels[static_cast<size_t>(band)]) ++band;
                // Band middle between its two bounding levels (lo / hi at the ends).
                const double below = band == 0 ? lo : p.contour_levels[static_cast<size_t>(band - 1)];
                const double above = band == levels ? hi : p.contour_levels[static_cast<size_t>(band)];
                p.band_grid[static_cast<size_t>(R) * static_cast<size_t>(p.band_cols) + static_cast<size_t>(C)] = (below + above) / 2.0;
            }
    }
    p.label = {DataLabel::State::Exact, pts.size(), pts.size()};
}

// ---- P2b group 2 (approved board 9) ----

constexpr double kTwoPi = 6.283185307179586;

// Polar: angle (0 at the top, clockwise) and radius per row, one series per
// colour group. Categories share the turn and close each line.
void PreparePolar(Prepared& p, const SourceColumn* acol, const SourceColumn* rcol, const Groups& groups) {
    PlotSpec::AngleUnit unit = p.spec.angle_unit;
    if (unit == PlotSpec::AngleUnit::Auto) unit = acol->numeric ? PlotSpec::AngleUnit::Degrees : PlotSpec::AngleUnit::Categories;
    std::vector<double> angle(acol->size(), NAN);
    if (unit == PlotSpec::AngleUnit::Categories) {
        // Categories in numeric order when they are all numbers, else as they come.
        std::vector<std::string> names;
        std::unordered_map<std::string, size_t> index;
        for (size_t r = 0; r < acol->size(); ++r) {
            const std::string k = CellText(*acol, r);
            if (!index.count(k)) {
                index[k] = names.size();
                names.push_back(k);
            }
        }
        std::vector<double> values(names.size());
        bool numbers = true;
        for (size_t i = 0; i < names.size() && numbers; ++i) numbers = ParseNumber(Trim(names[i]), values[i]);
        std::vector<size_t> order(names.size());
        std::iota(order.begin(), order.end(), 0);
        if (numbers) std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
        std::vector<size_t> at(names.size());
        for (size_t i = 0; i < order.size(); ++i) {
            at[order[i]] = i;
            p.polar_names.push_back(names[order[i]]);
        }
        const double n = static_cast<double>(std::max<size_t>(1, names.size()));
        for (size_t r = 0; r < acol->size(); ++r) angle[r] = static_cast<double>(at[index[CellText(*acol, r)]]) / n * kTwoPi;
        p.polar_closed = true;
    } else {
        const double k = unit == PlotSpec::AngleUnit::Degrees ? kTwoPi / 360.0 : 1.0;
        for (size_t r = 0; r < acol->numbers.size(); ++r) angle[r] = acol->numbers[r] * k;
    }
    size_t total = 0;
    for (size_t g = 0; g < groups.names.size(); ++g) {
        Series s;
        s.label = groups.names[g].empty() ? rcol->name : groups.names[g];
        std::vector<std::pair<double, double>> pts;
        for (size_t r = 0; r < rcol->numbers.size() && r < angle.size(); ++r) {
            if (groups.of_row[r] != static_cast<int>(g)) continue;
            const double a = angle[r], rr = rcol->numbers[r];
            if (!std::isfinite(a) || !std::isfinite(rr)) continue;
            pts.emplace_back(a, rr);
        }
        // Categories go round in angle order (one point per category and group).
        if (p.polar_closed) std::stable_sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [a, rr] : pts) {
            s.x.push_back(a);
            s.y.push_back(rr);
            p.polar_rmax = std::max(p.polar_rmax, std::fabs(rr));
        }
        total += pts.size();
        if (!s.x.empty()) {
            s.colour = static_cast<int>(g);
            p.series.push_back(std::move(s));
        }
    }
    if (p.polar_rmax <= 0) p.polar_rmax = 1.0;
    p.stats = Summarize(rcol->numbers);
    p.label = {DataLabel::State::Exact, total, total};
}

// Arrow components of a row: u, v as given, or a compass direction
// (degrees clockwise from north) and a length; wind directions say where
// the wind comes from, so they are turned round.
bool ArrowOf(const PlotSpec& spec, double a, double b, double& u, double& v) {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    if (spec.vector_from == PlotSpec::VectorFrom::UV) {
        u = a;
        v = b;
        return true;
    }
    const double t = a * kTwoPi / 360.0;
    u = b * std::sin(t);
    v = b * std::cos(t);
    if (spec.wind_from) {
        u = -u;
        v = -v;
    }
    return true;
}

// The distinct values of a column, sorted (a grid's x or y positions).
std::vector<double> Distinct(const std::vector<double>& v, size_t cap) {
    std::vector<double> d;
    for (double x : v)
        if (std::isfinite(x)) d.push_back(x);
    std::sort(d.begin(), d.end());
    d.erase(std::unique(d.begin(), d.end()), d.end());
    if (d.size() > cap) d.clear();
    return d;
}

void PrepareQuiver(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const SourceColumn* ucol, const SourceColumn* vcol) {
    const size_t n = std::min({xcol->numbers.size(), ycol->numbers.size(), ucol->numbers.size(), vcol->numbers.size()});
    double maxlen = 0;
    for (size_t r = 0; r < n; ++r) {
        double u, v;
        if (!std::isfinite(xcol->numbers[r]) || !std::isfinite(ycol->numbers[r]) ||
            !ArrowOf(p.spec, ucol->numbers[r], vcol->numbers[r], u, v))
            continue;
        p.qx.push_back(xcol->numbers[r]);
        p.qy.push_back(ycol->numbers[r]);
        p.qu.push_back(u);
        p.qv.push_back(v);
        maxlen = std::max(maxlen, std::hypot(u, v));
    }
    if (p.qx.empty()) {
        p.problem = "No rows have numbers in all four columns.";
        return;
    }
    p.x_min = *std::min_element(p.qx.begin(), p.qx.end());
    p.x_max = *std::max_element(p.qx.begin(), p.qx.end());
    p.y_min = *std::min_element(p.qy.begin(), p.qy.end());
    p.y_max = *std::max_element(p.qy.begin(), p.qy.end());
    if (p.x_max <= p.x_min) { p.x_min -= 0.5; p.x_max += 0.5; }
    if (p.y_max <= p.y_min) { p.y_min -= 0.5; p.y_max += 0.5; }
    // A grid (x and y take few values) thins every Nth each way; other
    // tables every Nth row. As many as fit: about 600 arrows.
    constexpr size_t kMaxArrows = 600;
    const auto xs = Distinct(p.qx, 2000), ys = Distinct(p.qy, 2000);
    const bool grid = !xs.empty() && !ys.empty() && xs.size() * ys.size() <= p.qx.size() * 2;
    int every = p.spec.arrow_every;
    if (every <= 0) {
        every = 1;
        const double ratio = static_cast<double>(p.qx.size()) / kMaxArrows;
        if (ratio > 1) every = static_cast<int>(std::ceil(grid ? std::sqrt(ratio) : ratio));
    }
    p.q_every = every;
    for (size_t i = 0; i < p.qx.size(); ++i) {
        bool keep;
        if (grid) {
            const size_t ix = static_cast<size_t>(std::lower_bound(xs.begin(), xs.end(), p.qx[i]) - xs.begin());
            const size_t iy = static_cast<size_t>(std::lower_bound(ys.begin(), ys.end(), p.qy[i]) - ys.begin());
            keep = ix % static_cast<size_t>(every) == 0 && iy % static_cast<size_t>(every) == 0;
        } else {
            keep = i % static_cast<size_t>(every) == 0;
        }
        if (keep) p.q_drawn.push_back(i);
    }
    // The longest arrow spans 90% of the spacing of the drawn arrows.
    const double spacing = grid ? std::min((p.x_max - p.x_min) / std::max<size_t>(1, xs.size() - 1),
                                           (p.y_max - p.y_min) / std::max<size_t>(1, ys.size() - 1)) * every
                                : std::sqrt((p.x_max - p.x_min) * (p.y_max - p.y_min) / static_cast<double>(p.q_drawn.size()));
    p.q_scale = maxlen > 0 ? 0.9 * spacing / maxlen : 1.0;
    p.grid_lo = 0.0;
    p.grid_hi = maxlen > 0 ? maxlen : 1.0;
    p.label = p.q_drawn.size() < p.qx.size() ? DataLabel{DataLabel::State::Sampled, p.q_drawn.size(), p.qx.size()}
                                             : DataLabel{DataLabel::State::Exact, p.qx.size(), p.qx.size()};
    if (p.q_drawn.size() < p.qx.size()) p.label.selection = "every " + std::to_string(every) + (grid ? " each way" : " rows");
}

// Stream: the field on a grid (a grid table as it is, other tables binned
// to 40 x 40 means), then lines traced both ways from evenly spaced starts,
// each stopping where another line already runs.
void PrepareStream(Prepared& p, const SourceColumn* xcol, const SourceColumn* ycol, const SourceColumn* ucol, const SourceColumn* vcol) {
    std::vector<std::array<double, 4>> rows;
    const size_t n = std::min({xcol->numbers.size(), ycol->numbers.size(), ucol->numbers.size(), vcol->numbers.size()});
    for (size_t r = 0; r < n; ++r) {
        double u, v;
        if (!std::isfinite(xcol->numbers[r]) || !std::isfinite(ycol->numbers[r]) ||
            !ArrowOf(p.spec, ucol->numbers[r], vcol->numbers[r], u, v))
            continue;
        rows.push_back({xcol->numbers[r], ycol->numbers[r], u, v});
    }
    if (rows.size() < 4) {
        p.problem = "A stream needs a field: at least 4 rows with numbers in all four columns.";
        return;
    }
    std::vector<double> xv, yv;
    for (const auto& q : rows) {
        xv.push_back(q[0]);
        yv.push_back(q[1]);
    }
    auto xs = Distinct(xv, 400), ys = Distinct(yv, 400);
    const bool grid = xs.size() >= 2 && ys.size() >= 2 && xs.size() * ys.size() <= rows.size() * 2;
    int C, R;
    double x0, x1, y0, y1;
    if (grid) {
        C = static_cast<int>(xs.size());
        R = static_cast<int>(ys.size());
        x0 = xs.front();
        x1 = xs.back();
        y0 = ys.front();
        y1 = ys.back();
    } else {
        C = R = 40;
        x0 = *std::min_element(xv.begin(), xv.end());
        x1 = *std::max_element(xv.begin(), xv.end());
        y0 = *std::min_element(yv.begin(), yv.end());
        y1 = *std::max_element(yv.begin(), yv.end());
    }
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;
    // Field values at grid points (r = 0 is the lowest y here; flipped to
    // the top-first convention for hover at the end).
    std::vector<double> U(static_cast<size_t>(R * C), 0.0), V(U.size(), 0.0), cnt(U.size(), 0.0);
    for (const auto& q : rows) {
        int c, r;
        if (grid) {
            c = static_cast<int>(std::lower_bound(xs.begin(), xs.end(), q[0]) - xs.begin());
            r = static_cast<int>(std::lower_bound(ys.begin(), ys.end(), q[1]) - ys.begin());
        } else {
            c = std::clamp(static_cast<int>((q[0] - x0) / (x1 - x0) * (C - 1) + 0.5), 0, C - 1);
            r = std::clamp(static_cast<int>((q[1] - y0) / (y1 - y0) * (R - 1) + 0.5), 0, R - 1);
        }
        const size_t k = static_cast<size_t>(r * C + c);
        U[k] += q[2];
        V[k] += q[3];
        cnt[k] += 1.0;
    }
    double vmax = 0;
    for (size_t k = 0; k < U.size(); ++k) {
        if (cnt[k] > 0) {
            U[k] /= cnt[k];
            V[k] /= cnt[k];
        }
        vmax = std::max(vmax, std::hypot(U[k], V[k]));
    }
    // Bilinear field in grid units (gx 0..C-1, gy 0..R-1).
    const auto field = [&](double gx, double gy, double& u, double& v) {
        if (gx < 0 || gy < 0 || gx > C - 1 || gy > R - 1) return false;
        const int c0 = std::min(static_cast<int>(gx), C - 2), r0 = std::min(static_cast<int>(gy), R - 2);
        const double fx = gx - c0, fy = gy - r0;
        const auto at = [&](const std::vector<double>& F, int r, int c) { return F[static_cast<size_t>(r * C + c)]; };
        u = (1 - fy) * ((1 - fx) * at(U, r0, c0) + fx * at(U, r0, c0 + 1)) + fy * ((1 - fx) * at(U, r0 + 1, c0) + fx * at(U, r0 + 1, c0 + 1));
        v = (1 - fy) * ((1 - fx) * at(V, r0, c0) + fx * at(V, r0, c0 + 1)) + fy * ((1 - fx) * at(V, r0 + 1, c0) + fx * at(V, r0 + 1, c0 + 1));
        return true;
    };
    // Occupancy: a coarse mask of cells a line already runs through.
    const int M = std::clamp(static_cast<int>(30 * p.spec.stream_density), 6, 150);
    std::vector<int> mask(static_cast<size_t>(M * M), -1);
    const auto cell = [&](double gx, double gy) {
        const int mx = std::clamp(static_cast<int>(gx / (C - 1) * M), 0, M - 1);
        const int my = std::clamp(static_cast<int>(gy / (R - 1) * M), 0, M - 1);
        return my * M + mx;
    };
    // Steps in grid units, with u, v turned into grid units per unit time.
    const double kx = (C - 1) / (x1 - x0), ky = (R - 1) / (y1 - y0);
    const double step = 0.25 * std::min((C - 1.0) / M, (R - 1.0) / M) + 0.05;
    const auto trace = [&](double gx, double gy, double dir, int line, std::vector<std::pair<double, double>>& out) {
        for (int k = 0; k < 4000; ++k) {
            double u, v;
            if (!field(gx, gy, u, v)) break;
            double sx = u * kx, sy = v * ky;
            const double sp = std::hypot(sx, sy);
            if (sp < 1e-12 * (vmax * (kx + ky) + 1e-300)) break;
            // Midpoint (RK2) step of fixed length along the field.
            double mx = gx + dir * 0.5 * step * sx / sp, my = gy + dir * 0.5 * step * sy / sp;
            if (!field(mx, my, u, v)) break;
            sx = u * kx;
            sy = v * ky;
            const double sp2 = std::hypot(sx, sy);
            if (sp2 <= 0) break;
            gx += dir * step * sx / sp2;
            gy += dir * step * sy / sp2;
            if (gx < 0 || gy < 0 || gx > C - 1 || gy > R - 1) break;
            const int m = cell(gx, gy);
            if (mask[static_cast<size_t>(m)] >= 0 && mask[static_cast<size_t>(m)] != line) break;
            mask[static_cast<size_t>(m)] = line;
            out.emplace_back(gx, gy);
        }
    };
    int line = 0;
    for (int my = 0; my < M; ++my)
        for (int mx = 0; mx < M; ++mx) {
            if (mask[static_cast<size_t>(my * M + mx)] >= 0) continue;
            const double sx0 = (mx + 0.5) / M * (C - 1), sy0 = (my + 0.5) / M * (R - 1);
            double u0, v0;
            if (!field(sx0, sy0, u0, v0) || std::hypot(u0, v0) <= vmax * 1e-6) continue;
            mask[static_cast<size_t>(my * M + mx)] = line;
            std::vector<std::pair<double, double>> back, fwd;
            trace(sx0, sy0, -1.0, line, back);
            trace(sx0, sy0, 1.0, line, fwd);
            std::vector<double> pts;
            for (auto it = back.rbegin(); it != back.rend(); ++it) pts.insert(pts.end(), {x0 + it->first / kx, y0 + it->second / ky});
            pts.insert(pts.end(), {x0 + sx0 / kx, y0 + sy0 / ky});
            for (const auto& q : fwd) pts.insert(pts.end(), {x0 + q.first / kx, y0 + q.second / ky});
            if (pts.size() >= 6) {
                const size_t mid = (pts.size() / 4) * 2;
                double um, vm;
                field((pts[mid] - x0) * kx, (pts[mid + 1] - y0) * ky, um, vm);
                p.stream_lines.push_back(std::move(pts));
                p.stream_speed.push_back(std::hypot(um, vm));
            }
            ++line;
        }
    // The field for hover, row 0 at the top (highest y), like the grids.
    p.grid_rows = R;
    p.grid_cols = C;
    p.field_u.assign(static_cast<size_t>(R * C), 0.0);
    p.field_v.assign(p.field_u.size(), 0.0);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            p.field_u[static_cast<size_t>((R - 1 - r) * C + c)] = U[static_cast<size_t>(r * C + c)];
            p.field_v[static_cast<size_t>((R - 1 - r) * C + c)] = V[static_cast<size_t>(r * C + c)];
        }
    p.x_min = x0;
    p.x_max = x1;
    p.y_min = y0;
    p.y_max = y1;
    p.grid_lo = 0.0;
    p.grid_hi = vmax > 0 ? vmax : 1.0;
    p.label = {DataLabel::State::Exact, rows.size(), rows.size()};
}

}  // namespace

std::vector<std::string> ColumnsNeeded(const PlotSpec& spec) {
    std::vector<std::string> out;
    const auto add = [&](const std::string& name) {
        if (!name.empty() && std::find(out.begin(), out.end(), name) == out.end()) out.push_back(name);
    };
    add(spec.x_column);
    for (const auto& y : spec.y_columns) add(y);
    add(spec.color_column);
    add(spec.value_column);
    if (Info(spec.kind).required & kEncVector) {
        add(spec.u_column);
        add(spec.v_column);
    }
    if (spec.rows == RowMode::Filter)
        for (const auto& c : spec.conditions) add(c.column);
    return out;
}

RowSelection SelectRows(const PlotSpec& spec, const Source& src) {
    RowSelection out;
    const size_t rows = src.Rows();
    out.total = rows;
    const std::string of_all = " of " + Thousands(static_cast<long long>(rows)) + " rows";
    std::vector<size_t> keep;
    switch (spec.rows) {
        case RowMode::All: out.all = true; return out;
        case RowMode::First: {
            const size_t n = std::min(std::max<size_t>(1, spec.first_rows), rows);
            if (n == rows) {
                out.all = true;
                return out;
            }
            for (size_t r = 0; r < n; ++r) keep.push_back(r);
            out.text = "first " + Thousands(static_cast<long long>(n)) + of_all;
            break;
        }
        case RowMode::Range: {
            const size_t from = std::max<size_t>(1, spec.row_from);
            const size_t to = std::min(std::max(from, spec.row_to), rows);
            if (from > rows) {
                out.problem = "The table has " + Thousands(static_cast<long long>(rows)) + " rows; the range starts at row " +
                              Thousands(static_cast<long long>(from)) + ".";
                return out;
            }
            for (size_t r = from - 1; r < to; ++r) keep.push_back(r);
            out.text = "rows " + Thousands(static_cast<long long>(from)) + " to " + Thousands(static_cast<long long>(to)) + of_all;
            break;
        }
        case RowMode::Filter: {
            std::vector<Condition> conds;
            for (const auto& rc : spec.conditions) {
                if (rc.column.empty()) continue;
                Condition c;
                c.column = src.Find(rc.column);
                if (!c.column) {
                    out.problem = "Filter column '" + rc.column + "' is not in the table.";
                    return out;
                }
                c.op = rc.op;
                c.value = Trim(rc.value);
                c.as_number = ParseNumber(c.value, c.number);
                conds.push_back(std::move(c));
            }
            if (conds.empty()) {  // no condition yet: all rows
                out.all = true;
                return out;
            }
            for (size_t r = 0; r < rows; ++r) {
                bool all = true;
                for (const auto& c : conds) all = all && Matches(c, r);
                if (all) keep.push_back(r);
            }
            if (keep.empty()) {
                out.problem = "No rows match " + ConditionsText(spec.conditions) + ".";
                return out;
            }
            out.text = "filtered \xC2\xB7 " + Thousands(static_cast<long long>(keep.size())) + of_all;
            break;
        }
    }
    out.source.row_limit = src.row_limit;
    out.source.total_rows = src.total_rows;
    for (const auto& c : src.columns) {
        SourceColumn sc;
        sc.name = c.name;
        sc.numeric = c.numeric;
        for (size_t r : keep) {
            if (c.numeric) sc.numbers.push_back(r < c.numbers.size() ? c.numbers[r] : NAN);
            else sc.text.push_back(r < c.text.size() ? c.text[r] : std::string());
        }
        out.source.columns.push_back(std::move(sc));
    }
    for (size_t r : keep) out.source.row_index.push_back(r < src.row_index.size() ? src.row_index[r] : static_cast<double>(r));
    return out;
}

ColumnSummary SummarizeColumn(const SourceColumn& c) {
    ColumnSummary s;
    s.name = c.name;
    s.numeric = c.numeric;
    s.has_stats = true;
    s.distinct = CountDistinct(c, kMaxColorGroups);
    if (c.numeric) {
        size_t n = 0, not_zero = 0;
        for (double v : c.numbers) {
            if (!std::isfinite(v)) continue;
            s.min = n == 0 ? v : std::min(s.min, v);
            s.max = n == 0 ? v : std::max(s.max, v);
            ++n;
            if (v != 0.0) ++not_zero;
        }
        s.not_zero = n > 0 ? static_cast<double>(not_zero) / static_cast<double>(n) : 0.0;
    }
    return s;
}

std::string ColumnSummary::Text() const {
    if (!has_stats) return "";
    if (numeric) {
        if (distinct == 0) return "no numbers";
        if (distinct == 1) return "always " + Short(min);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s to %s \xC2\xB7 %.1f%% not 0", Short(min).c_str(), Short(max).c_str(),
                      100.0 * not_zero);
        return buf;
    }
    if (distinct > kMaxColorGroups) return "more than " + std::to_string(kMaxColorGroups) + " values";
    return std::to_string(distinct) + (distinct == 1 ? " value" : " values");
}

Prepared Prepare(const PlotSpec& spec, const Source& all_rows) {
    Prepared p;
    p.spec = spec;
    const KindInfo& kind = Info(spec.kind);
    p.problem = MissingEncoding(spec);
    if (!p.problem.empty()) return p;
    // The rows the spec chose; the rest of the plot sees only them.
    const RowSelection selection = SelectRows(spec, all_rows);
    p.rows_total = selection.total;
    if (!selection.problem.empty()) {
        p.problem = selection.problem;
        return p;
    }
    const Source& src = selection.all ? all_rows : selection.source;
    p.rows_selected = src.Rows();

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
                           spec.kind != Kind::Heatmap && spec.kind != Kind::Polar;
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
    const SourceColumn* ucol = nullptr;
    const SourceColumn* vcol = nullptr;
    if (kind.required & kEncVector) {
        ucol = column(spec.u_column, true);
        if (!ucol) return p;
        vcol = column(spec.v_column, true);
        if (!vcol) return p;
    }
    const SourceColumn* value = nullptr;
    if (!spec.value_column.empty() && (kind.optional & kEncValue)) {
        value = column(spec.value_column, true);
        if (!value) return p;
    }
    const SourceColumn* colour = nullptr;
    if (!spec.color_column.empty() && (kind.optional & kEncColor)) {
        colour = column(spec.color_column, false);
        if (!colour) return p;
    }
    size_t rows = 0;
    if (xcol) rows = xcol->size();
    for (const auto* y : ys) rows = std::max(rows, y->size());
    // A scatter coloured by a number column with many values draws a scale
    // (or when asked); everything else colours groups.
    const SourceColumn* scale = nullptr;
    if (colour && colour->numeric && spec.kind == Kind::Scatter &&
        (spec.color_mode == ColourMode::Scale ||
         (spec.color_mode == ColourMode::Auto && CountDistinct(*colour, kMaxColorGroups) > kMaxColorGroups))) {
        scale = colour;
        colour = nullptr;
    }
    const Groups groups = GroupRows(colour, rows);

    switch (spec.kind) {
        case Kind::Line:
        case Kind::Area:
        case Kind::Step:
        case Kind::Stem: PrepareLines(p, src, xcol, ys, groups); break;
        case Kind::Scatter: PrepareScatter(p, xcol, ys, groups, scale); break;
        case Kind::Histogram: PrepareHistogram(p, xcol, groups); break;
        case Kind::Bar:
            if (colour) PrepareGroupedBars(p, xcol, ys.empty() ? nullptr : ys.front(), groups);
            else PrepareCategories(p, xcol, ys.empty() ? nullptr : ys.front());
            break;
        case Kind::Pie: PrepareCategories(p, xcol, ys.empty() ? nullptr : ys.front()); break;
        case Kind::Kde: PrepareKde(p, ys, groups); break;
        case Kind::Matrix: PrepareMatrix(p, ys); break;
        case Kind::Hexbin: PrepareHexbin(p, xcol, ys.front(), value); break;
        case Kind::Contour:
        case Kind::FilledContour: PrepareContour(p, xcol, ys.front(), value); break;
        case Kind::Polar: PreparePolar(p, xcol, ys.front(), groups); break;
        case Kind::Quiver: PrepareQuiver(p, xcol, ys.front(), ucol, vcol); break;
        case Kind::Stream: PrepareStream(p, xcol, ys.front(), ucol, vcol); break;
        case Kind::Box:
        case Kind::Violin: PrepareBoxes(p, ys, groups); break;
        case Kind::ErrorBars: PrepareErrorBars(p, xcol, ys.front()); break;
        case Kind::Heatmap: PrepareHeatmap(p, xcol, ys.front(), value); break;
        case Kind::Histogram2D: PrepareHistogram2D(p, xcol, ys.front()); break;
        default: break;
    }
    // Heatmap and 2D histogram: a scale from 0 to the largest cell.
    if ((spec.kind == Kind::Heatmap || spec.kind == Kind::Histogram2D) && !p.grid.empty()) {
        p.grid_lo = 0.0;
        p.grid_hi = *std::max_element(p.grid.begin(), p.grid.end());
        if (p.grid_hi <= 0.0) p.grid_hi = 1.0;
    }
    if (spec.kind != Kind::Histogram && spec.kind != Kind::Box && spec.kind != Kind::Violin && spec.kind != Kind::Kde &&
        spec.kind != Kind::Matrix && spec.kind != Kind::Polar && !ys.empty())
        p.stats = Summarize(ys.front()->numbers);
    // A source read with a row limit says so, whatever the kind did.
    if (src.row_limit > 0) p.label = {DataLabel::State::Truncated, src.row_limit, src.total_rows};
    // The rows chosen; a kind that said what it drew (quiver: every Nth arrow) keeps that.
    if (!selection.text.empty()) p.label.selection = selection.text;
    if (p.problem.empty() && p.series.empty() && p.grid.empty() && p.hex_x.empty() && p.qx.empty() && p.stream_lines.empty())
        p.problem = "No values to draw.";
    return p;
}

}  // namespace cyxwiz::plot
