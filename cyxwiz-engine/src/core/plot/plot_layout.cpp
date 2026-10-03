#include "plot_layout.h"

#include <algorithm>
#include <map>
#include <numeric>

namespace cyxwiz::plot {

std::vector<TreeRect> Squarify(const std::vector<double>& values, double x, double y, double w, double h) {
    std::vector<TreeRect> out(values.size());
    double total = 0;
    for (double v : values) total += std::max(0.0, v);
    if (total <= 0 || w <= 0 || h <= 0) {
        for (auto& r : out) {
            r.x = x;
            r.y = y;
        }
        return out;
    }
    const double scale = w * h / total;
    std::vector<double> area(values.size());
    for (size_t i = 0; i < values.size(); ++i) area[i] = std::max(0.0, values[i]) * scale;
    // The worst aspect ratio of a row laid along a side of length `side`.
    const auto worst = [](double sum, double lo, double hi, double side) {
        const double s2 = sum * sum, w2 = side * side;
        return std::max(w2 * hi / s2, s2 / (w2 * lo));
    };
    size_t i = 0;
    while (i < area.size()) {
        const double side = std::min(w, h);
        size_t j = i + 1;
        double sum = area[i], lo = area[i], hi = area[i];
        while (j < area.size() && area[j] > 0) {
            const double ns = sum + area[j], nl = std::min(lo, area[j]), nh = std::max(hi, area[j]);
            if (worst(ns, nl, nh, side) > worst(sum, lo, hi, side)) break;
            sum = ns;
            lo = nl;
            hi = nh;
            ++j;
        }
        if (sum <= 0) {  // only zeros left
            for (; i < area.size(); ++i) {
                out[i].x = x;
                out[i].y = y;
            }
            break;
        }
        if (w >= h) {  // a column on the left
            const double cw = sum / h;
            double yy = y;
            for (size_t k = i; k < j; ++k) {
                out[k] = TreeRect{x, yy, cw, area[k] / cw};
                yy += area[k] / cw;
            }
            x += cw;
            w -= cw;
        } else {  // a row on top
            const double rh = sum / w;
            double xx = x;
            for (size_t k = i; k < j; ++k) {
                out[k] = TreeRect{xx, y, area[k] / rh, rh};
                xx += area[k] / rh;
            }
            y += rh;
            h -= rh;
        }
        i = j;
    }
    return out;
}

namespace {

void LayOut(const Prepared& p, const std::vector<size_t>& leaves, size_t depth, int out_depth, double x, double y, double w,
            double h, double header, std::vector<TreeRect>& out) {
    const size_t levels = p.tree_levels.size();
    // Group the leaves by their name at this depth.
    std::map<std::string, std::vector<size_t>> by_name;
    for (size_t i : leaves) by_name[p.tree_leaves[i].path[depth]].push_back(i);
    std::vector<std::pair<double, std::string>> order;
    for (const auto& [name, members] : by_name) {
        double size = 0;
        for (size_t i : members) size += p.tree_leaves[i].size;
        if (size > 0) order.emplace_back(size, name);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<double> sizes;
    for (const auto& o : order) sizes.push_back(o.first);
    const std::vector<TreeRect> rects = Squarify(sizes, x, y, w, h);
    for (size_t k = 0; k < order.size(); ++k) {
        const auto& members = by_name[order[k].second];
        TreeRect r = rects[k];
        r.depth = out_depth;
        r.size = order[k].first;
        const Prepared::TreeLeaf& first = p.tree_leaves[members.front()];
        r.path.assign(first.path.begin(), first.path.begin() + static_cast<std::ptrdiff_t>(depth) + 1);
        r.top = first.top;
        if (depth + 1 == levels) {
            r.leaf = true;
            // A leaf is one path; its colour is the leaf's.
            r.colour = first.colour;
            out.push_back(std::move(r));
            continue;
        }
        out.push_back(r);
        // Children inside, under the header when it fits.
        const double pad = header * 0.12;
        const double top = r.h > header * 2.6 && r.w > header * 3.0 ? header : pad;
        if (r.w > pad * 2 && r.h > top + pad)
            LayOut(p, members, depth + 1, out_depth + 1, r.x + pad, r.y + top, r.w - pad * 2, r.h - top - pad, header, out);
    }
}

}  // namespace

std::vector<TreeRect> TreemapLayout(const Prepared& p, const std::vector<std::string>& zoom, double w, double h, double header) {
    std::vector<TreeRect> out;
    const size_t levels = p.tree_levels.size();
    if (levels == 0 || zoom.size() >= levels) return out;
    std::vector<size_t> leaves;
    for (size_t i = 0; i < p.tree_leaves.size(); ++i) {
        const auto& path = p.tree_leaves[i].path;
        if (path.size() == levels && std::equal(zoom.begin(), zoom.end(), path.begin())) leaves.push_back(i);
    }
    if (!leaves.empty()) LayOut(p, leaves, zoom.size(), 0, 0, 0, w, h, header, out);
    return out;
}

}  // namespace cyxwiz::plot
