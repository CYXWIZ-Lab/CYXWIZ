#include "plot_graph.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>

namespace cyxwiz::plot {

namespace {

// Positions scaled into 0..1 (both axes by the same factor keeps the shape).
void Normalise(std::vector<Point2>& p, bool keep_aspect) {
    if (p.empty()) return;
    double x0 = p[0].x, x1 = x0, y0 = p[0].y, y1 = y0;
    for (const auto& q : p) {
        x0 = std::min(x0, q.x);
        x1 = std::max(x1, q.x);
        y0 = std::min(y0, q.y);
        y1 = std::max(y1, q.y);
    }
    double sx = x1 - x0, sy = y1 - y0;
    if (keep_aspect) sx = sy = std::max(sx, sy);
    for (auto& q : p) {
        q.x = sx > 0 ? (q.x - x0) / sx + (keep_aspect ? (1.0 - (x1 - x0) / sx) / 2 : 0.0) : 0.5;
        q.y = sy > 0 ? (q.y - y0) / sy + (keep_aspect ? (1.0 - (y1 - y0) / sy) / 2 : 0.0) : 0.5;
    }
}

}  // namespace

namespace {

// Louvain's local moving on one level: `self` is each node's own internal
// weight (aggregated groups), `edges` the links between different nodes.
std::vector<int> LocalMoving(int n, const std::vector<Edge>& edges, const std::vector<double>& self) {
    std::vector<std::vector<std::pair<int, double>>> adj(static_cast<size_t>(n));
    std::vector<double> k(static_cast<size_t>(n), 0.0);
    double m2 = 0;  // twice the total weight
    for (int i = 0; i < n; ++i) {
        k[static_cast<size_t>(i)] += 2 * self[static_cast<size_t>(i)];
        m2 += 2 * self[static_cast<size_t>(i)];
    }
    for (const auto& e : edges) {
        adj[static_cast<size_t>(e.a)].push_back({e.b, e.weight});
        adj[static_cast<size_t>(e.b)].push_back({e.a, e.weight});
        k[static_cast<size_t>(e.a)] += e.weight;
        k[static_cast<size_t>(e.b)] += e.weight;
        m2 += 2 * e.weight;
    }
    std::vector<int> label(static_cast<size_t>(n));
    std::iota(label.begin(), label.end(), 0);
    if (m2 <= 0) return label;
    std::vector<double> tot(k);  // the sum of degrees per group
    for (int pass = 0; pass < 50; ++pass) {
        bool moved = false;
        for (int i = 0; i < n; ++i) {
            const size_t si = static_cast<size_t>(i);
            if (adj[si].empty()) continue;
            std::map<int, double> to;  // weight from i into each neighbouring group
            for (const auto& [j, w] : adj[si]) to[label[static_cast<size_t>(j)]] += w;
            const int own = label[si];
            tot[static_cast<size_t>(own)] -= k[si];
            int best = own;
            double best_gain = (to.count(own) ? to[own] : 0.0) - tot[static_cast<size_t>(own)] * k[si] / m2;
            for (const auto& [g, w] : to) {
                const double gain = w - tot[static_cast<size_t>(g)] * k[si] / m2;
                if (gain > best_gain + 1e-12) {
                    best = g;
                    best_gain = gain;
                }
            }
            tot[static_cast<size_t>(best)] += k[si];
            if (best != own) {
                label[si] = best;
                moved = true;
            }
        }
        if (!moved) break;
    }
    return label;
}

}  // namespace

std::vector<int> FindGroups(int n, const std::vector<Edge>& edges) {
    // Louvain: local moving, then each group becomes one node (its inside
    // weight kept) and the moving runs again, while groups still merge.
    std::vector<int> label(static_cast<size_t>(std::max(0, n)));
    std::iota(label.begin(), label.end(), 0);
    std::vector<Edge> level;
    for (const auto& e : edges)
        if (e.a >= 0 && e.b >= 0 && e.a < n && e.b < n && e.a != e.b && e.weight > 0) level.push_back(e);
    std::vector<double> self(static_cast<size_t>(std::max(0, n)), 0.0);
    int nodes = n;
    for (int round = 0; round < 10 && nodes > 1; ++round) {
        const auto moved = LocalMoving(nodes, level, self);
        // Compact the group numbers of this level.
        std::map<int, int> compact;
        for (int g : moved) compact.emplace(g, static_cast<int>(compact.size()));
        const int groups = static_cast<int>(compact.size());
        for (int& l : label) l = compact[moved[static_cast<size_t>(l)]];
        if (groups == nodes) break;
        // The next level: a node per group, links summed, inside links as self weight.
        std::vector<double> next_self(static_cast<size_t>(groups), 0.0);
        for (int i = 0; i < nodes; ++i) next_self[static_cast<size_t>(compact[moved[static_cast<size_t>(i)]])] += self[static_cast<size_t>(i)];
        std::map<std::pair<int, int>, double> sum;
        for (const auto& e : level) {
            int a = compact[moved[static_cast<size_t>(e.a)]], b = compact[moved[static_cast<size_t>(e.b)]];
            if (a == b) next_self[static_cast<size_t>(a)] += e.weight;
            else sum[{std::min(a, b), std::max(a, b)}] += e.weight;
        }
        level.clear();
        for (const auto& [ab, w] : sum) level.push_back({ab.first, ab.second, w});
        self = std::move(next_self);
        nodes = groups;
    }
    // Renumber by size (largest first, then the smaller first label).
    std::map<int, int> size;
    for (int l : label) ++size[l];
    std::vector<std::pair<int, int>> order(size.begin(), size.end());
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::map<int, int> rename;
    for (size_t g = 0; g < order.size(); ++g) rename[order[g].first] = static_cast<int>(g);
    for (int& l : label) l = rename[l];
    return label;
}

std::vector<Point2> ForceLayout(int n, const std::vector<Edge>& edges, unsigned seed) {
    std::vector<Point2> p(static_cast<size_t>(std::max(0, n)));
    if (n <= 0) return p;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (auto& q : p) q = {u(rng), u(rng)};
    if (n == 1) {
        p[0] = {0.5, 0.5};
        return p;
    }
    double mean_w = 0;
    size_t m = 0;
    for (const auto& e : edges)
        if (e.weight > 0) {
            mean_w += e.weight;
            ++m;
        }
    mean_w = m ? mean_w / static_cast<double>(m) : 1.0;
    const double k = std::sqrt(1.0 / n);
    const int iterations = n < 500 ? 300 : n < 2000 ? 150 : 60;
    const bool grid = n > 1500;
    std::vector<Point2> disp(p.size());
    for (int it = 0; it < iterations; ++it) {
        const double temp = 0.1 * (1.0 - static_cast<double>(it) / iterations) + 0.002;
        std::fill(disp.begin(), disp.end(), Point2{});
        const auto repel = [&](size_t i, size_t j) {
            double dx = p[i].x - p[j].x, dy = p[i].y - p[j].y;
            double d2 = dx * dx + dy * dy;
            if (d2 < 1e-12) {
                dx = 1e-4 * static_cast<double>((i % 7) + 1);
                dy = 1e-4 * static_cast<double>((j % 5) + 1);
                d2 = dx * dx + dy * dy;
            }
            const double f = k * k / d2;  // k^2 / d along the unit vector
            disp[i].x += dx * f;
            disp[i].y += dy * f;
            disp[j].x -= dx * f;
            disp[j].y -= dy * f;
        };
        if (!grid) {
            for (size_t i = 0; i < p.size(); ++i)
                for (size_t j = i + 1; j < p.size(); ++j) repel(i, j);
        } else {
            // Repulsion only from nodes in the same and neighbouring cells.
            const double cell = 2 * k;
            std::unordered_map<long long, std::vector<size_t>> cells;
            const auto key = [&](double x, double y) {
                return (static_cast<long long>(std::floor(x / cell)) << 32) ^ (static_cast<long long>(std::floor(y / cell)) & 0xffffffffLL);
            };
            for (size_t i = 0; i < p.size(); ++i) cells[key(p[i].x, p[i].y)].push_back(i);
            for (size_t i = 0; i < p.size(); ++i) {
                const long long cx = static_cast<long long>(std::floor(p[i].x / cell)), cy = static_cast<long long>(std::floor(p[i].y / cell));
                for (long long ox = -1; ox <= 1; ++ox)
                    for (long long oy = -1; oy <= 1; ++oy) {
                        auto c = cells.find(((cx + ox) << 32) ^ ((cy + oy) & 0xffffffffLL));
                        if (c == cells.end()) continue;
                        for (size_t j : c->second)
                            if (j > i) repel(i, j);
                    }
            }
        }
        for (const auto& e : edges) {
            if (e.a < 0 || e.b < 0 || e.a >= n || e.b >= n || e.a == e.b) continue;
            const size_t a = static_cast<size_t>(e.a), b = static_cast<size_t>(e.b);
            const double dx = p[a].x - p[b].x, dy = p[a].y - p[b].y;
            const double d = std::sqrt(dx * dx + dy * dy) + 1e-12;
            const double w = std::clamp(e.weight / mean_w, 0.5, 3.0);
            const double f = d / k * w;  // d^2 / k along the unit vector
            disp[a].x -= dx * f;
            disp[a].y -= dy * f;
            disp[b].x += dx * f;
            disp[b].y += dy * f;
        }
        for (size_t i = 0; i < p.size(); ++i) {
            // Gravity towards the middle keeps separate parts in view.
            disp[i].x -= (p[i].x - 0.5) * 2.0 * k;
            disp[i].y -= (p[i].y - 0.5) * 2.0 * k;
            const double len = std::sqrt(disp[i].x * disp[i].x + disp[i].y * disp[i].y);
            if (len > 0) {
                const double step = std::min(len, temp);
                p[i].x += disp[i].x / len * step;
                p[i].y += disp[i].y / len * step;
            }
        }
    }
    Normalise(p, true);
    return p;
}

std::vector<Point2> LayeredLayout(int n, const std::vector<Edge>& edges) {
    std::vector<Point2> p(static_cast<size_t>(std::max(0, n)));
    if (n <= 0) return p;
    std::vector<std::vector<int>> out(static_cast<size_t>(n));
    for (const auto& e : edges)
        if (e.a >= 0 && e.b >= 0 && e.a < n && e.b < n && e.a != e.b) out[static_cast<size_t>(e.a)].push_back(e.b);
    // Break cycles: drop edges back to a node on the current depth-first path.
    std::vector<int> state(static_cast<size_t>(n), 0);  // 0 new, 1 on path, 2 done
    std::vector<std::vector<int>> dag(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
        if (state[static_cast<size_t>(s)]) continue;
        std::vector<std::pair<int, size_t>> stack{{s, 0}};
        state[static_cast<size_t>(s)] = 1;
        while (!stack.empty()) {
            auto& [v, i] = stack.back();
            if (i < out[static_cast<size_t>(v)].size()) {
                const int w = out[static_cast<size_t>(v)][i++];
                if (state[static_cast<size_t>(w)] == 1) continue;  // a cycle: leave this edge out
                dag[static_cast<size_t>(v)].push_back(w);
                if (state[static_cast<size_t>(w)] == 0) {
                    state[static_cast<size_t>(w)] = 1;
                    stack.push_back({w, 0});
                }
            } else {
                state[static_cast<size_t>(v)] = 2;
                stack.pop_back();
            }
        }
    }
    // Longest path from a source, in topological order.
    std::vector<int> indeg(static_cast<size_t>(n), 0), layer(static_cast<size_t>(n), 0);
    for (int v = 0; v < n; ++v)
        for (int w : dag[static_cast<size_t>(v)]) ++indeg[static_cast<size_t>(w)];
    std::vector<int> queue;
    for (int v = 0; v < n; ++v)
        if (!indeg[static_cast<size_t>(v)]) queue.push_back(v);
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int v = queue[qi];
        for (int w : dag[static_cast<size_t>(v)]) {
            layer[static_cast<size_t>(w)] = std::max(layer[static_cast<size_t>(w)], layer[static_cast<size_t>(v)] + 1);
            if (--indeg[static_cast<size_t>(w)] == 0) queue.push_back(w);
        }
    }
    const int layers = *std::max_element(layer.begin(), layer.end()) + 1;
    std::vector<std::vector<int>> rows(static_cast<size_t>(layers));
    for (int v = 0; v < n; ++v) rows[static_cast<size_t>(layer[static_cast<size_t>(v)])].push_back(v);
    std::vector<double> order(static_cast<size_t>(n), 0.0);
    const auto renumber = [&](std::vector<int>& row) {
        for (size_t i = 0; i < row.size(); ++i) order[static_cast<size_t>(row[i])] = static_cast<double>(i);
    };
    for (auto& row : rows) renumber(row);
    std::vector<std::vector<int>> in(static_cast<size_t>(n));
    for (int v = 0; v < n; ++v)
        for (int w : dag[static_cast<size_t>(v)]) in[static_cast<size_t>(w)].push_back(v);
    // Barycentre sweeps, down then up.
    for (int sweep = 0; sweep < 8; ++sweep) {
        const bool down = sweep % 2 == 0;
        for (int li = 0; li < layers; ++li) {
            auto& row = rows[static_cast<size_t>(down ? li : layers - 1 - li)];
            std::vector<std::pair<double, int>> keyed;
            for (int v : row) {
                const auto& nb = down ? in[static_cast<size_t>(v)] : dag[static_cast<size_t>(v)];
                double sum = 0;
                for (int w : nb) sum += order[static_cast<size_t>(w)];
                keyed.push_back({nb.empty() ? order[static_cast<size_t>(v)] : sum / static_cast<double>(nb.size()), v});
            }
            std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < keyed.size(); ++i) row[i] = keyed[i].second;
            renumber(row);
        }
    }
    for (int li = 0; li < layers; ++li) {
        const auto& row = rows[static_cast<size_t>(li)];
        for (size_t i = 0; i < row.size(); ++i)
            p[static_cast<size_t>(row[i])] = {layers > 1 ? static_cast<double>(li) / (layers - 1) : 0.5,
                                              row.size() > 1 ? static_cast<double>(i) / static_cast<double>(row.size() - 1) : 0.5};
    }
    return p;
}

std::vector<Point2> CircleLayout(const std::vector<int>& groups) {
    const size_t n = groups.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return groups[a] < groups[b]; });
    std::vector<Point2> p(n);
    for (size_t i = 0; i < n; ++i) {
        const double a = 2.0 * 3.14159265358979323846 * static_cast<double>(i) / static_cast<double>(std::max<size_t>(1, n)) - 3.14159265358979323846 / 2;
        p[order[i]] = {0.5 + 0.5 * std::cos(a), 0.5 + 0.5 * std::sin(a)};
    }
    return p;
}

std::vector<Point2> TreeLayout(const std::vector<int>& parent, std::vector<int>* depth_out) {
    const int n = static_cast<int>(parent.size());
    std::vector<Point2> p(parent.size());
    std::vector<int> depth(parent.size(), 0);
    std::vector<std::vector<int>> children(parent.size());
    std::vector<int> roots;
    // A cycle is cut where it closes: walk up from each node, at most n steps.
    std::vector<int> par(parent.begin(), parent.end());
    for (int i = 0; i < n; ++i) {
        int v = i, steps = 0;
        while (v >= 0 && v < n && par[static_cast<size_t>(v)] >= 0 && steps <= n) {
            v = par[static_cast<size_t>(v)];
            ++steps;
            if (v == i) {
                par[static_cast<size_t>(i)] = -1;  // i closes a loop: make it a root
                break;
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const int q = par[static_cast<size_t>(i)];
        if (q < 0 || q >= n || q == i) roots.push_back(i);
        else children[static_cast<size_t>(q)].push_back(i);
    }
    double next_leaf = 0;
    int max_depth = 0;
    std::vector<char> seen(parent.size(), 0);
    // Iterative post-order: leaves get the next slot, parents the middle of their children.
    for (int r : roots) {
        std::vector<std::pair<int, size_t>> stack{{r, 0}};
        seen[static_cast<size_t>(r)] = 1;
        depth[static_cast<size_t>(r)] = 0;
        while (!stack.empty()) {
            auto& [v, i] = stack.back();
            const auto& ch = children[static_cast<size_t>(v)];
            if (i < ch.size()) {
                const int c = ch[i++];
                if (seen[static_cast<size_t>(c)]) continue;
                seen[static_cast<size_t>(c)] = 1;
                depth[static_cast<size_t>(c)] = depth[static_cast<size_t>(v)] + 1;
                max_depth = std::max(max_depth, depth[static_cast<size_t>(c)]);
                stack.push_back({c, 0});
            } else {
                if (ch.empty()) p[static_cast<size_t>(v)].x = next_leaf++;
                else p[static_cast<size_t>(v)].x = (p[static_cast<size_t>(ch.front())].x + p[static_cast<size_t>(ch.back())].x) / 2;
                stack.pop_back();
            }
        }
    }
    const double span = std::max(1.0, next_leaf - 1);
    for (int i = 0; i < n; ++i) {
        p[static_cast<size_t>(i)].x = next_leaf > 1 ? p[static_cast<size_t>(i)].x / span : 0.5;
        p[static_cast<size_t>(i)].y = max_depth > 0 ? static_cast<double>(depth[static_cast<size_t>(i)]) / max_depth : 0.0;
    }
    if (depth_out) *depth_out = depth;
    return p;
}

std::vector<std::array<int, 3>> Triangulate(const std::vector<Point2>& in) {
    std::vector<std::array<int, 3>> out;
    const int n = static_cast<int>(in.size());
    if (n < 3) return out;
    std::vector<Point2> pts(in);
    Normalise(pts, false);
    // Super triangle around 0..1 (indices n, n+1, n+2).
    pts.push_back({-10, -10});
    pts.push_back({11, -10});
    pts.push_back({0.5, 20});
    struct Tri {
        int v[3];
        double cx, cy, r2;
    };
    const auto make = [&](int a, int b, int c) {
        const Point2 &A = pts[static_cast<size_t>(a)], &B = pts[static_cast<size_t>(b)], &C = pts[static_cast<size_t>(c)];
        const double d = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        Tri t{{a, b, c}, 0, 0, -1};
        if (std::fabs(d) < 1e-18) return t;  // degenerate: never contains a point
        const double a2 = A.x * A.x + A.y * A.y, b2 = B.x * B.x + B.y * B.y, c2 = C.x * C.x + C.y * C.y;
        t.cx = (a2 * (B.y - C.y) + b2 * (C.y - A.y) + c2 * (A.y - B.y)) / d;
        t.cy = (a2 * (C.x - B.x) + b2 * (A.x - C.x) + c2 * (B.x - A.x)) / d;
        t.r2 = (A.x - t.cx) * (A.x - t.cx) + (A.y - t.cy) * (A.y - t.cy);
        return t;
    };
    std::vector<Tri> tris{make(n, n + 1, n + 2)};
    // Insert in x order: the triangles near the new point are recent.
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pts[static_cast<size_t>(a)].x != pts[static_cast<size_t>(b)].x ? pts[static_cast<size_t>(a)].x < pts[static_cast<size_t>(b)].x
                                                                               : pts[static_cast<size_t>(a)].y < pts[static_cast<size_t>(b)].y;
    });
    for (int i : order) {
        const Point2& P = pts[static_cast<size_t>(i)];
        std::vector<std::pair<int, int>> hole;
        std::vector<Tri> keep;
        keep.reserve(tris.size() + 2);
        for (const auto& t : tris) {
            const double dx = P.x - t.cx, dy = P.y - t.cy;
            if (t.r2 >= 0 && dx * dx + dy * dy < t.r2 * (1 + 1e-12)) {
                for (int e = 0; e < 3; ++e) hole.push_back({t.v[e], t.v[(e + 1) % 3]});
            } else {
                keep.push_back(t);
            }
        }
        // The hole's boundary: edges that belong to one removed triangle only.
        for (size_t a = 0; a < hole.size(); ++a) {
            bool shared = false;
            for (size_t b = 0; b < hole.size() && !shared; ++b)
                shared = a != b && hole[a].first == hole[b].second && hole[a].second == hole[b].first;
            if (!shared) keep.push_back(make(hole[a].first, hole[a].second, i));
        }
        tris.swap(keep);
    }
    for (const auto& t : tris)
        if (t.v[0] < n && t.v[1] < n && t.v[2] < n) out.push_back({t.v[0], t.v[1], t.v[2]});
    return out;
}

}  // namespace cyxwiz::plot
