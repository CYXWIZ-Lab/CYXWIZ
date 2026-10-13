// Graph and mesh algorithms (TOFIX134 P4 group 2): groups, force / layered /
// circle layouts, the tidy tree, Delaunay triangles.

#include "../src/core/plot/plot_graph.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

double Dist(const Point2& a, const Point2& b) { return std::hypot(a.x - b.x, a.y - b.y); }
}  // namespace

int main() {
    // Two cliques of 5 joined by one light edge.
    std::vector<Edge> e;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 5; ++i)
            for (int j = i + 1; j < 5; ++j) e.push_back({c * 5 + i, c * 5 + j, 2.0});
    e.push_back({4, 5, 0.5});
    const auto groups = FindGroups(10, e);
    Check(groups[0] == groups[4] && groups[5] == groups[9] && groups[0] != groups[5], "two cliques, two groups");
    Check((groups[0] == 0 || groups[0] == 1) && (groups[5] == 0 || groups[5] == 1), "groups numbered 0 and 1");

    const auto p = ForceLayout(10, e);
    const auto again = ForceLayout(10, e);
    Check(p.size() == 10 && p[3].x == again[3].x && p[7].y == again[7].y, "seeded: the same picture twice");
    double inside = 0, across = 0;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            if (i != j) inside += Dist(p[static_cast<size_t>(i)], p[static_cast<size_t>(j)]);
            across += Dist(p[static_cast<size_t>(i)], p[static_cast<size_t>(5 + j)]);
        }
    Check(inside / 20 < across / 25, "a clique sits together, apart from the other");
    for (const auto& q : p) Check(q.x >= -1e-9 && q.x <= 1 + 1e-9 && q.y >= -1e-9 && q.y <= 1 + 1e-9, "inside 0..1");

    // Layers: a -> b -> c, a -> c; a cycle c -> a is broken.
    const auto l = LayeredLayout(3, {{0, 1}, {1, 2}, {0, 2}, {2, 0}});
    Check(l[0].x == 0 && l[1].x == 0.5 && l[2].x == 1, "longest path layers: " + std::to_string(l[1].x));
    const auto wide = LayeredLayout(4, {{0, 2}, {1, 3}});
    Check(wide[0].x == 0 && wide[2].x == 1 && wide[0].y != wide[1].y, "two sources share the first layer");

    const auto circle = CircleLayout({1, 0, 1, 0});
    Check(std::fabs(Dist(circle[0], {0.5, 0.5}) - 0.5) < 1e-9 && std::fabs(circle[1].y - 0.0) < 1e-9, "on the circle, group 0 first (at the top)");

    // Tree: 0 -> {1, 2}, 1 -> {3, 4}; a loop 5 <-> 6 is cut.
    std::vector<int> depth;
    const auto t = TreeLayout({-1, 0, 0, 1, 1, 6, 5}, &depth);
    Check(t[3].x < t[4].x && std::fabs(t[1].x - (t[3].x + t[4].x) / 2) < 1e-12 && std::fabs(t[0].x - (t[1].x + t[2].x) / 2) < 1e-12,
          "parents centred over their children");
    Check(depth[0] == 0 && depth[3] == 2 && t[3].y == 1.0 && t[0].y == 0.0, "depth down the page");
    Check((depth[5] == 0) != (depth[6] == 0), "a loop becomes a root and its child");

    // Delaunay: a 5 x 5 grid has 2 (n - 1)^2 triangles; random points 2n - 2 - h at most.
    std::vector<Point2> grid;
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x) grid.push_back({static_cast<double>(x), static_cast<double>(y) * 10});
    Check(Triangulate(grid).size() == 32, "a 5 x 5 grid: 32 triangles, got " + std::to_string(Triangulate(grid).size()));
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<Point2> pts;
    for (int i = 0; i < 300; ++i) pts.push_back({u(rng), u(rng)});
    const auto tri = Triangulate(pts);
    Check(tri.size() >= 300 && tri.size() <= 2 * 300 - 5, "300 random points: " + std::to_string(tri.size()) + " triangles");
    // Every triangle has positive area and no other point inside its circle (spot check the first 50).
    for (size_t k = 0; k < 50; ++k) {
        const auto& A = pts[static_cast<size_t>(tri[k][0])];
        const auto& B = pts[static_cast<size_t>(tri[k][1])];
        const auto& C = pts[static_cast<size_t>(tri[k][2])];
        const double area = (B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y);
        Check(std::fabs(area) > 0, "a real triangle");
        const double d = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        const double a2 = A.x * A.x + A.y * A.y, b2 = B.x * B.x + B.y * B.y, c2 = C.x * C.x + C.y * C.y;
        const double cx = (a2 * (B.y - C.y) + b2 * (C.y - A.y) + c2 * (A.y - B.y)) / d;
        const double cy = (a2 * (C.x - B.x) + b2 * (A.x - C.x) + c2 * (B.x - A.x)) / d;
        const double r2 = (A.x - cx) * (A.x - cx) + (A.y - cy) * (A.y - cy);
        for (const auto& q : pts) Check((q.x - cx) * (q.x - cx) + (q.y - cy) * (q.y - cy) >= r2 * (1 - 1e-9), "empty circumcircle");
    }
    Check(Triangulate({{0, 0}, {1, 1}}).empty(), "fewer than three points: none");

    std::cout << "plot graph: groups, seeded force layout, layers with a broken cycle, circle, tidy tree with a cut loop, "
                 "Delaunay (grid count, empty circles). OK\n";
    return 0;
}
