#pragma once

// Graph and mesh algorithms for the plot kinds Network, Tree and Mesh
// (TOFIX134 P4 group 2, approved board 17). Pure and deterministic: the
// same input gives the same layout every time. Positions are in 0..1.

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace cyxwiz::plot {

struct Edge {
    int a = 0, b = 0;      // node indices (a -> b when directed)
    double weight = 1.0;
};

struct Point2 {
    double x = 0, y = 0;
};

// Groups of nodes that link more to each other than to the rest (weighted
// modularity: Louvain, nodes in index order, so the same every time),
// numbered by size: group 0 is the largest.
std::vector<int> FindGroups(int nodes, const std::vector<Edge>& edges);

// Force-directed layout (Fruchterman-Reingold with a little gravity so
// separate parts stay in view), seeded: same input, same picture.
std::vector<Point2> ForceLayout(int nodes, const std::vector<Edge>& edges, unsigned seed = 7);

// Layers for a directed graph (pipelines, model graphs): x = the layer
// (longest path from a source; cycles broken), y = the order in the layer
// after a few barycentre sweeps that reduce crossings.
std::vector<Point2> LayeredLayout(int nodes, const std::vector<Edge>& edges);

// Nodes on a circle, in the order of their group then index.
std::vector<Point2> CircleLayout(const std::vector<int>& groups);

// Tidy tree: `parent[i]` is a node index or -1 (a root). Leaves left to
// right in depth-first order, a parent centred over its children, y = the
// depth / the deepest level. Several roots sit side by side; a cycle is
// cut where it closes. `depth` gets each node's level.
std::vector<Point2> TreeLayout(const std::vector<int>& parent, std::vector<int>* depth = nullptr);

// Delaunay triangles (Bowyer-Watson) of the points, as index triples. The
// points should be distinct; x and y are scaled to 0..1 each first, so
// columns in different units triangulate evenly.
std::vector<std::array<int, 3>> Triangulate(const std::vector<Point2>& points);

}  // namespace cyxwiz::plot
