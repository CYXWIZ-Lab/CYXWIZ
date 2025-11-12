#pragma once

#include <vector>
#include <string>

namespace cyxwiz::plotting {

/**
 * Plot3D - Utilities and data structures for 3D plotting
 * Supports surface plots, 3D scatter, 3D lines
 */
class Plot3D {
public:
    // 3D Point
    struct Point3D {
        double x, y, z;
        Point3D(double x_ = 0, double y_ = 0, double z_ = 0)
            : x(x_), y(y_), z(z_) {}
    };

    // Surface data (Z = f(X, Y))
    struct SurfaceData {
        std::vector<double> x_grid;  // X coordinates (rows)
        std::vector<double> y_grid;  // Y coordinates (columns)
        std::vector<double> z_values; // Z values (x_grid.size() * y_grid.size())

        size_t rows() const { return x_grid.size(); }
        size_t cols() const { return y_grid.size(); }

        // Access Z value at (row, col)
        double& at(size_t row, size_t col) {
            return z_values[row * cols() + col];
        }

        const double& at(size_t row, size_t col) const {
            return z_values[row * cols() + col];
        }
    };

    // 3D Scatter data
    struct ScatterData3D {
        std::vector<Point3D> points;
        std::vector<int> labels;  // Optional cluster/group labels

        void AddPoint(double x, double y, double z, int label = 0) {
            points.emplace_back(x, y, z);
            labels.push_back(label);
        }

        size_t size() const { return points.size(); }
    };

    // 3D Line/Path data
    struct LineData3D {
        std::vector<Point3D> points;

        void AddPoint(double x, double y, double z) {
            points.emplace_back(x, y, z);
        }

        size_t size() const { return points.size(); }
    };

    // Mesh data (triangulated surface)
    struct MeshData {
        std::vector<Point3D> vertices;
        std::vector<int> indices;  // Triangle indices (3 per triangle)
        std::vector<Point3D> normals;  // Vertex normals (optional)

        void AddTriangle(const Point3D& p1, const Point3D& p2, const Point3D& p3) {
            size_t base = vertices.size();
            vertices.push_back(p1);
            vertices.push_back(p2);
            vertices.push_back(p3);
            indices.push_back(static_cast<int>(base));
            indices.push_back(static_cast<int>(base + 1));
            indices.push_back(static_cast<int>(base + 2));
        }

        size_t triangle_count() const { return indices.size() / 3; }
    };

    // Camera/View settings
    struct ViewSettings {
        double azimuth = 45.0;    // Rotation around Z-axis (degrees)
        double elevation = 30.0;  // Angle from XY-plane (degrees)
        double distance = 5.0;    // Camera distance from origin
        Point3D look_at{0, 0, 0}; // Point to look at
        bool orthographic = false; // Orthographic vs perspective projection

        // Field of view for perspective (degrees)
        double fov = 60.0;

        // Clipping planes
        double near_clip = 0.1;
        double far_clip = 100.0;
    };

    // Axis/Grid settings
    struct AxisSettings {
        bool show_grid = true;
        bool show_axes = true;
        bool show_box = true;  // Bounding box
        bool equal_aspect = false;  // Equal aspect ratio for all axes

        // Axis limits (auto if min == max)
        double x_min = 0, x_max = 0;
        double y_min = 0, y_max = 0;
        double z_min = 0, z_max = 0;

        // Labels
        std::string x_label = "X";
        std::string y_label = "Y";
        std::string z_label = "Z";
    };

    // Colormap for surface plots
    enum class Colormap {
        Viridis,
        Plasma,
        Inferno,
        Magma,
        Jet,
        Hot,
        Cool,
        Rainbow,
        Grayscale
    };

    // Surface rendering mode
    enum class SurfaceMode {
        Surface,      // Solid surface
        Wireframe,    // Wireframe only
        Both,         // Surface + wireframe overlay
        Contour       // Contour lines
    };
};

} // namespace cyxwiz::plotting
