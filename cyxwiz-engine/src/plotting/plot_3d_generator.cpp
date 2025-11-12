#define _USE_MATH_DEFINES
#include "plot_3d_generator.h"
#include <cmath>
#include <random>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz::plotting {

// ============================================================================
// Random Number Generator
// ============================================================================

std::mt19937& Plot3DGenerator::GetRNG() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

// ============================================================================
// Mathematical Surfaces
// ============================================================================

Plot3D::SurfaceData Plot3DGenerator::GenerateSurface(
    double x_min, double x_max, size_t x_steps,
    double y_min, double y_max, size_t y_steps,
    std::function<double(double, double)> func) {

    Plot3D::SurfaceData surface;
    surface.x_grid = Linspace(x_min, x_max, x_steps);
    surface.y_grid = Linspace(y_min, y_max, y_steps);
    surface.z_values.resize(x_steps * y_steps);

    for (size_t i = 0; i < x_steps; ++i) {
        for (size_t j = 0; j < y_steps; ++j) {
            surface.at(i, j) = func(surface.x_grid[i], surface.y_grid[j]);
        }
    }

    return surface;
}

Plot3D::SurfaceData Plot3DGenerator::GenerateGaussian(
    double amplitude, double x0, double y0, double sigma, size_t resolution) {

    return GenerateSurface(
        -5.0, 5.0, resolution,
        -5.0, 5.0, resolution,
        [=](double x, double y) {
            double dx = x - x0;
            double dy = y - y0;
            double r_sq = dx * dx + dy * dy;
            return amplitude * std::exp(-r_sq / (2.0 * sigma * sigma));
        });
}

Plot3D::SurfaceData Plot3DGenerator::GenerateSineWave(
    double wavelength, double amplitude, size_t resolution) {

    double k = 2.0 * M_PI / wavelength;
    return GenerateSurface(
        -5.0, 5.0, resolution,
        -5.0, 5.0, resolution,
        [=](double x, double y) {
            return amplitude * std::sin(k * x) * std::cos(k * y);
        });
}

Plot3D::SurfaceData Plot3DGenerator::GenerateSaddle(size_t resolution) {
    return GenerateSurface(
        -2.0, 2.0, resolution,
        -2.0, 2.0, resolution,
        [](double x, double y) {
            return x * x - y * y;
        });
}

Plot3D::SurfaceData Plot3DGenerator::GenerateSphere(
    double radius, size_t resolution) {

    Plot3D::SurfaceData surface;

    // Parametric sphere: theta (azimuth), phi (elevation)
    size_t theta_steps = resolution;
    size_t phi_steps = resolution / 2;

    surface.x_grid.resize(theta_steps);
    surface.y_grid.resize(phi_steps);
    surface.z_values.resize(theta_steps * phi_steps);

    for (size_t i = 0; i < theta_steps; ++i) {
        double theta = 2.0 * M_PI * i / theta_steps;
        surface.x_grid[i] = theta;

        for (size_t j = 0; j < phi_steps; ++j) {
            double phi = M_PI * j / phi_steps;
            surface.y_grid[j] = phi;

            // Sphere parametric equations
            double x = radius * std::sin(phi) * std::cos(theta);
            double y = radius * std::sin(phi) * std::sin(theta);
            double z = radius * std::cos(phi);

            surface.z_values[i * phi_steps + j] = z;
        }
    }

    return surface;
}

Plot3D::SurfaceData Plot3DGenerator::GenerateTorus(
    double major_radius, double minor_radius, size_t resolution) {

    Plot3D::SurfaceData surface;

    size_t u_steps = resolution;
    size_t v_steps = resolution / 2;

    surface.x_grid.resize(u_steps);
    surface.y_grid.resize(v_steps);
    surface.z_values.resize(u_steps * v_steps);

    for (size_t i = 0; i < u_steps; ++i) {
        double u = 2.0 * M_PI * i / u_steps;
        surface.x_grid[i] = u;

        for (size_t j = 0; j < v_steps; ++j) {
            double v = 2.0 * M_PI * j / v_steps;
            surface.y_grid[j] = v;

            // Torus parametric equations
            double x = (major_radius + minor_radius * std::cos(v)) * std::cos(u);
            double y = (major_radius + minor_radius * std::cos(v)) * std::sin(u);
            double z = minor_radius * std::sin(v);

            surface.z_values[i * v_steps + j] = z;
        }
    }

    return surface;
}

Plot3D::SurfaceData Plot3DGenerator::GenerateParaboloid(
    double scale, size_t resolution) {

    return GenerateSurface(
        -2.0, 2.0, resolution,
        -2.0, 2.0, resolution,
        [=](double x, double y) {
            return scale * (x * x + y * y);
        });
}

// ============================================================================
// ML/Data Science Surfaces
// ============================================================================

Plot3D::SurfaceData Plot3DGenerator::GenerateLossLandscape(
    size_t num_minima, double noise_level, size_t resolution) {

    std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> depth_dist(0.5, 2.0);
    std::normal_distribution<double> noise(0.0, noise_level);

    // Generate random minima positions and depths
    struct Minimum {
        double x, y, depth, width;
    };
    std::vector<Minimum> minima;

    for (size_t i = 0; i < num_minima; ++i) {
        minima.push_back({
            pos_dist(GetRNG()),
            pos_dist(GetRNG()),
            depth_dist(GetRNG()),
            0.5 + depth_dist(GetRNG()) * 0.5
        });
    }

    return GenerateSurface(
        -6.0, 6.0, resolution,
        -6.0, 6.0, resolution,
        [&](double x, double y) {
            // Sum of Gaussians (inverted for minima)
            double z = 3.0;  // Base height

            for (const auto& min : minima) {
                double dx = x - min.x;
                double dy = y - min.y;
                double r_sq = dx * dx + dy * dy;
                z -= min.depth * std::exp(-r_sq / (2.0 * min.width * min.width));
            }

            // Add noise
            z += noise(GetRNG());

            return z;
        });
}

Plot3D::LineData3D Plot3DGenerator::GenerateGradientDescentPath(
    const Plot3D::SurfaceData& surface,
    double start_x, double start_y,
    double learning_rate,
    size_t num_steps) {

    Plot3D::LineData3D path;

    double x = start_x;
    double y = start_y;

    for (size_t step = 0; step < num_steps; ++step) {
        double z = EvaluateSurface(surface, x, y);
        path.AddPoint(x, y, z);

        // Numerical gradient (finite differences)
        double h = 0.01;
        double dz_dx = (EvaluateSurface(surface, x + h, y) -
                        EvaluateSurface(surface, x - h, y)) / (2.0 * h);
        double dz_dy = (EvaluateSurface(surface, x, y + h) -
                        EvaluateSurface(surface, x, y - h)) / (2.0 * h);

        // Update position (gradient descent)
        x -= learning_rate * dz_dx;
        y -= learning_rate * dz_dy;

        // Clamp to bounds
        x = std::clamp(x, surface.x_grid.front(), surface.x_grid.back());
        y = std::clamp(y, surface.y_grid.front(), surface.y_grid.back());
    }

    return path;
}

Plot3D::SurfaceData Plot3DGenerator::GenerateDecisionBoundary(size_t resolution) {
    // Simple XOR-like decision boundary
    return GenerateSurface(
        -2.0, 2.0, resolution,
        -2.0, 2.0, resolution,
        [](double x, double y) {
            // Sigmoid of XOR-like function
            double val = x * y;
            return 1.0 / (1.0 + std::exp(-val));
        });
}

// ============================================================================
// 3D Scatter Data
// ============================================================================

Plot3D::ScatterData3D Plot3DGenerator::Generate3DClusters(
    size_t points_per_cluster, size_t num_clusters, double cluster_spread) {

    Plot3D::ScatterData3D data;

    std::uniform_real_distribution<double> center_dist(-5.0, 5.0);

    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        double cx = center_dist(GetRNG());
        double cy = center_dist(GetRNG());
        double cz = center_dist(GetRNG());

        std::normal_distribution<double> dist(0.0, cluster_spread);

        for (size_t i = 0; i < points_per_cluster; ++i) {
            data.AddPoint(
                cx + dist(GetRNG()),
                cy + dist(GetRNG()),
                cz + dist(GetRNG()),
                static_cast<int>(cluster));
        }
    }

    return data;
}

Plot3D::LineData3D Plot3DGenerator::Generate3DSpiral(
    size_t points, double height, double radius, double turns) {

    Plot3D::LineData3D line;

    for (size_t i = 0; i < points; ++i) {
        double t = static_cast<double>(i) / (points - 1);
        double theta = 2.0 * M_PI * turns * t;

        line.AddPoint(
            radius * std::cos(theta),
            radius * std::sin(theta),
            height * t);
    }

    return line;
}

Plot3D::LineData3D Plot3DGenerator::Generate3DHelix(
    size_t points, double height, double radius) {

    return Generate3DSpiral(points, height, radius, 3.0);
}

Plot3D::LineData3D Plot3DGenerator::GenerateLissajous(
    size_t points, double a, double b, double c) {

    Plot3D::LineData3D line;

    for (size_t i = 0; i < points; ++i) {
        double t = 2.0 * M_PI * i / points;

        line.AddPoint(
            std::sin(a * t),
            std::sin(b * t),
            std::sin(c * t));
    }

    return line;
}

Plot3D::LineData3D Plot3DGenerator::Generate3DRandomWalk(
    size_t steps, double step_size) {

    Plot3D::LineData3D line;

    std::normal_distribution<double> step_dist(0.0, step_size);

    double x = 0, y = 0, z = 0;
    line.AddPoint(x, y, z);

    for (size_t i = 1; i < steps; ++i) {
        x += step_dist(GetRNG());
        y += step_dist(GetRNG());
        z += step_dist(GetRNG());
        line.AddPoint(x, y, z);
    }

    return line;
}

Plot3D::ScatterData3D Plot3DGenerator::GenerateSpherePoints(
    size_t points, double radius) {

    Plot3D::ScatterData3D data;

    std::uniform_real_distribution<double> u_dist(0.0, 1.0);

    for (size_t i = 0; i < points; ++i) {
        // Uniform distribution on sphere using rejection sampling
        double theta = 2.0 * M_PI * u_dist(GetRNG());
        double phi = std::acos(2.0 * u_dist(GetRNG()) - 1.0);

        double x = radius * std::sin(phi) * std::cos(theta);
        double y = radius * std::sin(phi) * std::sin(theta);
        double z = radius * std::cos(phi);

        data.AddPoint(x, y, z);
    }

    return data;
}

Plot3D::ScatterData3D Plot3DGenerator::GenerateCubePoints(
    size_t points, double size) {

    Plot3D::ScatterData3D data;

    std::uniform_real_distribution<double> coord_dist(-size / 2, size / 2);

    for (size_t i = 0; i < points; ++i) {
        data.AddPoint(
            coord_dist(GetRNG()),
            coord_dist(GetRNG()),
            coord_dist(GetRNG()));
    }

    return data;
}

// ============================================================================
// Physics/Engineering
// ============================================================================

Plot3D::SurfaceData Plot3DGenerator::GenerateWaveInterference(
    double wavelength, size_t resolution) {

    // Two point sources
    double source1_x = -2.0, source1_y = 0.0;
    double source2_x = 2.0, source2_y = 0.0;
    double k = 2.0 * M_PI / wavelength;

    return GenerateSurface(
        -5.0, 5.0, resolution,
        -5.0, 5.0, resolution,
        [=](double x, double y) {
            double r1 = std::sqrt((x - source1_x) * (x - source1_x) +
                                  (y - source1_y) * (y - source1_y));
            double r2 = std::sqrt((x - source2_x) * (x - source2_x) +
                                  (y - source2_y) * (y - source2_y));

            double wave1 = std::sin(k * r1) / (r1 + 0.1);
            double wave2 = std::sin(k * r2) / (r2 + 0.1);

            return wave1 + wave2;
        });
}

Plot3D::LineData3D Plot3DGenerator::GenerateFieldLines(
    const Plot3D::Point3D& source,
    size_t num_lines,
    size_t points_per_line) {

    // Simplified electric field line (radial from point source)
    Plot3D::LineData3D lines;

    for (size_t line = 0; line < num_lines; ++line) {
        double theta = 2.0 * M_PI * line / num_lines;
        double phi = M_PI * (line % 2 == 0 ? 0.25 : 0.75);

        for (size_t pt = 0; pt < points_per_line; ++pt) {
            double r = 0.5 + 5.0 * pt / points_per_line;

            double x = source.x + r * std::sin(phi) * std::cos(theta);
            double y = source.y + r * std::sin(phi) * std::sin(theta);
            double z = source.z + r * std::cos(phi);

            lines.AddPoint(x, y, z);
        }
    }

    return lines;
}

// ============================================================================
// Utility Functions
// ============================================================================

void Plot3DGenerator::AddNoise(Plot3D::SurfaceData& surface, double amplitude) {
    std::normal_distribution<double> noise(0.0, amplitude);

    for (double& z : surface.z_values) {
        z += noise(GetRNG());
    }
}

void Plot3DGenerator::NormalizeSurface(Plot3D::SurfaceData& surface) {
    if (surface.z_values.empty()) return;

    double min_z = *std::min_element(surface.z_values.begin(), surface.z_values.end());
    double max_z = *std::max_element(surface.z_values.begin(), surface.z_values.end());
    double range = max_z - min_z;

    if (range > 0.0) {
        for (double& z : surface.z_values) {
            z = (z - min_z) / range;
        }
    }
}

void Plot3DGenerator::CalculateNormals(Plot3D::MeshData& mesh) {
    mesh.normals.resize(mesh.vertices.size(), Plot3D::Point3D(0, 0, 0));

    // Calculate face normals and accumulate to vertices
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        const auto& v0 = mesh.vertices[mesh.indices[i]];
        const auto& v1 = mesh.vertices[mesh.indices[i + 1]];
        const auto& v2 = mesh.vertices[mesh.indices[i + 2]];

        // Edge vectors
        double e1x = v1.x - v0.x, e1y = v1.y - v0.y, e1z = v1.z - v0.z;
        double e2x = v2.x - v0.x, e2y = v2.y - v0.y, e2z = v2.z - v0.z;

        // Cross product (face normal)
        double nx = e1y * e2z - e1z * e2y;
        double ny = e1z * e2x - e1x * e2z;
        double nz = e1x * e2y - e1y * e2x;

        // Accumulate to vertex normals
        for (size_t j = 0; j < 3; ++j) {
            auto& n = mesh.normals[mesh.indices[i + j]];
            n.x += nx;
            n.y += ny;
            n.z += nz;
        }
    }

    // Normalize vertex normals
    for (auto& n : mesh.normals) {
        double len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (len > 0.0) {
            n.x /= len;
            n.y /= len;
            n.z /= len;
        }
    }
}

Plot3D::MeshData Plot3DGenerator::SurfaceToMesh(const Plot3D::SurfaceData& surface) {
    Plot3D::MeshData mesh;

    // Create vertices
    for (size_t i = 0; i < surface.rows(); ++i) {
        for (size_t j = 0; j < surface.cols(); ++j) {
            mesh.vertices.emplace_back(
                surface.x_grid[i],
                surface.y_grid[j],
                surface.at(i, j));
        }
    }

    // Create triangles
    for (size_t i = 0; i < surface.rows() - 1; ++i) {
        for (size_t j = 0; j < surface.cols() - 1; ++j) {
            int v0 = static_cast<int>(i * surface.cols() + j);
            int v1 = static_cast<int>(i * surface.cols() + (j + 1));
            int v2 = static_cast<int>((i + 1) * surface.cols() + j);
            int v3 = static_cast<int>((i + 1) * surface.cols() + (j + 1));

            // Two triangles per quad
            mesh.indices.push_back(v0);
            mesh.indices.push_back(v1);
            mesh.indices.push_back(v2);

            mesh.indices.push_back(v1);
            mesh.indices.push_back(v3);
            mesh.indices.push_back(v2);
        }
    }

    CalculateNormals(mesh);

    return mesh;
}

// ============================================================================
// Helpers
// ============================================================================

double Plot3DGenerator::EvaluateSurface(
    const Plot3D::SurfaceData& surface,
    double x, double y) {

    // Bilinear interpolation
    // Find grid cell
    auto it_x = std::lower_bound(surface.x_grid.begin(), surface.x_grid.end(), x);
    auto it_y = std::lower_bound(surface.y_grid.begin(), surface.y_grid.end(), y);

    if (it_x == surface.x_grid.end()) it_x--;
    if (it_y == surface.y_grid.end()) it_y--;
    if (it_x == surface.x_grid.begin()) it_x++;
    if (it_y == surface.y_grid.begin()) it_y++;

    size_t i1 = std::distance(surface.x_grid.begin(), it_x) - 1;
    size_t j1 = std::distance(surface.y_grid.begin(), it_y) - 1;
    size_t i2 = i1 + 1;
    size_t j2 = j1 + 1;

    double x1 = surface.x_grid[i1], x2 = surface.x_grid[i2];
    double y1 = surface.y_grid[j1], y2 = surface.y_grid[j2];

    double z11 = surface.at(i1, j1);
    double z12 = surface.at(i1, j2);
    double z21 = surface.at(i2, j1);
    double z22 = surface.at(i2, j2);

    double tx = (x - x1) / (x2 - x1);
    double ty = (y - y1) / (y2 - y1);

    double z1 = z11 * (1 - tx) + z21 * tx;
    double z2 = z12 * (1 - tx) + z22 * tx;

    return z1 * (1 - ty) + z2 * ty;
}

std::vector<double> Plot3DGenerator::Linspace(double start, double end, size_t count) {
    std::vector<double> result(count);

    if (count == 1) {
        result[0] = start;
        return result;
    }

    double step = (end - start) / (count - 1);
    for (size_t i = 0; i < count; ++i) {
        result[i] = start + i * step;
    }

    return result;
}

} // namespace cyxwiz::plotting
