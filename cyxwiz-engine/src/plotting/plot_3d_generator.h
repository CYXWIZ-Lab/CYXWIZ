#pragma once

#include "plot_3d.h"
#include <functional>

namespace cyxwiz::plotting {

/**
 * Plot3DGenerator - Generate test data for 3D plots
 */
class Plot3DGenerator {
public:
    // ========================================================================
    // Mathematical Surfaces
    // ========================================================================

    /**
     * Generate surface from function Z = f(X, Y)
     * @param x_min, x_max X-axis range
     * @param y_min, y_max Y-axis range
     * @param x_steps, y_steps Grid resolution
     * @param func Function(x, y) -> z
     */
    static Plot3D::SurfaceData GenerateSurface(
        double x_min, double x_max, size_t x_steps,
        double y_min, double y_max, size_t y_steps,
        std::function<double(double, double)> func);

    /**
     * Generate Gaussian bell surface
     * Z = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
     */
    static Plot3D::SurfaceData GenerateGaussian(
        double amplitude = 1.0,
        double x0 = 0.0, double y0 = 0.0,
        double sigma = 1.0,
        size_t resolution = 50);

    /**
     * Generate sine wave surface
     * Z = sin(k*X) * cos(k*Y)
     */
    static Plot3D::SurfaceData GenerateSineWave(
        double wavelength = 2.0,
        double amplitude = 1.0,
        size_t resolution = 50);

    /**
     * Generate saddle surface (hyperbolic paraboloid)
     * Z = X^2 - Y^2
     */
    static Plot3D::SurfaceData GenerateSaddle(size_t resolution = 50);

    /**
     * Generate sphere surface
     * X^2 + Y^2 + Z^2 = R^2
     */
    static Plot3D::SurfaceData GenerateSphere(
        double radius = 1.0,
        size_t resolution = 50);

    /**
     * Generate torus surface
     */
    static Plot3D::SurfaceData GenerateTorus(
        double major_radius = 1.0,
        double minor_radius = 0.3,
        size_t resolution = 50);

    /**
     * Generate paraboloid
     * Z = X^2 + Y^2
     */
    static Plot3D::SurfaceData GenerateParaboloid(
        double scale = 1.0,
        size_t resolution = 50);

    // ========================================================================
    // ML/Data Science Surfaces
    // ========================================================================

    /**
     * Generate loss landscape (2D parameter space)
     * Simulates neural network loss surface with multiple local minima
     */
    static Plot3D::SurfaceData GenerateLossLandscape(
        size_t num_minima = 5,
        double noise_level = 0.1,
        size_t resolution = 100);

    /**
     * Generate gradient descent path on surface
     * @param surface The surface to descend on
     * @param start_x, start_y Starting position
     * @param learning_rate Step size
     * @param num_steps Number of iterations
     */
    static Plot3D::LineData3D GenerateGradientDescentPath(
        const Plot3D::SurfaceData& surface,
        double start_x, double start_y,
        double learning_rate = 0.1,
        size_t num_steps = 100);

    /**
     * Generate decision boundary surface for classification
     */
    static Plot3D::SurfaceData GenerateDecisionBoundary(
        size_t resolution = 50);

    // ========================================================================
    // 3D Scatter Data
    // ========================================================================

    /**
     * Generate 3D clustered data (K-means style)
     */
    static Plot3D::ScatterData3D Generate3DClusters(
        size_t points_per_cluster,
        size_t num_clusters,
        double cluster_spread = 1.0);

    /**
     * Generate 3D spiral
     */
    static Plot3D::LineData3D Generate3DSpiral(
        size_t points = 500,
        double height = 10.0,
        double radius = 5.0,
        double turns = 3.0);

    /**
     * Generate 3D helix
     */
    static Plot3D::LineData3D Generate3DHelix(
        size_t points = 500,
        double height = 10.0,
        double radius = 1.0);

    /**
     * Generate 3D Lissajous curve
     */
    static Plot3D::LineData3D GenerateLissajous(
        size_t points = 1000,
        double a = 3.0, double b = 2.0, double c = 1.0);

    /**
     * Generate 3D random walk
     */
    static Plot3D::LineData3D Generate3DRandomWalk(
        size_t steps = 1000,
        double step_size = 0.1);

    /**
     * Generate sphere point cloud
     */
    static Plot3D::ScatterData3D GenerateSpherePoints(
        size_t points = 1000,
        double radius = 1.0);

    /**
     * Generate cube point cloud
     */
    static Plot3D::ScatterData3D GenerateCubePoints(
        size_t points = 1000,
        double size = 2.0);

    // ========================================================================
    // Physics/Engineering
    // ========================================================================

    /**
     * Generate wave interference pattern
     * Simulates two point sources creating interference
     */
    static Plot3D::SurfaceData GenerateWaveInterference(
        double wavelength = 1.0,
        size_t resolution = 100);

    /**
     * Generate electric/magnetic field lines
     */
    static Plot3D::LineData3D GenerateFieldLines(
        const Plot3D::Point3D& source,
        size_t num_lines = 10,
        size_t points_per_line = 100);

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * Add noise to surface data
     */
    static void AddNoise(Plot3D::SurfaceData& surface, double amplitude);

    /**
     * Normalize surface Z-values to [0, 1]
     */
    static void NormalizeSurface(Plot3D::SurfaceData& surface);

    /**
     * Calculate surface normals for mesh
     */
    static void CalculateNormals(Plot3D::MeshData& mesh);

    /**
     * Convert surface data to mesh
     */
    static Plot3D::MeshData SurfaceToMesh(const Plot3D::SurfaceData& surface);

private:
    static std::mt19937& GetRNG();

    // Helper: Evaluate surface Z value at grid position
    static double EvaluateSurface(
        const Plot3D::SurfaceData& surface,
        double x, double y);

    // Helper: Create linear spaced grid
    static std::vector<double> Linspace(double start, double end, size_t count);
};

} // namespace cyxwiz::plotting
