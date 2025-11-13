#pragma once

#include <vector>
#include <string>
#include <random>
#include <functional>

namespace cyxwiz::plotting {

/**
 * TestDataGenerator - Utilities for generating fake data for testing plots
 */
class TestDataGenerator {
public:
    struct DataSeries {
        std::vector<double> x;
        std::vector<double> y;
    };

    // ========================================================================
    // Statistical Distributions
    // ========================================================================

    /**
     * Generate data from normal distribution
     * @param count Number of samples
     * @param mean Mean of distribution
     * @param std_dev Standard deviation
     */
    static std::vector<double> GenerateNormal(size_t count, double mean = 0.0,
                                              double std_dev = 1.0);

    /**
     * Generate data from uniform distribution
     * @param count Number of samples
     * @param min Minimum value
     * @param max Maximum value
     */
    static std::vector<double> GenerateUniform(size_t count, double min = 0.0,
                                               double max = 1.0);

    /**
     * Generate data from exponential distribution
     * @param count Number of samples
     * @param lambda Rate parameter
     */
    static std::vector<double> GenerateExponential(size_t count, double lambda = 1.0);

    /**
     * Generate bimodal distribution (mixture of two normals)
     */
    static std::vector<double> GenerateBimodal(size_t count,
                                               double mean1 = -2.0, double std1 = 1.0,
                                               double mean2 = 2.0, double std2 = 1.0);

    // ========================================================================
    // Mathematical Functions (Real Calculations)
    // ========================================================================

    /**
     * Plot any mathematical function y = f(x)
     * @param func Function to plot
     * @param x_min Start of x range
     * @param x_max End of x range
     * @param count Number of points to calculate
     */
    static DataSeries PlotFunction(std::function<double(double)> func,
                                   double x_min, double x_max, size_t count);

    /**
     * Plot polynomial: y = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0
     * @param coefficients Coefficients [a_0, a_1, ..., a_n]
     */
    static DataSeries PlotPolynomial(const std::vector<double>& coefficients,
                                     double x_min, double x_max, size_t count);

    /**
     * Plot trigonometric functions
     */
    static DataSeries PlotSine(double amplitude, double frequency, double phase,
                               double x_min, double x_max, size_t count);
    static DataSeries PlotCosine(double amplitude, double frequency, double phase,
                                 double x_min, double x_max, size_t count);
    static DataSeries PlotTangent(double amplitude, double frequency,
                                  double x_min, double x_max, size_t count);

    /**
     * Plot exponential: y = a * exp(b * x)
     */
    static DataSeries PlotExponential(double a, double b,
                                      double x_min, double x_max, size_t count);

    /**
     * Plot logarithmic: y = a * log(b * x)
     */
    static DataSeries PlotLogarithmic(double a, double b,
                                      double x_min, double x_max, size_t count);

    /**
     * Plot power function: y = a * x^b
     */
    static DataSeries PlotPower(double a, double b,
                                double x_min, double x_max, size_t count);

    /**
     * Plot rational function: y = (a*x + b) / (c*x + d)
     */
    static DataSeries PlotRational(double a, double b, double c, double d,
                                   double x_min, double x_max, size_t count);

    /**
     * Plot hyperbolic functions
     */
    static DataSeries PlotSinh(double amplitude, double x_min, double x_max, size_t count);
    static DataSeries PlotCosh(double amplitude, double x_min, double x_max, size_t count);
    static DataSeries PlotTanh(double amplitude, double x_min, double x_max, size_t count);

    /**
     * Plot Gaussian (bell curve): y = a * exp(-(x-mu)^2 / (2*sigma^2))
     */
    static DataSeries PlotGaussian(double amplitude, double mean, double std_dev,
                                   double x_min, double x_max, size_t count);

    /**
     * Plot parametric curve (x(t), y(t))
     */
    struct ParametricData {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> t;  // Parameter values
    };
    static ParametricData PlotParametric(
        std::function<double(double)> x_func,
        std::function<double(double)> y_func,
        double t_min, double t_max, size_t count);

    /**
     * Plot circle: x^2 + y^2 = r^2
     */
    static ParametricData PlotCircle(double radius, size_t count);

    /**
     * Plot ellipse: (x/a)^2 + (y/b)^2 = 1
     */
    static ParametricData PlotEllipse(double a, double b, size_t count);

    /**
     * Plot lissajous curve: x = A*sin(a*t + delta), y = B*sin(b*t)
     */
    static ParametricData PlotLissajous(double A, double B, double a, double b,
                                        double delta, size_t count);

    // ========================================================================
    // Time Series / Signals
    // ========================================================================

    /**
     * Generate sine wave
     * @param count Number of points
     * @param frequency Frequency in Hz
     * @param noise_amplitude Noise level (0 = no noise)
     */
    static DataSeries GenerateSineWave(size_t count, double frequency = 1.0,
                                       double noise_amplitude = 0.0);

    /**
     * Generate cosine wave
     */
    static DataSeries GenerateCosineWave(size_t count, double frequency = 1.0,
                                         double noise_amplitude = 0.0);

    /**
     * Generate composite signal (sum of multiple frequencies)
     */
    static DataSeries GenerateCompositeSignal(size_t count,
                                              const std::vector<double>& frequencies,
                                              const std::vector<double>& amplitudes);

    /**
     * Generate random walk
     */
    static DataSeries GenerateRandomWalk(size_t count, double step_size = 0.1);

    // ========================================================================
    // ML Training Curves
    // ========================================================================

    /**
     * Generate realistic training loss curve
     * @param epochs Number of training epochs
     * @param initial_loss Starting loss value
     * @param final_loss Final loss value
     * @param noise_level Noise/variance in the curve
     */
    static DataSeries GenerateTrainingCurve(size_t epochs, double initial_loss = 2.5,
                                           double final_loss = 0.1,
                                           double noise_level = 0.05);

    /**
     * Generate accuracy curve (inverse of loss)
     */
    static DataSeries GenerateAccuracyCurve(size_t epochs, double initial_acc = 0.3,
                                           double final_acc = 0.95,
                                           double noise_level = 0.02);

    /**
     * Generate overfitting scenario (train vs validation curves)
     */
    struct OverfitData {
        DataSeries train_loss;
        DataSeries val_loss;
    };
    static OverfitData GenerateOverfittingCurves(size_t epochs);

    // ========================================================================
    // Categorical Data
    // ========================================================================

    /**
     * Generate categorical data for bar charts
     */
    struct CategoricalData {
        std::vector<std::string> categories;
        std::vector<double> values;
    };
    static CategoricalData GenerateCategoricalData(size_t num_categories,
                                                   double min_val = 0.0,
                                                   double max_val = 100.0);

    /**
     * Generate confusion matrix data
     */
    static std::vector<double> GenerateConfusionMatrix(size_t num_classes,
                                                       double accuracy = 0.85);

    // ========================================================================
    // 2D Patterns
    // ========================================================================

    /**
     * Generate 2D scatter data with clusters
     */
    struct ScatterData {
        std::vector<double> x;
        std::vector<double> y;
        std::vector<int> labels;  // Cluster labels
    };
    static ScatterData GenerateClusteredData(size_t points_per_cluster,
                                            size_t num_clusters);

    /**
     * Generate spiral pattern
     */
    static ScatterData GenerateSpiralData(size_t points, size_t num_spirals = 2);

    /**
     * Generate XOR pattern (non-linearly separable)
     */
    static ScatterData GenerateXORData(size_t points_per_quadrant);

    // ========================================================================
    // Heatmap Data
    // ========================================================================

    /**
     * Generate 2D heatmap data (e.g., for weight visualization)
     */
    static std::vector<double> GenerateHeatmapData(size_t rows, size_t cols,
                                                   double min = -1.0,
                                                   double max = 1.0);

    /**
     * Generate correlation matrix
     */
    static std::vector<double> GenerateCorrelationMatrix(size_t dimensions);

    // ========================================================================
    // Special Datasets
    // ========================================================================

    /**
     * Generate data with outliers
     */
    static std::vector<double> GenerateDataWithOutliers(size_t count,
                                                        double outlier_fraction = 0.05);

    /**
     * Generate multi-modal distribution
     */
    static std::vector<double> GenerateMultiModal(size_t count,
                                                  const std::vector<double>& means,
                                                  const std::vector<double>& std_devs);

    /**
     * Generate skewed distribution
     */
    static std::vector<double> GenerateSkewedData(size_t count, double skewness = 2.0);

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /**
     * Add noise to existing data
     */
    static void AddNoise(std::vector<double>& data, double noise_amplitude);

    /**
     * Normalize data to [0, 1]
     */
    static void Normalize(std::vector<double>& data);

    /**
     * Standardize data (mean=0, std=1)
     */
    static void Standardize(std::vector<double>& data);

private:
    // Random number generator (thread_local for thread safety)
    static std::mt19937& GetRNG();

    // Helper: Generate linearly spaced values
    static std::vector<double> Linspace(double start, double end, size_t count);
};

} // namespace cyxwiz::plotting
