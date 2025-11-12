#define _USE_MATH_DEFINES
#include "test_data_generator.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz::plotting {

// ============================================================================
// Random Number Generator
// ============================================================================

std::mt19937& TestDataGenerator::GetRNG() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

// ============================================================================
// Statistical Distributions
// ============================================================================

std::vector<double> TestDataGenerator::GenerateNormal(size_t count, double mean,
                                                      double std_dev) {
    std::normal_distribution<double> dist(mean, std_dev);
    std::vector<double> result(count);

    for (size_t i = 0; i < count; ++i) {
        result[i] = dist(GetRNG());
    }

    return result;
}

std::vector<double> TestDataGenerator::GenerateUniform(size_t count, double min,
                                                       double max) {
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> result(count);

    for (size_t i = 0; i < count; ++i) {
        result[i] = dist(GetRNG());
    }

    return result;
}

std::vector<double> TestDataGenerator::GenerateExponential(size_t count, double lambda) {
    std::exponential_distribution<double> dist(lambda);
    std::vector<double> result(count);

    for (size_t i = 0; i < count; ++i) {
        result[i] = dist(GetRNG());
    }

    return result;
}

std::vector<double> TestDataGenerator::GenerateBimodal(size_t count,
                                                       double mean1, double std1,
                                                       double mean2, double std2) {
    std::normal_distribution<double> dist1(mean1, std1);
    std::normal_distribution<double> dist2(mean2, std2);
    std::bernoulli_distribution coin(0.5);  // 50/50 split

    std::vector<double> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = coin(GetRNG()) ? dist1(GetRNG()) : dist2(GetRNG());
    }

    return result;
}

// ============================================================================
// Time Series / Signals
// ============================================================================

TestDataGenerator::DataSeries TestDataGenerator::GenerateSineWave(
    size_t count, double frequency, double noise_amplitude) {

    DataSeries series;
    series.x.resize(count);
    series.y.resize(count);

    std::normal_distribution<double> noise(0.0, noise_amplitude);

    for (size_t i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / count;
        series.x[i] = t;
        series.y[i] = std::sin(2.0 * M_PI * frequency * t);

        if (noise_amplitude > 0.0) {
            series.y[i] += noise(GetRNG());
        }
    }

    return series;
}

TestDataGenerator::DataSeries TestDataGenerator::GenerateCosineWave(
    size_t count, double frequency, double noise_amplitude) {

    DataSeries series;
    series.x.resize(count);
    series.y.resize(count);

    std::normal_distribution<double> noise(0.0, noise_amplitude);

    for (size_t i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / count;
        series.x[i] = t;
        series.y[i] = std::cos(2.0 * M_PI * frequency * t);

        if (noise_amplitude > 0.0) {
            series.y[i] += noise(GetRNG());
        }
    }

    return series;
}

TestDataGenerator::DataSeries TestDataGenerator::GenerateCompositeSignal(
    size_t count,
    const std::vector<double>& frequencies,
    const std::vector<double>& amplitudes) {

    DataSeries series;
    series.x.resize(count);
    series.y.resize(count, 0.0);

    for (size_t i = 0; i < count; ++i) {
        double t = static_cast<double>(i) / count;
        series.x[i] = t;

        for (size_t f = 0; f < frequencies.size(); ++f) {
            double amp = (f < amplitudes.size()) ? amplitudes[f] : 1.0;
            series.y[i] += amp * std::sin(2.0 * M_PI * frequencies[f] * t);
        }
    }

    return series;
}

TestDataGenerator::DataSeries TestDataGenerator::GenerateRandomWalk(
    size_t count, double step_size) {

    DataSeries series;
    series.x.resize(count);
    series.y.resize(count);

    std::normal_distribution<double> step(0.0, step_size);

    series.x[0] = 0.0;
    series.y[0] = 0.0;

    for (size_t i = 1; i < count; ++i) {
        series.x[i] = static_cast<double>(i);
        series.y[i] = series.y[i - 1] + step(GetRNG());
    }

    return series;
}

// ============================================================================
// ML Training Curves
// ============================================================================

TestDataGenerator::DataSeries TestDataGenerator::GenerateTrainingCurve(
    size_t epochs, double initial_loss, double final_loss, double noise_level) {

    DataSeries series;
    series.x.resize(epochs);
    series.y.resize(epochs);

    std::normal_distribution<double> noise(0.0, noise_level);

    for (size_t i = 0; i < epochs; ++i) {
        double t = static_cast<double>(i) / epochs;
        series.x[i] = static_cast<double>(i + 1);

        // Exponential decay with diminishing returns
        double progress = 1.0 - std::exp(-3.0 * t);
        series.y[i] = initial_loss - (initial_loss - final_loss) * progress;

        // Add noise
        series.y[i] += noise(GetRNG());

        // Ensure loss doesn't go negative
        series.y[i] = std::max(0.0, series.y[i]);
    }

    return series;
}

TestDataGenerator::DataSeries TestDataGenerator::GenerateAccuracyCurve(
    size_t epochs, double initial_acc, double final_acc, double noise_level) {

    DataSeries series;
    series.x.resize(epochs);
    series.y.resize(epochs);

    std::normal_distribution<double> noise(0.0, noise_level);

    for (size_t i = 0; i < epochs; ++i) {
        double t = static_cast<double>(i) / epochs;
        series.x[i] = static_cast<double>(i + 1);

        // Logarithmic growth with saturation
        double progress = std::log1p(3.0 * t) / std::log1p(3.0);
        series.y[i] = initial_acc + (final_acc - initial_acc) * progress;

        // Add noise
        series.y[i] += noise(GetRNG());

        // Clamp to [0, 1]
        series.y[i] = std::clamp(series.y[i], 0.0, 1.0);
    }

    return series;
}

TestDataGenerator::OverfitData TestDataGenerator::GenerateOverfittingCurves(
    size_t epochs) {

    OverfitData data;

    // Training loss: continues to decrease
    data.train_loss = GenerateTrainingCurve(epochs, 2.5, 0.05, 0.03);

    // Validation loss: decreases then increases (overfitting)
    data.val_loss.x.resize(epochs);
    data.val_loss.y.resize(epochs);

    std::normal_distribution<double> noise(0.0, 0.05);
    size_t overfit_start = epochs / 2;  // Start overfitting at 50%

    for (size_t i = 0; i < epochs; ++i) {
        data.val_loss.x[i] = static_cast<double>(i + 1);

        if (i < overfit_start) {
            // Normal decrease
            double t = static_cast<double>(i) / overfit_start;
            double progress = 1.0 - std::exp(-3.0 * t);
            data.val_loss.y[i] = 2.5 - (2.5 - 0.3) * progress;
        } else {
            // Start increasing (overfitting)
            double t = static_cast<double>(i - overfit_start) /
                      static_cast<double>(epochs - overfit_start);
            data.val_loss.y[i] = 0.3 + 0.5 * t;
        }

        data.val_loss.y[i] += noise(GetRNG());
    }

    return data;
}

// ============================================================================
// Categorical Data
// ============================================================================

TestDataGenerator::CategoricalData TestDataGenerator::GenerateCategoricalData(
    size_t num_categories, double min_val, double max_val) {

    CategoricalData data;
    data.categories.resize(num_categories);
    data.values.resize(num_categories);

    std::uniform_real_distribution<double> dist(min_val, max_val);

    for (size_t i = 0; i < num_categories; ++i) {
        data.categories[i] = "Cat_" + std::to_string(i + 1);
        data.values[i] = dist(GetRNG());
    }

    return data;
}

std::vector<double> TestDataGenerator::GenerateConfusionMatrix(
    size_t num_classes, double accuracy) {

    size_t total = num_classes * num_classes;
    std::vector<double> matrix(total, 0.0);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < num_classes; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            if (i == j) {
                // Diagonal (correct predictions)
                matrix[i * num_classes + j] = accuracy + dist(GetRNG()) * 0.1;
            } else {
                // Off-diagonal (incorrect predictions)
                matrix[i * num_classes + j] = (1.0 - accuracy) / (num_classes - 1);
                matrix[i * num_classes + j] += dist(GetRNG()) * 0.05;
            }
        }
    }

    return matrix;
}

// ============================================================================
// 2D Patterns
// ============================================================================

TestDataGenerator::ScatterData TestDataGenerator::GenerateClusteredData(
    size_t points_per_cluster, size_t num_clusters) {

    ScatterData data;
    size_t total_points = points_per_cluster * num_clusters;
    data.x.reserve(total_points);
    data.y.reserve(total_points);
    data.labels.reserve(total_points);

    std::uniform_real_distribution<double> center_dist(-5.0, 5.0);

    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        double cx = center_dist(GetRNG());
        double cy = center_dist(GetRNG());

        std::normal_distribution<double> x_dist(cx, 0.5);
        std::normal_distribution<double> y_dist(cy, 0.5);

        for (size_t i = 0; i < points_per_cluster; ++i) {
            data.x.push_back(x_dist(GetRNG()));
            data.y.push_back(y_dist(GetRNG()));
            data.labels.push_back(static_cast<int>(cluster));
        }
    }

    return data;
}

TestDataGenerator::ScatterData TestDataGenerator::GenerateSpiralData(
    size_t points, size_t num_spirals) {

    ScatterData data;
    data.x.reserve(points * num_spirals);
    data.y.reserve(points * num_spirals);
    data.labels.reserve(points * num_spirals);

    std::normal_distribution<double> noise(0.0, 0.1);

    for (size_t spiral = 0; spiral < num_spirals; ++spiral) {
        double offset = (2.0 * M_PI * spiral) / num_spirals;

        for (size_t i = 0; i < points; ++i) {
            double t = static_cast<double>(i) / points;
            double r = t * 5.0;
            double theta = t * 4.0 * M_PI + offset;

            data.x.push_back(r * std::cos(theta) + noise(GetRNG()));
            data.y.push_back(r * std::sin(theta) + noise(GetRNG()));
            data.labels.push_back(static_cast<int>(spiral));
        }
    }

    return data;
}

TestDataGenerator::ScatterData TestDataGenerator::GenerateXORData(
    size_t points_per_quadrant) {

    ScatterData data;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Quadrant 1 (++): label 1
    for (size_t i = 0; i < points_per_quadrant; ++i) {
        data.x.push_back(dist(GetRNG()));
        data.y.push_back(dist(GetRNG()));
        data.labels.push_back(1);
    }

    // Quadrant 2 (-+): label 0
    for (size_t i = 0; i < points_per_quadrant; ++i) {
        data.x.push_back(-dist(GetRNG()));
        data.y.push_back(dist(GetRNG()));
        data.labels.push_back(0);
    }

    // Quadrant 3 (--): label 1
    for (size_t i = 0; i < points_per_quadrant; ++i) {
        data.x.push_back(-dist(GetRNG()));
        data.y.push_back(-dist(GetRNG()));
        data.labels.push_back(1);
    }

    // Quadrant 4 (+-): label 0
    for (size_t i = 0; i < points_per_quadrant; ++i) {
        data.x.push_back(dist(GetRNG()));
        data.y.push_back(-dist(GetRNG()));
        data.labels.push_back(0);
    }

    return data;
}

// ============================================================================
// Heatmap Data
// ============================================================================

std::vector<double> TestDataGenerator::GenerateHeatmapData(size_t rows, size_t cols,
                                                           double min, double max) {
    std::vector<double> data(rows * cols);
    std::uniform_real_distribution<double> dist(min, max);

    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] = dist(GetRNG());
    }

    return data;
}

std::vector<double> TestDataGenerator::GenerateCorrelationMatrix(size_t dimensions) {
    std::vector<double> matrix(dimensions * dimensions);

    std::uniform_real_distribution<double> dist(-0.8, 0.8);

    for (size_t i = 0; i < dimensions; ++i) {
        for (size_t j = 0; j < dimensions; ++j) {
            if (i == j) {
                matrix[i * dimensions + j] = 1.0;  // Perfect self-correlation
            } else if (i > j) {
                // Copy from upper triangle (symmetric)
                matrix[i * dimensions + j] = matrix[j * dimensions + i];
            } else {
                matrix[i * dimensions + j] = dist(GetRNG());
            }
        }
    }

    return matrix;
}

// ============================================================================
// Special Datasets
// ============================================================================

std::vector<double> TestDataGenerator::GenerateDataWithOutliers(
    size_t count, double outlier_fraction) {

    auto data = GenerateNormal(count, 0.0, 1.0);

    size_t num_outliers = static_cast<size_t>(count * outlier_fraction);
    std::uniform_int_distribution<size_t> idx_dist(0, count - 1);
    std::uniform_real_distribution<double> outlier_dist(-10.0, 10.0);

    for (size_t i = 0; i < num_outliers; ++i) {
        size_t idx = idx_dist(GetRNG());
        data[idx] = outlier_dist(GetRNG());
    }

    return data;
}

std::vector<double> TestDataGenerator::GenerateMultiModal(
    size_t count,
    const std::vector<double>& means,
    const std::vector<double>& std_devs) {

    std::vector<double> result;
    result.reserve(count);

    std::uniform_int_distribution<size_t> mode_dist(0, means.size() - 1);

    for (size_t i = 0; i < count; ++i) {
        size_t mode = mode_dist(GetRNG());
        double mean = means[mode];
        double std_dev = (mode < std_devs.size()) ? std_devs[mode] : 1.0;

        std::normal_distribution<double> dist(mean, std_dev);
        result.push_back(dist(GetRNG()));
    }

    return result;
}

std::vector<double> TestDataGenerator::GenerateSkewedData(size_t count,
                                                          double skewness) {
    // Use log-normal distribution for positive skew
    std::lognormal_distribution<double> dist(0.0, skewness);
    std::vector<double> result(count);

    for (size_t i = 0; i < count; ++i) {
        result[i] = dist(GetRNG());
    }

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

void TestDataGenerator::AddNoise(std::vector<double>& data, double noise_amplitude) {
    std::normal_distribution<double> noise(0.0, noise_amplitude);

    for (double& val : data) {
        val += noise(GetRNG());
    }
}

void TestDataGenerator::Normalize(std::vector<double>& data) {
    if (data.empty()) return;

    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;

    if (range > 0.0) {
        for (double& val : data) {
            val = (val - min_val) / range;
        }
    }
}

void TestDataGenerator::Standardize(std::vector<double>& data) {
    if (data.empty()) return;

    // Calculate mean
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    // Calculate standard deviation
    double sq_sum = 0.0;
    for (double val : data) {
        sq_sum += (val - mean) * (val - mean);
    }
    double std_dev = std::sqrt(sq_sum / data.size());

    // Standardize
    if (std_dev > 0.0) {
        for (double& val : data) {
            val = (val - mean) / std_dev;
        }
    }
}

std::vector<double> TestDataGenerator::Linspace(double start, double end, size_t count) {
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
