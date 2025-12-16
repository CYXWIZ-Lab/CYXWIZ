// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/data_transform.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Ensure Windows min/max macros are undefined after all includes
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

// GPU availability check (cached)
static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool CheckGPUAvailable() {
    if (s_gpu_checked) return s_use_gpu;
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        if (s_use_gpu) {
            spdlog::info("[DataTransform] GPU acceleration enabled");
        }
    } catch (const af::exception& e) {
        spdlog::warn("[DataTransform] GPU check failed: {}", e.what());
        s_use_gpu = false;
    }
#endif
    return s_use_gpu;
}

// ============================================================================
// Utility Functions
// ============================================================================

ColumnStats DataTransform::ComputeColumnStats(const std::vector<double>& data) {
    ColumnStats stats;
    if (data.empty()) return stats;

    stats.count = static_cast<int>(data.size());

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());

            stats.min = af::min<double>(gpu_data);
            stats.max = af::max<double>(gpu_data);
            stats.mean = af::mean<double>(gpu_data);
            stats.std_dev = af::stdev<double>(gpu_data, AF_VARIANCE_SAMPLE);

            // Sort for percentiles
            af::array sorted = af::sort(gpu_data);
            std::vector<double> sorted_cpu(data.size());
            sorted.host(sorted_cpu.data());

            stats.median = GetPercentile(sorted_cpu, 50.0);
            stats.q1 = GetPercentile(sorted_cpu, 25.0);
            stats.q3 = GetPercentile(sorted_cpu, 75.0);
            stats.iqr = stats.q3 - stats.q1;

            // Check for negatives and zeros (CPU)
            for (const auto& x : data) {
                if (x < 0) stats.has_negatives = true;
                if (x == 0) stats.has_zeros = true;
            }

            return stats;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU stats failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    stats.min = *std::min_element(data.begin(), data.end());
    stats.max = *std::max_element(data.begin(), data.end());
    stats.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    // Standard deviation
    double variance = 0.0;
    for (const auto& x : data) {
        variance += (x - stats.mean) * (x - stats.mean);
    }
    stats.std_dev = std::sqrt(variance / data.size());

    // Median and quartiles
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    stats.median = GetPercentile(sorted, 50.0);
    stats.q1 = GetPercentile(sorted, 25.0);
    stats.q3 = GetPercentile(sorted, 75.0);
    stats.iqr = stats.q3 - stats.q1;

    // Check for negatives and zeros
    for (const auto& x : data) {
        if (x < 0) stats.has_negatives = true;
        if (x == 0) stats.has_zeros = true;
    }

    return stats;
}

double DataTransform::GetPercentile(std::vector<double> data, double percentile) {
    if (data.empty()) return 0.0;

    std::sort(data.begin(), data.end());

    double idx = (percentile / 100.0) * (data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(idx));
    size_t upper = static_cast<size_t>(std::ceil(idx));

    if (lower == upper) return data[lower];

    double frac = idx - lower;
    return data[lower] * (1.0 - frac) + data[upper] * frac;
}

std::vector<int> DataTransform::DetectOutliers(const std::vector<double>& data, double iqr_multiplier) {
    std::vector<int> outliers;
    if (data.size() < 4) return outliers;

    auto stats = ComputeColumnStats(data);
    double lower_bound = stats.q1 - iqr_multiplier * stats.iqr;
    double upper_bound = stats.q3 + iqr_multiplier * stats.iqr;

    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < lower_bound || data[i] > upper_bound) {
            outliers.push_back(static_cast<int>(i));
        }
    }

    return outliers;
}

bool DataTransform::CanApplyLogTransform(const std::vector<double>& data, bool use_log1p) {
    for (const auto& x : data) {
        if (use_log1p) {
            if (x < -1.0) return false;  // log1p requires x > -1
        } else {
            if (x <= 0.0) return false;  // log requires x > 0
        }
    }
    return true;
}

bool DataTransform::CanApplyBoxCox(const std::vector<double>& data) {
    for (const auto& x : data) {
        if (x <= 0.0) return false;  // Box-Cox requires strictly positive
    }
    return true;
}

// ============================================================================
// Normalization (Min-Max Scaling)
// ============================================================================

TransformResult DataTransform::NormalizeColumn(const std::vector<double>& data,
                                                double range_min, double range_max) {
    TransformResult result;
    result.method = "normalize";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());

            double data_min = af::min<double>(gpu_data);
            double data_max = af::max<double>(gpu_data);
            double data_range = data_max - data_min;

            if (data_range < 1e-10) {
                result.error_message = "Data has zero range (constant values)";
                return result;
            }

            result.params["data_min"] = data_min;
            result.params["data_max"] = data_max;
            result.params["range_min"] = range_min;
            result.params["range_max"] = range_max;

            double target_range = range_max - range_min;
            af::array transformed = ((gpu_data - data_min) / data_range) * target_range + range_min;

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU normalize failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    double data_min = *std::min_element(data.begin(), data.end());
    double data_max = *std::max_element(data.begin(), data.end());
    double data_range = data_max - data_min;

    if (data_range < 1e-10) {
        result.error_message = "Data has zero range (constant values)";
        return result;
    }

    result.params["data_min"] = data_min;
    result.params["data_max"] = data_max;
    result.params["range_min"] = range_min;
    result.params["range_max"] = range_max;

    std::vector<double> transformed(data.size());
    double target_range = range_max - range_min;

    for (size_t i = 0; i < data.size(); ++i) {
        transformed[i] = ((data[i] - data_min) / data_range) * target_range + range_min;
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::Normalize(const std::vector<std::vector<double>>& data,
                                         double range_min, double range_max) {
    TransformResult result;
    result.method = "normalize";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    result.params["range_min"] = range_min;
    result.params["range_max"] = range_max;

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = NormalizeColumn(data[col], range_min, range_max);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_min"] = col_result.params["data_min"];
        result.params["col" + std::to_string(col) + "_max"] = col_result.params["data_max"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Standardization (Z-Score)
// ============================================================================

TransformResult DataTransform::StandardizeColumn(const std::vector<double>& data) {
    TransformResult result;
    result.method = "standardize";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());

            double mean = af::mean<double>(gpu_data);
            double std_dev = af::stdev<double>(gpu_data, AF_VARIANCE_SAMPLE);

            if (std_dev < 1e-10) {
                result.error_message = "Data has zero variance (constant values)";
                return result;
            }

            result.params["mean"] = mean;
            result.params["std_dev"] = std_dev;

            af::array transformed = (gpu_data - mean) / std_dev;

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU standardize failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    double variance = 0.0;
    for (const auto& x : data) {
        variance += (x - mean) * (x - mean);
    }
    double std_dev = std::sqrt(variance / data.size());

    if (std_dev < 1e-10) {
        result.error_message = "Data has zero variance (constant values)";
        return result;
    }

    result.params["mean"] = mean;
    result.params["std_dev"] = std_dev;

    std::vector<double> transformed(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        transformed[i] = (data[i] - mean) / std_dev;
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::Standardize(const std::vector<std::vector<double>>& data) {
    TransformResult result;
    result.method = "standardize";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = StandardizeColumn(data[col]);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_mean"] = col_result.params["mean"];
        result.params["col" + std::to_string(col) + "_std"] = col_result.params["std_dev"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Log Transforms
// ============================================================================

TransformResult DataTransform::LogTransformColumn(const std::vector<double>& data,
                                                   const std::string& base,
                                                   bool use_log1p) {
    TransformResult result;
    result.method = "log_transform";
    result.params["base"] = (base == "natural") ? 0.0 : (base == "log10") ? 10.0 : 2.0;
    result.params["use_log1p"] = use_log1p ? 1.0 : 0.0;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    if (!CanApplyLogTransform(data, use_log1p)) {
        result.error_message = use_log1p ?
            "Data contains values <= -1 (log1p requires x > -1)" :
            "Data contains non-positive values (log requires x > 0)";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());

            if (use_log1p) {
                gpu_data = gpu_data + 1.0;
            }

            af::array transformed;
            if (base == "natural") {
                transformed = af::log(gpu_data);
            } else if (base == "log10") {
                transformed = af::log10(gpu_data);
            } else {  // log2
                transformed = af::log(gpu_data) / std::log(2.0);
            }

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU log transform failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    std::vector<double> transformed(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        double x = use_log1p ? (data[i] + 1.0) : data[i];

        if (base == "natural") {
            transformed[i] = std::log(x);
        } else if (base == "log10") {
            transformed[i] = std::log10(x);
        } else {  // log2
            transformed[i] = std::log2(x);
        }
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::LogTransform(const std::vector<std::vector<double>>& data,
                                            const std::string& base,
                                            bool use_log1p) {
    TransformResult result;
    result.method = "log_transform";
    result.params["base"] = (base == "natural") ? 0.0 : (base == "log10") ? 10.0 : 2.0;
    result.params["use_log1p"] = use_log1p ? 1.0 : 0.0;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = LogTransformColumn(data[col], base, use_log1p);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
    }

    result.success = true;
    return result;
}

// ============================================================================
// Box-Cox Transform
// ============================================================================

BoxCoxLambdaResult DataTransform::FindOptimalLambda(const std::vector<double>& data,
                                                     double lambda_min,
                                                     double lambda_max,
                                                     int n_steps) {
    BoxCoxLambdaResult result;

    if (data.empty() || !CanApplyBoxCox(data)) {
        return result;
    }

    double step = (lambda_max - lambda_min) / n_steps;
    double best_ll = -std::numeric_limits<double>::infinity();
    double best_lambda = 0.0;

    int n = static_cast<int>(data.size());

    // Compute geometric mean for normalization
    double log_sum = 0.0;
    for (const auto& x : data) {
        log_sum += std::log(x);
    }
    double geom_mean = std::exp(log_sum / n);

    for (int i = 0; i <= n_steps; ++i) {
        double lambda = lambda_min + i * step;
        result.lambdas_tested.push_back(lambda);

        // Transform data
        std::vector<double> transformed(n);
        for (int j = 0; j < n; ++j) {
            if (std::abs(lambda) < 1e-10) {
                transformed[j] = geom_mean * std::log(data[j]);
            } else {
                transformed[j] = (std::pow(data[j], lambda) - 1.0) / (lambda * std::pow(geom_mean, lambda - 1));
            }
        }

        // Compute variance of transformed data
        double mean = std::accumulate(transformed.begin(), transformed.end(), 0.0) / n;
        double variance = 0.0;
        for (const auto& t : transformed) {
            variance += (t - mean) * (t - mean);
        }
        variance /= n;

        // Log-likelihood (proportional)
        double ll = -0.5 * n * std::log(variance);
        result.log_likelihoods.push_back(ll);

        if (ll > best_ll) {
            best_ll = ll;
            best_lambda = lambda;
        }
    }

    result.optimal_lambda = best_lambda;
    result.best_log_likelihood = best_ll;
    result.success = true;
    return result;
}

TransformResult DataTransform::BoxCoxColumn(const std::vector<double>& data,
                                            double lambda,
                                            bool auto_lambda) {
    TransformResult result;
    result.method = "boxcox";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    if (!CanApplyBoxCox(data)) {
        result.error_message = "Box-Cox requires strictly positive data";
        return result;
    }

    double use_lambda = lambda;
    if (auto_lambda) {
        auto lambda_result = FindOptimalLambda(data);
        if (!lambda_result.success) {
            result.error_message = "Failed to find optimal lambda";
            return result;
        }
        use_lambda = lambda_result.optimal_lambda;
    }

    result.params["lambda"] = use_lambda;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());

            af::array transformed;
            if (std::abs(use_lambda) < 1e-10) {
                transformed = af::log(gpu_data);
            } else {
                transformed = (af::pow(gpu_data, use_lambda) - 1.0) / use_lambda;
            }

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU BoxCox failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    std::vector<double> transformed(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        if (std::abs(use_lambda) < 1e-10) {
            transformed[i] = std::log(data[i]);
        } else {
            transformed[i] = (std::pow(data[i], use_lambda) - 1.0) / use_lambda;
        }
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::BoxCox(const std::vector<std::vector<double>>& data,
                                      double lambda,
                                      bool auto_lambda) {
    TransformResult result;
    result.method = "boxcox";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = BoxCoxColumn(data[col], lambda, auto_lambda);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_lambda"] = col_result.params["lambda"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Yeo-Johnson Transform
// ============================================================================

TransformResult DataTransform::YeoJohnsonColumn(const std::vector<double>& data,
                                                 double lambda,
                                                 bool auto_lambda) {
    TransformResult result;
    result.method = "yeojohnson";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    double use_lambda = lambda;
    if (auto_lambda) {
        // Simple search for optimal lambda (similar to Box-Cox but supports negatives)
        double best_ll = -std::numeric_limits<double>::infinity();
        for (double l = -5.0; l <= 5.0; l += 0.1) {
            std::vector<double> temp(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                double x = data[i];
                if (x >= 0) {
                    if (std::abs(l) < 1e-10) {
                        temp[i] = std::log(x + 1);
                    } else {
                        temp[i] = (std::pow(x + 1, l) - 1) / l;
                    }
                } else {
                    if (std::abs(l - 2) < 1e-10) {
                        temp[i] = -std::log(-x + 1);
                    } else {
                        temp[i] = -(std::pow(-x + 1, 2 - l) - 1) / (2 - l);
                    }
                }
            }

            double mean = std::accumulate(temp.begin(), temp.end(), 0.0) / temp.size();
            double var = 0;
            for (const auto& t : temp) var += (t - mean) * (t - mean);
            var /= temp.size();

            double ll = -0.5 * data.size() * std::log(var + 1e-10);
            if (ll > best_ll) {
                best_ll = ll;
                use_lambda = l;
            }
        }
    }

    result.params["lambda"] = use_lambda;

    std::vector<double> transformed(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double x = data[i];
        if (x >= 0) {
            if (std::abs(use_lambda) < 1e-10) {
                transformed[i] = std::log(x + 1);
            } else {
                transformed[i] = (std::pow(x + 1, use_lambda) - 1) / use_lambda;
            }
        } else {
            if (std::abs(use_lambda - 2) < 1e-10) {
                transformed[i] = -std::log(-x + 1);
            } else {
                transformed[i] = -(std::pow(-x + 1, 2 - use_lambda) - 1) / (2 - use_lambda);
            }
        }
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::YeoJohnson(const std::vector<std::vector<double>>& data,
                                          double lambda,
                                          bool auto_lambda) {
    TransformResult result;
    result.method = "yeojohnson";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = YeoJohnsonColumn(data[col], lambda, auto_lambda);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_lambda"] = col_result.params["lambda"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Robust Scaling
// ============================================================================

TransformResult DataTransform::RobustScaleColumn(const std::vector<double>& data) {
    TransformResult result;
    result.method = "robust_scale";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    auto stats = ComputeColumnStats(data);

    if (stats.iqr < 1e-10) {
        result.error_message = "Data has zero IQR";
        return result;
    }

    result.params["median"] = stats.median;
    result.params["iqr"] = stats.iqr;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());
            af::array transformed = (gpu_data - stats.median) / stats.iqr;

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU robust scale failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    std::vector<double> transformed(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        transformed[i] = (data[i] - stats.median) / stats.iqr;
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::RobustScale(const std::vector<std::vector<double>>& data) {
    TransformResult result;
    result.method = "robust_scale";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = RobustScaleColumn(data[col]);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_median"] = col_result.params["median"];
        result.params["col" + std::to_string(col) + "_iqr"] = col_result.params["iqr"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Max Abs Scaling
// ============================================================================

TransformResult DataTransform::MaxAbsScaleColumn(const std::vector<double>& data) {
    TransformResult result;
    result.method = "maxabs_scale";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && data.size() > 1000) {
        try {
            af::array gpu_data(static_cast<dim_t>(data.size()), data.data());
            double max_abs = af::max<double>(af::abs(gpu_data));

            if (max_abs < 1e-10) {
                result.error_message = "Data is all zeros";
                return result;
            }

            result.params["max_abs"] = max_abs;

            af::array transformed = gpu_data / max_abs;

            std::vector<double> transformed_cpu(data.size());
            transformed.host(transformed_cpu.data());

            result.transformed_data.push_back(transformed_cpu);
            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU maxabs scale failed, fallback to CPU: {}", e.what());
        }
    }
#endif

    // CPU fallback
    double max_abs = 0.0;
    for (const auto& x : data) {
        max_abs = std::max(max_abs, std::abs(x));
    }

    if (max_abs < 1e-10) {
        result.error_message = "Data is all zeros";
        return result;
    }

    result.params["max_abs"] = max_abs;

    std::vector<double> transformed(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        transformed[i] = data[i] / max_abs;
    }

    result.transformed_data.push_back(transformed);
    result.success = true;
    return result;
}

TransformResult DataTransform::MaxAbsScale(const std::vector<std::vector<double>>& data) {
    TransformResult result;
    result.method = "maxabs_scale";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        auto col_result = MaxAbsScaleColumn(data[col]);
        if (!col_result.success) {
            result.error_message = "Column " + std::to_string(col) + ": " + col_result.error_message;
            return result;
        }
        result.transformed_data.push_back(col_result.transformed_data[0]);
        result.params["col" + std::to_string(col) + "_maxabs"] = col_result.params["max_abs"];
    }

    result.success = true;
    return result;
}

// ============================================================================
// Quantile Transform
// ============================================================================

TransformResult DataTransform::QuantileTransform(const std::vector<std::vector<double>>& data,
                                                  const std::string& output_distribution,
                                                  int n_quantiles) {
    TransformResult result;
    result.method = "quantile_transform";
    result.params["output_distribution"] = (output_distribution == "normal") ? 1.0 : 0.0;
    result.params["n_quantiles"] = static_cast<double>(n_quantiles);

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    for (size_t col = 0; col < data.size(); ++col) {
        const auto& column = data[col];
        std::vector<double> transformed(column.size());

        // Sort indices
        std::vector<size_t> indices(column.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return column[a] < column[b];
        });

        // Assign ranks
        for (size_t i = 0; i < indices.size(); ++i) {
            double quantile = static_cast<double>(i) / (indices.size() - 1);

            if (output_distribution == "normal") {
                // Inverse CDF of standard normal
                // Using approximation
                double t = quantile;
                if (t <= 0.0) t = 0.0001;
                if (t >= 1.0) t = 0.9999;

                // Approximation for inverse normal CDF
                double a0 = 2.515517, a1 = 0.802853, a2 = 0.010328;
                double b1 = 1.432788, b2 = 0.189269, b3 = 0.001308;

                double p = t < 0.5 ? t : 1 - t;
                double s = std::sqrt(-2.0 * std::log(p));
                double z = s - (a0 + a1 * s + a2 * s * s) / (1 + b1 * s + b2 * s * s + b3 * s * s * s);

                transformed[indices[i]] = (t < 0.5) ? -z : z;
            } else {
                transformed[indices[i]] = quantile;
            }
        }

        result.transformed_data.push_back(transformed);
    }

    result.success = true;
    return result;
}

// ============================================================================
// Power Transform
// ============================================================================

TransformResult DataTransform::PowerTransform(const std::vector<std::vector<double>>& data,
                                               double power) {
    TransformResult result;
    result.method = "power_transform";
    result.params["power"] = power;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (CheckGPUAvailable() && !data.empty() && data[0].size() > 1000) {
        try {
            for (size_t col = 0; col < data.size(); ++col) {
                af::array gpu_data(static_cast<dim_t>(data[col].size()), data[col].data());

                // Check for negative values with fractional power
                if (power != std::floor(power)) {
                    double min_val = af::min<double>(gpu_data);
                    if (min_val < 0) {
                        result.error_message = "Cannot apply fractional power to negative values";
                        return result;
                    }
                }

                af::array transformed = af::pow(gpu_data, power);

                std::vector<double> transformed_cpu(data[col].size());
                transformed.host(transformed_cpu.data());

                result.transformed_data.push_back(transformed_cpu);
            }

            result.success = true;
            return result;
        } catch (const af::exception& e) {
            spdlog::warn("[DataTransform] GPU power transform failed, fallback to CPU: {}", e.what());
            result.transformed_data.clear();
        }
    }
#endif

    // CPU fallback
    for (size_t col = 0; col < data.size(); ++col) {
        std::vector<double> transformed(data[col].size());

        for (size_t i = 0; i < data[col].size(); ++i) {
            double x = data[col][i];
            if (x < 0 && power != std::floor(power)) {
                result.error_message = "Cannot apply fractional power to negative values";
                return result;
            }
            transformed[i] = std::pow(x, power);
        }

        result.transformed_data.push_back(transformed);
    }

    result.success = true;
    return result;
}

// ============================================================================
// Inverse Transforms
// ============================================================================

std::vector<double> DataTransform::InverseTransformColumn(
    const std::vector<double>& data,
    const TransformResult& original_result) {

    std::vector<double> result(data.size());

    if (original_result.method == "normalize") {
        double data_min = original_result.params.at("data_min");
        double data_max = original_result.params.at("data_max");
        double range_min = original_result.params.at("range_min");
        double range_max = original_result.params.at("range_max");
        double data_range = data_max - data_min;
        double target_range = range_max - range_min;

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = ((data[i] - range_min) / target_range) * data_range + data_min;
        }
    } else if (original_result.method == "standardize") {
        double mean = original_result.params.at("mean");
        double std_dev = original_result.params.at("std_dev");

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * std_dev + mean;
        }
    } else if (original_result.method == "log_transform") {
        double base_param = original_result.params.at("base");
        bool use_log1p = original_result.params.at("use_log1p") > 0.5;

        for (size_t i = 0; i < data.size(); ++i) {
            double val;
            if (base_param < 1.0) {  // natural
                val = std::exp(data[i]);
            } else if (base_param > 9.0) {  // log10
                val = std::pow(10.0, data[i]);
            } else {  // log2
                val = std::pow(2.0, data[i]);
            }
            result[i] = use_log1p ? (val - 1.0) : val;
        }
    } else if (original_result.method == "boxcox") {
        double lambda = original_result.params.at("lambda");

        for (size_t i = 0; i < data.size(); ++i) {
            if (std::abs(lambda) < 1e-10) {
                result[i] = std::exp(data[i]);
            } else {
                double temp = data[i] * lambda + 1.0;
                if (temp <= 0) temp = 1e-10;
                result[i] = std::pow(temp, 1.0 / lambda);
            }
        }
    } else if (original_result.method == "robust_scale") {
        double median = original_result.params.at("median");
        double iqr = original_result.params.at("iqr");

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * iqr + median;
        }
    } else if (original_result.method == "maxabs_scale") {
        double max_abs = original_result.params.at("max_abs");

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = data[i] * max_abs;
        }
    } else if (original_result.method == "power_transform") {
        double power = original_result.params.at("power");

        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = std::pow(data[i], 1.0 / power);
        }
    }

    return result;
}

std::vector<std::vector<double>> DataTransform::InverseTransform(
    const std::vector<std::vector<double>>& data,
    const TransformResult& original_result) {

    std::vector<std::vector<double>> result;

    for (size_t col = 0; col < data.size(); ++col) {
        TransformResult col_result;
        col_result.method = original_result.method;

        // Extract column-specific params
        if (original_result.method == "normalize") {
            col_result.params["data_min"] = original_result.params.at("col" + std::to_string(col) + "_min");
            col_result.params["data_max"] = original_result.params.at("col" + std::to_string(col) + "_max");
            col_result.params["range_min"] = original_result.params.at("range_min");
            col_result.params["range_max"] = original_result.params.at("range_max");
        } else if (original_result.method == "standardize") {
            col_result.params["mean"] = original_result.params.at("col" + std::to_string(col) + "_mean");
            col_result.params["std_dev"] = original_result.params.at("col" + std::to_string(col) + "_std");
        } else if (original_result.method == "boxcox" || original_result.method == "yeojohnson") {
            col_result.params["lambda"] = original_result.params.at("col" + std::to_string(col) + "_lambda");
        } else if (original_result.method == "robust_scale") {
            col_result.params["median"] = original_result.params.at("col" + std::to_string(col) + "_median");
            col_result.params["iqr"] = original_result.params.at("col" + std::to_string(col) + "_iqr");
        } else if (original_result.method == "maxabs_scale") {
            col_result.params["max_abs"] = original_result.params.at("col" + std::to_string(col) + "_maxabs");
        } else {
            col_result.params = original_result.params;
        }

        result.push_back(InverseTransformColumn(data[col], col_result));
    }

    return result;
}

// ============================================================================
// Normality Tests
// ============================================================================

NormalityTestResult DataTransform::ShapiroWilkTest(const std::vector<double>& data) {
    NormalityTestResult result;
    result.test_name = "Shapiro-Wilk";

    // Simplified Shapiro-Wilk implementation
    // For a full implementation, you'd need the coefficients table
    if (data.size() < 3 || data.size() > 5000) {
        result.statistic = 0;
        result.p_value = 0;
        return result;
    }

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    int n = static_cast<int>(data.size());
    double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;

    double ss = 0.0;
    for (const auto& x : sorted) {
        ss += (x - mean) * (x - mean);
    }

    // Simplified W statistic calculation
    double b = 0.0;
    for (int i = 0; i < n / 2; ++i) {
        double a_coef = 0.7071;  // Simplified coefficient
        b += a_coef * (sorted[n - 1 - i] - sorted[i]);
    }

    double W = (b * b) / ss;
    result.statistic = W;

    // Approximate p-value (simplified)
    result.p_value = W > 0.95 ? 0.1 : (W > 0.9 ? 0.05 : 0.01);
    result.is_normal = result.p_value > 0.05;

    return result;
}

NormalityTestResult DataTransform::DAgostinoPearsonTest(const std::vector<double>& data) {
    NormalityTestResult result;
    result.test_name = "D'Agostino-Pearson";

    if (data.size() < 20) {
        result.statistic = 0;
        result.p_value = 0;
        return result;
    }

    int n = static_cast<int>(data.size());
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;

    // Calculate moments
    double m2 = 0, m3 = 0, m4 = 0;
    for (const auto& x : data) {
        double d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    // Skewness and kurtosis
    double skew = m3 / std::pow(m2, 1.5);
    double kurt = m4 / (m2 * m2) - 3;

    // Z-scores for skewness and kurtosis
    double z_skew = std::abs(skew) / std::sqrt(6.0 / n);
    double z_kurt = std::abs(kurt) / std::sqrt(24.0 / n);

    // Combined K^2 statistic
    result.statistic = z_skew * z_skew + z_kurt * z_kurt;

    // Approximate p-value (chi-squared with 2 df)
    result.p_value = std::exp(-result.statistic / 2);
    result.is_normal = result.p_value > 0.05;

    return result;
}

} // namespace cyxwiz
