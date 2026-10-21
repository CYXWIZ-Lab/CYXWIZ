// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <cmath>
#include <random>
#include <vector>

namespace cyxwiz {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// Synthetic Data Generation
// ============================================================================

std::vector<double> TimeSeries::GenerateWhiteNoise(int n, double mean, double std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, std);

    for (int i = 0; i < n; i++) {
        result[i] = dist(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateRandomWalk(int n, double start, double std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std);

    result[0] = start;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + dist(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateTrendSeasonal(int n, double trend_slope,
                                                       double seasonal_amplitude,
                                                       int period, double noise_std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < n; i++) {
        double trend = trend_slope * i;
        double seasonal = seasonal_amplitude * std::sin(TWO_PI * i / period);
        result[i] = trend + seasonal + noise(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateAR(int n, const std::vector<double>& coeffs, double noise_std) {
    int p = static_cast<int>(coeffs.size());
    std::vector<double> result(n, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    // Initialize with noise
    for (int i = 0; i < p; i++) {
        result[i] = noise(gen);
    }

    for (int i = p; i < n; i++) {
        double val = noise(gen);
        for (int j = 0; j < p; j++) {
            val += coeffs[j] * result[i - 1 - j];
        }
        result[i] = val;
    }
    return result;
}

std::vector<double> TimeSeries::GenerateMA(int n, const std::vector<double>& coeffs, double noise_std) {
    int q = static_cast<int>(coeffs.size());
    std::vector<double> result(n);
    std::vector<double> errors(n + q);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < n + q; i++) {
        errors[i] = noise(gen);
    }

    for (int i = 0; i < n; i++) {
        double val = errors[i + q];
        for (int j = 0; j < q; j++) {
            val += coeffs[j] * errors[i + q - 1 - j];
        }
        result[i] = val;
    }
    return result;
}

std::vector<double> TimeSeries::GenerateARIMA(int n, const std::vector<double>& ar_coeffs,
                                               const std::vector<double>& ma_coeffs,
                                               int d, double noise_std) {
    // Generate ARMA first
    int p = static_cast<int>(ar_coeffs.size());
    int q = static_cast<int>(ma_coeffs.size());

    std::vector<double> errors(n + q);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (size_t i = 0; i < errors.size(); i++) {
        errors[i] = noise(gen);
    }

    std::vector<double> arma(n, 0.0);
    int start = (p > 1) ? p : 1;
    for (int i = 0; i < start; i++) {
        arma[i] = errors[i + q];
    }

    for (int i = start; i < n; i++) {
        double val = errors[i + q];
        for (int j = 0; j < p; j++) {
            val += ar_coeffs[j] * arma[i - 1 - j];
        }
        for (int j = 0; j < q; j++) {
            val += ma_coeffs[j] * errors[i + q - 1 - j];
        }
        arma[i] = val;
    }

    // Integrate d times
    std::vector<double> result = arma;
    for (int i = 0; i < d; i++) {
        for (int j = 1; j < static_cast<int>(result.size()); j++) {
            result[j] += result[j - 1];
        }
    }

    return result;
}

// ============================================================================

} // namespace cyxwiz
