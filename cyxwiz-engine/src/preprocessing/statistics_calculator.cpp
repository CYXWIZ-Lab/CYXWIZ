#include "statistics_calculator.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cctype>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// Static member initialization
std::unordered_map<std::string, DatasetStatistics> StatisticsCalculator::cache_;
std::mutex StatisticsCalculator::cache_mutex_;

// ============================================================================
// Public Methods
// ============================================================================

DatasetStatistics StatisticsCalculator::Compute(
    const std::string& dataset_id,
    DataRegistry* registry,
    std::function<void(float)> progress_callback
) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(dataset_id);
        if (it != cache_.end()) {
            spdlog::info("StatisticsCalculator: Using cached statistics for dataset '{}'", dataset_id);
            return it->second;
        }
    }

    spdlog::info("StatisticsCalculator: Computing statistics for dataset '{}'", dataset_id);

    // Get dataset from registry
    auto dataset = registry->GetDataset(dataset_id);
    if (!dataset.IsValid()) {
        spdlog::error("StatisticsCalculator: Dataset '{}' not found", dataset_id);
        return DatasetStatistics{};
    }

    const size_t num_samples = dataset.Size();
    if (num_samples == 0) {
        spdlog::warn("StatisticsCalculator: Dataset '{}' has no samples", dataset_id);
        return DatasetStatistics{};
    }

    auto info = dataset.GetInfo();
    DatasetStatistics stats;
    stats.num_samples = num_samples;
    stats.shape = info.shape;

    // Determine number of channels
    size_t num_channels = 1;
    if (stats.shape.size() >= 3) {
        num_channels = stats.shape[2];  // Assuming [H, W, C] format
    }

    // Initialize statistics vectors
    stats.mean.resize(num_channels, 0.0f);
    stats.std.resize(num_channels, 0.0f);
    stats.min.resize(num_channels, std::numeric_limits<float>::max());
    stats.max.resize(num_channels, std::numeric_limits<float>::lowest());
    stats.median.resize(num_channels, 0.0f);
    stats.q25.resize(num_channels, 0.0f);
    stats.q75.resize(num_channels, 0.0f);
    stats.histograms.resize(num_channels);

    // Collect data per channel
    std::vector<std::vector<float>> channel_data(num_channels);
    for (size_t ch = 0; ch < num_channels; ++ch) {
        channel_data[ch].reserve(num_samples * (stats.shape.size() >= 2 ? stats.shape[0] * stats.shape[1] : 1));
    }

    // First pass: Collect all data, compute min/max/mean
    for (size_t i = 0; i < num_samples; ++i) {
        if (progress_callback && i % 100 == 0) {
            progress_callback(static_cast<float>(i) / num_samples * 0.5f);  // 0-50%
        }

        auto [data, label] = dataset.GetSample(i);

        // Process each pixel/value
        const size_t values_per_sample = data.size() / num_channels;
        for (size_t j = 0; j < values_per_sample; ++j) {
            for (size_t ch = 0; ch < num_channels; ++ch) {
                const size_t idx = j * num_channels + ch;
                if (idx < data.size()) {
                    const float value = data[idx];
                    channel_data[ch].push_back(value);
                    stats.min[ch] = std::min(stats.min[ch], value);
                    stats.max[ch] = std::max(stats.max[ch], value);
                    stats.mean[ch] += value;
                }
            }
        }
    }

    // Finalize mean
    for (size_t ch = 0; ch < num_channels; ++ch) {
        if (!channel_data[ch].empty()) {
            stats.mean[ch] /= channel_data[ch].size();
        }
    }

    // Second pass: Compute std, median, percentiles, histogram
    for (size_t ch = 0; ch < num_channels; ++ch) {
        if (progress_callback) {
            progress_callback(0.5f + (static_cast<float>(ch) / num_channels) * 0.5f);  // 50-100%
        }

        auto& data = channel_data[ch];
        if (data.empty()) continue;

        // Compute std
        stats.std[ch] = ComputeStd(data, stats.mean[ch]);

        // Sort for percentiles (expensive!)
        std::sort(data.begin(), data.end());

        // Compute percentiles
        stats.median[ch] = ComputePercentile(data, 50.0f);
        stats.q25[ch] = ComputePercentile(data, 25.0f);
        stats.q75[ch] = ComputePercentile(data, 75.0f);

        // Compute histogram
        stats.histograms[ch] = ComputeHistogram(data, 256);
    }

    // Compute PCA for whitening using ArrayFire SVD
    spdlog::info("Computing PCA components for whitening...");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        // 1. Calculate total features per sample from shape
        size_t n_features = 1;
        for (size_t dim : stats.shape) {
            n_features *= dim;
        }
        size_t n_samples = num_samples;

        // 2. Build data matrix [n_samples x n_features]
        std::vector<float> data_flat;
        data_flat.reserve(n_samples * n_features);

        for (size_t i = 0; i < n_samples; ++i) {
            auto [sample_data, label] = dataset.GetSample(i);

            for (size_t j = 0; j < n_features; ++j) {
                data_flat.push_back(sample_data[j]);
            }

            // Update progress every 10% during data loading
            if (progress_callback && i % std::max(1ULL, n_samples / 10) == 0) {
                progress_callback(0.75f + 0.1f * (float(i) / n_samples));
            }
        }

        // 3. Create ArrayFire matrix [n_samples x n_features]
        af::array data_af(n_features, n_samples, data_flat.data());  // Column-major
        data_af = af::transpose(data_af);  // Now [n_samples x n_features]

        // 4. Center the data (subtract mean)
        af::array mean_af = af::mean(data_af, 0);  // Mean along rows (per feature)
        af::array centered = data_af - af::tile(mean_af, static_cast<unsigned>(n_samples), 1);

        // 5. Perform SVD: X = U Σ V^T
        //    - Eigenvectors of covariance = columns of V
        //    - Eigenvalues = (σ² / n_samples)
        af::array U, S, Vt;
        af::svd(U, S, Vt, centered);

        // 6. Extract results to CPU
        // Eigenvectors (columns of V = rows of Vt transposed)
        af::array V = af::transpose(Vt);
        std::vector<float> V_host(n_features * n_features);
        V.host(V_host.data());

        // Singular values
        std::vector<float> S_host(n_features);
        S.host(S_host.data());

        // Mean vector
        std::vector<float> mean_host(n_features);
        mean_af.host(mean_host.data());

        // 7. Store in stats structure
        stats.pca_components.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            stats.pca_components[i].resize(n_features);
            for (size_t j = 0; j < n_features; ++j) {
                // V is column-major from ArrayFire
                stats.pca_components[i][j] = V_host[j * n_features + i];
            }
        }

        // Eigenvalues = (σ² / n_samples)
        stats.pca_eigenvalues.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            float sigma = S_host[i];
            stats.pca_eigenvalues[i] = (sigma * sigma) / n_samples;
        }

        // Store mean for centering
        stats.pca_mean.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            stats.pca_mean[i] = mean_host[i];
        }

        stats.pca_computed = true;

        spdlog::info("PCA computed (GPU-accelerated): {} components", n_features);

    } catch (const af::exception& e) {
        spdlog::error("ArrayFire PCA computation failed: {}", e.what());
        stats.pca_computed = false;
    } catch (const std::exception& e) {
        spdlog::error("PCA computation failed: {}", e.what());
        stats.pca_computed = false;
    }
#else
    spdlog::warn("PCA computation requires ArrayFire (GPU support not available)");
    stats.pca_computed = false;
#endif

    stats.is_valid = true;

    if (progress_callback) {
        progress_callback(1.0f);
    }

    spdlog::info("StatisticsCalculator: Computed statistics for dataset '{}' ({} channels, {} samples)",
                 dataset_id, num_channels, num_samples);

    // Cache the result
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cache_[dataset_id] = stats;
    }

    return stats;
}

NormalizationStrategy StatisticsCalculator::AutoDetectStrategy(
    const DatasetStatistics& stats,
    const std::string& dataset_name
) {
    if (!stats.is_valid) {
        return NormalizationStrategy::None;
    }

    // Check dataset name for known patterns
    std::string name_lower = ToLowerCase(dataset_name);

    if (name_lower.find("mnist") != std::string::npos) {
        spdlog::info("StatisticsCalculator: Auto-detected MNIST dataset");
        return NormalizationStrategy::MNIST;
    }
    if (name_lower.find("cifar") != std::string::npos) {
        spdlog::info("StatisticsCalculator: Auto-detected CIFAR-10 dataset");
        return NormalizationStrategy::CIFAR10;
    }
    if (name_lower.find("imagenet") != std::string::npos) {
        spdlog::info("StatisticsCalculator: Auto-detected ImageNet dataset");
        return NormalizationStrategy::ImageNet;
    }

    // Check statistics
    // If data is already in [0, 1] range, likely pre-normalized
    if (stats.IsNormalized()) {
        spdlog::info("StatisticsCalculator: Data already normalized to [0, 1], using Custom strategy with computed stats");
        return NormalizationStrategy::Custom;
    }

    // Default: ImageNet for 3-channel, MNIST for 1-channel
    if (stats.GetNumChannels() == 3) {
        spdlog::info("StatisticsCalculator: Auto-detected 3-channel data, using ImageNet preset");
        return NormalizationStrategy::ImageNet;
    } else if (stats.GetNumChannels() == 1) {
        spdlog::info("StatisticsCalculator: Auto-detected 1-channel data, using MNIST preset");
        return NormalizationStrategy::MNIST;
    }

    spdlog::info("StatisticsCalculator: No specific preset detected, using None");
    return NormalizationStrategy::None;
}

DatasetStatistics StatisticsCalculator::GetCached(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_.find(dataset_id);
    if (it != cache_.end()) {
        return it->second;
    }
    return DatasetStatistics{};
}

void StatisticsCalculator::ClearCache(const std::string& dataset_id) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.erase(dataset_id);
    spdlog::info("StatisticsCalculator: Cleared cache for dataset '{}'", dataset_id);
}

void StatisticsCalculator::ClearAllCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    spdlog::info("StatisticsCalculator: Cleared all cached statistics");
}

// ============================================================================
// Private Helper Methods
// ============================================================================

std::vector<int> StatisticsCalculator::ComputeHistogram(const std::vector<float>& data, int num_bins) {
    if (data.empty()) {
        return std::vector<int>(num_bins, 0);
    }

    std::vector<int> histogram(num_bins, 0);

    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    if (max_val - min_val < 1e-6f) {
        // All values are the same
        histogram[0] = static_cast<int>(data.size());
        return histogram;
    }

    float bin_width = (max_val - min_val) / num_bins;

    for (float value : data) {
        int bin = static_cast<int>((value - min_val) / bin_width);
        bin = std::clamp(bin, 0, num_bins - 1);
        histogram[bin]++;
    }

    return histogram;
}

float StatisticsCalculator::ComputePercentile(std::vector<float> data, float percentile) {
    if (data.empty()) {
        return 0.0f;
    }

    // Assume data is already sorted
    size_t index = static_cast<size_t>((percentile / 100.0f) * (data.size() - 1));
    index = std::min(index, data.size() - 1);
    return data[index];
}

float StatisticsCalculator::ComputeMean(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f;
    }

    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return static_cast<float>(sum / data.size());
}

float StatisticsCalculator::ComputeStd(const std::vector<float>& data, float mean) {
    if (data.empty()) {
        return 1.0f;
    }

    double variance = 0.0;
    for (float value : data) {
        double diff = value - mean;
        variance += diff * diff;
    }
    variance /= data.size();

    return static_cast<float>(std::sqrt(variance));
}

std::string StatisticsCalculator::ToLowerCase(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

} // namespace cyxwiz
