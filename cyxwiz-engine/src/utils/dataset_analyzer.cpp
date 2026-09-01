#include "dataset_analyzer.h"
#include "../core/data_registry.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <future>

namespace cyxwiz {

// ============================================================================
// ClassDistribution Methods
// ============================================================================

float ClassDistribution::GetImbalanceRatio() const {
    if (counts.empty()) return 1.0f;

    auto [min_it, max_it] = std::minmax_element(
        counts.begin(), counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    if (min_it->second == 0) return std::numeric_limits<float>::infinity();
    return static_cast<float>(max_it->second) / static_cast<float>(min_it->second);
}

bool ClassDistribution::IsBalanced(float threshold) const {
    return GetImbalanceRatio() < threshold;
}

// ============================================================================
// DatasetAnalyzer - Main Analytics
// ============================================================================

DatasetAnalytics DatasetAnalyzer::ComputeAnalytics(
    const std::string& dataset_name,
    DataRegistry* registry,
    std::function<void(float, const std::string&)> progress_callback
) {
    DatasetAnalytics analytics;

    if (!registry) {
        analytics.error_message = "Invalid registry pointer";
        return analytics;
    }

    try {
        auto handle = registry->GetDataset(dataset_name);
        if (!handle) {
            analytics.error_message = "Dataset not found: " + dataset_name;
            return analytics;
        }

        size_t total_samples = handle.Size();
        if (total_samples == 0) {
            analytics.error_message = "Dataset is empty";
            return analytics;
        }

        auto info = handle.GetInfo();

        // Extract dimensions from shape [H, W, C] or [H, W] or [H*W]
        int height = 0, width = 0, channels = 1;
        if (info.shape.size() >= 3) {
            height = static_cast<int>(info.shape[0]);
            width = static_cast<int>(info.shape[1]);
            channels = static_cast<int>(info.shape[2]);
        } else if (info.shape.size() == 2) {
            height = static_cast<int>(info.shape[0]);
            width = static_cast<int>(info.shape[1]);
            channels = 1;  // Grayscale
        } else if (info.shape.size() == 1) {
            // Flattened, assume square image
            size_t total = info.shape[0];
            height = width = static_cast<int>(std::sqrt(total));
            channels = 1;
        }

        if (progress_callback) {
            progress_callback(0.0f, "Computing class distribution...");
        }

        // 1. Compute class distribution (parallelized)
        std::vector<int> labels(total_samples);

        // Use multiple threads for label collection
        const size_t num_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size_t(12));
        const size_t chunk_size = (total_samples + num_threads - 1) / num_threads;

        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, total_samples);
            if (start >= total_samples) break;

            futures.push_back(std::async(std::launch::async, [&handle, &labels, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    auto [_, label] = handle.GetSample(i);
                    labels[i] = label;
                }
            }));
        }

        // Wait for all threads
        for (auto& f : futures) {
            f.get();
        }

        if (progress_callback) {
            progress_callback(0.15f, "Computing class distribution...");
        }

        analytics.class_distribution = ComputeClassDistribution(labels);

        // 2. Compute dimension statistics
        if (progress_callback) {
            progress_callback(0.2f, "Computing dimension statistics...");
        }
        analytics.dimension_stats = ComputeDimensionStats(dataset_name, registry);

        // 3. Load samples for histogram and outlier detection
        if (progress_callback) {
            progress_callback(0.4f, "Loading samples...");
        }

        // Sample up to 1000 images for analysis (full dataset may be too large)
        size_t sample_size = std::min(total_samples, size_t(1000));
        size_t step = (total_samples > sample_size) ? (total_samples / sample_size) : 1;

        std::vector<std::vector<float>> images;
        std::vector<int> sampled_labels;
        std::vector<size_t> sampled_indices;

        images.reserve(sample_size);
        sampled_labels.reserve(sample_size);
        sampled_indices.reserve(sample_size);

        for (size_t i = 0; i < total_samples; i += step) {
            if (images.size() >= sample_size) break;

            auto [sample, label] = handle.GetSample(i);
            images.push_back(sample);
            sampled_labels.push_back(label);
            sampled_indices.push_back(i);

            if (progress_callback && images.size() % 100 == 0) {
                float prog = 0.4f + 0.3f * (static_cast<float>(images.size()) / sample_size);
                progress_callback(prog, std::format("Loaded {}/{} samples", images.size(), sample_size));
            }
        }

        // 4. Compute color histogram
        if (progress_callback) {
            progress_callback(0.7f, "Computing color histogram...");
        }
        analytics.color_histogram = ComputeColorHistogram(images, width, height, channels);

        // 5. Compute brightness/contrast statistics (parallelized)
        if (progress_callback) {
            progress_callback(0.8f, "Computing brightness/contrast...");
        }

        std::vector<float> brightness_values(images.size());
        std::vector<float> contrast_values(images.size());

        // Parallelize brightness/contrast computation
        {
            const size_t img_count = images.size();
            const size_t bc_threads = std::min(static_cast<size_t>(std::thread::hardware_concurrency()), size_t(8));
            const size_t bc_chunk = (img_count + bc_threads - 1) / bc_threads;

            std::vector<std::future<void>> bc_futures;
            for (size_t t = 0; t < bc_threads; ++t) {
                size_t start = t * bc_chunk;
                size_t end = std::min(start + bc_chunk, img_count);
                if (start >= img_count) break;

                bc_futures.push_back(std::async(std::launch::async,
                    [&images, &brightness_values, &contrast_values, channels, start, end]() {
                        for (size_t i = start; i < end; ++i) {
                            brightness_values[i] = ComputeBrightness(images[i], channels);
                            contrast_values[i] = ComputeContrast(images[i]);
                        }
                    }));
            }
            for (auto& f : bc_futures) f.get();
        }

        // Compute mean/std
        float sum_brightness = std::accumulate(brightness_values.begin(), brightness_values.end(), 0.0f);
        analytics.mean_brightness = sum_brightness / brightness_values.size();

        float sum_contrast = std::accumulate(contrast_values.begin(), contrast_values.end(), 0.0f);
        analytics.mean_contrast = sum_contrast / contrast_values.size();

        float var_brightness = 0.0f, var_contrast = 0.0f;
        for (size_t i = 0; i < brightness_values.size(); ++i) {
            var_brightness += std::pow(brightness_values[i] - analytics.mean_brightness, 2);
            var_contrast += std::pow(contrast_values[i] - analytics.mean_contrast, 2);
        }
        analytics.std_brightness = std::sqrt(var_brightness / brightness_values.size());
        analytics.std_contrast = std::sqrt(var_contrast / contrast_values.size());

        // 6. Detect outliers
        if (progress_callback) {
            progress_callback(0.9f, "Detecting outliers...");
        }
        analytics.outliers = DetectOutliers(images, channels);

        if (progress_callback) {
            progress_callback(1.0f, "Analytics complete!");
        }

        analytics.success = true;
        return analytics;

    } catch (const std::exception& e) {
        analytics.error_message = std::string("Exception: ") + e.what();
        spdlog::error("DatasetAnalyzer: {}", analytics.error_message);
        return analytics;
    }
}

// ============================================================================
// Color Histogram
// ============================================================================

ColorHistogram DatasetAnalyzer::ComputeColorHistogram(
    const std::vector<std::vector<float>>& images,
    int width, int height, int channels
) {
    ColorHistogram hist;
    hist.is_grayscale = (channels == 1);

    for (const auto& image : images) {
        size_t pixel_count = width * height;

        if (channels == 1) {
            // Grayscale
            for (size_t i = 0; i < pixel_count; ++i) {
                int bin = static_cast<int>(std::clamp(image[i] * 255.0f, 0.0f, 255.0f));
                hist.gray_bins[bin]++;
                hist.total_pixels++;
            }
        } else if (channels == 3) {
            // RGB
            for (size_t i = 0; i < pixel_count; ++i) {
                int r = static_cast<int>(std::clamp(image[i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                int g = static_cast<int>(std::clamp(image[i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                int b = static_cast<int>(std::clamp(image[i * 3 + 2] * 255.0f, 0.0f, 255.0f));

                hist.red_bins[r]++;
                hist.green_bins[g]++;
                hist.blue_bins[b]++;
                hist.total_pixels++;
            }
        }
    }

    return hist;
}

// ============================================================================
// Class Distribution
// ============================================================================

ClassDistribution DatasetAnalyzer::ComputeClassDistribution(
    const std::vector<int>& labels
) {
    ClassDistribution dist;
    dist.total_samples = labels.size();

    // Count occurrences
    for (int label : labels) {
        dist.counts[label]++;
    }

    dist.num_classes = static_cast<int>(dist.counts.size());

    // Compute percentages
    for (const auto& [label, count] : dist.counts) {
        dist.percentages[label] = (static_cast<float>(count) / dist.total_samples) * 100.0f;
    }

    return dist;
}

// ============================================================================
// Dimension Statistics
// ============================================================================

DimensionStatistics DatasetAnalyzer::ComputeDimensionStats(
    const std::string& dataset_name,
    DataRegistry* registry
) {
    DimensionStatistics stats;

    auto handle = registry->GetDataset(dataset_name);
    if (!handle) return stats;

    auto info = handle.GetInfo();

    // Extract dimensions from shape
    int height = 0, width = 0, channels = 1;
    if (info.shape.size() >= 3) {
        height = static_cast<int>(info.shape[0]);
        width = static_cast<int>(info.shape[1]);
        channels = static_cast<int>(info.shape[2]);
    } else if (info.shape.size() == 2) {
        height = static_cast<int>(info.shape[0]);
        width = static_cast<int>(info.shape[1]);
        channels = 1;
    } else if (info.shape.size() == 1) {
        size_t total = info.shape[0];
        height = width = static_cast<int>(std::sqrt(total));
        channels = 1;
    }

    // For most datasets, dimensions are uniform
    stats.min_width = stats.max_width = stats.mean_width = width;
    stats.min_height = stats.max_height = stats.mean_height = height;
    stats.min_channels = stats.max_channels = stats.mode_channels = channels;

    stats.dimension_counts[{width, height}] = handle.Size();

    return stats;
}

// ============================================================================
// Outlier Detection
// ============================================================================

OutlierInfo DatasetAnalyzer::DetectOutliers(
    const std::vector<std::vector<float>>& images,
    int channels,
    float iqr_multiplier
) {
    OutlierInfo outliers;

    if (images.empty()) return outliers;

    // Compute brightness for all images
    std::vector<float> brightness_values;
    brightness_values.reserve(images.size());
    for (const auto& img : images) {
        brightness_values.push_back(ComputeBrightness(img, channels));
    }

    // Compute quartiles for IQR method
    std::vector<float> sorted_brightness = brightness_values;
    std::sort(sorted_brightness.begin(), sorted_brightness.end());

    size_t n = sorted_brightness.size();
    float q1 = sorted_brightness[n / 4];
    float q3 = sorted_brightness[3 * n / 4];
    float iqr = q3 - q1;

    float lower_bound = q1 - iqr_multiplier * iqr;
    float upper_bound = q3 + iqr_multiplier * iqr;

    // Find outliers
    for (size_t i = 0; i < brightness_values.size(); ++i) {
        float brightness = brightness_values[i];

        if (brightness < lower_bound || brightness > upper_bound) {
            outliers.outlier_indices.push_back(i);
            outliers.outlier_scores.push_back(
                std::abs(brightness - ((lower_bound + upper_bound) / 2.0f)) / iqr
            );
            outliers.outlier_reasons[i] = OutlierInfo::OutlierReason::Brightness;
        }
    }

    return outliers;
}

// ============================================================================
// Helper Methods
// ============================================================================

float DatasetAnalyzer::ComputeBrightness(const std::vector<float>& image, int channels) {
    if (image.empty()) return 0.0f;

    float sum = 0.0f;
    size_t pixel_count = image.size() / channels;

    if (channels == 1) {
        // Grayscale: average all pixels
        sum = std::accumulate(image.begin(), image.end(), 0.0f);
        return sum / pixel_count;
    } else if (channels == 3) {
        // RGB: compute luminosity
        for (size_t i = 0; i < pixel_count; ++i) {
            float r = image[i * 3 + 0];
            float g = image[i * 3 + 1];
            float b = image[i * 3 + 2];
            sum += 0.299f * r + 0.587f * g + 0.114f * b;
        }
        return sum / pixel_count;
    }

    return 0.0f;
}

float DatasetAnalyzer::ComputeContrast(const std::vector<float>& image) {
    if (image.empty()) return 0.0f;

    // Compute standard deviation as contrast measure
    float mean = std::accumulate(image.begin(), image.end(), 0.0f) / image.size();

    float variance = 0.0f;
    for (float val : image) {
        variance += std::pow(val - mean, 2);
    }
    variance /= image.size();

    return std::sqrt(variance);
}

} // namespace cyxwiz
