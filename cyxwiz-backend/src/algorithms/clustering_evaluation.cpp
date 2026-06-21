// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Ensure Windows min/max macros are undefined
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE

// ==================== Cluster Evaluation ====================

ClusterMetrics Clustering::EvaluateClustering(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    ClusterMetrics result;

    if (data.empty() || labels.empty()) {
        result.error_message = "Empty data or labels";
        return result;
    }

    try {
        af::array af_data = ToAfArray(data);
        af::array dist_matrix = ComputeEuclideanDistanceMatrix(af_data);

        int n = static_cast<int>(data.size());
        std::set<int> unique_labels(labels.begin(), labels.end());
        unique_labels.erase(-1);  // Remove noise label if present
        int n_clusters = static_cast<int>(unique_labels.size());

        if (n_clusters < 2) {
            result.error_message = "Need at least 2 clusters";
            return result;
        }

        // Convert labels to af::array
        af::array af_labels = af::array(n, labels.data()).as(s32);

        // Compute silhouette
        af::array silhouette = ComputeSilhouetteCoefficients(dist_matrix, af_labels, n_clusters);
        result.per_sample_silhouette = AfArrayToDoubleVector(silhouette);
        result.silhouette_score = af::mean<double>(silhouette);

        // Compute cluster silhouettes
        result.cluster_silhouettes.resize(n_clusters, 0.0);
        std::vector<int> cluster_counts(n_clusters, 0);
        for (int i = 0; i < n; ++i) {
            if (labels[i] >= 0 && labels[i] < n_clusters) {
                result.cluster_silhouettes[labels[i]] += result.per_sample_silhouette[i];
                cluster_counts[labels[i]]++;
            }
        }
        for (int k = 0; k < n_clusters; ++k) {
            if (cluster_counts[k] > 0) {
                result.cluster_silhouettes[k] /= cluster_counts[k];
            }
        }

        // Davies-Bouldin and Calinski-Harabasz
        result.davies_bouldin_index = ComputeDaviesBouldinIndex(data, labels);
        result.calinski_harabasz_score = ComputeCalinskiHarabaszScore(data, labels);

        result.n_clusters = n_clusters;
        result.n_samples = n;
        result.success = true;

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
    }

    return result;
}

af::array Clustering::ComputeSilhouetteCoefficients(const af::array& dist_matrix, const af::array& labels, int n_clusters) {
    int n = static_cast<int>(dist_matrix.dims(0));

    std::vector<int> cpu_labels = AfArrayToIntVector(labels);
    std::vector<double> dist_flat(n * n);
    dist_matrix.host(dist_flat.data());

    std::vector<double> silhouettes(n);

    for (int i = 0; i < n; ++i) {
        int cluster_i = cpu_labels[i];
        if (cluster_i < 0) {
            silhouettes[i] = 0.0;
            continue;
        }

        // Compute a(i): mean distance to same cluster
        double a_i = 0.0;
        int same_count = 0;
        for (int j = 0; j < n; ++j) {
            if (j != i && cpu_labels[j] == cluster_i) {
                a_i += dist_flat[j * n + i];
                same_count++;
            }
        }
        a_i = (same_count > 0) ? a_i / same_count : 0.0;

        // Compute b(i): minimum mean distance to other clusters
        double b_i = std::numeric_limits<double>::max();
        for (int k = 0; k < n_clusters; ++k) {
            if (k == cluster_i) continue;

            double dist_k = 0.0;
            int count_k = 0;
            for (int j = 0; j < n; ++j) {
                if (cpu_labels[j] == k) {
                    dist_k += dist_flat[j * n + i];
                    count_k++;
                }
            }
            if (count_k > 0) {
                b_i = std::min(b_i, dist_k / count_k);
            }
        }

        if (b_i == std::numeric_limits<double>::max()) b_i = 0.0;

        double max_ab = std::max(a_i, b_i);
        silhouettes[i] = (max_ab > 0) ? (b_i - a_i) / max_ab : 0.0;
    }

    return af::array(n, silhouettes.data());
}

double Clustering::ComputeSilhouetteScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    auto metrics = EvaluateClustering(data, labels);
    return metrics.silhouette_score;
}

double Clustering::ComputeDaviesBouldinIndex(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    if (data.empty()) return 0.0;

    std::set<int> unique_labels(labels.begin(), labels.end());
    unique_labels.erase(-1);
    int n_clusters = static_cast<int>(unique_labels.size());

    if (n_clusters < 2) return 0.0;

    int n_features = static_cast<int>(data[0].size());

    // Compute cluster centroids and scatter
    std::vector<std::vector<double>> centroids(n_clusters, std::vector<double>(n_features, 0.0));
    std::vector<int> cluster_sizes(n_clusters, 0);
    std::vector<double> scatter(n_clusters, 0.0);

    for (size_t i = 0; i < data.size(); ++i) {
        int k = labels[i];
        if (k < 0 || k >= n_clusters) continue;
        cluster_sizes[k]++;
        for (int f = 0; f < n_features; ++f) {
            centroids[k][f] += data[i][f];
        }
    }

    for (int k = 0; k < n_clusters; ++k) {
        if (cluster_sizes[k] > 0) {
            for (int f = 0; f < n_features; ++f) {
                centroids[k][f] /= cluster_sizes[k];
            }
        }
    }

    // Compute scatter (average distance to centroid)
    for (size_t i = 0; i < data.size(); ++i) {
        int k = labels[i];
        if (k < 0 || k >= n_clusters) continue;
        double dist = 0.0;
        for (int f = 0; f < n_features; ++f) {
            double d = data[i][f] - centroids[k][f];
            dist += d * d;
        }
        scatter[k] += std::sqrt(dist);
    }
    for (int k = 0; k < n_clusters; ++k) {
        if (cluster_sizes[k] > 0) scatter[k] /= cluster_sizes[k];
    }

    // Compute Davies-Bouldin index
    double db = 0.0;
    for (int i = 0; i < n_clusters; ++i) {
        double max_ratio = 0.0;
        for (int j = 0; j < n_clusters; ++j) {
            if (i == j) continue;
            double centroid_dist = 0.0;
            for (int f = 0; f < n_features; ++f) {
                double d = centroids[i][f] - centroids[j][f];
                centroid_dist += d * d;
            }
            centroid_dist = std::sqrt(centroid_dist);
            if (centroid_dist > 0) {
                double ratio = (scatter[i] + scatter[j]) / centroid_dist;
                max_ratio = std::max(max_ratio, ratio);
            }
        }
        db += max_ratio;
    }

    return db / n_clusters;
}

double Clustering::ComputeCalinskiHarabaszScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    if (data.empty()) return 0.0;

    int n_samples = static_cast<int>(data.size());
    std::set<int> unique_labels(labels.begin(), labels.end());
    unique_labels.erase(-1);
    int n_clusters = static_cast<int>(unique_labels.size());

    if (n_clusters < 2) return 0.0;

    int n_features = static_cast<int>(data[0].size());

    // Global mean
    std::vector<double> global_mean(n_features, 0.0);
    for (const auto& point : data) {
        for (int f = 0; f < n_features; ++f) {
            global_mean[f] += point[f];
        }
    }
    for (int f = 0; f < n_features; ++f) {
        global_mean[f] /= n_samples;
    }

    // Cluster centroids and sizes
    std::vector<std::vector<double>> centroids(n_clusters, std::vector<double>(n_features, 0.0));
    std::vector<int> cluster_sizes(n_clusters, 0);

    for (size_t i = 0; i < data.size(); ++i) {
        int k = labels[i];
        if (k < 0 || k >= n_clusters) continue;
        cluster_sizes[k]++;
        for (int f = 0; f < n_features; ++f) {
            centroids[k][f] += data[i][f];
        }
    }

    for (int k = 0; k < n_clusters; ++k) {
        if (cluster_sizes[k] > 0) {
            for (int f = 0; f < n_features; ++f) {
                centroids[k][f] /= cluster_sizes[k];
            }
        }
    }

    // Between-cluster dispersion (BGSS)
    double bgss = 0.0;
    for (int k = 0; k < n_clusters; ++k) {
        double dist_sq = 0.0;
        for (int f = 0; f < n_features; ++f) {
            double d = centroids[k][f] - global_mean[f];
            dist_sq += d * d;
        }
        bgss += cluster_sizes[k] * dist_sq;
    }

    // Within-cluster dispersion (WGSS)
    double wgss = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        int k = labels[i];
        if (k < 0 || k >= n_clusters) continue;
        double dist_sq = 0.0;
        for (int f = 0; f < n_features; ++f) {
            double d = data[i][f] - centroids[k][f];
            dist_sq += d * d;
        }
        wgss += dist_sq;
    }

    if (wgss == 0.0) return 0.0;

    return (bgss / (n_clusters - 1)) / (wgss / (n_samples - n_clusters));
}


#else  // No ArrayFire - CPU fallback stubs

ClusterMetrics Clustering::EvaluateClustering(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    ClusterMetrics result;
    result.error_message = "ArrayFire not available";
    return result;
}

double Clustering::ComputeSilhouetteScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    return 0.0;
}

double Clustering::ComputeDaviesBouldinIndex(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    return 0.0;
}

double Clustering::ComputeCalinskiHarabaszScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    return 0.0;
}


#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz
