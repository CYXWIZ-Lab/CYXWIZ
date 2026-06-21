// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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

// ==================== Hierarchical Clustering ====================

HierarchicalResult Clustering::Hierarchical(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    const std::string& linkage,
    const std::string& metric
) {
    HierarchicalResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());

    if (n_clusters <= 0 || n_clusters > n) {
        result.error_message = "Invalid number of clusters";
        return result;
    }

    try {
        // Compute distance matrix on GPU
        af::array af_data = ToAfArray(data);
        af::array dist_matrix = ComputeDistanceMatrix(af_data, metric);

        // Transfer to CPU for agglomerative clustering
        std::vector<double> dist_flat(n * n);
        dist_matrix.host(dist_flat.data());

        // Build distance matrix on CPU
        std::vector<std::vector<double>> cpu_dist(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                cpu_dist[i][j] = dist_flat[j * n + i];
            }
        }

        // Cluster tracking
        std::vector<std::set<int>> clusters(n);
        for (int i = 0; i < n; ++i) {
            clusters[i].insert(i);
        }
        std::vector<bool> active(n, true);

        // Build linkage matrix
        result.linkage_matrix.reserve(n - 1);

        for (int step = 0; step < n - 1; ++step) {
            // Find minimum distance pair
            double min_dist = std::numeric_limits<double>::max();
            int min_i = -1, min_j = -1;

            for (int i = 0; i < n + step; ++i) {
                if (!active[i]) continue;
                for (int j = i + 1; j < n + step; ++j) {
                    if (!active[j]) continue;
                    if (cpu_dist[i][j] < min_dist) {
                        min_dist = cpu_dist[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            if (min_i < 0) break;

            // Merge clusters
            int new_cluster_idx = n + step;
            clusters.push_back(std::set<int>());
            clusters[new_cluster_idx].insert(clusters[min_i].begin(), clusters[min_i].end());
            clusters[new_cluster_idx].insert(clusters[min_j].begin(), clusters[min_j].end());

            // Record linkage
            result.linkage_matrix.push_back({
                static_cast<double>(min_i),
                static_cast<double>(min_j),
                min_dist,
                static_cast<double>(clusters[new_cluster_idx].size())
            });

            // Update distance matrix
            cpu_dist.push_back(std::vector<double>(new_cluster_idx + 1, 0.0));
            for (auto& row : cpu_dist) {
                row.resize(new_cluster_idx + 1, 0.0);
            }

            active.push_back(true);
            active[min_i] = false;
            active[min_j] = false;

            // Compute distances to new cluster
            for (int k = 0; k < new_cluster_idx; ++k) {
                if (!active[k]) continue;

                double new_dist = 0.0;

                if (linkage == "single") {
                    new_dist = std::min(cpu_dist[min_i][k], cpu_dist[min_j][k]);
                } else if (linkage == "complete") {
                    new_dist = std::max(cpu_dist[min_i][k], cpu_dist[min_j][k]);
                } else if (linkage == "average") {
                    double n_i = static_cast<double>(clusters[min_i].size());
                    double n_j = static_cast<double>(clusters[min_j].size());
                    new_dist = (n_i * cpu_dist[min_i][k] + n_j * cpu_dist[min_j][k]) / (n_i + n_j);
                } else {  // ward
                    double n_i = static_cast<double>(clusters[min_i].size());
                    double n_j = static_cast<double>(clusters[min_j].size());
                    double n_k = static_cast<double>(clusters[k].size());
                    double d_ik = cpu_dist[min_i][k];
                    double d_jk = cpu_dist[min_j][k];
                    double d_ij = cpu_dist[min_i][min_j];
                    new_dist = std::sqrt(((n_i + n_k) * d_ik * d_ik + (n_j + n_k) * d_jk * d_jk - n_k * d_ij * d_ij) / (n_i + n_j + n_k));
                }

                cpu_dist[new_cluster_idx][k] = new_dist;
                cpu_dist[k][new_cluster_idx] = new_dist;
            }
        }

        // Cut dendrogram to get n_clusters
        result.labels = CutDendrogram(result.linkage_matrix, 0.0, n);

        // Adjust to get desired number of clusters
        if (!result.linkage_matrix.empty()) {
            // Find cut height for n_clusters
            int target_merges = n - n_clusters;
            if (target_merges > 0 && target_merges <= static_cast<int>(result.linkage_matrix.size())) {
                double cut_height = result.linkage_matrix[target_merges - 1][2] + 0.001;
                result.labels = CutDendrogram(result.linkage_matrix, cut_height, n);
            }
        }

        result.n_clusters = n_clusters;
        result.success = true;

        spdlog::info("Hierarchical clustering completed: {} clusters", n_clusters);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        spdlog::error("Hierarchical clustering failed: {}", result.error_message);
    }

    return result;
}

std::vector<int> Clustering::CutDendrogram(
    const std::vector<std::vector<double>>& linkage_matrix,
    double height,
    int n_samples
) {
    // Initialize each sample in its own cluster
    std::vector<int> labels(n_samples);
    std::iota(labels.begin(), labels.end(), 0);

    // Union-find data structure
    std::vector<int> parent(2 * n_samples);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    };

    // Apply merges up to cut height
    for (size_t i = 0; i < linkage_matrix.size(); ++i) {
        if (linkage_matrix[i][2] > height) break;

        int c1 = static_cast<int>(linkage_matrix[i][0]);
        int c2 = static_cast<int>(linkage_matrix[i][1]);
        int new_cluster = n_samples + static_cast<int>(i);

        parent[find(c1)] = new_cluster;
        parent[find(c2)] = new_cluster;
    }

    // Assign final labels
    std::map<int, int> root_to_label;
    int next_label = 0;

    for (int i = 0; i < n_samples; ++i) {
        int root = find(i);
        if (root_to_label.find(root) == root_to_label.end()) {
            root_to_label[root] = next_label++;
        }
        labels[i] = root_to_label[root];
    }

    return labels;
}


#else  // No ArrayFire - CPU fallback stubs

HierarchicalResult Clustering::Hierarchical(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    const std::string& linkage,
    const std::string& metric
) {
    HierarchicalResult result;
    result.error_message = "ArrayFire not available";
    return result;
}

std::vector<int> Clustering::CutDendrogram(
    const std::vector<std::vector<double>>& linkage_matrix,
    double height,
    int n_samples
) {
    return {};
}


#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz

