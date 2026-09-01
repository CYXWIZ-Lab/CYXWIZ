// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include "arrayfire_backend_utils.h"
#include "arrayfire_host_materialization.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <queue>
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

static std::string BuildClusteringContext(
    const std::vector<std::vector<double>>& data,
    const std::string& extra = {})
{
    std::string context = "samples=" + std::to_string(data.size()) +
        "; features=" + std::to_string(data.empty() ? 0 : data[0].size());
    if (!extra.empty()) {
        context += "; ";
        context += extra;
    }
    return context;
}

static void LogClusteringBackendFailureOnce(
    const char* operation_name,
    const char* error_message,
    const std::string& data_context)
{
    const BackendFallbackReason reason = ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(data_context);
    if (!ShouldLogArrayFireBackendFallbackOnce(operation_name, reason, context)) {
        return;
    }

    std::string message = std::string("ArrayFire ") +
        (operation_name ? operation_name : "clustering operation") +
        " failed (reason=" + BackendFallbackReasonName(reason) +
        "); clustering operation cannot complete on the current ArrayFire backend.";
    if (!context.empty()) {
        message += " Context: ";
        message += context;
        message += ".";
    }
    if (reason != BackendFallbackReason::CudaJitParamOverflow &&
        error_message != nullptr &&
        error_message[0] != '\0') {
        message += " Error: ";
        message += error_message;
    }
    spdlog::error("{}", message);
}

// ==================== DBSCAN Implementation ====================

DBSCANResult Clustering::DBSCAN(
    const std::vector<std::vector<double>>& data,
    double eps,
    int min_samples,
    const std::string& metric
) {
    DBSCANResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());

    try {
        // Compute distance matrix on GPU
        af::array af_data = ToAfArray(data);
        af::array dist_matrix = ComputeDistanceMatrix(af_data, metric);

        // Transfer distance matrix to CPU for DBSCAN logic
        // (DBSCAN requires sequential cluster expansion which is hard to parallelize)
        std::vector<double> dist_flat(n * n);
        dist_matrix.eval();
        MaterializeArrayFireToHost(
            dist_matrix,
            dist_flat.data(),
            ArrayFireHostSyncCategory::AlgorithmCpuPath,
            "Clustering::DBSCAN::DistanceMatrix",
            "arrayfire_column_major");

        // Initialize labels
        result.labels.assign(n, -1);  // -1 = unvisited/noise
        result.core_samples.assign(n, false);

        int cluster_id = 0;
        std::vector<bool> visited(n, false);

        // Find core samples and their neighbors using GPU-computed distances
        std::vector<std::vector<int>> neighborhoods(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist_flat[j * n + i] <= eps) {  // Column-major
                    neighborhoods[i].push_back(j);
                }
            }
            if (static_cast<int>(neighborhoods[i].size()) >= min_samples) {
                result.core_samples[i] = true;
            }
        }

        // DBSCAN clustering
        for (int i = 0; i < n; ++i) {
            if (visited[i]) continue;
            visited[i] = true;

            if (!result.core_samples[i]) {
                // Noise point (may be reassigned later)
                continue;
            }

            // Expand cluster from core point
            result.labels[i] = cluster_id;
            std::queue<int> to_expand;

            for (int neighbor : neighborhoods[i]) {
                if (result.labels[neighbor] == -1) {
                    result.labels[neighbor] = cluster_id;
                }
                if (!visited[neighbor]) {
                    to_expand.push(neighbor);
                }
            }

            while (!to_expand.empty()) {
                int p = to_expand.front();
                to_expand.pop();

                if (visited[p]) continue;
                visited[p] = true;

                result.labels[p] = cluster_id;

                if (result.core_samples[p]) {
                    for (int neighbor : neighborhoods[p]) {
                        if (result.labels[neighbor] == -1) {
                            result.labels[neighbor] = cluster_id;
                        }
                        if (!visited[neighbor]) {
                            to_expand.push(neighbor);
                        }
                    }
                }
            }

            cluster_id++;
        }

        result.n_clusters = cluster_id;
        result.n_noise_points = static_cast<int>(std::count(result.labels.begin(), result.labels.end(), -1));
        result.success = true;

        spdlog::info("DBSCAN completed: {} clusters, {} noise points",
                     result.n_clusters, result.n_noise_points);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        LogClusteringBackendFailureOnce(
            "Clustering::DBSCAN",
            e.what(),
            BuildClusteringContext(
                data,
                "eps=" + std::to_string(eps) +
                "; min_samples=" + std::to_string(min_samples) +
                "; metric=" + metric));
    }

    return result;
}

std::vector<double> Clustering::ComputeKDistances(
    const std::vector<std::vector<double>>& data,
    int k
) {
    if (data.empty()) return {};

    try {
        af::array af_data = ToAfArray(data);
        af::array dist_matrix = ComputeEuclideanDistanceMatrix(af_data);

        int n = static_cast<int>(data.size());

        // Sort each row and get k-th distance
        af::array sorted_dists;
        af::array indices;
        af::sort(sorted_dists, indices, dist_matrix, 1);
        sorted_dists.eval();

        // Get k-th column (k-th nearest neighbor distance)
        int k_idx = std::min(k, n - 1);
        af::array k_distances = sorted_dists(af::span, k_idx);
        k_distances.eval();

        // Sort k-distances for plotting
        af::array sorted_k_dists;
        af::sort(sorted_k_dists, indices, k_distances);
        sorted_k_dists.eval();

        return AfArrayToDoubleVector(sorted_k_dists);

    } catch (const af::exception& e) {
        LogClusteringBackendFailureOnce(
            "Clustering::ComputeKDistances",
            e.what(),
            BuildClusteringContext(data, "k=" + std::to_string(k)));
        return {};
    }
}


#else  // No ArrayFire - CPU fallback stubs

DBSCANResult Clustering::DBSCAN(
    const std::vector<std::vector<double>>& data,
    double eps,
    int min_samples,
    const std::string& metric
) {
    DBSCANResult result;
    result.error_message = "ArrayFire not available";
    return result;
}

std::vector<double> Clustering::ComputeKDistances(
    const std::vector<std::vector<double>>& data,
    int k
) {
    return {};
}


#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz
