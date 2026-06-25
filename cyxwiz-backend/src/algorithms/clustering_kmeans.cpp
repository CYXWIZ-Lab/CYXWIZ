// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include "arrayfire_backend_utils.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <limits>
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

// ==================== K-Means GPU Helpers ====================

af::array Clustering::InitializeCentroidsRandom(const af::array& data, int n_clusters, unsigned int seed) {
    int n = static_cast<int>(data.dims(0));

    if (seed != 0) af::setSeed(seed);

    // Generate random indices
    af::array rand_vals = af::randu(n, f64);
    rand_vals.eval();
    af::array sorted_vals, indices;
    af::sort(sorted_vals, indices, rand_vals);
    indices.eval();

    // Take first n_clusters indices
    af::array selected_indices = indices(af::seq(0, n_clusters - 1));
    selected_indices.eval();

    // Gather centroids
    af::array centroids = data(selected_indices, af::span);
    centroids.eval();
    return centroids;
}

af::array Clustering::InitializeCentroidsKMeansPP(const af::array& data, int n_clusters, unsigned int seed) {
    int n = static_cast<int>(data.dims(0));
    int d = static_cast<int>(data.dims(1));

    if (seed != 0) af::setSeed(seed);

    // Store centroids
    std::vector<af::array> centroid_list;

    // Choose first centroid randomly
    af::array first_idx_array = af::randu(1, u32);
    first_idx_array.eval();
    int first_idx = static_cast<int>(first_idx_array.scalar<unsigned int>() % n);
    centroid_list.push_back(data(first_idx, af::span));

    // Distance to nearest centroid for each point
    af::array min_distances = af::constant(std::numeric_limits<float>::max(), n, f64);

    for (int c = 1; c < n_clusters; ++c) {
        // Update distances to last added centroid
        af::array last_centroid = centroid_list.back();
        af::array last_centroid_tile = af::tile(last_centroid, n, 1);
        last_centroid_tile.eval();

        af::array sq_dist = af::sum(af::pow(data - last_centroid_tile, 2), 1);
        sq_dist.eval();
        min_distances = af::min(min_distances, sq_dist);
        min_distances.eval();

        // Sample proportional to squared distance
        double distance_sum = af::sum<double>(min_distances);
        af::array probs = min_distances / distance_sum;
        probs.eval();
        af::array cum_probs = af::accum(probs);
        cum_probs.eval();

        af::array random_value = af::randu(1, f64);
        random_value.eval();
        double r = random_value.scalar<double>();
        af::array mask = cum_probs >= r;
        mask.eval();

        // Find first true index
        unsigned int next_idx = 0;
        af::array true_indices = af::where(mask);
        true_indices.eval();
        if (!true_indices.isempty()) {
            next_idx = true_indices(0).scalar<unsigned int>();
        }

        centroid_list.push_back(data(next_idx, af::span));
    }

    // Stack centroids into matrix
    af::array centroids = af::constant(0.0, n_clusters, d, f64);
    for (int i = 0; i < n_clusters; ++i) {
        centroids(i, af::span) = centroid_list[i];
    }
    centroids.eval();

    return centroids;
}

af::array Clustering::AssignClusters(const af::array& data, const af::array& centroids) {
    af::array distances = ComputePointToCentroidDistances(data, centroids);
    af::array min_vals, labels;
    af::min(min_vals, labels, distances, 1);
    labels.eval();
    return labels;
}

af::array Clustering::UpdateCentroids(const af::array& data, const af::array& labels, int n_clusters) {
    int n = static_cast<int>(data.dims(0));
    (void)n;  // Suppress unused variable warning
    int d = static_cast<int>(data.dims(1));

    af::array new_centroids = af::constant(0.0, n_clusters, d, f64);

    for (int k = 0; k < n_clusters; ++k) {
        af::array mask = (labels == k);
        mask.eval();
        af::array cluster_points = af::where(mask);
        cluster_points.eval();

        if (!cluster_points.isempty()) {
            int cluster_size = static_cast<int>(cluster_points.elements());
            af::array cluster_data = data(cluster_points, af::span);
            cluster_data.eval();

            // Mean of cluster points
            af::array centroid = af::sum(cluster_data, 0) / static_cast<double>(cluster_size);
            centroid.eval();
            new_centroids(k, af::span) = centroid;
        }
    }
    new_centroids.eval();

    return new_centroids;
}

double Clustering::ComputeInertia(const af::array& data, const af::array& labels, const af::array& centroids) {
    int k = static_cast<int>(centroids.dims(0));

    double inertia = 0.0;

    for (int c = 0; c < k; ++c) {
        af::array mask = (labels == c);
        mask.eval();
        af::array cluster_indices = af::where(mask);
        cluster_indices.eval();

        if (!cluster_indices.isempty()) {
            af::array cluster_data = data(cluster_indices, af::span);
            cluster_data.eval();
            af::array centroid = centroids(c, af::span);
            af::array centroid_tile = af::tile(centroid, static_cast<int>(cluster_data.dims(0)), 1);
            centroid_tile.eval();

            af::array sq_dist = af::sum(af::pow(cluster_data - centroid_tile, 2), 1);
            sq_dist.eval();
            inertia += af::sum<double>(sq_dist);
        }
    }

    return inertia;
}

// ==================== K-Means Main Algorithm ====================

KMeansResult Clustering::KMeans(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    int max_iter,
    const std::string& init,
    int n_init,
    double tol,
    unsigned int seed,
    std::function<void(int, double)> progress_callback
) {
    KMeansResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n_samples = static_cast<int>(data.size());

    if (n_clusters <= 0 || n_clusters > n_samples) {
        result.error_message = "Invalid number of clusters";
        return result;
    }

    try {
        // Convert to ArrayFire
        af::array af_data = ToAfArray(data);

        // Best result across n_init runs
        double best_inertia = std::numeric_limits<double>::max();
        af::array best_labels;
        af::array best_centroids;

        for (int init_run = 0; init_run < n_init; ++init_run) {
            unsigned int run_seed = (seed == 0) ? 0 : seed + init_run;

            // Initialize centroids
            af::array centroids;
            if (init == "random") {
                centroids = InitializeCentroidsRandom(af_data, n_clusters, run_seed);
            } else {
                centroids = InitializeCentroidsKMeansPP(af_data, n_clusters, run_seed);
            }

            af::array labels;
            double prev_inertia = std::numeric_limits<double>::max();
            int iterations = 0;
            bool converged = false;

            for (int iter = 0; iter < max_iter; ++iter) {
                // Assign clusters
                labels = AssignClusters(af_data, centroids);

                // Update centroids
                af::array new_centroids = UpdateCentroids(af_data, labels, n_clusters);

                // Compute inertia
                double inertia = ComputeInertia(af_data, labels, new_centroids);

                if (init_run == 0 && progress_callback) {
                    progress_callback(iter + 1, inertia);
                }

                // Check convergence
                if (std::abs(prev_inertia - inertia) < tol) {
                    converged = true;
                    centroids = new_centroids;
                    iterations = iter + 1;
                    break;
                }

                centroids = new_centroids;
                prev_inertia = inertia;
                iterations = iter + 1;
            }

            double final_inertia = ComputeInertia(af_data, labels, centroids);

            if (final_inertia < best_inertia) {
                best_inertia = final_inertia;
                best_labels = labels;
                best_centroids = centroids;
                result.n_iterations = iterations;
                result.converged = converged;
            }
        }

        // Convert results back
        result.labels = AfArrayToIntVector(best_labels);
        result.centroids = FromAfArray(best_centroids);
        result.inertia = best_inertia;
        result.n_clusters = n_clusters;
        result.success = true;

        spdlog::info("K-Means completed: {} clusters, inertia={:.4f}, iterations={}",
                     n_clusters, best_inertia, result.n_iterations);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        LogClusteringBackendFailureOnce(
            "Clustering::KMeans",
            e.what(),
            BuildClusteringContext(
                data,
                "clusters=" + std::to_string(n_clusters)));
    }

    return result;
}

ElbowAnalysis Clustering::ComputeElbowAnalysis(
    const std::vector<std::vector<double>>& data,
    int k_min,
    int k_max,
    std::function<void(int, int)> progress_callback
) {
    ElbowAnalysis result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    try {
        for (int k = k_min; k <= k_max; ++k) {
            if (progress_callback) {
                progress_callback(k - k_min + 1, k_max - k_min + 1);
            }

            auto kmeans_result = KMeans(data, k, 100, "kmeans++", 3, 1e-4, 0, nullptr);

            if (kmeans_result.success) {
                result.k_values.push_back(k);
                result.inertias.push_back(kmeans_result.inertia);

                // Compute silhouette score
                double silhouette = ComputeSilhouetteScore(data, kmeans_result.labels);
                result.silhouette_scores.push_back(silhouette);
            }
        }

        // Find elbow point using second derivative
        if (result.inertias.size() >= 3) {
            double max_curvature = 0.0;
            int elbow_idx = 0;

            for (size_t i = 1; i < result.inertias.size() - 1; ++i) {
                double curvature = std::abs(result.inertias[i - 1] - 2 * result.inertias[i] + result.inertias[i + 1]);
                if (curvature > max_curvature) {
                    max_curvature = curvature;
                    elbow_idx = static_cast<int>(i);
                }
            }
            result.suggested_k = result.k_values[elbow_idx];
        } else if (!result.k_values.empty()) {
            result.suggested_k = result.k_values[0];
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}


#else  // No ArrayFire - CPU fallback stubs

KMeansResult Clustering::KMeans(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    int max_iter,
    const std::string& init,
    int n_init,
    double tol,
    unsigned int seed,
    std::function<void(int, double)> progress_callback
) {
    KMeansResult result;
    result.error_message = "ArrayFire not available - GPU acceleration required for clustering";
    spdlog::warn("Clustering::KMeans called without ArrayFire support");
    return result;
}

ElbowAnalysis Clustering::ComputeElbowAnalysis(
    const std::vector<std::vector<double>>& data,
    int k_min,
    int k_max,
    std::function<void(int, int)> progress_callback
) {
    ElbowAnalysis result;
    result.error_message = "ArrayFire not available";
    return result;
}


#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz
