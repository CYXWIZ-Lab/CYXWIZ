#include "clustering.h"
#include <arrayfire.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <set>
#include <queue>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ==================== ArrayFire Conversion Helpers ====================

af::array Clustering::ToAfArray(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return af::array();

    int n_samples = static_cast<int>(data.size());
    int n_features = static_cast<int>(data[0].size());

    // Flatten data in column-major order for ArrayFire
    std::vector<double> flat_data(n_samples * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            flat_data[j * n_samples + i] = data[i][j];
        }
    }

    return af::array(n_samples, n_features, flat_data.data());
}

std::vector<std::vector<double>> Clustering::FromAfArray(const af::array& arr) {
    if (arr.isempty()) return {};

    int n_samples = static_cast<int>(arr.dims(0));
    int n_features = static_cast<int>(arr.dims(1));

    std::vector<double> flat_data(n_samples * n_features);
    arr.host(flat_data.data());

    std::vector<std::vector<double>> result(n_samples, std::vector<double>(n_features));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            result[i][j] = flat_data[j * n_samples + i];
        }
    }

    return result;
}

std::vector<int> Clustering::AfArrayToIntVector(const af::array& arr) {
    if (arr.isempty()) return {};

    int n = static_cast<int>(arr.elements());
    std::vector<int> result(n);

    // Convert to int array on host
    af::array int_arr = arr.as(s32);
    int_arr.host(result.data());

    return result;
}

std::vector<double> Clustering::AfArrayToDoubleVector(const af::array& arr) {
    if (arr.isempty()) return {};

    int n = static_cast<int>(arr.elements());
    std::vector<double> result(n);

    af::array double_arr = arr.as(f64);
    double_arr.host(result.data());

    return result;
}

// ==================== GPU Distance Functions ====================

af::array Clustering::ComputeEuclideanDistanceMatrix(const af::array& data) {
    // data: [n_samples x n_features]
    // Output: [n_samples x n_samples] distance matrix

    int n = static_cast<int>(data.dims(0));

    // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
    af::array sq_norms = af::sum(data * data, 1);  // [n x 1]

    // Expand for broadcasting
    af::array sq_norms_row = af::tile(sq_norms, 1, n);          // [n x n]
    af::array sq_norms_col = af::tile(sq_norms.T(), n, 1);      // [n x n]

    // Compute dot products: data @ data.T
    af::array dot_products = af::matmul(data, data.T());  // [n x n]

    // Squared distances
    af::array sq_distances = sq_norms_row + sq_norms_col - 2.0 * dot_products;

    // Clamp negative values (numerical errors) and take sqrt
    sq_distances = af::max(sq_distances, 0.0);
    return af::sqrt(sq_distances);
}

af::array Clustering::ComputeManhattanDistanceMatrix(const af::array& data) {
    int n = static_cast<int>(data.dims(0));
    int d = static_cast<int>(data.dims(1));

    // Expand data for pairwise computation
    // data_i: [n x 1 x d], data_j: [1 x n x d]
    af::array data_i = af::moddims(data, n, 1, d);
    af::array data_j = af::moddims(data, 1, n, d);

    // Tile for broadcasting
    data_i = af::tile(data_i, 1, n, 1);
    data_j = af::tile(data_j, n, 1, 1);

    // Manhattan distance: sum(|a - b|)
    af::array diff = af::abs(data_i - data_j);
    return af::sum(diff, 2);  // [n x n]
}

af::array Clustering::ComputeCosineDistanceMatrix(const af::array& data) {
    int n = static_cast<int>(data.dims(0));

    // Normalize data
    af::array norms = af::sqrt(af::sum(data * data, 1));  // [n x 1]
    norms = af::max(norms, 1e-10);  // Avoid division by zero
    af::array normalized = data / af::tile(norms, 1, static_cast<int>(data.dims(1)));

    // Cosine similarity = normalized @ normalized.T
    af::array similarity = af::matmul(normalized, normalized.T());

    // Cosine distance = 1 - similarity
    return 1.0 - similarity;
}

af::array Clustering::ComputeDistanceMatrix(const af::array& data, const std::string& metric) {
    if (metric == "manhattan") return ComputeManhattanDistanceMatrix(data);
    if (metric == "cosine") return ComputeCosineDistanceMatrix(data);
    return ComputeEuclideanDistanceMatrix(data);
}

af::array Clustering::ComputePointToCentroidDistances(const af::array& data, const af::array& centroids) {
    // data: [n_samples x n_features]
    // centroids: [n_clusters x n_features]
    // Output: [n_samples x n_clusters]

    int n = static_cast<int>(data.dims(0));
    int k = static_cast<int>(centroids.dims(0));
    int d = static_cast<int>(data.dims(1));

    // ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
    af::array data_sq = af::sum(data * data, 1);          // [n x 1]
    af::array cent_sq = af::sum(centroids * centroids, 1); // [k x 1]

    af::array data_sq_tile = af::tile(data_sq, 1, k);      // [n x k]
    af::array cent_sq_tile = af::tile(cent_sq.T(), n, 1);  // [n x k]

    af::array dot = af::matmul(data, centroids.T());       // [n x k]

    af::array sq_distances = data_sq_tile + cent_sq_tile - 2.0 * dot;
    sq_distances = af::max(sq_distances, 0.0);

    return af::sqrt(sq_distances);
}

// ==================== K-Means GPU Helpers ====================

af::array Clustering::InitializeCentroidsRandom(const af::array& data, int n_clusters, unsigned int seed) {
    int n = static_cast<int>(data.dims(0));

    if (seed != 0) af::setSeed(seed);

    // Generate random indices
    af::array rand_vals = af::randu(n, f64);
    af::array sorted_vals, indices;
    af::sort(sorted_vals, indices, rand_vals);

    // Take first n_clusters indices
    af::array selected_indices = indices(af::seq(0, n_clusters - 1));

    // Gather centroids
    return data(selected_indices, af::span);
}

af::array Clustering::InitializeCentroidsKMeansPP(const af::array& data, int n_clusters, unsigned int seed) {
    int n = static_cast<int>(data.dims(0));
    int d = static_cast<int>(data.dims(1));

    if (seed != 0) af::setSeed(seed);

    // Store centroids
    std::vector<af::array> centroid_list;

    // Choose first centroid randomly
    int first_idx = static_cast<int>(af::randu(1, u32).scalar<unsigned int>() % n);
    centroid_list.push_back(data(first_idx, af::span));

    // Distance to nearest centroid for each point
    af::array min_distances = af::constant(std::numeric_limits<float>::max(), n, f64);

    for (int c = 1; c < n_clusters; ++c) {
        // Update distances to last added centroid
        af::array last_centroid = centroid_list.back();
        af::array last_centroid_tile = af::tile(last_centroid, n, 1);

        af::array sq_dist = af::sum(af::pow(data - last_centroid_tile, 2), 1);
        min_distances = af::min(min_distances, sq_dist);

        // Sample proportional to squared distance
        af::array probs = min_distances / af::sum(min_distances);
        af::array cum_probs = af::accum(probs);

        double r = af::randu(1, f64).scalar<double>();
        af::array mask = cum_probs >= r;

        // Find first true index
        unsigned int next_idx = 0;
        af::array true_indices = af::where(mask);
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

    return centroids;
}

af::array Clustering::AssignClusters(const af::array& data, const af::array& centroids) {
    af::array distances = ComputePointToCentroidDistances(data, centroids);
    af::array min_vals, labels;
    af::min(min_vals, labels, distances, 1);
    return labels;
}

af::array Clustering::UpdateCentroids(const af::array& data, const af::array& labels, int n_clusters) {
    int n = static_cast<int>(data.dims(0));
    int d = static_cast<int>(data.dims(1));

    af::array new_centroids = af::constant(0.0, n_clusters, d, f64);

    for (int k = 0; k < n_clusters; ++k) {
        af::array mask = (labels == k);
        af::array cluster_points = af::where(mask);

        if (!cluster_points.isempty()) {
            int cluster_size = static_cast<int>(cluster_points.elements());
            af::array cluster_data = data(cluster_points, af::span);

            // Mean of cluster points
            af::array centroid = af::sum(cluster_data, 0) / static_cast<double>(cluster_size);
            new_centroids(k, af::span) = centroid;
        }
    }

    return new_centroids;
}

double Clustering::ComputeInertia(const af::array& data, const af::array& labels, const af::array& centroids) {
    int n = static_cast<int>(data.dims(0));
    int k = static_cast<int>(centroids.dims(0));

    double inertia = 0.0;

    for (int c = 0; c < k; ++c) {
        af::array mask = (labels == c);
        af::array cluster_indices = af::where(mask);

        if (!cluster_indices.isempty()) {
            af::array cluster_data = data(cluster_indices, af::span);
            af::array centroid = centroids(c, af::span);
            af::array centroid_tile = af::tile(centroid, static_cast<int>(cluster_data.dims(0)), 1);

            af::array sq_dist = af::sum(af::pow(cluster_data - centroid_tile, 2), 1);
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
    int n_features = static_cast<int>(data[0].size());

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
        spdlog::error("K-Means failed: {}", result.error_message);
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
        dist_matrix.host(dist_flat.data());

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
        spdlog::error("DBSCAN failed: {}", result.error_message);
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

        // Get k-th column (k-th nearest neighbor distance)
        int k_idx = std::min(k, n - 1);
        af::array k_distances = sorted_dists(af::span, k_idx);

        // Sort k-distances for plotting
        af::array sorted_k_dists;
        af::sort(sorted_k_dists, indices, k_distances);

        return AfArrayToDoubleVector(sorted_k_dists);

    } catch (const af::exception& e) {
        spdlog::error("ComputeKDistances failed: {}", e.what());
        return {};
    }
}

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

// ==================== GMM Implementation ====================

GMMResult Clustering::GMM(
    const std::vector<std::vector<double>>& data,
    int n_components,
    const std::string& covariance_type,
    int max_iter,
    double tol,
    int n_init,
    unsigned int seed,
    std::function<void(int, double)> progress_callback
) {
    GMMResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n_samples = static_cast<int>(data.size());
    int n_features = static_cast<int>(data[0].size());

    if (n_components <= 0 || n_components > n_samples) {
        result.error_message = "Invalid number of components";
        return result;
    }

    try {
        af::array af_data = ToAfArray(data);

        double best_ll = -std::numeric_limits<double>::max();
        af::array best_means;
        std::vector<af::array> best_covs;
        af::array best_weights;
        af::array best_resp;
        int best_iters = 0;
        bool best_converged = false;

        for (int init_run = 0; init_run < n_init; ++init_run) {
            unsigned int run_seed = (seed == 0) ? 0 : seed + init_run;

            // Initialize parameters
            af::array means;
            std::vector<af::array> covariances;
            af::array weights;
            InitializeGMM(af_data, n_components, means, covariances, weights, covariance_type, run_seed);

            double prev_ll = -std::numeric_limits<double>::max();
            af::array responsibilities;
            int iterations = 0;
            bool converged = false;

            for (int iter = 0; iter < max_iter; ++iter) {
                // E-Step
                responsibilities = EStep(af_data, means, covariances, weights);

                // M-Step
                MStep(af_data, responsibilities, means, covariances, weights, covariance_type);

                // Compute log-likelihood
                double ll = ComputeLogLikelihood(af_data, means, covariances, weights);

                if (init_run == 0 && progress_callback) {
                    progress_callback(iter + 1, ll);
                }

                if (std::abs(ll - prev_ll) < tol) {
                    converged = true;
                    iterations = iter + 1;
                    break;
                }

                prev_ll = ll;
                iterations = iter + 1;
            }

            double final_ll = ComputeLogLikelihood(af_data, means, covariances, weights);

            if (final_ll > best_ll) {
                best_ll = final_ll;
                best_means = means;
                best_covs = covariances;
                best_weights = weights;
                best_resp = responsibilities;
                best_iters = iterations;
                best_converged = converged;
            }
        }

        // Convert results
        result.means = FromAfArray(best_means);

        // Convert responsibilities
        result.responsibilities.resize(n_samples);
        std::vector<double> resp_flat(n_samples * n_components);
        best_resp.host(resp_flat.data());
        for (int i = 0; i < n_samples; ++i) {
            result.responsibilities[i].resize(n_components);
            for (int j = 0; j < n_components; ++j) {
                result.responsibilities[i][j] = resp_flat[j * n_samples + i];
            }
        }

        // Hard labels
        af::array max_vals, label_indices;
        af::max(max_vals, label_indices, best_resp, 1);
        result.labels = AfArrayToIntVector(label_indices);

        // Weights
        result.weights = AfArrayToDoubleVector(best_weights);

        // Covariances
        result.covariances.resize(n_components);
        for (int k = 0; k < n_components; ++k) {
            result.covariances[k] = FromAfArray(best_covs[k]);
        }

        result.log_likelihood = best_ll;
        result.n_components = n_components;
        result.n_iterations = best_iters;
        result.converged = best_converged;

        // BIC and AIC
        int n_params = n_components * n_features + n_components * n_features * (n_features + 1) / 2 + n_components - 1;
        result.bic = -2 * best_ll + n_params * std::log(static_cast<double>(n_samples));
        result.aic = -2 * best_ll + 2 * n_params;

        result.success = true;

        spdlog::info("GMM completed: {} components, log-likelihood={:.4f}", n_components, best_ll);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        spdlog::error("GMM failed: {}", result.error_message);
    }

    return result;
}

void Clustering::InitializeGMM(
    const af::array& data,
    int n_components,
    af::array& means,
    std::vector<af::array>& covariances,
    af::array& weights,
    const std::string& covariance_type,
    unsigned int seed
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));

    // Initialize means using k-means++
    means = InitializeCentroidsKMeansPP(data, n_components, seed);

    // Initialize weights uniformly
    weights = af::constant(1.0 / n_components, n_components, f64);

    // Initialize covariances
    covariances.clear();
    af::array data_centered = data - af::tile(af::mean(data, 0), n_samples, 1);
    af::array global_cov = af::matmul(data_centered.T(), data_centered) / static_cast<double>(n_samples - 1);

    for (int k = 0; k < n_components; ++k) {
        if (covariance_type == "spherical") {
            double var = af::mean<double>(af::diag(global_cov));
            covariances.push_back(af::identity(n_features, n_features, f64) * var);
        } else if (covariance_type == "diag") {
            covariances.push_back(af::diag(af::diag(global_cov), 0, false));
        } else {
            covariances.push_back(global_cov.copy());
        }
    }
}

af::array Clustering::EStep(
    const af::array& data,
    const af::array& means,
    const std::vector<af::array>& covariances,
    const af::array& weights
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_components = static_cast<int>(means.dims(0));

    af::array log_probs = af::constant(0.0, n_samples, n_components, f64);

    for (int k = 0; k < n_components; ++k) {
        af::array pdf = GaussianPDF(data, means(k, af::span), covariances[k]);
        log_probs(af::span, k) = af::log(pdf + 1e-300) + std::log(weights(k).scalar<double>());
    }

    // Log-sum-exp for numerical stability
    af::array max_log = af::max(log_probs, 1);
    af::array log_probs_shifted = log_probs - af::tile(max_log, 1, n_components);
    af::array sum_exp = af::sum(af::exp(log_probs_shifted), 1);
    af::array log_sum = max_log + af::log(sum_exp);

    // Responsibilities
    af::array responsibilities = af::exp(log_probs - af::tile(log_sum, 1, n_components));

    return responsibilities;
}

void Clustering::MStep(
    const af::array& data,
    const af::array& responsibilities,
    af::array& means,
    std::vector<af::array>& covariances,
    af::array& weights,
    const std::string& covariance_type
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));
    int n_components = static_cast<int>(means.dims(0));

    // Update weights
    af::array nk = af::sum(responsibilities, 0).T();  // [n_components x 1]
    weights = nk / static_cast<double>(n_samples);

    // Update means
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);  // [n_samples x 1]
        af::array weighted_sum = af::sum(data * af::tile(resp_k, 1, n_features), 0);
        means(k, af::span) = weighted_sum / nk(k).scalar<double>();
    }

    // Update covariances
    for (int k = 0; k < n_components; ++k) {
        af::array resp_k = responsibilities(af::span, k);
        af::array mean_k = means(k, af::span);
        af::array diff = data - af::tile(mean_k, n_samples, 1);

        if (covariance_type == "spherical") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            double var = af::sum<double>(weighted_sq) / (nk(k).scalar<double>() * n_features);
            covariances[k] = af::identity(n_features, n_features, f64) * var;
        } else if (covariance_type == "diag") {
            af::array weighted_sq = af::sum(diff * diff * af::tile(resp_k, 1, n_features), 0);
            af::array variances = weighted_sq / nk(k).scalar<double>();
            covariances[k] = af::diag(variances.T(), 0, false);
        } else {
            af::array weighted_diff = diff * af::tile(af::sqrt(resp_k), 1, n_features);
            covariances[k] = af::matmul(weighted_diff.T(), weighted_diff) / nk(k).scalar<double>();
            // Add regularization
            covariances[k] += af::identity(n_features, n_features, f64) * 1e-6;
        }
    }
}

af::array Clustering::GaussianPDF(
    const af::array& data,
    const af::array& mean,
    const af::array& covariance
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_features = static_cast<int>(data.dims(1));

    af::array diff = data - af::tile(mean, n_samples, 1);

    // Compute (x - mu)^T @ Sigma^-1 @ (x - mu) for each sample
    af::array cov_inv = af::inverse(covariance);
    af::array mahalanobis = af::sum(af::matmul(diff, cov_inv) * diff, 1);

    double log_det = af::sum<double>(af::log(af::abs(af::diag(covariance))));
    double log_norm = -0.5 * (n_features * std::log(2 * M_PI) + log_det);

    return af::exp(log_norm - 0.5 * mahalanobis);
}

double Clustering::ComputeLogLikelihood(
    const af::array& data,
    const af::array& means,
    const std::vector<af::array>& covariances,
    const af::array& weights
) {
    int n_samples = static_cast<int>(data.dims(0));
    int n_components = static_cast<int>(means.dims(0));

    af::array weighted_probs = af::constant(0.0, n_samples, f64);

    for (int k = 0; k < n_components; ++k) {
        af::array pdf = GaussianPDF(data, means(k, af::span), covariances[k]);
        weighted_probs += weights(k).scalar<double>() * pdf;
    }

    return af::sum<double>(af::log(weighted_probs + 1e-300));
}

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

} // namespace cyxwiz
