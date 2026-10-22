// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <string>
#include <functional>

namespace cyxwiz {

/**
 * KMeansResult - Result of K-Means clustering
 */
struct CYXWIZ_API KMeansResult {
    std::vector<int> labels;                          // Cluster assignment for each point
    std::vector<std::vector<double>> centroids;       // Cluster centroids
    double inertia = 0.0;                             // Sum of squared distances to centroids
    int n_clusters = 0;
    int n_iterations = 0;
    bool converged = false;
    std::vector<double> inertia_history;              // Inertia at each iteration of the best run

    bool success = false;
    std::string error_message;
};

/**
 * DBSCANResult - Result of DBSCAN clustering
 */
struct CYXWIZ_API DBSCANResult {
    std::vector<int> labels;                          // Cluster labels (-1 = noise)
    int n_clusters = 0;
    int n_noise_points = 0;
    std::vector<bool> core_samples;                   // Whether each point is a core sample

    bool success = false;
    std::string error_message;
};

/**
 * HierarchicalResult - Result of Hierarchical clustering
 */
struct CYXWIZ_API HierarchicalResult {
    // One row per merge, as scipy.cluster.hierarchy.linkage: [id1, id2, distance, count];
    // ids < n are samples, id n + i is the cluster made by row i; id1 < id2.
    std::vector<std::vector<double>> linkage_matrix;
    std::vector<int> labels;                          // Cluster labels after n - n_clusters merges
    int n_clusters = 0;
    double cophenetic_correlation = 0.0;              // scipy cophenet against the input distances

    bool success = false;
    std::string error_message;
};

/**
 * GMMResult - Result of Gaussian Mixture Model clustering
 */
struct CYXWIZ_API GMMResult {
    std::vector<int> labels;                          // Hard cluster assignments
    std::vector<std::vector<double>> responsibilities; // Soft assignments [n_samples x n_components]
    std::vector<std::vector<double>> means;           // Component means
    std::vector<std::vector<std::vector<double>>> covariances; // d x d per component (tied: the shared one)
    std::vector<double> weights;                      // Component weights
    double log_likelihood = 0.0;                      // Total log-likelihood of the data
    double bic = 0.0;                                 // Bayesian Information Criterion
    double aic = 0.0;                                 // Akaike Information Criterion
    int n_components = 0;
    int n_iterations = 0;
    bool converged = false;

    bool success = false;
    std::string error_message;
};

/**
 * ClusterMetrics - Cluster quality evaluation metrics. Noise points (label -1)
 * are left out; labels may be any integers.
 */
struct CYXWIZ_API ClusterMetrics {
    double silhouette_score = 0.0;                    // -1 to 1, higher is better
    double davies_bouldin_index = 0.0;                // Lower is better
    double calinski_harabasz_score = 0.0;             // Higher is better
    std::vector<double> per_sample_silhouette;        // One per sample; 0 for noise points
    std::vector<double> cluster_silhouettes;          // Mean per cluster, in ascending label order

    int n_clusters = 0;
    int n_samples = 0;                                // Samples evaluated (noise excluded)

    bool success = false;
    std::string error_message;
};

/**
 * ElbowAnalysis - For determining optimal k in K-Means
 */
struct CYXWIZ_API ElbowAnalysis {
    std::vector<int> k_values;
    std::vector<double> inertias;
    std::vector<double> silhouette_scores;
    int suggested_k = 0;                              // Largest second difference of the inertias

    bool success = false;
    std::string error_message;
};

/**
 * Clustering - K-Means, DBSCAN, hierarchical, Gaussian mixture and cluster
 * metrics on ArrayFire on the active device (its CPU backend is the CPU
 * option; TOFIX140). Semantics
 * follow scikit-learn (KMeans, DBSCAN, GaussianMixture, silhouette_score,
 * davies_bouldin_score, calinski_harabasz_score) and scipy linkage / cophenet.
 * Rows are samples; every row needs the same number of features.
 */
class CYXWIZ_API Clustering {
public:
    // ==================== K-Means ====================

    /**
     * Lloyd's K-Means. Stops when the total squared centroid shift is at most
     * tol * mean feature variance (scikit-learn). An empty cluster keeps its
     * previous centroid. The best of n_init runs (lowest inertia) is returned.
     *
     * @param init "kmeans++" or "random"
     * @param seed Random seed (0 = a fresh random seed)
     * @param progress_callback Optional (iteration, inertia) of the first run
     */
    static KMeansResult KMeans(
        const std::vector<std::vector<double>>& data,
        int n_clusters,
        int max_iter = 300,
        const std::string& init = "kmeans++",
        int n_init = 10,
        double tol = 1e-4,
        unsigned int seed = 0,
        std::function<void(int, double)> progress_callback = nullptr
    );

    /**
     * K-Means inertia and silhouette for k = k_min .. k_max (k_max is capped
     * at n_samples - 1).
     *
     * @param progress_callback Optional (current_k_index, total_k)
     */
    static ElbowAnalysis ComputeElbowAnalysis(
        const std::vector<std::vector<double>>& data,
        int k_min = 2,
        int k_max = 10,
        std::function<void(int, int)> progress_callback = nullptr
    );

    // ==================== DBSCAN ====================

    /**
     * DBSCAN as scikit-learn: a point is core when at least min_samples points
     * (itself included) lie within eps (inclusive); clusters are numbered in
     * order of their lowest-index core point, and a border point joins the
     * earliest cluster that reaches it.
     *
     * @param metric "euclidean", "manhattan" or "cosine"
     */
    static DBSCANResult DBSCAN(
        const std::vector<std::vector<double>>& data,
        double eps,
        int min_samples = 5,
        const std::string& metric = "euclidean"
    );

    /**
     * Each point's euclidean distance to its k-th nearest other point, sorted
     * ascending (the k-distance graph for choosing eps). Empty on failure; the
     * reason is logged.
     */
    static std::vector<double> ComputeKDistances(
        const std::vector<std::vector<double>>& data,
        int k = 5
    );

    /** As above, with the distance "euclidean", "manhattan" or "cosine" (DBSCAN's metrics). */
    static std::vector<double> ComputeKDistances(
        const std::vector<std::vector<double>>& data,
        int k,
        const std::string& metric
    );

    // ==================== Hierarchical ====================

    /**
     * Agglomerative clustering (scipy linkage via Lance-Williams updates).
     *
     * @param linkage "ward" (euclidean only), "complete", "average" or "single"
     * @param metric "euclidean", "manhattan" or "cosine"
     */
    static HierarchicalResult Hierarchical(
        const std::vector<std::vector<double>>& data,
        int n_clusters,
        const std::string& linkage = "ward",
        const std::string& metric = "euclidean"
    );

    /**
     * Labels after applying every merge whose distance is at most `height`
     * (scipy fcluster, criterion "distance"); clusters are numbered in order of
     * first appearance.
     */
    static std::vector<int> CutDendrogram(
        const std::vector<std::vector<double>>& linkage_matrix,
        double height,
        int n_samples
    );

    // ==================== Gaussian Mixture Model ====================

    /**
     * Gaussian mixture by EM (scikit-learn GaussianMixture: reg_covar 1e-6,
     * k-means++ seeded start). Stops when the mean per-sample log-likelihood
     * changes by less than tol; the best of n_init runs is returned.
     *
     * @param covariance_type "full", "tied", "diag" or "spherical"
     * @param seed Random seed (0 = a fresh random seed)
     * @param progress_callback Optional (iteration, total log-likelihood) of the first run
     */
    static GMMResult GMM(
        const std::vector<std::vector<double>>& data,
        int n_components,
        const std::string& covariance_type = "full",
        int max_iter = 100,
        double tol = 1e-3,
        int n_init = 1,
        unsigned int seed = 0,
        std::function<void(int, double)> progress_callback = nullptr
    );

    // ==================== Cluster Evaluation ====================

    /**
     * Silhouette, Davies-Bouldin and Calinski-Harabasz scores (euclidean).
     * Needs 2 to n - 1 clusters among the non-noise points.
     */
    static ClusterMetrics EvaluateClustering(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );

    /**
     * Mean silhouette, or NaN when it cannot be computed.
     */
    static double ComputeSilhouetteScore(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );
};

} // namespace cyxwiz
