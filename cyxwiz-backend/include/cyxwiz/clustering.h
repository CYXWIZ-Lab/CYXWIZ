#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <string>
#include <functional>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>

// Forward declare ArrayFire array to avoid header dependency in public interface
namespace af {
    class array;
}

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
    std::vector<double> inertia_history;              // Inertia per iteration (for elbow plot)

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
    std::vector<std::vector<double>> linkage_matrix;  // For dendrogram: [idx1, idx2, distance, count]
    std::vector<int> labels;                          // Cluster labels at cut level
    int n_clusters = 0;
    double cophenetic_correlation = 0.0;              // Quality measure

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
    std::vector<std::vector<std::vector<double>>> covariances; // Covariance matrices
    std::vector<double> weights;                      // Component weights
    double log_likelihood = 0.0;
    double bic = 0.0;                                 // Bayesian Information Criterion
    double aic = 0.0;                                 // Akaike Information Criterion
    int n_components = 0;
    int n_iterations = 0;
    bool converged = false;

    bool success = false;
    std::string error_message;
};

/**
 * ClusterMetrics - Cluster quality evaluation metrics
 */
struct CYXWIZ_API ClusterMetrics {
    double silhouette_score = 0.0;                    // -1 to 1, higher is better
    double davies_bouldin_index = 0.0;                // Lower is better
    double calinski_harabasz_score = 0.0;             // Higher is better
    std::vector<double> per_sample_silhouette;        // Silhouette for each sample
    std::vector<double> cluster_silhouettes;          // Average silhouette per cluster

    int n_clusters = 0;
    int n_samples = 0;

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
    int suggested_k = 0;                              // Elbow point

    bool success = false;
    std::string error_message;
};

/**
 * Clustering - Static class with GPU-accelerated clustering algorithms using ArrayFire
 *
 * All algorithms use ArrayFire for GPU computation (CUDA/OpenCL/CPU backends).
 * Public interface uses std::vector for easy integration with GUI panels.
 */
class CYXWIZ_API Clustering {
public:
    // ==================== K-Means ====================

    /**
     * K-Means clustering algorithm (GPU-accelerated)
     *
     * @param data Input data [n_samples x n_features]
     * @param n_clusters Number of clusters (k)
     * @param max_iter Maximum iterations
     * @param init Initialization method: "random" or "kmeans++"
     * @param n_init Number of random initializations to try
     * @param tol Convergence tolerance
     * @param seed Random seed (0 = random)
     * @param progress_callback Optional callback (iteration, inertia)
     * @return KMeansResult
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
     * Elbow method analysis for optimal k selection
     *
     * @param data Input data
     * @param k_min Minimum k to try
     * @param k_max Maximum k to try
     * @param progress_callback Optional callback (current_k, total_k)
     * @return ElbowAnalysis with inertias and silhouette scores
     */
    static ElbowAnalysis ComputeElbowAnalysis(
        const std::vector<std::vector<double>>& data,
        int k_min = 2,
        int k_max = 10,
        std::function<void(int, int)> progress_callback = nullptr
    );

    // ==================== DBSCAN ====================

    /**
     * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
     * GPU-accelerated distance matrix computation
     *
     * @param data Input data [n_samples x n_features]
     * @param eps Maximum distance for neighborhood
     * @param min_samples Minimum points to form a dense region
     * @param metric Distance metric: "euclidean", "manhattan", "cosine"
     * @return DBSCANResult
     */
    static DBSCANResult DBSCAN(
        const std::vector<std::vector<double>>& data,
        double eps,
        int min_samples = 5,
        const std::string& metric = "euclidean"
    );

    /**
     * Suggest epsilon for DBSCAN using k-distance graph (GPU-accelerated)
     *
     * @param data Input data
     * @param k Number of nearest neighbors (usually min_samples)
     * @return Sorted k-distances for plotting
     */
    static std::vector<double> ComputeKDistances(
        const std::vector<std::vector<double>>& data,
        int k = 5
    );

    // ==================== Hierarchical ====================

    /**
     * Agglomerative Hierarchical Clustering (GPU-accelerated distance matrix)
     *
     * @param data Input data [n_samples x n_features]
     * @param n_clusters Number of clusters (for cutting dendrogram)
     * @param linkage Linkage method: "ward", "complete", "average", "single"
     * @param metric Distance metric: "euclidean", "manhattan", "cosine"
     * @return HierarchicalResult with linkage matrix for dendrogram
     */
    static HierarchicalResult Hierarchical(
        const std::vector<std::vector<double>>& data,
        int n_clusters,
        const std::string& linkage = "ward",
        const std::string& metric = "euclidean"
    );

    /**
     * Cut dendrogram at a specific height
     *
     * @param linkage_matrix Linkage matrix from Hierarchical()
     * @param height Cut height
     * @param n_samples Original number of samples
     * @return Cluster labels
     */
    static std::vector<int> CutDendrogram(
        const std::vector<std::vector<double>>& linkage_matrix,
        double height,
        int n_samples
    );

    // ==================== Gaussian Mixture Model ====================

    /**
     * Gaussian Mixture Model (EM algorithm, GPU-accelerated)
     *
     * @param data Input data [n_samples x n_features]
     * @param n_components Number of Gaussian components
     * @param covariance_type "full", "tied", "diag", "spherical"
     * @param max_iter Maximum EM iterations
     * @param tol Convergence tolerance
     * @param n_init Number of random initializations
     * @param seed Random seed
     * @param progress_callback Optional callback (iteration, log_likelihood)
     * @return GMMResult
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
     * Compute clustering quality metrics (GPU-accelerated)
     *
     * @param data Input data [n_samples x n_features]
     * @param labels Cluster labels
     * @return ClusterMetrics with silhouette, Davies-Bouldin, etc.
     */
    static ClusterMetrics EvaluateClustering(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );

    /**
     * Compute silhouette score only
     */
    static double ComputeSilhouetteScore(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );

    /**
     * Compute Davies-Bouldin index only
     */
    static double ComputeDaviesBouldinIndex(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );

    /**
     * Compute Calinski-Harabasz score (Variance Ratio Criterion)
     */
    static double ComputeCalinskiHarabaszScore(
        const std::vector<std::vector<double>>& data,
        const std::vector<int>& labels
    );

private:
    // ==================== ArrayFire Conversion Helpers ====================

    // Convert std::vector<std::vector<double>> to af::array [n_samples x n_features]
    static af::array ToAfArray(const std::vector<std::vector<double>>& data);

    // Convert af::array to std::vector<std::vector<double>>
    static std::vector<std::vector<double>> FromAfArray(const af::array& arr);

    // Convert af::array to std::vector<int> (for labels)
    static std::vector<int> AfArrayToIntVector(const af::array& arr);

    // Convert af::array to std::vector<double>
    static std::vector<double> AfArrayToDoubleVector(const af::array& arr);

    // ==================== GPU Distance Functions ====================

    // Compute pairwise Euclidean distance matrix using GPU
    static af::array ComputeEuclideanDistanceMatrix(const af::array& data);

    // Compute pairwise Manhattan distance matrix using GPU
    static af::array ComputeManhattanDistanceMatrix(const af::array& data);

    // Compute pairwise Cosine distance matrix using GPU
    static af::array ComputeCosineDistanceMatrix(const af::array& data);

    // Compute distance matrix based on metric name
    static af::array ComputeDistanceMatrix(const af::array& data, const std::string& metric);

    // Compute distances from points to centroids
    static af::array ComputePointToCentroidDistances(const af::array& data, const af::array& centroids);

    // ==================== K-Means GPU Helpers ====================

    // Initialize centroids using k-means++ on GPU
    static af::array InitializeCentroidsKMeansPP(const af::array& data, int n_clusters, unsigned int seed);

    // Initialize centroids randomly on GPU
    static af::array InitializeCentroidsRandom(const af::array& data, int n_clusters, unsigned int seed);

    // Assign points to nearest centroid (returns label indices)
    static af::array AssignClusters(const af::array& data, const af::array& centroids);

    // Update centroids based on cluster assignments
    static af::array UpdateCentroids(const af::array& data, const af::array& labels, int n_clusters);

    // Compute inertia (sum of squared distances to centroids)
    static double ComputeInertia(const af::array& data, const af::array& labels, const af::array& centroids);

    // ==================== GMM GPU Helpers ====================

    // Initialize GMM parameters
    static void InitializeGMM(
        const af::array& data,
        int n_components,
        af::array& means,
        std::vector<af::array>& covariances,
        af::array& weights,
        const std::string& covariance_type,
        unsigned int seed
    );

    // E-Step: Compute responsibilities
    static af::array EStep(
        const af::array& data,
        const af::array& means,
        const std::vector<af::array>& covariances,
        const af::array& weights
    );

    // M-Step: Update parameters
    static void MStep(
        const af::array& data,
        const af::array& responsibilities,
        af::array& means,
        std::vector<af::array>& covariances,
        af::array& weights,
        const std::string& covariance_type
    );

    // Compute multivariate Gaussian PDF for each component
    static af::array GaussianPDF(
        const af::array& data,
        const af::array& mean,
        const af::array& covariance
    );

    // Compute log-likelihood
    static double ComputeLogLikelihood(
        const af::array& data,
        const af::array& means,
        const std::vector<af::array>& covariances,
        const af::array& weights
    );

    // ==================== Cluster Evaluation GPU Helpers ====================

    // Compute silhouette coefficients for all samples
    static af::array ComputeSilhouetteCoefficients(const af::array& dist_matrix, const af::array& labels, int n_clusters);
};

} // namespace cyxwiz
