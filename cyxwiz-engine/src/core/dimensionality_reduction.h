#pragma once

#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace cyxwiz {

/**
 * PCAResult - Result of Principal Component Analysis
 */
struct PCAResult {
    std::vector<std::vector<double>> components;      // Principal components [n_components x n_features]
    std::vector<std::vector<double>> transformed;     // Projected data [n_samples x n_components]
    std::vector<double> explained_variance;           // Variance per component
    std::vector<double> explained_variance_ratio;     // Percentage of variance
    std::vector<double> singular_values;
    int n_components = 0;
    double total_variance_explained = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * tSNEResult - Result of t-SNE dimensionality reduction
 */
struct tSNEResult {
    std::vector<std::vector<double>> embeddings;      // 2D/3D coordinates [n_samples x n_dims]
    std::vector<double> kl_divergence_history;        // Convergence tracking
    double final_kl_divergence = 0.0;
    int n_iterations = 0;
    int perplexity = 0;
    bool success = false;
    std::string error_message;
};

/**
 * UMAPResult - Result of UMAP dimensionality reduction
 */
struct UMAPResult {
    std::vector<std::vector<double>> embeddings;      // 2D/3D coordinates
    int n_neighbors = 0;
    double min_dist = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * DimensionalityReduction - Algorithms for reducing data dimensions
 *
 * Implements:
 * - PCA (Principal Component Analysis)
 * - t-SNE (t-Distributed Stochastic Neighbor Embedding)
 * - UMAP (Uniform Manifold Approximation and Projection) - simplified
 */
class DimensionalityReduction {
public:
    /**
     * Compute PCA on input data
     * @param data Input data [n_samples x n_features]
     * @param n_components Number of components to keep (default 2)
     * @param center Whether to center data (default true)
     * @param scale Whether to scale to unit variance (default false)
     * @return PCAResult with components and transformed data
     */
    static PCAResult ComputePCA(
        const std::vector<std::vector<double>>& data,
        int n_components = 2,
        bool center = true,
        bool scale = false
    );

    /**
     * Compute t-SNE embedding
     * @param data Input data [n_samples x n_features]
     * @param n_dims Output dimensions (2 or 3)
     * @param perplexity Perplexity parameter (default 30)
     * @param learning_rate Learning rate (default 200)
     * @param n_iterations Number of iterations (default 1000)
     * @param progress_callback Optional callback for progress updates (iteration, kl_divergence)
     * @return tSNEResult with embeddings
     */
    static tSNEResult ComputetSNE(
        const std::vector<std::vector<double>>& data,
        int n_dims = 2,
        int perplexity = 30,
        double learning_rate = 200.0,
        int n_iterations = 1000,
        std::function<void(int, double)> progress_callback = nullptr
    );

    /**
     * Compute UMAP embedding (simplified version)
     * @param data Input data [n_samples x n_features]
     * @param n_dims Output dimensions (2 or 3)
     * @param n_neighbors Number of neighbors (default 15)
     * @param min_dist Minimum distance (default 0.1)
     * @return UMAPResult with embeddings
     */
    static UMAPResult ComputeUMAP(
        const std::vector<std::vector<double>>& data,
        int n_dims = 2,
        int n_neighbors = 15,
        double min_dist = 0.1
    );

private:
    // === Helper methods for PCA ===

    /**
     * Center data by subtracting column means
     */
    static std::vector<std::vector<double>> CenterData(
        const std::vector<std::vector<double>>& data
    );

    /**
     * Scale data to unit variance
     */
    static std::vector<std::vector<double>> ScaleData(
        const std::vector<std::vector<double>>& data
    );

    /**
     * Compute covariance matrix
     */
    static std::vector<std::vector<double>> ComputeCovarianceMatrix(
        const std::vector<std::vector<double>>& data
    );

    /**
     * Power iteration for computing dominant eigenvector
     */
    static std::vector<double> PowerIteration(
        const std::vector<std::vector<double>>& matrix,
        int max_iterations = 1000,
        double tolerance = 1e-10
    );

    /**
     * Deflate matrix by removing contribution of eigenvector
     */
    static void DeflateMatrix(
        std::vector<std::vector<double>>& matrix,
        const std::vector<double>& eigenvector,
        double eigenvalue
    );

    // === Helper methods for t-SNE ===

    /**
     * Compute pairwise squared Euclidean distances
     */
    static std::vector<std::vector<double>> ComputeSquaredDistances(
        const std::vector<std::vector<double>>& data
    );

    /**
     * Compute conditional probabilities P(j|i) with given perplexity
     */
    static std::vector<std::vector<double>> ComputeConditionalProbabilities(
        const std::vector<std::vector<double>>& squared_distances,
        int perplexity
    );

    /**
     * Binary search to find sigma that gives target entropy (perplexity)
     */
    static double FindSigma(
        const std::vector<double>& distances_i,
        int i,
        double target_entropy,
        double tolerance = 1e-5,
        int max_iterations = 50
    );

    /**
     * Compute joint probabilities P_ij = (P(j|i) + P(i|j)) / (2n)
     */
    static std::vector<std::vector<double>> ComputeJointProbabilities(
        const std::vector<std::vector<double>>& cond_probs
    );

    /**
     * Compute Q distribution (Student's t with df=1)
     */
    static std::vector<std::vector<double>> ComputeQDistribution(
        const std::vector<std::vector<double>>& embeddings,
        double& sum_q
    );

    /**
     * Compute KL divergence gradient
     */
    static std::vector<std::vector<double>> ComputetSNEGradient(
        const std::vector<std::vector<double>>& P,
        const std::vector<std::vector<double>>& Q,
        const std::vector<std::vector<double>>& embeddings,
        double sum_q
    );

    /**
     * Compute KL divergence
     */
    static double ComputeKLDivergence(
        const std::vector<std::vector<double>>& P,
        const std::vector<std::vector<double>>& Q
    );

    // === Helper methods for UMAP ===

    /**
     * Find k nearest neighbors
     */
    static std::vector<std::vector<int>> FindKNearestNeighbors(
        const std::vector<std::vector<double>>& squared_distances,
        int k
    );

    /**
     * Compute UMAP graph weights
     */
    static std::vector<std::vector<double>> ComputeUMAPWeights(
        const std::vector<std::vector<double>>& squared_distances,
        const std::vector<std::vector<int>>& neighbors,
        int n_neighbors
    );

    // === Utility methods ===

    /**
     * Matrix-vector multiplication
     */
    static std::vector<double> MatVecMul(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& vec
    );

    /**
     * Vector dot product
     */
    static double DotProduct(
        const std::vector<double>& a,
        const std::vector<double>& b
    );

    /**
     * Vector L2 norm
     */
    static double Norm(const std::vector<double>& vec);

    /**
     * Normalize vector to unit length
     */
    static void Normalize(std::vector<double>& vec);
};

} // namespace cyxwiz
