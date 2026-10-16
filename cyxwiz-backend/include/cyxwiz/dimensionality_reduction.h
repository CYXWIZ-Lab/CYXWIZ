#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <functional>
#include <string>

namespace cyxwiz {

/**
 * PCAResult - Result of Principal Component Analysis
 */
struct CYXWIZ_API PCAResult {
    std::vector<std::vector<double>> components;      // Principal axes [n_components x n_features], unit rows
    std::vector<std::vector<double>> transformed;     // Projected data [n_samples x n_components]
    std::vector<double> explained_variance;           // Variance per component (divides by n - 1)
    std::vector<double> explained_variance_ratio;     // Fraction of the total variance
    std::vector<double> singular_values;
    int n_components = 0;
    double total_variance_explained = 0.0;
    bool success = false;
    std::string error_message;
};

/**
 * tSNEResult - Result of t-SNE dimensionality reduction
 */
struct CYXWIZ_API tSNEResult {
    std::vector<std::vector<double>> embeddings;      // [n_samples x n_dims]
    std::vector<double> kl_divergence_history;        // KL every 50 iterations
    double final_kl_divergence = 0.0;                 // KL at the last iteration
    int n_iterations = 0;                             // Iterations run
    int perplexity = 0;
    bool success = false;
    std::string error_message;
};

/**
 * UMAPResult - Result of UMAP dimensionality reduction
 */
struct CYXWIZ_API UMAPResult {
    std::vector<std::vector<double>> embeddings;      // [n_samples x n_dims]
    int n_neighbors = 0;
    double min_dist = 0.0;
    double a = 0.0;                                   // Low-dimensional curve 1 / (1 + a d^(2b))
    double b = 0.0;
    int n_epochs = 0;
    bool success = false;
    std::string error_message;
};

/**
 * DimensionalityReduction - PCA, t-SNE and UMAP on ArrayFire on the active
 * device (its CPU backend is the CPU option; TOFIX140). A build without
 * ArrayFire returns an error. Rows are samples; every row needs the same number
 * of features. t-SNE and UMAP hold n x n matrices on the device.
 */
class CYXWIZ_API DimensionalityReduction {
public:
    /**
     * PCA as scikit-learn (eigen-decomposition of the n - 1 covariance; each
     * component's largest-magnitude entry is positive).
     *
     * @param n_components Components to keep; 0 or more than min(samples, features) keeps that minimum
     * @param center Subtract column means (without it, variances are second moments about 0)
     * @param scale Divide each feature by its standard deviation (divides by n; constant features are left as they are)
     */
    static PCAResult ComputePCA(
        const std::vector<std::vector<double>>& data,
        int n_components = 2,
        bool center = true,
        bool scale = false
    );

    /**
     * Exact t-SNE as scikit-learn's method="exact", init="pca": perplexity
     * calibrated per point by binary search, early exaggeration 12 for the first
     * 250 iterations (momentum 0.5, then 0.8), gains with minimum 0.01, Student-t
     * kernel with max(n_dims - 1, 1) degrees of freedom.
     *
     * @param perplexity At least 1 and below the number of samples
     * @param progress_callback Optional (iteration, kl_divergence), every 50 iterations
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
     * UMAP: umap-learn's fuzzy graph (exact k-nearest neighbours, smooth_knn_dist,
     * fuzzy union) and a / b curve fit, a PCA start (umap-learn init="pca" without
     * its 1e-4 noise), and its layout updates in expectation over each epoch
     * (edges weighted by w / w_max, 5 negative samples per edge, coordinates
     * clipped to +-4, learning rate falling linearly from 1). The fuzzy graph and a
     * / b match umap-learn; layouts differ from its sampled updates.
     *
     * @param n_neighbors Neighbours including the point itself, 2 to n_samples
     * @param n_epochs 0: 500 up to 10,000 samples, else 200
     */
    static UMAPResult ComputeUMAP(
        const std::vector<std::vector<double>>& data,
        int n_dims = 2,
        int n_neighbors = 15,
        double min_dist = 0.1,
        int n_epochs = 0
    );
};

} // namespace cyxwiz
