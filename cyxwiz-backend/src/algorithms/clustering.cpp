// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/clustering.h"
#include <spdlog/spdlog.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <set>
#include <queue>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    (void)n;  // Suppress unused variable warning

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

#else  // No ArrayFire - CPU fallback stubs

#endif  // CYXWIZ_HAS_ARRAYFIRE

} // namespace cyxwiz
