#pragma once

// Device helpers shared by the clustering sources (TOFIX140): host rows in and
// out of ArrayFire, pairwise distances in row blocks, seeded random engines and
// k-means++ seeding. Data matrices are [n_samples, n_features] in the compute
// precision (float64 where the device has it).

#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_ARRAYFIRE

#include "arrayfire_host_values.h"

#include <arrayfire.h>

#include <stdexcept>

namespace cyxwiz::clustering_detail {

using Matrix = std::vector<std::vector<double>>;

// An input the algorithm cannot use; its message becomes the result's error.
struct Refusal : std::runtime_error {
    using std::runtime_error::runtime_error;
};

af::array MatrixToDevice(const Matrix& data);
Matrix MatrixFromDevice(const af::array& source, const char* operation);
std::vector<int> LabelsFromDevice(const af::array& source, const char* operation);

// Rows per block so a [rows, n] intermediate stays within the block budget.
dim_t RowBlock(dim_t n);

// Distances from each row of `rows` [b, d] to each row of `all` [n, d]: [b, n].
// Euclidean and Manhattan are accumulated feature by feature (no |a|^2 + |b|^2
// - 2ab cancellation); cosine is 1 - u.v / (|u| |v|).
af::array PairwiseDistances(const af::array& rows, const af::array& all, const std::string& metric);

// Squared euclidean distances [b, k] from rows to centers, feature by feature.
af::array SquaredDistances(const af::array& rows, const af::array& centers);

// Engine seeded with `seed`, or from std::random_device when seed is 0.
af::randomEngine SeededEngine(unsigned int seed);

// k-means++ seeding: k rows of data, each next one drawn with probability
// proportional to its squared distance to the nearest chosen row.
af::array KMeansPlusPlus(const af::array& data, int k, af::randomEngine& engine);

}  // namespace cyxwiz::clustering_detail

#endif

namespace cyxwiz::clustering_detail {

// Empty when data is a non-empty rectangular matrix with at least one feature.
std::string MatrixRefusal(const std::vector<std::vector<double>>& data);
bool IsKnownMetric(const std::string& metric);
std::string NoArrayFire(const char* operation);

}  // namespace cyxwiz::clustering_detail
