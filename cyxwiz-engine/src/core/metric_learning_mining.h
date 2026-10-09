#pragma once

// Online mining for metric learning (TOFIX140 A5). With mining set to hard or
// semi-hard on a Pair / Triplet Dataset Builder, the batch goes through the
// encoder once as it is ([N, D] embeddings, class ids), and the loss picks its
// pairs or triplets from the [N, N] distance matrix of those embeddings. The
// picks are not differentiated (torch: chosen from detached distances).
// Header-only: the losses and the accuracy decision both use it.

#include "metric_learning_batch.h"

#include <cyxwiz/tensor.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

// Class ids of a batch's labels: Int32/Int64/Float32, shape [N] or [N, 1].
inline std::vector<int64_t> ReadBatchClassIds(const Tensor& labels, size_t rows) {
    const std::string builder = "The Pair / Triplet Dataset Builder";
    if (labels.NumElements() != rows) {
        throw std::runtime_error(builder + " needs one class id per row; the labels have " +
                                 std::to_string(labels.NumElements()) + " values for " + std::to_string(rows) +
                                 " rows");
    }
    std::vector<int64_t> ids(rows);
    switch (labels.GetDataType()) {
        case DataType::Int32: {
            const int32_t* values = labels.ReadData<int32_t>();
            for (size_t i = 0; i < rows; ++i) ids[i] = values[i];
            break;
        }
        case DataType::Int64: {
            const int64_t* values = labels.ReadData<int64_t>();
            for (size_t i = 0; i < rows; ++i) ids[i] = values[i];
            break;
        }
        case DataType::Float32: {
            const float* values = labels.ReadData<float>();
            for (size_t i = 0; i < rows; ++i) {
                if (!std::isfinite(values[i]) || values[i] != std::round(values[i])) {
                    throw std::runtime_error(builder + " needs whole-number class ids; a label is " +
                                             std::to_string(values[i]));
                }
                ids[i] = static_cast<int64_t>(values[i]);
            }
            break;
        }
        default:
            throw std::runtime_error(builder + " needs Int32, Int64 or Float32 class ids");
    }
    return ids;
}

enum class MiningDistance {
    Euclidean,  // Triplet, Contrastive
    Cosine,     // Cosine Embedding: 1 - cos(a, b)
};

// [N, N] distances between the rows of a row-major [N, D] matrix.
inline std::vector<double> MiningDistances(const float* x, size_t n, size_t d, MiningDistance kind) {
    // As the backend Cosine Embedding loss.
    constexpr double kCosineEpsilon = 1.0e-12;
    std::vector<double> distances(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double value = 0.0;
            if (kind == MiningDistance::Euclidean) {
                for (size_t k = 0; k < d; ++k) {
                    const double diff = static_cast<double>(x[i * d + k]) - x[j * d + k];
                    value += diff * diff;
                }
                value = std::sqrt(value);
            } else {
                double dot = 0.0, aa = 0.0, bb = 0.0;
                for (size_t k = 0; k < d; ++k) {
                    dot += static_cast<double>(x[i * d + k]) * x[j * d + k];
                    aa += static_cast<double>(x[i * d + k]) * x[i * d + k];
                    bb += static_cast<double>(x[j * d + k]) * x[j * d + k];
                }
                value = 1.0 - dot / std::sqrt((aa + kCosineEpsilon) * (bb + kCosineEpsilon));
            }
            distances[i * n + j] = value;
            distances[j * n + i] = value;
        }
    }
    return distances;
}

// {anchor, positive, negative} rows, in anchor order.
//   Hard (batch-hard, Hermans et al. 2017): for every row with a positive and
//     a negative, its farthest same-class row and its closest other-class row.
//   SemiHard (FaceNet; the TF Addons triplet_semihard_loss rule): for every
//     same-class pair (anchor, positive), the closest negative farther from
//     the anchor than the positive; when there is none, the farthest negative.
// Ties go to the lowest row index.
inline std::vector<std::array<int, 3>> MineTriplets(const std::vector<int64_t>& class_ids,
                                                     const std::vector<double>& distances,
                                                     MetricMining mining) {
    const size_t n = class_ids.size();
    std::vector<std::array<int, 3>> triplets;
    for (size_t a = 0; a < n; ++a) {
        const double* row = distances.data() + a * n;
        int farthest_positive = -1, closest_negative = -1, farthest_negative = -1;
        for (size_t j = 0; j < n; ++j) {
            if (j == a) continue;
            if (class_ids[j] == class_ids[a]) {
                if (farthest_positive < 0 || row[j] > row[farthest_positive]) farthest_positive = static_cast<int>(j);
            } else {
                if (closest_negative < 0 || row[j] < row[closest_negative]) closest_negative = static_cast<int>(j);
                if (farthest_negative < 0 || row[j] > row[farthest_negative]) farthest_negative = static_cast<int>(j);
            }
        }
        if (farthest_positive < 0 || closest_negative < 0) continue;
        if (mining == MetricMining::Hard) {
            triplets.push_back({static_cast<int>(a), farthest_positive, closest_negative});
            continue;
        }
        for (size_t p = 0; p < n; ++p) {
            if (p == a || class_ids[p] != class_ids[a]) continue;
            int semi_hard = -1;
            for (size_t j = 0; j < n; ++j) {
                if (class_ids[j] == class_ids[a] || row[j] <= row[p]) continue;
                if (semi_hard < 0 || row[j] < row[semi_hard]) semi_hard = static_cast<int>(j);
            }
            triplets.push_back({static_cast<int>(a), static_cast<int>(p), semi_hard >= 0 ? semi_hard : farthest_negative});
        }
    }
    return triplets;
}

// Hard pairs {row, partner, similar}: for every row, its farthest same-class
// row (similar = 1) and its closest other-class row (similar = 0), when it has
// them, in row order. Ties go to the lowest row index.
inline std::vector<std::array<int, 3>> MinePairs(const std::vector<int64_t>& class_ids,
                                                  const std::vector<double>& distances) {
    const size_t n = class_ids.size();
    std::vector<std::array<int, 3>> pairs;
    for (size_t a = 0; a < n; ++a) {
        const double* row = distances.data() + a * n;
        int farthest_positive = -1, closest_negative = -1;
        for (size_t j = 0; j < n; ++j) {
            if (j == a) continue;
            if (class_ids[j] == class_ids[a]) {
                if (farthest_positive < 0 || row[j] > row[farthest_positive]) farthest_positive = static_cast<int>(j);
            } else if (closest_negative < 0 || row[j] < row[closest_negative]) {
                closest_negative = static_cast<int>(j);
            }
        }
        if (farthest_positive >= 0) pairs.push_back({static_cast<int>(a), farthest_positive, 1});
        if (closest_negative >= 0) pairs.push_back({static_cast<int>(a), closest_negative, 0});
    }
    return pairs;
}

}  // namespace cyxwiz
