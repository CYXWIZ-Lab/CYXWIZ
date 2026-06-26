#pragma once

#include "metric_learning_batch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

struct PairMetricResult {
    size_t pair_count = 0;
    double threshold = 0.0;
    double accuracy = 0.0;
    double positive_distance_mean = 0.0;
    double negative_distance_mean = 0.0;
    size_t positive_count = 0;
    size_t negative_count = 0;
};

struct RetrievalMetricResult {
    size_t query_count = 0;
    size_t k = 0;
    double recall_at_k = 0.0;
    double mean_reciprocal_rank = 0.0;
    double nearest_neighbor_class_agreement = 0.0;
};

inline size_t FlattenedEmbeddingDim(const Tensor& embeddings) {
    const auto& shape = embeddings.Shape();
    if (shape.size() < 2 || shape[0] == 0) {
        throw std::invalid_argument(
            "metric-learning embeddings must be [batch, ...features]");
    }
    return embeddings.NumElements() / shape[0];
}

inline double EuclideanDistanceRow(const Tensor& left,
                                   const Tensor& right,
                                   size_t row,
                                   size_t dim) {
    const float* left_data = left.Data<float>();
    const float* right_data = right.Data<float>();
    double sum = 0.0;
    const size_t offset = row * dim;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(left_data[offset + i]) -
                            static_cast<double>(right_data[offset + i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline double EuclideanDistanceRows(const Tensor& embeddings,
                                    size_t left_row,
                                    size_t right_row,
                                    size_t dim) {
    const float* data = embeddings.Data<float>();
    double sum = 0.0;
    const size_t left_offset = left_row * dim;
    const size_t right_offset = right_row * dim;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(data[left_offset + i]) -
                            static_cast<double>(data[right_offset + i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline bool PairLabelMeansSimilar(MetricLearningLabelConvention convention,
                                  double label) {
    if (!IsValidMetricLearningLabel(convention, label)) {
        throw std::invalid_argument(
            "metric-learning pair label does not match convention");
    }
    switch (convention) {
        case MetricLearningLabelConvention::
            ContrastiveZeroSimilarOneDissimilar:
            return label == 0.0;
        case MetricLearningLabelConvention::
            CosineOneSimilarNegativeOneDissimilar:
            return label == 1.0;
        case MetricLearningLabelConvention::TripletNoLabels:
            throw std::invalid_argument(
                "triplet label convention cannot score pair metrics");
    }
    return false;
}

inline PairMetricResult ComputePairDistanceMetrics(
    const Tensor& embedding_a,
    const Tensor& embedding_b,
    const Tensor& pair_label,
    MetricLearningLabelConvention convention,
    double threshold) {
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "pair metric embeddings must have identical shapes");
    }

    const auto& shape = embedding_a.Shape();
    const size_t pair_count = shape.empty() ? 0 : shape[0];
    if (pair_count == 0) {
        throw std::invalid_argument("pair metric embeddings cannot be empty");
    }
    if (!TensorIsBatchVector(pair_label, pair_count)) {
        throw std::invalid_argument(
            "pair metric labels must be [batch] or [batch, 1]");
    }

    const size_t dim = FlattenedEmbeddingDim(embedding_a);
    const float* labels = pair_label.Data<float>();

    PairMetricResult result;
    result.pair_count = pair_count;
    result.threshold = threshold;

    size_t correct = 0;
    double positive_sum = 0.0;
    double negative_sum = 0.0;

    for (size_t row = 0; row < pair_count; ++row) {
        const double distance =
            EuclideanDistanceRow(embedding_a, embedding_b, row, dim);
        const bool similar = PairLabelMeansSimilar(
            convention, static_cast<double>(labels[row]));
        const bool predicted_similar = distance <= threshold;
        if (predicted_similar == similar) {
            ++correct;
        }
        if (similar) {
            positive_sum += distance;
            ++result.positive_count;
        } else {
            negative_sum += distance;
            ++result.negative_count;
        }
    }

    result.accuracy = static_cast<double>(correct) /
                      static_cast<double>(pair_count);
    result.positive_distance_mean =
        result.positive_count == 0
            ? 0.0
            : positive_sum / static_cast<double>(result.positive_count);
    result.negative_distance_mean =
        result.negative_count == 0
            ? 0.0
            : negative_sum / static_cast<double>(result.negative_count);
    return result;
}

inline int64_t ClassIdAt(const Tensor& class_ids, size_t row) {
    switch (class_ids.GetDataType()) {
        case DataType::Int64:
            return class_ids.Data<int64_t>()[row];
        case DataType::Int32:
            return static_cast<int64_t>(class_ids.Data<int32_t>()[row]);
        case DataType::Float32:
            return static_cast<int64_t>(class_ids.Data<float>()[row]);
        default:
            throw std::invalid_argument(
                "retrieval class IDs must be Int64, Int32, or Float32");
    }
}

inline RetrievalMetricResult ComputeRetrievalMetrics(
    const Tensor& embeddings,
    const Tensor& class_ids,
    size_t k) {
    const auto& shape = embeddings.Shape();
    const size_t sample_count = shape.empty() ? 0 : shape[0];
    if (sample_count < 2) {
        throw std::invalid_argument(
            "retrieval metrics require at least two embeddings");
    }
    if (k == 0) {
        throw std::invalid_argument("retrieval k must be non-zero");
    }
    if (!TensorIsBatchVector(class_ids, sample_count)) {
        throw std::invalid_argument(
            "retrieval class IDs must be [batch] or [batch, 1]");
    }

    const size_t dim = FlattenedEmbeddingDim(embeddings);
    const size_t effective_k = std::min(k, sample_count - 1);

    RetrievalMetricResult result;
    result.query_count = sample_count;
    result.k = effective_k;

    double recall_hits = 0.0;
    double reciprocal_rank_sum = 0.0;
    double nearest_match_sum = 0.0;

    for (size_t query = 0; query < sample_count; ++query) {
        std::vector<std::pair<double, size_t>> distances;
        distances.reserve(sample_count - 1);
        for (size_t candidate = 0; candidate < sample_count; ++candidate) {
            if (candidate == query) {
                continue;
            }
            distances.emplace_back(
                EuclideanDistanceRows(embeddings, query, candidate, dim),
                candidate);
        }
        std::sort(distances.begin(), distances.end(),
                  [](const auto& left, const auto& right) {
                      if (left.first == right.first) {
                          return left.second < right.second;
                      }
                      return left.first < right.first;
                  });

        const int64_t query_class = ClassIdAt(class_ids, query);
        size_t first_relevant_rank = 0;
        for (size_t rank = 0; rank < distances.size(); ++rank) {
            const size_t candidate = distances[rank].second;
            if (ClassIdAt(class_ids, candidate) == query_class) {
                first_relevant_rank = rank + 1;
                break;
            }
        }

        if (first_relevant_rank > 0 && first_relevant_rank <= effective_k) {
            recall_hits += 1.0;
        }
        if (first_relevant_rank > 0) {
            reciprocal_rank_sum +=
                1.0 / static_cast<double>(first_relevant_rank);
        }
        if (!distances.empty() &&
            ClassIdAt(class_ids, distances.front().second) == query_class) {
            nearest_match_sum += 1.0;
        }
    }

    const double denom = static_cast<double>(sample_count);
    result.recall_at_k = recall_hits / denom;
    result.mean_reciprocal_rank = reciprocal_rank_sum / denom;
    result.nearest_neighbor_class_agreement = nearest_match_sum / denom;
    return result;
}

}  // namespace cyxwiz
