#pragma once

#include "metric_learning_batch.h"

#include <cstddef>
#include <cstdint>
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

// Metric-learning evaluation runs on the ArrayFire device (TOFIX140; the CPU
// is ArrayFire's CPU backend): embeddings of any rank [batch, ...features],
// results read back once as a few aggregates.

size_t FlattenedEmbeddingDim(const Tensor& embeddings);

// One id of a [batch] / [batch, 1] Int64 / Int32 / exact-integer Float32
// tensor, read on the host (output metadata).
int64_t ClassIdAt(const Tensor& class_ids, size_t row);

PairMetricResult
ComputePairDistanceMetrics(const Tensor& embedding_a, const Tensor& embedding_b,
                           const Tensor& pair_label,
                           MetricLearningLabelConvention convention,
                           double threshold);

RetrievalMetricResult ComputeRetrievalMetrics(const Tensor& embeddings,
                                              const Tensor& class_ids,
                                              size_t k);

// Per pair row of two [N, ...features] batches: the Euclidean distance and,
// when asked, the cosine similarity (zero vectors refused), computed on the
// device in float64 where it has it and downloaded once.
struct PairRowScores {
    std::vector<double> distances;
    std::vector<double> cosines;
};
PairRowScores ComputePairRowScores(const Tensor& embedding_a,
                                   const Tensor& embedding_b,
                                   bool with_cosine);

} // namespace cyxwiz
