#pragma once

#include "metric_learning_batch.h"

#include <cstddef>
#include <cstdint>

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

size_t FlattenedEmbeddingDim(const Tensor& embeddings);

double EuclideanDistanceRow(const Tensor& left, const Tensor& right, size_t row,
                            size_t dim);

double EuclideanDistanceRows(const Tensor& embeddings, size_t left_row,
                             size_t right_row, size_t dim);

bool PairLabelMeansSimilar(MetricLearningLabelConvention convention,
                           double label);

int64_t ClassIdAt(const Tensor& class_ids, size_t row);

PairMetricResult
ComputePairDistanceMetrics(const Tensor& embedding_a, const Tensor& embedding_b,
                           const Tensor& pair_label,
                           MetricLearningLabelConvention convention,
                           double threshold);

RetrievalMetricResult ComputeRetrievalMetrics(const Tensor& embeddings,
                                              const Tensor& class_ids,
                                              size_t k);

} // namespace cyxwiz
