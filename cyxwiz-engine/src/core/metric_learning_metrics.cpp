#include "metric_learning_metrics.h"

#include "algorithms/arrayfire_host_materialization.h"
#include "algorithms/arrayfire_host_values.h"

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

size_t FlattenedEmbeddingDim(const Tensor& embeddings) {
    const auto& shape = embeddings.Shape();
    if (shape.size() < 2 || shape[0] == 0) {
        throw std::invalid_argument(
            "metric-learning embeddings must be [batch, ...features]");
    }
    const size_t dim = embeddings.NumElements() / shape[0];
    if (dim == 0) {
        throw std::invalid_argument(
            "metric-learning embeddings must have non-empty features");
    }
    return dim;
}

int64_t ClassIdAt(const Tensor& class_ids, size_t row) {
    switch (class_ids.GetDataType()) {
    case DataType::Int64:
        return class_ids.ReadData<int64_t>()[row];
    case DataType::Int32:
        return static_cast<int64_t>(class_ids.ReadData<int32_t>()[row]);
    case DataType::Float32: {
        const float value = class_ids.ReadData<float>()[row];
        constexpr float kLargestExactFloatInteger = 16777216.0f;
        if (!std::isfinite(value) || std::trunc(value) != value ||
            std::fabs(value) > kLargestExactFloatInteger) {
            throw std::invalid_argument(
                "Float32 retrieval class IDs must be exact finite integers");
        }
        return static_cast<int64_t>(value);
    }
    default:
        throw std::invalid_argument(
            "retrieval class IDs must be Int64, Int32, or Float32");
    }
}

namespace {

constexpr const char* kPairMetricOperation = "MetricLearning::PairDistance";
constexpr const char* kRetrievalMetricOperation = "MetricLearning::Retrieval";

struct EmbeddingShape {
    size_t batch = 0;
    size_t dim = 0;
};

EmbeddingShape ValidatePairInputs(const Tensor& embedding_a,
                                  const Tensor& embedding_b,
                                  const Tensor& pair_label,
                                  MetricLearningLabelConvention convention,
                                  double threshold) {
    if (embedding_a.GetDataType() != DataType::Float32 ||
        embedding_b.GetDataType() != DataType::Float32) {
        throw std::invalid_argument("pair metric embeddings must be Float32");
    }
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "pair metric embeddings must have identical shapes");
    }
    const auto& shape = embedding_a.Shape();
    const size_t batch = shape.empty() ? 0 : shape[0];
    if (batch == 0) {
        throw std::invalid_argument("pair metric embeddings cannot be empty");
    }
    if (pair_label.GetDataType() != DataType::Float32 ||
        !TensorIsBatchVector(pair_label, batch)) {
        throw std::invalid_argument(
            "pair metric labels must be Float32 [batch] or [batch, 1]");
    }
    if (convention == MetricLearningLabelConvention::TripletNoLabels) {
        throw std::invalid_argument(
            "triplet label convention cannot score pair metrics");
    }
    if (!std::isfinite(threshold)) {
        throw std::invalid_argument("pair metric threshold must be finite");
    }
    return {batch, FlattenedEmbeddingDim(embedding_a)};
}

EmbeddingShape ValidateRetrievalInputs(const Tensor& embeddings,
                                       const Tensor& class_ids, size_t k) {
    if (embeddings.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(
            "retrieval metric embeddings must be Float32");
    }
    const auto& shape = embeddings.Shape();
    const size_t batch = shape.empty() ? 0 : shape[0];
    if (batch < 2) {
        throw std::invalid_argument(
            "retrieval metrics require at least two embeddings");
    }
    if (k == 0) {
        throw std::invalid_argument("retrieval k must be non-zero");
    }
    if (batch > std::numeric_limits<unsigned>::max()) {
        throw std::invalid_argument(
            "retrieval metrics take at most 2^32 - 1 embeddings");
    }
    if (!TensorIsBatchVector(class_ids, batch)) {
        throw std::invalid_argument(
            "retrieval class IDs must be [batch] or [batch, 1]");
    }
    if (class_ids.GetDataType() != DataType::Int64 &&
        class_ids.GetDataType() != DataType::Int32 &&
        class_ids.GetDataType() != DataType::Float32) {
        throw std::invalid_argument(
            "retrieval class IDs must be Int64, Int32, or Float32");
    }
    return {batch, FlattenedEmbeddingDim(embeddings)};
}

// [batch, dim] on the device for embeddings of any rank (Reshape stays on the
// device).
af::array EmbeddingMatrix(const Tensor& embeddings, EmbeddingShape shape) {
    return embeddings.Reshape({shape.batch, shape.dim}).GetSemanticArray().as(f32);
}

template <typename Fn>
auto OnDevice(const char* operation, Fn&& fn) {
    try {
        return fn();
    } catch (const af::exception& error) {
        throw std::runtime_error(DeviceError(operation, error));
    }
}

void ValidatePairLabels(const af::array& labels,
                        MetricLearningLabelConvention convention) {
    const af::array valid = convention ==
                                    MetricLearningLabelConvention::
                                        ContrastiveZeroSimilarOneDissimilar
                                ? (labels == 0.0f) + (labels == 1.0f)
                                : (labels == 1.0f) + (labels == -1.0f);
    const af::array invalid_count = af::sum((valid == 0).as(u32));
    uint32_t invalid = 0;
    MaterializeArrayFireToHost(
        invalid_count, &invalid,
        ArrayFireHostSyncCategory::MetricInputValidation,
        "MetricLearning::PairDistance::LabelValidation", "arrayfire_scalar",
        "bounded_validation_readback", "valid label convention count");
    if (invalid != 0) {
        throw std::invalid_argument(
            "metric-learning pair label does not match convention");
    }
}

void ValidateRetrievalClassIds(const Tensor& class_ids) {
    if (class_ids.GetDataType() != DataType::Float32) {
        return;
    }
    constexpr float kLargestExactFloatInteger = 16777216.0f;
    const af::array values = af::flat(class_ids.GetSemanticArray()).as(f32);
    const af::array invalid = af::isNaN(values) || af::isInf(values) ||
                              values != af::trunc(values) ||
                              af::abs(values) > kLargestExactFloatInteger;
    const af::array invalid_count = af::sum(invalid.as(u32));
    uint32_t count = 0;
    MaterializeArrayFireToHost(
        invalid_count, &count, ArrayFireHostSyncCategory::MetricInputValidation,
        "MetricLearning::Retrieval::ClassIdValidation", "arrayfire_scalar",
        "bounded_validation_readback", "exact finite class ID count");
    if (count != 0) {
        throw std::invalid_argument(
            "Float32 retrieval class IDs must be exact finite integers");
    }
}

PairMetricResult ComputePairArrayFire(const Tensor& embedding_a,
                                      const Tensor& embedding_b,
                                      const Tensor& pair_label,
                                      MetricLearningLabelConvention convention,
                                      double threshold, EmbeddingShape shape) {
    const af::array left = EmbeddingMatrix(embedding_a, shape);
    const af::array right = EmbeddingMatrix(embedding_b, shape);
    const af::array labels = af::flat(pair_label.GetSemanticArray()).as(f32);
    ValidatePairLabels(labels, convention);

    const af::array difference = left - right;
    const af::array distances = af::sqrt(af::sum(difference * difference, 1));
    const af::array similar = convention ==
                                      MetricLearningLabelConvention::
                                          ContrastiveZeroSimilarOneDissimilar
                                  ? labels == 0.0f
                                  : labels == 1.0f;
    const af::array similar_f32 = similar.as(f32);
    const af::array dissimilar_f32 = 1.0f - similar_f32;
    const af::array predicted_similar =
        distances <= static_cast<float>(threshold);
    af::array aggregates = af::join(
        0, af::sum((predicted_similar == similar).as(f32)),
        af::sum(distances * similar_f32), af::sum(distances * dissimilar_f32));
    aggregates =
        af::join(0, aggregates, af::sum(similar_f32), af::sum(dissimilar_f32));
    aggregates.eval();

    float values[5] = {};
    MaterializeArrayFireToHost(
        aggregates, values, ArrayFireHostSyncCategory::MetricScalarReadback,
        kPairMetricOperation, "arrayfire_aggregate_vector",
        "bounded_metric_readback",
        "correct,positive_sum,negative_sum,positive_count,negative_count");

    PairMetricResult result;
    result.pair_count = shape.batch;
    result.threshold = threshold;
    result.positive_count = static_cast<size_t>(std::llround(values[3]));
    result.negative_count = static_cast<size_t>(std::llround(values[4]));
    result.accuracy =
        static_cast<double>(values[0]) / static_cast<double>(shape.batch);
    result.positive_distance_mean =
        result.positive_count == 0
            ? 0.0
            : static_cast<double>(values[1]) /
                  static_cast<double>(result.positive_count);
    result.negative_distance_mean =
        result.negative_count == 0
            ? 0.0
            : static_cast<double>(values[2]) /
                  static_cast<double>(result.negative_count);
    return result;
}

RetrievalMetricResult ComputeRetrievalArrayFire(const Tensor& embeddings,
                                                const Tensor& class_ids,
                                                size_t k,
                                                EmbeddingShape shape) {
    ValidateRetrievalClassIds(class_ids);
    const unsigned count = static_cast<unsigned>(shape.batch);
    const af::array matrix = EmbeddingMatrix(embeddings, shape);
    const af::array squared_norms = af::sum(matrix * matrix, 1);
    af::array squared_distances =
        af::tile(squared_norms, 1U, count) +
        af::tile(af::transpose(squared_norms), count, 1U) -
        2.0f * af::matmul(matrix, matrix, AF_MAT_NONE, AF_MAT_TRANS);
    squared_distances = af::max(squared_distances, 0.0f);

    const dim_t dimension = static_cast<dim_t>(shape.batch);
    const af::array identity = af::identity(dimension, dimension, f32);
    const af::array infinity = af::constant(
        std::numeric_limits<float>::infinity(), dimension, dimension, f32);
    squared_distances =
        af::select(identity > 0.0f, infinity, squared_distances);

    const af::array classes = af::flat(class_ids.GetSemanticArray()).as(s64);
    const af::array query_classes =
        af::tile(af::moddims(classes, dimension, 1), 1U, count);
    const af::array candidate_classes =
        af::tile(af::moddims(classes, 1, dimension), count, 1U);
    const af::array relevant = ((query_classes == candidate_classes).as(u8) *
                                (identity == 0.0f).as(u8)) > 0;
    const af::array masked_distances =
        af::select(relevant, squared_distances, infinity);

    const af::array nearest_relevant_distance = af::min(masked_distances, 1);
    const af::array candidate_indices =
        af::iota(af::dim4(1, dimension), af::dim4(1), u32);
    const af::array candidate_index_grid =
        af::tile(candidate_indices, count, 1U);
    const af::array nearest_distance_grid =
        af::tile(nearest_relevant_distance, 1U, count);
    // The first relevant candidate in the native order (distance, then
    // index) is the lowest index among the relevant ones at the nearest
    // distance; af::min's index for tied values need not be that one.
    const af::array nearest_relevant_index = af::min(
        af::select(relevant && masked_distances == nearest_distance_grid,
                   candidate_index_grid,
                   af::constant(static_cast<unsigned>(dimension), dimension,
                                dimension, u32)),
        1);
    const af::array nearest_index_grid =
        af::tile(nearest_relevant_index, 1U, count);
    // Only other-class candidates can come before the first relevant one.
    const af::array precedes_relevant =
        (!relevant) &&
        ((squared_distances < nearest_distance_grid) ||
         ((squared_distances == nearest_distance_grid) &&
          (candidate_index_grid < nearest_index_grid)));
    const af::array first_relevant_rank =
        af::sum(precedes_relevant.as(f32), 1) + 1.0f;
    const af::array has_relevant = af::sum(relevant.as(u32), 1) > 0;
    const size_t effective_k = std::min(k, shape.batch - 1);
    const af::array recall_hit =
        (has_relevant.as(u8) *
         (first_relevant_rank <= static_cast<float>(effective_k)).as(u8)) > 0;
    const af::array nearest_match =
        (has_relevant.as(u8) * (first_relevant_rank == 1.0f).as(u8)) > 0;
    const af::array reciprocal_rank =
        af::select(has_relevant, 1.0f / first_relevant_rank,
                   af::constant(0.0f, first_relevant_rank.dims()));
    af::array aggregates =
        af::join(0, af::sum(recall_hit.as(f32)), af::sum(reciprocal_rank),
                 af::sum(nearest_match.as(f32)));
    aggregates.eval();

    float values[3] = {};
    MaterializeArrayFireToHost(
        aggregates, values, ArrayFireHostSyncCategory::MetricScalarReadback,
        kRetrievalMetricOperation, "arrayfire_aggregate_vector",
        "bounded_metric_readback",
        "recall_hits,reciprocal_rank_sum,nearest_match_sum");

    const double denominator = static_cast<double>(shape.batch);
    RetrievalMetricResult result;
    result.query_count = shape.batch;
    result.k = effective_k;
    result.recall_at_k = static_cast<double>(values[0]) / denominator;
    result.mean_reciprocal_rank = static_cast<double>(values[1]) / denominator;
    result.nearest_neighbor_class_agreement =
        static_cast<double>(values[2]) / denominator;
    return result;
}

} // namespace

PairMetricResult
ComputePairDistanceMetrics(const Tensor& embedding_a, const Tensor& embedding_b,
                           const Tensor& pair_label,
                           MetricLearningLabelConvention convention,
                           double threshold) {
    const EmbeddingShape shape = ValidatePairInputs(
        embedding_a, embedding_b, pair_label, convention, threshold);
    return OnDevice(kPairMetricOperation, [&] {
        return ComputePairArrayFire(embedding_a, embedding_b, pair_label,
                                    convention, threshold, shape);
    });
}

RetrievalMetricResult ComputeRetrievalMetrics(const Tensor& embeddings,
                                              const Tensor& class_ids,
                                              size_t k) {
    const EmbeddingShape shape =
        ValidateRetrievalInputs(embeddings, class_ids, k);
    return OnDevice(kRetrievalMetricOperation, [&] {
        return ComputeRetrievalArrayFire(embeddings, class_ids, k, shape);
    });
}

PairRowScores ComputePairRowScores(const Tensor& embedding_a,
                                   const Tensor& embedding_b,
                                   bool with_cosine) {
    constexpr const char* operation = "MetricLearning::PairScoreOutput";
    if (embedding_a.Shape() != embedding_b.Shape()) {
        throw std::invalid_argument(
            "PairScoreOutput embeddings must have identical shapes");
    }
    const auto& shape = embedding_a.Shape();
    const size_t rows = shape.empty() ? 0 : shape[0];
    if (rows == 0) {
        throw std::invalid_argument("PairScoreOutput embeddings cannot be empty");
    }
    const size_t dim = FlattenedEmbeddingDim(embedding_a);
    const std::vector<double> values = OnDevice(operation, [&] {
        const af::dtype type = DeviceHasDouble() ? f64 : f32;
        const af::array left =
            embedding_a.Reshape({rows, dim}).GetSemanticArray().as(type);
        const af::array right =
            embedding_b.Reshape({rows, dim}).GetSemanticArray().as(type);
        const af::array difference = left - right;
        af::array out = af::sqrt(af::sum(difference * difference, 1));
        if (with_cosine) {
            const af::array left_norm = af::sqrt(af::sum(left * left, 1));
            const af::array right_norm = af::sqrt(af::sum(right * right, 1));
            const af::array zero = (left_norm == 0.0) || (right_norm == 0.0);
            const af::array cosine =
                af::select(zero, af::constant(0.0, zero.dims(), type),
                           af::sum(left * right, 1) / (left_norm * right_norm));
            out = af::join(0, out, cosine, af::sum(zero.as(type)));
        }
        return DeviceToHostValues(out, operation);
    });
    PairRowScores scores;
    scores.distances.assign(values.begin(), values.begin() + rows);
    if (with_cosine) {
        if (values.back() != 0.0) {
            throw std::invalid_argument(
                "PairScoreOutput cosine similarity requires non-zero embeddings");
        }
        scores.cosines.assign(values.begin() + rows, values.begin() + 2 * rows);
    }
    return scores;
}

} // namespace cyxwiz
