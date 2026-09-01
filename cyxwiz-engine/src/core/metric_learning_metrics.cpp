#include "metric_learning_metrics.h"

#include "algorithms/arrayfire_backend_utils.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include "algorithms/arrayfire_host_materialization.h"

#include <arrayfire.h>
#endif

#include <spdlog/spdlog.h>

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

double EuclideanDistanceRow(const Tensor& left, const Tensor& right, size_t row,
                            size_t dim) {
    const float* left_data = left.ReadData<float>();
    const float* right_data = right.ReadData<float>();
    double sum = 0.0;
    const size_t offset = row * dim;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = static_cast<double>(left_data[offset + i]) -
                            static_cast<double>(right_data[offset + i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double EuclideanDistanceRows(const Tensor& embeddings, size_t left_row,
                             size_t right_row, size_t dim) {
    const float* data = embeddings.ReadData<float>();
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

bool PairLabelMeansSimilar(MetricLearningLabelConvention convention,
                           double label) {
    if (!IsValidMetricLearningLabel(convention, label)) {
        throw std::invalid_argument(
            "metric-learning pair label does not match convention");
    }
    switch (convention) {
    case MetricLearningLabelConvention::ContrastiveZeroSimilarOneDissimilar:
        return label == 0.0;
    case MetricLearningLabelConvention::CosineOneSimilarNegativeOneDissimilar:
        return label == 1.0;
    case MetricLearningLabelConvention::TripletNoLabels:
        throw std::invalid_argument(
            "triplet label convention cannot score pair metrics");
    }
    return false;
}

namespace {

PairMetricResult ComputePairDistanceMetricsNative(
    const Tensor& embedding_a, const Tensor& embedding_b,
    const Tensor& pair_label, MetricLearningLabelConvention convention,
    double threshold) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::MetricCpuPath,
        "MetricLearning::PairDistance");
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
    const float* labels = pair_label.ReadData<float>();

    PairMetricResult result;
    result.pair_count = pair_count;
    result.threshold = threshold;

    size_t correct = 0;
    double positive_sum = 0.0;
    double negative_sum = 0.0;

    for (size_t row = 0; row < pair_count; ++row) {
        const double distance =
            EuclideanDistanceRow(embedding_a, embedding_b, row, dim);
        const bool similar =
            PairLabelMeansSimilar(convention, static_cast<double>(labels[row]));
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

    result.accuracy =
        static_cast<double>(correct) / static_cast<double>(pair_count);
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

} // namespace

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

RetrievalMetricResult ComputeRetrievalMetricsNative(const Tensor& embeddings,
                                                    const Tensor& class_ids,
                                                    size_t k) {
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::MetricCpuPath, "MetricLearning::Retrieval");
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

std::string PairFallbackContext(const Tensor& embedding_a,
                                const Tensor& embedding_b,
                                const Tensor& pair_label) {
    return BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("embedding_a", embedding_a.Shape()) + "; " +
        BuildTensorShapeContext("embedding_b", embedding_b.Shape()) + "; " +
        BuildTensorShapeContext("pair_label", pair_label.Shape()));
}

std::string RetrievalFallbackContext(const Tensor& embeddings,
                                     const Tensor& class_ids) {
    return BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("embeddings", embeddings.Shape()) + "; " +
        BuildTensorShapeContext("class_ids", class_ids.Shape()));
}

void ReportNativeFallback(const char* operation_name,
                          BackendFallbackReason reason,
                          const char* error_message,
                          const std::string& context) {
    ThrowIfArrayFireNativeCpuFallbackForbidden(operation_name, reason,
                                               error_message, context);
    if (ShouldLogArrayFireBackendFallbackOnce(operation_name, reason,
                                              context)) {
        spdlog::warn("{}",
                     BuildArrayFireBackendFallbackMessage(
                         operation_name, reason,
                         reason != BackendFallbackReason::CudaJitParamOverflow,
                         error_message, context));
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool SupportsArrayFireShape(EmbeddingShape shape, size_t rank) {
    return rank <= 4 &&
           shape.batch <=
               static_cast<size_t>(std::numeric_limits<dim_t>::max()) &&
           shape.dim <= static_cast<size_t>(std::numeric_limits<dim_t>::max());
}

af::array EmbeddingMatrix(const Tensor& embeddings, EmbeddingShape shape) {
    return af::moddims(embeddings.GetSemanticArray().as(f32),
                       static_cast<dim_t>(shape.batch),
                       static_cast<dim_t>(shape.dim));
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

    af::array nearest_relevant_distance;
    af::array nearest_relevant_index;
    af::min(nearest_relevant_distance, nearest_relevant_index, masked_distances,
            1);
    const af::array candidate_indices =
        af::iota(af::dim4(1, dimension), af::dim4(1), u32);
    const af::array candidate_index_grid =
        af::tile(candidate_indices, count, 1U);
    const af::array nearest_index_grid =
        af::tile(nearest_relevant_index, 1U, count);
    const af::array nearest_distance_grid =
        af::tile(nearest_relevant_distance, 1U, count);
    const af::array precedes_relevant =
        ((squared_distances < nearest_distance_grid).as(u8) +
         ((squared_distances == nearest_distance_grid).as(u8) *
          (candidate_index_grid < nearest_index_grid).as(u8))) > 0;
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
#endif

} // namespace

PairMetricResult
ComputePairDistanceMetrics(const Tensor& embedding_a, const Tensor& embedding_b,
                           const Tensor& pair_label,
                           MetricLearningLabelConvention convention,
                           double threshold) {
    const EmbeddingShape shape = ValidatePairInputs(
        embedding_a, embedding_b, pair_label, convention, threshold);
    const std::string context =
        PairFallbackContext(embedding_a, embedding_b, pair_label);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (!SupportsArrayFireShape(shape, embedding_a.Shape().size())) {
        ReportNativeFallback(
            kPairMetricOperation, BackendFallbackReason::UnsupportedShape,
            "ArrayFire metric embeddings support ranks 2 through 4", context);
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   kPairMetricOperation)) {
        ReportNativeFallback(
            kPairMetricOperation, BackendFallbackReason::GpuBackendException,
            "forced ArrayFire backend fallback test hook", context);
    } else if (!IsCurrentArrayFireBackendAvailable()) {
        ReportNativeFallback(kPairMetricOperation,
                             BackendFallbackReason::BackendUnavailable,
                             "ArrayFire backend unavailable", context);
    } else {
        try {
            return ComputePairArrayFire(embedding_a, embedding_b, pair_label,
                                        convention, threshold, shape);
        } catch (const af::exception& error) {
            ReportNativeFallback(
                kPairMetricOperation,
                ClassifyArrayFireBackendFallbackReason(error.what()),
                error.what(), context);
        }
    }
#else
    ReportNativeFallback(kPairMetricOperation,
                         BackendFallbackReason::BackendUnavailable,
                         "ArrayFire support is unavailable", context);
#endif
    return ComputePairDistanceMetricsNative(embedding_a, embedding_b,
                                            pair_label, convention, threshold);
}

RetrievalMetricResult ComputeRetrievalMetrics(const Tensor& embeddings,
                                              const Tensor& class_ids,
                                              size_t k) {
    const EmbeddingShape shape =
        ValidateRetrievalInputs(embeddings, class_ids, k);
    const std::string context = RetrievalFallbackContext(embeddings, class_ids);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const bool supported =
        SupportsArrayFireShape(shape, embeddings.Shape().size()) &&
        shape.batch <= std::numeric_limits<unsigned>::max();
    if (!supported) {
        ReportNativeFallback(
            kRetrievalMetricOperation, BackendFallbackReason::UnsupportedShape,
            "ArrayFire retrieval embeddings support ranks 2 through 4",
            context);
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   kRetrievalMetricOperation)) {
        ReportNativeFallback(kRetrievalMetricOperation,
                             BackendFallbackReason::GpuBackendException,
                             "forced ArrayFire backend fallback test hook",
                             context);
    } else if (!IsCurrentArrayFireBackendAvailable()) {
        ReportNativeFallback(kRetrievalMetricOperation,
                             BackendFallbackReason::BackendUnavailable,
                             "ArrayFire backend unavailable", context);
    } else {
        try {
            return ComputeRetrievalArrayFire(embeddings, class_ids, k, shape);
        } catch (const af::exception& error) {
            ReportNativeFallback(
                kRetrievalMetricOperation,
                ClassifyArrayFireBackendFallbackReason(error.what()),
                error.what(), context);
        }
    }
#else
    ReportNativeFallback(kRetrievalMetricOperation,
                         BackendFallbackReason::BackendUnavailable,
                         "ArrayFire support is unavailable", context);
#endif
    return ComputeRetrievalMetricsNative(embeddings, class_ids, k);
}

} // namespace cyxwiz
