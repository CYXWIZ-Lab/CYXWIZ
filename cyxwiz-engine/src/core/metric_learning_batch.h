#pragma once

#include <cyxwiz/tensor.h>

#include <cstddef>

namespace cyxwiz {

enum class MetricLearningLabelConvention {
    ContrastiveZeroSimilarOneDissimilar,
    CosineOneSimilarNegativeOneDissimilar,
    TripletNoLabels,
};

inline bool IsValidMetricLearningLabel(
    MetricLearningLabelConvention convention,
    double label) {
    switch (convention) {
        case MetricLearningLabelConvention::
            ContrastiveZeroSimilarOneDissimilar:
            return label == 0.0 || label == 1.0;
        case MetricLearningLabelConvention::
            CosineOneSimilarNegativeOneDissimilar:
            return label == 1.0 || label == -1.0;
        case MetricLearningLabelConvention::TripletNoLabels:
            return false;
    }
    return false;
}

inline bool TensorIsBatchVector(const Tensor& tensor, size_t size) {
    const auto& shape = tensor.Shape();
    return (shape.size() == 1 && shape[0] == size) ||
           (shape.size() == 2 && shape[0] == size && shape[1] == 1);
}

inline bool TensorIsEmpty(const Tensor& tensor) {
    return tensor.Shape().empty() || tensor.NumElements() == 0;
}

}  // namespace cyxwiz
