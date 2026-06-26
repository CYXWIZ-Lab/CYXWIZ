#pragma once

#include <cyxwiz/tensor.h>

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

enum class MetricLearningLabelConvention {
    ContrastiveZeroSimilarOneDissimilar,
    CosineOneSimilarNegativeOneDissimilar,
    TripletNoLabels,
};

struct PairBatch;
struct TripletBatch;

inline bool ValidatePairBatchShape(const PairBatch& batch,
                                   std::string* error = nullptr);
inline bool ValidateTripletBatchShape(const TripletBatch& batch,
                                      std::string* error = nullptr);

inline bool MetricLearningConventionRequiresLabels(
    MetricLearningLabelConvention convention) {
    return convention != MetricLearningLabelConvention::TripletNoLabels;
}

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

struct PairBatch {
    Tensor input_a;       // [batch, ...] first item in each pair
    Tensor input_b;       // [batch, ...] second item in each pair
    Tensor pair_label;    // [batch] or [batch, 1]
    Tensor sample_id_a;   // optional [batch]
    Tensor sample_id_b;   // optional [batch]
    Tensor class_id_a;    // optional [batch]
    Tensor class_id_b;    // optional [batch]
    size_t size = 0;

    bool HasInputA() const { return HasTensor(input_a); }
    bool HasInputB() const { return HasTensor(input_b); }
    bool HasPairLabels() const { return HasTensor(pair_label); }
    bool HasSampleIds() const {
        return HasTensor(sample_id_a) && HasTensor(sample_id_b);
    }
    bool HasClassIds() const {
        return HasTensor(class_id_a) && HasTensor(class_id_b);
    }
    bool IsValid() const { return ValidatePairBatchShape(*this); }

private:
    static bool HasTensor(const Tensor& tensor) {
        return !tensor.Shape().empty() && tensor.NumElements() > 0;
    }
};

struct TripletBatch {
    Tensor anchor;             // [batch, ...]
    Tensor positive;           // [batch, ...]
    Tensor negative;           // [batch, ...]
    Tensor anchor_sample_id;   // optional [batch]
    Tensor positive_sample_id; // optional [batch]
    Tensor negative_sample_id; // optional [batch]
    Tensor anchor_class_id;    // optional [batch]
    Tensor positive_class_id;  // optional [batch]
    Tensor negative_class_id;  // optional [batch]
    size_t size = 0;

    bool HasAnchor() const { return HasTensor(anchor); }
    bool HasPositive() const { return HasTensor(positive); }
    bool HasNegative() const { return HasTensor(negative); }
    bool HasSampleIds() const {
        return HasTensor(anchor_sample_id) &&
               HasTensor(positive_sample_id) &&
               HasTensor(negative_sample_id);
    }
    bool HasClassIds() const {
        return HasTensor(anchor_class_id) &&
               HasTensor(positive_class_id) &&
               HasTensor(negative_class_id);
    }
    bool IsValid() const { return ValidateTripletBatchShape(*this); }

private:
    static bool HasTensor(const Tensor& tensor) {
        return !tensor.Shape().empty() && tensor.NumElements() > 0;
    }
};

inline bool TensorBatchDimensionMatches(const Tensor& tensor, size_t size) {
    return !tensor.Shape().empty() && tensor.Shape()[0] == size;
}

inline bool TensorIsBatchVector(const Tensor& tensor, size_t size) {
    const auto& shape = tensor.Shape();
    return (shape.size() == 1 && shape[0] == size) ||
           (shape.size() == 2 && shape[0] == size && shape[1] == 1);
}

inline bool TensorIsEmpty(const Tensor& tensor) {
    return tensor.Shape().empty() || tensor.NumElements() == 0;
}

inline bool OptionalTensorIsEmptyOrBatchVector(const Tensor& tensor,
                                               size_t size) {
    return TensorIsEmpty(tensor) || TensorIsBatchVector(tensor, size);
}

inline bool ValidatePairBatchShape(const PairBatch& batch,
                                   std::string* error) {
    if (batch.size == 0) {
        if (error) *error = "PairBatch size must be non-zero";
        return false;
    }
    if (!batch.HasInputA() || !batch.HasInputB()) {
        if (error) *error = "PairBatch requires input_a and input_b";
        return false;
    }
    if (batch.input_a.Shape() != batch.input_b.Shape()) {
        if (error) *error = "PairBatch input_a and input_b shapes must match";
        return false;
    }
    if (!TensorBatchDimensionMatches(batch.input_a, batch.size)) {
        if (error) *error = "PairBatch inputs must start with batch size";
        return false;
    }
    if (!batch.HasPairLabels() ||
        !TensorIsBatchVector(batch.pair_label, batch.size)) {
        if (error) *error = "PairBatch labels must be [batch] or [batch, 1]";
        return false;
    }
    if (!OptionalTensorIsEmptyOrBatchVector(batch.sample_id_a, batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.sample_id_b, batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.class_id_a, batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.class_id_b, batch.size)) {
        if (error) *error = "PairBatch optional IDs must be batch vectors";
        return false;
    }
    return true;
}

inline bool ValidateTripletBatchShape(const TripletBatch& batch,
                                      std::string* error) {
    if (batch.size == 0) {
        if (error) *error = "TripletBatch size must be non-zero";
        return false;
    }
    if (!batch.HasAnchor() || !batch.HasPositive() || !batch.HasNegative()) {
        if (error) *error =
            "TripletBatch requires anchor, positive, and negative";
        return false;
    }
    if (batch.anchor.Shape() != batch.positive.Shape() ||
        batch.anchor.Shape() != batch.negative.Shape()) {
        if (error) *error =
            "TripletBatch branch input shapes must match";
        return false;
    }
    if (!TensorBatchDimensionMatches(batch.anchor, batch.size)) {
        if (error) *error = "TripletBatch inputs must start with batch size";
        return false;
    }
    if (!OptionalTensorIsEmptyOrBatchVector(batch.anchor_sample_id,
                                           batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.positive_sample_id,
                                           batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.negative_sample_id,
                                           batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.anchor_class_id,
                                           batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.positive_class_id,
                                           batch.size) ||
        !OptionalTensorIsEmptyOrBatchVector(batch.negative_class_id,
                                           batch.size)) {
        if (error) *error = "TripletBatch optional IDs must be batch vectors";
        return false;
    }
    return true;
}

class IPairBatcher {
public:
    virtual ~IPairBatcher() = default;

    virtual PairBatch GetNextPairBatch() = 0;
    virtual void Reset() = 0;
    virtual bool IsEpochComplete() const = 0;
    virtual size_t GetNumBatches() const = 0;
    virtual size_t GetNumSamples() const = 0;
};

class ITripletBatcher {
public:
    virtual ~ITripletBatcher() = default;

    virtual TripletBatch GetNextTripletBatch() = 0;
    virtual void Reset() = 0;
    virtual bool IsEpochComplete() const = 0;
    virtual size_t GetNumBatches() const = 0;
    virtual size_t GetNumSamples() const = 0;
};

}  // namespace cyxwiz
