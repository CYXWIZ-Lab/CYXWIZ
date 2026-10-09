#include "core/metric_learning_batch.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor FloatTensor(const std::vector<size_t>& shape,
                           const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data());
}

void TestMetricLearningLabelConventions() {
    using cyxwiz::MetricLearningLabelConvention;

    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              0.0),
          "contrastive should accept 0 = similar");
    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              1.0),
          "contrastive should accept 1 = dissimilar");
    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  ContrastiveZeroSimilarOneDissimilar,
              -1.0),
          "contrastive should reject cosine label convention");

    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              1.0),
          "cosine embedding should accept 1 = similar");
    Check(cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              -1.0),
          "cosine embedding should accept -1 = dissimilar");
    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::
                  CosineOneSimilarNegativeOneDissimilar,
              0.0),
          "cosine embedding should reject contrastive label convention");

    Check(!cyxwiz::IsValidMetricLearningLabel(
              MetricLearningLabelConvention::TripletNoLabels,
              1.0),
          "triplet convention should not validate scalar pair labels");
}

void TestBatchVectorShapes() {
    const std::vector<float> values = {1.0f, 2.0f, 3.0f};
    Check(cyxwiz::TensorIsBatchVector(FloatTensor({3}, values), 3),
          "[batch] should be a batch vector");
    Check(cyxwiz::TensorIsBatchVector(FloatTensor({3, 1}, values), 3),
          "[batch, 1] should be a batch vector");
    Check(!cyxwiz::TensorIsBatchVector(FloatTensor({1, 3}, values), 3),
          "[1, batch] should not be a batch vector");
    Check(!cyxwiz::TensorIsBatchVector(FloatTensor({3}, values), 2),
          "a batch vector must match the batch size");
    Check(cyxwiz::TensorIsEmpty(cyxwiz::Tensor()),
          "a default tensor should be empty");
    Check(!cyxwiz::TensorIsEmpty(FloatTensor({3}, values)),
          "a filled tensor should not be empty");
}

}  // namespace

int main() {
    TestMetricLearningLabelConventions();
    TestBatchVectorShapes();
    std::cout << "Metric-learning batch contracts passed\n";
    return 0;
}
