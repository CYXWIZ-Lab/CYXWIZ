#include "core/metric_learning_metrics.h"
#include "cyxwiz/loss.h"

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class ArrayFireCpuScope {
public:
    ArrayFireCpuScope() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        try {
            original_backend_ = af::getActiveBackend();
            has_original_backend_ = true;
            af::setBackend(AF_BACKEND_CPU);
        } catch (const af::exception& e) {
            std::cout << "NOTE: could not force ArrayFire CPU backend: "
                      << e.what() << "\n";
        }
#endif
    }

    ~ArrayFireCpuScope() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (has_original_backend_) {
            try {
                af::setBackend(original_backend_);
            } catch (const af::exception& e) {
                std::cout << "NOTE: could not restore ArrayFire backend: "
                          << e.what() << "\n";
            }
        }
#endif
    }

private:
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::Backend original_backend_ = AF_BACKEND_DEFAULT;
    bool has_original_backend_ = false;
#endif
};

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected << "\n";
        std::exit(1);
    }
}

cyxwiz::Tensor FloatTensor(const std::vector<size_t>& shape,
                           const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

cyxwiz::Tensor Int64Tensor(const std::vector<size_t>& shape,
                           const std::vector<int64_t>& values) {
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Int64);
}

void CheckFiniteTensor(const cyxwiz::Tensor& tensor,
                       const std::string& message) {
    Check(tensor.GetDataType() == cyxwiz::DataType::Float32,
          message + " should produce Float32");
    const float* data = tensor.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        Check(std::isfinite(data[i]), message + " should be finite");
    }
}

void RunLossCase(cyxwiz::Loss& loss,
                 const cyxwiz::Tensor& predictions,
                 const cyxwiz::Tensor& targets,
                 const std::string& name) {
    cyxwiz::Tensor value = loss.Forward(predictions, targets);
    Check(value.NumElements() >= 1, name + " forward should produce a value");
    CheckFiniteTensor(value, name + " forward");

    cyxwiz::Tensor grad = loss.Backward(predictions, targets);
    Check(grad.Shape() == predictions.Shape(),
          name + " backward should preserve prediction shape");
    CheckFiniteTensor(grad, name + " backward");
}

void TestCpuLossCoverage() {
    const auto vector_predictions =
        FloatTensor({4}, {0.8f, 0.2f, 0.4f, 0.7f});
    const auto vector_targets =
        FloatTensor({4}, {1.0f, 0.0f, 0.0f, 1.0f});

    cyxwiz::MSELoss mse;
    RunLossCase(mse, vector_predictions, vector_targets, "MSELoss");

    cyxwiz::L1Loss l1;
    RunLossCase(l1, vector_predictions, vector_targets, "L1Loss");

    cyxwiz::SmoothL1Loss smooth_l1(1.0f);
    RunLossCase(smooth_l1, vector_predictions, vector_targets,
                "SmoothL1Loss");

    cyxwiz::BCELoss bce;
    RunLossCase(bce, vector_predictions, vector_targets, "BCELoss");

    const auto logits = FloatTensor({4}, {1.2f, -0.8f, -0.2f, 1.0f});
    cyxwiz::BCEWithLogitsLoss bce_logits(cyxwiz::Reduction::Mean, 2.0f);
    RunLossCase(bce_logits, logits, vector_targets, "BCEWithLogitsLoss");

    const auto log_probs =
        FloatTensor({2, 2}, {-0.1053605f, -2.3025851f,
                             -1.2039728f, -0.3566749f});
    const auto probability_targets =
        FloatTensor({2, 2}, {0.9f, 0.1f, 0.3f, 0.7f});
    cyxwiz::KLDivLoss kl_div;
    RunLossCase(kl_div, log_probs, probability_targets, "KLDivLoss");

    const auto class_logits =
        FloatTensor({2, 3}, {2.0f, 0.0f, -1.0f,
                             -1.0f, 0.5f, 1.5f});
    const auto class_targets = Int64Tensor({2}, {0, 2});

    cyxwiz::CrossEntropyLoss ce(cyxwiz::Reduction::Mean, -100, {}, 0.1f);
    RunLossCase(ce, class_logits, class_targets,
                "CrossEntropyLoss label_smoothing");

    const auto nll_log_probs =
        FloatTensor({2, 3}, {-0.1f, -2.0f, -3.0f,
                             -2.5f, -1.5f, -0.2f});
    cyxwiz::NLLLoss nll;
    RunLossCase(nll, nll_log_probs, class_targets, "NLLLoss");

    cyxwiz::FocalLoss focal;
    RunLossCase(focal, class_logits, class_targets, "FocalLoss");

    const auto mask_predictions =
        FloatTensor({2, 2}, {0.8f, 0.2f, 0.4f, 0.9f});
    const auto mask_targets =
        FloatTensor({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});

    cyxwiz::SoftDiceLoss soft_dice(cyxwiz::Reduction::Mean, 1.0f);
    RunLossCase(soft_dice, mask_predictions, mask_targets, "SoftDiceLoss");

    cyxwiz::TverskyLoss tversky(cyxwiz::Reduction::Mean, 0.3f, 0.7f, 1.0f);
    RunLossCase(tversky, mask_predictions, mask_targets, "TverskyLoss");

    cyxwiz::JaccardLoss jaccard(cyxwiz::Reduction::Mean, 1.0f);
    RunLossCase(jaccard, mask_predictions, mask_targets, "JaccardLoss");

    cyxwiz::CosineEmbeddingLoss cosine(0.0f);
    cosine.SetLabels(FloatTensor({1}, {1.0f}));
    RunLossCase(cosine,
                FloatTensor({1, 2}, {1.0f, 0.0f}),
                FloatTensor({1, 2}, {0.0f, 1.0f}),
                "CosineEmbeddingLoss");

    cyxwiz::ContrastiveLoss contrastive(2.0f);
    contrastive.SetLabels(FloatTensor({1}, {0.0f}));
    RunLossCase(contrastive,
                FloatTensor({1, 2}, {1.0f, 0.0f}),
                FloatTensor({1, 2}, {0.0f, 0.0f}),
                "ContrastiveLoss");

    cyxwiz::TripletLoss triplet(1.0f);
    triplet.SetNegative(FloatTensor({1, 2}, {1.0f, 0.0f}));
    RunLossCase(triplet,
                FloatTensor({1, 2}, {0.0f, 0.0f}),
                FloatTensor({1, 2}, {0.5f, 0.0f}),
                "TripletLoss");
}

void TestCpuMetricCoverage() {
    const auto left = FloatTensor({2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});
    const auto right = FloatTensor({2, 2}, {0.1f, 0.0f, 2.0f, 0.0f});
    const auto labels = FloatTensor({2}, {0.0f, 1.0f});

    const auto pair_metrics = cyxwiz::ComputePairDistanceMetrics(
        left,
        right,
        labels,
        cyxwiz::MetricLearningLabelConvention::
            ContrastiveZeroSimilarOneDissimilar,
        0.5);
    Check(pair_metrics.pair_count == 2,
          "pair metric coverage should count pairs");
    CheckNear(static_cast<float>(pair_metrics.accuracy), 1.0f, 1e-6f,
              "pair metric coverage should compute accuracy");

    const auto embeddings =
        FloatTensor({3, 2}, {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f});
    const auto class_ids = Int64Tensor({3}, {1, 1, 2});
    const auto retrieval =
        cyxwiz::ComputeRetrievalMetrics(embeddings, class_ids, 1);
    Check(retrieval.query_count == 3,
          "retrieval metric coverage should count queries");
    Check(retrieval.k == 1, "retrieval metric coverage should preserve k");
}

bool ActiveArrayFireGpuBackend() {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::Backend backend = af::getActiveBackend();
        return backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL;
    } catch (const af::exception& e) {
        std::cout << "SKIP: ArrayFire backend unavailable: "
                  << e.what() << "\n";
        return false;
    }
#else
    std::cout << "SKIP: ArrayFire not compiled for this build\n";
    return false;
#endif
}

void TestArrayFireGpuLossCoverageOrSkip() {
    if (!ActiveArrayFireGpuBackend()) {
        std::cout << "SKIP: GPU loss coverage requires active CUDA/OpenCL "
                     "ArrayFire backend\n";
        return;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const float prediction_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float target_values[] = {1.0f, 1.0f, 2.0f, 2.0f};
    const af::array prediction_array(2, 2, prediction_values);
    const af::array target_array(2, 2, target_values);

    const cyxwiz::Tensor predictions(prediction_array);
    const cyxwiz::Tensor targets(target_array);
    cyxwiz::MSELoss mse;
    const cyxwiz::Tensor value = mse.Forward(predictions, targets);
    CheckFiniteTensor(value, "ArrayFire MSELoss forward");
    const cyxwiz::Tensor grad = mse.Backward(predictions, targets);
    Check(grad.Shape() == std::vector<size_t>({2, 2}),
          "ArrayFire MSELoss backward should preserve shape");
    CheckFiniteTensor(grad, "ArrayFire MSELoss backward");
#endif
}

void PrintCoverageMatrix() {
    std::cout
        << "ML primitive coverage matrix:\n"
        << "- CPU losses: MSE, L1, SmoothL1/Huber, BCE, BCEWithLogits, "
           "KLDiv, CrossEntropy(label_smoothing), NLL, Focal, SoftDice, "
           "Tversky, Jaccard, CosineEmbedding, Contrastive, Triplet\n"
        << "- CPU metrics: metric-learning pair distance, retrieval, "
           "pipeline ClassificationMetricsNode in operator-routing tests\n"
        << "- GPU losses: ArrayFire MSE smoke runs only on active CUDA/OpenCL; "
           "otherwise skipped explicitly\n"
        << "- GPU metrics: metric-learning pair/retrieval metrics are ArrayFire-first; "
           "Arrow/pipeline metrics remain native CPU by design\n";
}

}  // namespace

int main() {
    {
        ArrayFireCpuScope cpu_scope;
        TestCpuLossCoverage();
        TestCpuMetricCoverage();
    }
    TestArrayFireGpuLossCoverageOrSkip();
    PrintCoverageMatrix();
    std::cout << "ML primitive coverage matrix passed\n";
    return 0;
}
