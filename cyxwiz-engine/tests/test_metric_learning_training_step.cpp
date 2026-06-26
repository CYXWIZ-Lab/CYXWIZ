#include "core/metric_learning_dataset_builder.h"
#include "core/metric_learning_training_step.h"

#include <cyxwiz/tensor.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected << "\n";
        std::exit(1);
    }
}

class NoopOptimizer final : public cyxwiz::Optimizer {
public:
    void Step(std::map<std::string, cyxwiz::Tensor>&,
              const std::map<std::string, cyxwiz::Tensor>&) override {
        ++step_calls;
    }

    void ZeroGrad() override {}

    int step_calls = 0;
};

class CountingExecutableModel final : public cyxwiz::IExecutableModel {
public:
    explicit CountingExecutableModel(float scale) : scale_(scale) {}

    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        ++forward_calls;
        std::vector<float> output(input.NumElements(), 0.0f);
        const float* data = input.Data<float>();
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = data[i] * scale_;
        }
        return cyxwiz::Tensor(input.Shape(), output.data());
    }

    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad_output) override {
        ++backward_calls;
        std::vector<float> input_grad(grad_output.NumElements(), 0.0f);
        const float* data = grad_output.Data<float>();
        for (size_t i = 0; i < input_grad.size(); ++i) {
            accumulated_gradient += data[i];
            input_grad[i] = data[i] * scale_;
        }
        return cyxwiz::Tensor(grad_output.Shape(), input_grad.data());
    }

    void SetTraining(bool training) override { training_mode = training; }

    std::map<std::string, cyxwiz::Tensor> GetParameters() override {
        return {{"scale", cyxwiz::Tensor({1}, &scale_)}};
    }

    void SetParameters(
        const std::map<std::string, cyxwiz::Tensor>& params) override {
        const auto it = params.find("scale");
        if (it != params.end()) {
            scale_ = it->second.Data<float>()[0];
        }
    }

    std::map<std::string, cyxwiz::Tensor> GetGradients() override {
        return {{"scale", cyxwiz::Tensor({1}, &accumulated_gradient)}};
    }

    void UpdateParameters(cyxwiz::Optimizer* optimizer) override {
        ++update_calls;
        if (optimizer != nullptr) {
            auto params = GetParameters();
            const auto grads = GetGradients();
            optimizer->Step(params, grads);
        }
    }

    int forward_calls = 0;
    int backward_calls = 0;
    int update_calls = 0;
    bool training_mode = true;
    float accumulated_gradient = 0.0f;

private:
    float scale_;
};

cyxwiz::PairBatch MakePairBatch() {
    std::vector<cyxwiz::PairDatasetRow> rows = {
        {{1.0f, 0.0f}, {0.0f, 0.0f}, 0.0f},
        {{0.0f, 1.0f}, {2.0f, 1.0f}, 1.0f},
    };
    cyxwiz::PairDatasetBuilderConfig config;
    config.batcher.batch_size = 2;
    auto built = cyxwiz::BuildPairDataset(rows, config);
    return built.CreateBatcher().GetNextPairBatch();
}

cyxwiz::TripletBatch MakeTripletBatch() {
    std::vector<cyxwiz::TripletDatasetRow> rows = {
        {{0.0f, 0.0f}, {0.5f, 0.0f}, {1.0f, 0.0f}},
    };
    cyxwiz::TripletDatasetBuilderConfig config;
    config.batcher.batch_size = 1;
    auto built = cyxwiz::BuildTripletDataset(rows, config);
    return built.CreateBatcher().GetNextTripletBatch();
}

void TestPairMetricTrainingStep() {
    auto owned = std::make_unique<CountingExecutableModel>(1.0f);
    auto* raw = owned.get();
    cyxwiz::SharedEncoderRuntime runtime(std::move(owned));
    NoopOptimizer optimizer;

    cyxwiz::PairMetricTrainingStepConfig config;
    config.loss_kind = cyxwiz::MetricLearningPairLossKind::Contrastive;
    config.margin = 2.0f;
    config.update_parameters = true;

    const auto result = cyxwiz::RunPairMetricTrainingStep(
        runtime, MakePairBatch(), config, &optimizer);

    Check(raw->forward_calls == 2,
          "pair training step should run both branches through encoder");
    Check(raw->backward_calls == 2,
          "pair training step should backpropagate both branch gradients");
    Check(raw->update_calls == 1 && optimizer.step_calls == 1,
          "pair training step should perform one optional update");
    Check(result.loss.Shape() == std::vector<size_t>({1}),
          "pair training step should return scalar loss tensor");
    Check(result.embeddings.embedding_a.Shape() == std::vector<size_t>({2, 2}),
          "pair training step should return branch embeddings");
    Check(result.input_gradients.input_a.Shape() ==
              std::vector<size_t>({2, 2}),
          "pair training step should return input gradients");
}

void TestTripletMetricTrainingStep() {
    auto owned = std::make_unique<CountingExecutableModel>(1.0f);
    auto* raw = owned.get();
    cyxwiz::SharedEncoderRuntime runtime(std::move(owned));

    cyxwiz::TripletMetricTrainingStepConfig config;
    config.margin = 1.0f;

    const auto result = cyxwiz::RunTripletMetricTrainingStep(
        runtime, MakeTripletBatch(), config);

    Check(raw->forward_calls == 3,
          "triplet training step should run all branches through encoder");
    Check(raw->backward_calls == 3,
          "triplet training step should backpropagate all branch gradients");
    Check(raw->update_calls == 0,
          "triplet training step should not update unless requested");
    CheckNear(result.loss.Data<float>()[0], 0.5f, 1e-5f,
              "triplet training step should return metric loss");
    Check(result.input_gradients.negative.Shape() ==
              std::vector<size_t>({1, 2}),
          "triplet training step should return negative branch input gradient");
}

void TestTrainingStepValidation() {
    cyxwiz::SharedEncoderRuntime runtime(
        std::make_unique<CountingExecutableModel>(1.0f));
    cyxwiz::PairMetricTrainingStepConfig config;
    config.update_parameters = true;

    bool rejected_missing_optimizer = false;
    try {
        (void)cyxwiz::RunPairMetricTrainingStep(
            runtime, MakePairBatch(), config);
    } catch (const std::invalid_argument&) {
        rejected_missing_optimizer = true;
    }
    Check(rejected_missing_optimizer,
          "training step should reject update without optimizer");

    bool rejected_invalid_batch = false;
    try {
        config.update_parameters = false;
        cyxwiz::PairBatch invalid;
        (void)cyxwiz::RunPairMetricTrainingStep(runtime, invalid, config);
    } catch (const std::invalid_argument&) {
        rejected_invalid_batch = true;
    }
    Check(rejected_invalid_batch,
          "training step should reject invalid pair batches");
}

void TestPairTrainingStepWithDeterministicSequentialEncoder() {
    auto model = std::make_unique<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::LinearModule>(2, 2, false);
    cyxwiz::SharedEncoderRuntime runtime(
        std::make_unique<cyxwiz::SequentialExecutableModel>(std::move(model)));
    NoopOptimizer optimizer;

    cyxwiz::PairMetricTrainingStepConfig config;
    config.loss_kind = cyxwiz::MetricLearningPairLossKind::Contrastive;
    config.margin = 2.0f;
    config.update_parameters = true;

    const auto result = cyxwiz::RunPairMetricTrainingStep(
        runtime, MakePairBatch(), config, &optimizer);

    Check(optimizer.step_calls == 1,
          "sequential pair training step should update once with accumulated gradients");
    Check(result.input_gradients.input_a.Shape() ==
              std::vector<size_t>({2, 2}) &&
          result.input_gradients.input_b.Shape() ==
              std::vector<size_t>({2, 2}),
          "sequential pair training step should return both branch input gradients");
}

}  // namespace

int main() {
    TestPairMetricTrainingStep();
    TestTripletMetricTrainingStep();
    TestTrainingStepValidation();
    TestPairTrainingStepWithDeterministicSequentialEncoder();
    std::cout << "Metric-learning training step contracts passed\n";
    return 0;
}
