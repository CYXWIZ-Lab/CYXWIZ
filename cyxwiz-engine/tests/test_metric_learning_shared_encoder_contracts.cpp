#include "core/metric_learning_dataset_builder.h"
#include "core/metric_learning_shared_encoder.h"

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

cyxwiz::Tensor FloatTensor(const std::vector<size_t>& shape,
                           const std::vector<float>& values) {
    return cyxwiz::Tensor(shape, values.data());
}

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

    void SetTraining(bool training) override {
        training_mode = training;
    }

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

    void UpdateParameters(cyxwiz::Optimizer* /*optimizer*/) override {
        ++update_calls;
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
        {{1.0f, 2.0f}, {3.0f, 4.0f}, 0.0f},
        {{5.0f, 6.0f}, {7.0f, 8.0f}, 1.0f},
    };
    cyxwiz::PairDatasetBuilderConfig config;
    config.batcher.batch_size = 2;
    auto built = cyxwiz::BuildPairDataset(rows, config);
    return built.CreateBatcher().GetNextPairBatch();
}

cyxwiz::TripletBatch MakeTripletBatch() {
    std::vector<cyxwiz::TripletDatasetRow> rows = {
        {{1.0f, 0.0f}, {1.1f, 0.0f}, {0.0f, 1.0f}},
        {{0.0f, 1.0f}, {0.0f, 1.1f}, {1.0f, 0.0f}},
    };
    cyxwiz::TripletDatasetBuilderConfig config;
    config.batcher.batch_size = 2;
    auto built = cyxwiz::BuildTripletDataset(rows, config);
    return built.CreateBatcher().GetNextTripletBatch();
}

void TestSharedEncoderPairContract() {
    auto owned = std::make_unique<CountingExecutableModel>(2.0f);
    auto* raw = owned.get();
    cyxwiz::SharedEncoderRuntime runtime(std::move(owned));

    const auto batch = MakePairBatch();
    const auto embeddings = runtime.ForwardPair(batch);
    Check(raw->forward_calls == 2,
          "shared encoder pair forward should call one encoder twice");
    Check(embeddings.embedding_a.Shape() == std::vector<size_t>({2, 2}),
          "pair embedding_a shape should match branch output");
    CheckNear(embeddings.embedding_a.Data<float>()[0], 2.0f, 1e-5f,
              "pair encoder should scale first branch");
    CheckNear(embeddings.embedding_b.Data<float>()[3], 16.0f, 1e-5f,
              "pair encoder should scale second branch");

    const auto grads = runtime.BackwardPair(
        FloatTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}),
        FloatTensor({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f}));
    Check(raw->backward_calls == 2,
          "shared encoder pair backward should route both branch gradients");
    Check(grads.input_a.Shape() == std::vector<size_t>({2, 2}) &&
              grads.input_b.Shape() == std::vector<size_t>({2, 2}),
          "pair input gradients should preserve branch shapes");
    CheckNear(raw->accumulated_gradient, 12.0f, 1e-5f,
              "pair branch gradients should accumulate in one encoder");

    runtime.UpdateParameters(nullptr);
    Check(raw->update_calls == 1,
          "shared encoder pair step should update one encoder once");

    runtime.SetTraining(false);
    Check(!raw->training_mode,
          "shared encoder training mode should delegate to encoder");

    const auto params = runtime.GetParameters();
    Check(params.count("scale") == 1,
          "shared encoder should expose the encoder parameter set");
}

void TestSharedEncoderTripletContract() {
    auto owned = std::make_unique<CountingExecutableModel>(3.0f);
    auto* raw = owned.get();
    cyxwiz::SharedEncoderRuntime runtime(std::move(owned));

    const auto batch = MakeTripletBatch();
    const auto embeddings = runtime.ForwardTriplet(batch);
    Check(raw->forward_calls == 3,
          "shared encoder triplet forward should call one encoder three times");
    CheckNear(embeddings.anchor.Data<float>()[0], 3.0f, 1e-5f,
              "triplet encoder should scale anchor branch");
    CheckNear(embeddings.positive.Data<float>()[2], 0.0f, 1e-5f,
              "triplet encoder should scale positive branch");
    CheckNear(embeddings.negative.Data<float>()[1], 3.0f, 1e-5f,
              "triplet encoder should scale negative branch");

    const auto grads = runtime.BackwardTriplet(
        FloatTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}),
        FloatTensor({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f}),
        FloatTensor({2, 2}, {-1.0f, -1.0f, -1.0f, -1.0f}));
    Check(raw->backward_calls == 3,
          "shared encoder triplet backward should route all branch gradients");
    Check(grads.anchor.Shape() == std::vector<size_t>({2, 2}) &&
              grads.positive.Shape() == std::vector<size_t>({2, 2}) &&
              grads.negative.Shape() == std::vector<size_t>({2, 2}),
          "triplet input gradients should preserve branch shapes");
    CheckNear(raw->accumulated_gradient, 8.0f, 1e-5f,
              "triplet branch gradients should accumulate in one encoder");

    runtime.UpdateParameters(nullptr);
    Check(raw->update_calls == 1,
          "shared encoder triplet step should update one encoder once");
}

void TestSharedEncoderRejectsInvalidInput() {
    bool rejected_null_encoder = false;
    try {
        cyxwiz::SharedEncoderRuntime runtime(nullptr);
        (void)runtime;
    } catch (const std::invalid_argument&) {
        rejected_null_encoder = true;
    }
    Check(rejected_null_encoder,
          "shared encoder should reject null encoder ownership");

    auto owned = std::make_unique<CountingExecutableModel>(1.0f);
    cyxwiz::SharedEncoderRuntime runtime(std::move(owned));

    bool rejected_invalid_pair = false;
    try {
        cyxwiz::PairBatch invalid;
        (void)runtime.ForwardPair(invalid);
    } catch (const std::invalid_argument&) {
        rejected_invalid_pair = true;
    }
    Check(rejected_invalid_pair,
          "shared encoder should reject invalid PairBatch");
}

void TestSharedEncoderSequentialReplayAccumulatesGradients() {
    auto model = std::make_unique<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::LinearModule>(2, 2, false);
    auto executable =
        std::make_unique<cyxwiz::SequentialExecutableModel>(std::move(model));
    cyxwiz::SharedEncoderRuntime runtime(std::move(executable));

    const auto batch = MakePairBatch();
    const auto embeddings = runtime.ForwardPair(batch);
    Check(embeddings.embedding_a.Shape() == std::vector<size_t>({2, 2}),
          "sequential shared encoder should still support pair forward");

    const auto grads = runtime.BackwardPair(
        FloatTensor({2, 2}, {1.0f, 1.0f, 1.0f, 1.0f}),
        FloatTensor({2, 2}, {2.0f, 2.0f, 2.0f, 2.0f}));
    Check(grads.input_a.Shape() == std::vector<size_t>({2, 2}) &&
              grads.input_b.Shape() == std::vector<size_t>({2, 2}),
          "sequential replay backward should return both input gradients");

    const auto accumulated = runtime.GetGradients();
    const auto it = accumulated.find("layer0.weight");
    Check(it != accumulated.end(),
          "sequential replay backward should expose accumulated weight gradient");
    const float* weight_grad = it->second.Data<float>();
    CheckNear(weight_grad[0], 13.0f, 1e-4f,
              "sequential replay should sum first branch and second branch dW[0]");
    CheckNear(weight_grad[1], 16.0f, 1e-4f,
              "sequential replay should sum first branch and second branch dW[1]");
    CheckNear(weight_grad[2], 13.0f, 1e-4f,
              "sequential replay should sum accumulated gradients for output row 1");
    CheckNear(weight_grad[3], 16.0f, 1e-4f,
              "sequential replay should sum accumulated gradients for output row 1");
}

void TestSharedEncoderRejectsTrainingModeStatefulSequentialReplay() {
    auto model = std::make_unique<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::DropoutModule>(0.5f);
    auto executable =
        std::make_unique<cyxwiz::SequentialExecutableModel>(std::move(model));
    cyxwiz::SharedEncoderRuntime runtime(std::move(executable));

    bool rejected_stateful_replay = false;
    try {
        (void)runtime.ForwardPair(MakePairBatch());
    } catch (const std::logic_error& e) {
        rejected_stateful_replay =
            std::string(e.what()).find("activation snapshots") !=
            std::string::npos;
    }
    Check(rejected_stateful_replay,
          "training-mode stateful sequential modules should require snapshots");
}

}  // namespace

int main() {
    TestSharedEncoderPairContract();
    TestSharedEncoderTripletContract();
    TestSharedEncoderRejectsInvalidInput();
    TestSharedEncoderSequentialReplayAccumulatesGradients();
    TestSharedEncoderRejectsTrainingModeStatefulSequentialReplay();
    std::cout << "Metric-learning shared encoder contracts passed\n";
    return 0;
}
