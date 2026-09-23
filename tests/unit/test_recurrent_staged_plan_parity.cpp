// tofix67 slice 5: staged-ArrayFire vs native-CPU parity for the recurrent
// execution plan named RecurrentStagedArrayFirePlanName. The staged plan is
// the LSTM ArrayFire forward with per-timestep materialization boundaries;
// the native path is the CPU BPTT implementation. Same weights, same input:
// outputs and gradients must agree.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/recurrent_cuda_placement.h>
#include <cyxwiz/tensor.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

struct ScopedForceNativeRecurrentForward {
    ScopedForceNativeRecurrentForward() {
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    }
    ~ScopedForceNativeRecurrentForward() {
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    }
};

cyxwiz::Tensor MakeInput(size_t batch, size_t seq, size_t features) {
    cyxwiz::Tensor input(std::vector<size_t>{batch, seq, features});
    float* data = input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        data[i] = 0.2f * std::sin(0.31f * static_cast<float>(i)) +
                  0.05f * std::cos(0.07f * static_cast<float>(i));
    }
    return input;
}

cyxwiz::Tensor MakeUpstream(const std::vector<size_t>& shape) {
    cyxwiz::Tensor upstream(shape);
    float* data = upstream.Data<float>();
    for (size_t i = 0; i < upstream.NumElements(); ++i) {
        data[i] = 0.1f * std::sin(0.13f * static_cast<float>(i) + 0.5f);
    }
    return upstream;
}

void CheckTensorsNear(const cyxwiz::Tensor& actual,
                      const cyxwiz::Tensor& expected,
                      float tolerance,
                      const std::string& label) {
    REQUIRE(actual.Shape() == expected.Shape());
    const float* a = actual.ReadData<float>();
    const float* e = expected.ReadData<float>();
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.NumElements(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(a[i] - e[i]));
    }
    INFO(label << ": max_abs_diff=" << max_abs_diff);
    CHECK(max_abs_diff <= tolerance);
}

} // namespace

namespace {
// This test is the staged-ArrayFire vs native-CPU oracle; the neural
// provider must not hijack either leg.
struct ScopedDisableNeuralProviders {
    ScopedDisableNeuralProviders() {
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    }
    ~ScopedDisableNeuralProviders() {
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    }
};
} // namespace

TEST_CASE("LSTM staged ArrayFire plan matches native CPU forward and backward",
          "[gpu_execution][recurrent][parity]") {
    ScopedDisableNeuralProviders no_providers;
    constexpr size_t kBatch = 2;
    constexpr size_t kSeq = 5;
    constexpr size_t kInput = 6;
    constexpr int kHidden = 4;

    // Staged run first: default policy uses the ArrayFire recurrent forward
    // on non-CUDA backends (the staged per-timestep plan).
    cyxwiz::LSTMLayer staged(kInput, kHidden);
    const auto weights = staged.GetParameters();

    const auto input = MakeInput(kBatch, kSeq, kInput);
    const auto staged_output = staged.Forward(input);
    const auto upstream = MakeUpstream(staged_output.Shape());
    const auto staged_grad_input = staged.Backward(upstream);

    // Native run with identical weights.
    ScopedForceNativeRecurrentForward force_native;
    cyxwiz::LSTMLayer native(kInput, kHidden);
    native.SetParameters(weights);
    const auto native_output = native.Forward(input);
    const auto native_grad_input = native.Backward(upstream);

    CheckTensorsNear(staged_output, native_output, 1e-4f,
                     "forward output parity");
    CheckTensorsNear(staged_grad_input, native_grad_input, 1e-4f,
                     "backward input-gradient parity");

    // Parameter gradients must agree too — both paths feed the same CPU
    // BPTT backward through the caches their forward populated.
    const auto staged_params = staged.GetParameters();
    const auto native_params = native.GetParameters();
    size_t compared_gradients = 0;
    for (const auto& [name, value] : staged_params) {
        // LSTM gradient keys are spelled "layer<N>_grad_<param>".
        if (name.find("grad_") == std::string::npos) {
            continue;
        }
        const auto it = native_params.find(name);
        if (it == native_params.end() || value.NumElements() == 0 ||
            it->second.NumElements() == 0) {
            continue;
        }
        CheckTensorsNear(value, it->second, 1e-4f,
                         "parameter gradient parity: " + name);
        ++compared_gradients;
    }
    INFO("compared " << compared_gradients << " parameter gradients");
    CHECK(compared_gradients > 0);
}
