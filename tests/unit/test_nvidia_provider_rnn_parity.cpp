// tofix68 phase 3: the provider's pilot op (rnn_forward) must match the
// CPU reference RNNLayer exactly — same weights, same input. Runs only in
// provider-enabled builds; skips gracefully when no CUDA device let the
// provider register.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/recurrent_cuda_placement.h>
#include <cyxwiz/tensor.h>

#ifndef NOMINMAX
#define NOMINMAX  // arrayfire.h pulls windows.h; keep std::max usable
#endif
#include <arrayfire.h>

#include <cmath>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER

namespace {

cyxwiz::Tensor FilledTensor(const std::vector<size_t>& shape, float scale,
                            float phase) {
    cyxwiz::Tensor tensor(shape);
    float* data = tensor.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        data[i] = scale * std::sin(0.29f * static_cast<float>(i) + phase);
    }
    return tensor;
}

} // namespace

TEST_CASE("NVIDIA provider rnn_forward matches the CPU reference layer",
          "[gpu_execution][neural_provider][parity]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();

    for (const bool tanh_activation : {true, false}) {
        cyxwiz::NeuralOpRequest request;
        request.target = {cyxwiz::DeviceType::CUDA, 0};
        request.op = cyxwiz::NeuralOp::RnnForward;
        request.training = false;
        request.dtype = cyxwiz::DataType::Float32;
        request.batch = 4;
        request.seq = 12;
        request.input = 6;
        request.hidden = 16;
        request.activation = tanh_activation
            ? cyxwiz::NeuralActivation::Tanh
            : cyxwiz::NeuralActivation::Relu;

        auto provider = registry.FindSupporting(request);
        if (!provider) {
            WARN("no provider supports rnn_forward on this machine; "
                 "parity not exercised");
            return;
        }

        cyxwiz::RNNLayer reference(
            static_cast<int>(request.input),
            static_cast<int>(request.hidden), 1, true, false,
            tanh_activation ? "tanh" : "relu");
        auto params = reference.GetParameters();
        const auto input =
            FilledTensor({request.batch, request.seq, request.input},
                         0.7f, tanh_activation ? 0.1f : 0.9f);
        const auto expected = reference.Forward(input);

        cyxwiz::Tensor actual(
            std::vector<size_t>{request.batch, request.seq, request.hidden});
        const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
        const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
        const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
        const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&actual};

        const auto status = provider->Execute(request, buffers);
        INFO("provider: " << provider->Version()
                          << " detail: " << status.detail);
        REQUIRE(status.ok);

        const float* e = expected.ReadData<float>();
        const float* a = actual.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < expected.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(e[i] - a[i]));
        }
        INFO((tanh_activation ? "tanh" : "relu")
             << " parity max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 1e-4f);
    }
}

TEST_CASE("NVIDIA provider lstm_forward matches the native CPU LSTM",
          "[gpu_execution][neural_provider][parity]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();

    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::LstmForward;
    request.training = false;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = 4;
    request.seq = 10;
    request.input = 6;
    request.hidden = 16;
    request.activation = cyxwiz::NeuralActivation::None;

    auto provider = registry.FindSupporting(request);
    if (!provider) {
        WARN("no provider supports lstm_forward on this machine; parity "
             "not exercised");
        return;
    }

    // The oracle is the native CPU LSTM (forced past the ArrayFire path)
    // — the same reference the staged plan is parity-gated against. The
    // warm-up Forward matters: the native path lazily reinitializes
    // weights on first use, so parameters are captured only afterwards.
    cyxwiz::LSTMLayer reference(static_cast<int>(request.input),
                                static_cast<int>(request.hidden));
    const auto input = FilledTensor(
        {request.batch, request.seq, request.input}, 0.6f, 0.4f);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(input);  // warm-up: settles weight storage
    auto params = reference.GetParameters();
    reference.ResetState();
    const auto expected = reference.Forward(input);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);

    cyxwiz::Tensor actual(
        std::vector<size_t>{request.batch, request.seq, request.hidden});
    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    cyxwiz::NeuralOpBuffers buffers;
    buffers.inputs = {&input};
    buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
    buffers.outputs = {&actual};

    const auto status = provider->Execute(request, buffers);
    INFO("provider: " << provider->Version()
                      << " detail: " << status.detail);
    REQUIRE(status.ok);

    const float* e = expected.ReadData<float>();
    const float* a = actual.ReadData<float>();
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < expected.NumElements(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(e[i] - a[i]));
    }
    INFO("lstm parity max_abs_diff=" << max_abs_diff);
    CHECK(max_abs_diff <= 1e-4f);

    // LSTM owns its gate activations: a tuple that names one is refused.
    auto tanh_request = request;
    tanh_request.activation = cyxwiz::NeuralActivation::Tanh;
    CHECK_FALSE(provider->QueryCapability(tanh_request).supported);
}

TEST_CASE("NVIDIA provider lstm_backward matches the native CPU BPTT",
          "[gpu_execution][neural_provider][parity]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();

    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::LstmBackward;
    request.training = true;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = 4;
    request.seq = 10;
    request.input = 6;
    request.hidden = 16;
    request.activation = cyxwiz::NeuralActivation::None;

    auto provider = registry.FindSupporting(request);
    if (!provider) {
        WARN("no provider supports lstm_backward on this machine; parity "
             "not exercised");
        return;
    }

    cyxwiz::LSTMLayer reference(static_cast<int>(request.input),
                                static_cast<int>(request.hidden));
    const auto input = FilledTensor(
        {request.batch, request.seq, request.input}, 0.5f, 0.7f);
    const auto upstream = FilledTensor(
        {request.batch, request.seq, request.hidden}, 0.3f, 1.3f);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(input);  // warm-up: settles weight storage
    auto params = reference.GetParameters();
    reference.ResetState();
    reference.Forward(input);
    const auto expected_dx = reference.Backward(upstream);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    const auto grads = reference.GetParameters();

    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    cyxwiz::Tensor actual_dx(std::vector<size_t>{
        request.batch, request.seq, request.input});
    cyxwiz::Tensor dW_ih(std::vector<size_t>{4 * request.hidden,
                                             request.input});
    cyxwiz::Tensor dW_hh(std::vector<size_t>{4 * request.hidden,
                                             request.hidden});
    cyxwiz::Tensor db_ih(std::vector<size_t>{4 * request.hidden});
    cyxwiz::Tensor db_hh(std::vector<size_t>{4 * request.hidden});

    cyxwiz::NeuralOpBuffers buffers;
    buffers.inputs = {&input, &upstream};
    buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
    buffers.outputs = {&actual_dx};
    buffers.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};

    const auto status = provider->Execute(request, buffers);
    INFO("provider: " << provider->Version()
                      << " detail: " << status.detail);
    REQUIRE(status.ok);

    const auto compare = [](const cyxwiz::Tensor& actual,
                            const cyxwiz::Tensor& expected,
                            const char* label) {
        REQUIRE(actual.NumElements() == expected.NumElements());
        const float* a = actual.ReadData<float>();
        const float* e = expected.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < actual.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(a[i] - e[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    compare(actual_dx, expected_dx, "dx");
    compare(dW_ih, grads.at("layer0_grad_W_ih"), "dW_ih");
    compare(dW_hh, grads.at("layer0_grad_W_hh"), "dW_hh");
    compare(db_ih, grads.at("layer0_grad_b_ih"), "db_ih");
    compare(db_hh, grads.at("layer0_grad_b_hh"), "db_hh");
}

TEST_CASE("LSTMLayer routes training through the provider with parity",
          "[gpu_execution][neural_provider][parity]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest probe;
    probe.target = {cyxwiz::DeviceType::CUDA, 0};
    probe.op = cyxwiz::NeuralOp::LstmForward;
    probe.batch = 4;
    probe.seq = 9;
    probe.input = 5;
    probe.hidden = 12;
    if (!registry.FindSupporting(probe)) {
        WARN("no provider on this machine; layer routing not exercised");
        return;
    }

    // The layer dispatches on the run's SELECTED device (device-keyed
    // rule), and the test harness pins ArrayFire to CPU — under which the
    // layer must (and does) skip the CUDA provider. Select CUDA for this
    // test only, restoring the harness backend on exit.
    struct SelectedBackendGuard {
        af::Backend previous = AF_BACKEND_CPU;
        int previous_device = 0;
        bool active = false;
        ~SelectedBackendGuard() {
            if (active) {
                try {
                    af::setBackend(previous);
                    af::setDevice(previous_device);
                } catch (...) {
                }
            }
        }
    } backend_guard;
    try {
        backend_guard.previous = af::getActiveBackend();
        backend_guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(0);
        backend_guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend cannot be selected in this process; "
             "layer routing not exercised");
        return;
    }
    REQUIRE(cyxwiz::CaptureCurrentNeuralDeviceTarget().platform ==
            cyxwiz::DeviceType::CUDA);
    cyxwiz::LSTMLayer layer(5, 12);
    const auto input = FilledTensor({4, 9, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({4, 9, 12}, 0.25f, 1.1f);

    // Oracle: same layer, providers hidden, forced-native path.
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    layer.Forward(input);  // settles weights
    layer.ResetState();
    const auto expected = layer.Forward(input);
    const auto expected_h_n = layer.GetHiddenState();
    const auto expected_c_n = layer.GetCellState();
    const auto expected_dx = layer.Backward(upstream);
    const auto oracle_grads = layer.GetParameters();
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);

    // Routed: identical weights, provider active — the layer must pick
    // the provider for both Forward and Backward.
    layer.ResetState();
    const auto actual = layer.Forward(input);
    const auto actual_h_n = layer.GetHiddenState();
    const auto actual_c_n = layer.GetCellState();
    const auto actual_dx = layer.Backward(upstream);
    const auto routed_grads = layer.GetParameters();

    const auto compare = [](const cyxwiz::Tensor& a, const cyxwiz::Tensor& e,
                            const char* label) {
        REQUIRE(a.NumElements() == e.NumElements());
        const float* pa = a.ReadData<float>();
        const float* pe = e.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(pa[i] - pe[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    compare(actual, expected, "routed forward output");
    compare(actual_dx, expected_dx, "routed backward dx");
    for (const char* name :
         {"layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
          "layer0_grad_b_hh"}) {
        compare(routed_grads.at(name), oracle_grads.at(name), name);
    }
    // Final states: same [layers*dirs, batch, hidden] layout as the CPU
    // path and numerically equal, including the cell state the provider
    // now returns through its optional second output.
    CHECK(actual_h_n.Shape() == std::vector<size_t>{1, 4, 12});
    CHECK(actual_c_n.Shape() == std::vector<size_t>{1, 4, 12});
    REQUIRE(expected_h_n.Shape() == actual_h_n.Shape());
    REQUIRE(expected_c_n.Shape() == actual_c_n.Shape());
    compare(actual_h_n, expected_h_n, "routed final hidden state");
    compare(actual_c_n, expected_c_n, "routed final cell state");
}

TEST_CASE("NVIDIA provider refuses out-of-contract rnn_forward tuples",
          "[gpu_execution][neural_provider]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::RnnForward;
    request.batch = 4;
    request.seq = 8;
    request.input = 6;
    request.hidden = 16;
    request.activation = cyxwiz::NeuralActivation::Tanh;

    // rnn training is served since provider 0.7.0 (rnn_backward).
    auto training_request = request;
    training_request.training = true;
    CHECK(registry.FindSupporting(training_request) != nullptr);

    auto bidirectional_request = request;
    bidirectional_request.directions = 2;
    CHECK(registry.FindSupporting(bidirectional_request) == nullptr);

    auto gelu_request = request;
    gelu_request.activation = cyxwiz::NeuralActivation::Gelu;
    CHECK(registry.FindSupporting(gelu_request) == nullptr);

    // Device-keyed dispatch: the SAME in-contract tuple aimed at a
    // non-CUDA device must not reach this provider, hardware presence
    // notwithstanding (registry filter), and a direct query must refuse
    // truthfully too.
    for (const auto platform : {cyxwiz::DeviceType::OPENCL,
                                cyxwiz::DeviceType::ONEAPI,
                                cyxwiz::DeviceType::CPU}) {
        auto other_device_request = request;
        other_device_request.target = {platform, 0};
        // Another tenant (e.g. the OpenCL provider) may serve this
        // family; the invariant is that it is never the CUDA provider.
        if (const auto picked = registry.FindSupporting(other_device_request)) {
            CHECK(std::string(picked->ProviderId()) !=
                  "cyxwiz.nvidia-cublas-cell");
            CHECK(picked->Platform() == platform);
        }
        // Other test fixtures may register stubs for other families in this
        // process; the invariant is that the CUDA provider is never among
        // the providers serving a non-CUDA target.
        for (const auto& serving :
             registry.ListServing(other_device_request.target)) {
            CHECK(std::string(serving->ProviderId()) !=
                  "cyxwiz.nvidia-cublas-cell");
            CHECK(serving->Platform() == platform);
        }
        for (const auto& provider : registry.List()) {
            if (std::string(provider->ProviderId()) !=
                "cyxwiz.nvidia-cublas-cell") {
                continue;
            }
            CHECK(provider->Platform() == cyxwiz::DeviceType::CUDA);
            const auto capability =
                provider->QueryCapability(other_device_request);
            CHECK_FALSE(capability.supported);
            CHECK(capability.detail.find("serves cuda only") !=
                  std::string::npos);
        }
    }
    CHECK(registry.FindSupporting(request) != nullptr);
}


// ---------------------------------------------------------------- GRU (P3)

TEST_CASE("NVIDIA provider gru_forward matches the CPU GRU reference",
          "[gpu_execution][neural_provider][parity][gru]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::GruForward;
    request.training = false;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = 4;
    request.seq = 10;
    request.input = 6;
    request.hidden = 16;
    request.activation = cyxwiz::NeuralActivation::None;

    auto provider = registry.FindSupporting(request);
    if (!provider) {
        WARN("no provider supports gru_forward on this machine; parity not "
             "exercised");
        return;
    }
    // A GRU tuple naming an activation is refused (GRU owns its gates).
    auto activated = request;
    activated.activation = cyxwiz::NeuralActivation::Tanh;
    CHECK(registry.FindSupporting(activated) == nullptr);

    cyxwiz::GRULayer reference(static_cast<int>(request.input),
                               static_cast<int>(request.hidden));
    const auto input = FilledTensor(
        {request.batch, request.seq, request.input}, 0.5f, 0.7f);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(input);  // warm-up: settles weight storage
    reference.ResetState();
    const auto expected = reference.Forward(input);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    const auto params = reference.GetParameters();

    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    REQUIRE(W_ih.NumElements() == 3 * request.hidden * request.input);
    cyxwiz::Tensor actual(std::vector<size_t>{request.batch, request.seq,
                                              request.hidden});
    cyxwiz::NeuralOpBuffers buffers;
    buffers.inputs = {&input};
    buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
    buffers.outputs = {&actual};
    const auto status = provider->Execute(request, buffers);
    INFO("provider: " << provider->Version()
                      << " detail: " << status.detail);
    REQUIRE(status.ok);

    REQUIRE(actual.NumElements() == expected.NumElements());
    const float* a = actual.ReadData<float>();
    const float* e = expected.ReadData<float>();
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.NumElements(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(a[i] - e[i]));
    }
    INFO("gru_forward max_abs_diff=" << max_abs_diff);
    CHECK(max_abs_diff <= 1e-4f);

    // GRU accepts {y} or {y, h_n}; the three-output LSTM form (with a
    // cell state) is refused typed, not silently ignored.
    cyxwiz::Tensor bogus_hidden(std::vector<size_t>{1, request.batch, request.hidden});
    cyxwiz::Tensor bogus_cell(std::vector<size_t>{1, request.batch, request.hidden});
    cyxwiz::NeuralOpBuffers two_outputs;
    two_outputs.inputs = {&input};
    two_outputs.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
    two_outputs.outputs = {&actual, &bogus_hidden, &bogus_cell};
    const auto refused = provider->Execute(request, two_outputs);
    CHECK_FALSE(refused.ok);
    CHECK(refused.reason ==
          cyxwiz::BackendFallbackReason::NvidiaProviderUnsupportedContract);
}

TEST_CASE("NVIDIA provider gru_backward matches the CPU GRU BPTT reference",
          "[gpu_execution][neural_provider][parity][gru]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::GruBackward;
    request.training = true;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = 4;
    request.seq = 10;
    request.input = 6;
    request.hidden = 16;
    request.activation = cyxwiz::NeuralActivation::None;

    auto provider = registry.FindSupporting(request);
    if (!provider) {
        WARN("no provider supports gru_backward on this machine; parity "
             "not exercised");
        return;
    }

    cyxwiz::GRULayer reference(static_cast<int>(request.input),
                               static_cast<int>(request.hidden));
    const auto input = FilledTensor(
        {request.batch, request.seq, request.input}, 0.5f, 0.7f);
    const auto upstream = FilledTensor(
        {request.batch, request.seq, request.hidden}, 0.3f, 1.3f);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(input);  // warm-up: settles weight storage
    auto params = reference.GetParameters();
    reference.ResetState();
    reference.Forward(input);
    const auto expected_dx = reference.Backward(upstream);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    const auto grads = reference.GetParameters();

    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    cyxwiz::Tensor actual_dx(std::vector<size_t>{
        request.batch, request.seq, request.input});
    cyxwiz::Tensor dW_ih(std::vector<size_t>{3 * request.hidden,
                                             request.input});
    cyxwiz::Tensor dW_hh(std::vector<size_t>{3 * request.hidden,
                                             request.hidden});
    cyxwiz::Tensor db_ih(std::vector<size_t>{3 * request.hidden});
    cyxwiz::Tensor db_hh(std::vector<size_t>{3 * request.hidden});

    cyxwiz::NeuralOpBuffers buffers;
    buffers.inputs = {&input, &upstream};
    buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
    buffers.outputs = {&actual_dx};
    buffers.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};

    const auto status = provider->Execute(request, buffers);
    INFO("provider: " << provider->Version()
                      << " detail: " << status.detail);
    REQUIRE(status.ok);

    const auto compare = [](const cyxwiz::Tensor& actual,
                            const cyxwiz::Tensor& expected,
                            const char* label) {
        REQUIRE(actual.NumElements() == expected.NumElements());
        const float* a = actual.ReadData<float>();
        const float* e = expected.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < actual.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(a[i] - e[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    compare(actual_dx, expected_dx, "dx");
    compare(dW_ih, grads.at("layer0_grad_W_ih"), "dW_ih");
    compare(dW_hh, grads.at("layer0_grad_W_hh"), "dW_hh");
    compare(db_ih, grads.at("layer0_grad_b_ih"), "db_ih");
    compare(db_hh, grads.at("layer0_grad_b_hh"), "db_hh");

    // GRU-specific: the n-gate bias gradients differ by the reset gate, so
    // db_ih and db_hh must NOT be identical (an LSTM-style shortcut would
    // pass dx/dW parity and still be wrong here).
    {
        const float* ih = db_ih.ReadData<float>();
        const float* hh = db_hh.ReadData<float>();
        float n_slot_diff = 0.0f;
        for (size_t i = 2 * request.hidden; i < 3 * request.hidden; ++i) {
            n_slot_diff = std::max(n_slot_diff, std::fabs(ih[i] - hh[i]));
        }
        CHECK(n_slot_diff > 1e-6f);
    }
}

TEST_CASE("GRULayer routes training through the provider with parity",
          "[gpu_execution][neural_provider][parity][gru]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest probe;
    probe.target = {cyxwiz::DeviceType::CUDA, 0};
    probe.op = cyxwiz::NeuralOp::GruBackward;
    probe.training = true;
    probe.batch = 4;
    probe.seq = 9;
    probe.input = 5;
    probe.hidden = 12;
    if (!registry.FindSupporting(probe)) {
        WARN("no provider on this machine; GRU layer routing not exercised");
        return;
    }
    struct SelectedBackendGuard {
        af::Backend previous = AF_BACKEND_CPU;
        int previous_device = 0;
        bool active = false;
        ~SelectedBackendGuard() {
            if (active) {
                try {
                    af::setBackend(previous);
                    af::setDevice(previous_device);
                } catch (...) {
                }
            }
        }
    } backend_guard;
    try {
        backend_guard.previous = af::getActiveBackend();
        backend_guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(0);
        backend_guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend cannot be selected in this process; "
             "GRU layer routing not exercised");
        return;
    }
    REQUIRE(cyxwiz::CaptureCurrentNeuralDeviceTarget().platform ==
            cyxwiz::DeviceType::CUDA);
    cyxwiz::GRULayer layer(5, 12);
    const auto input = FilledTensor({4, 9, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({4, 9, 12}, 0.25f, 1.1f);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    layer.Forward(input);  // settles weights
    layer.ResetState();
    const auto expected = layer.Forward(input);
    const auto expected_dx = layer.Backward(upstream);
    const auto oracle_grads = layer.GetParameters();
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);

    layer.ResetState();
    const auto actual = layer.Forward(input);
    const auto actual_dx = layer.Backward(upstream);
    const auto routed_grads = layer.GetParameters();

    const auto compare = [](const cyxwiz::Tensor& a, const cyxwiz::Tensor& e,
                            const char* label) {
        REQUIRE(a.NumElements() == e.NumElements());
        const float* pa = a.ReadData<float>();
        const float* pe = e.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(pa[i] - pe[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    compare(actual, expected, "routed GRU forward output");
    compare(actual_dx, expected_dx, "routed GRU backward dx");
    for (const char* name :
         {"layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
          "layer0_grad_b_hh"}) {
        compare(routed_grads.at(name), oracle_grads.at(name), name);
    }
    // Final hidden state in the CPU path's [layers*dirs, batch, hidden]
    // layout, equal to the last output timestep.
    const auto h_n = layer.GetHiddenState();
    REQUIRE(h_n.Shape() == std::vector<size_t>{1, 4, 12});
    const float* hn = h_n.ReadData<float>();
    const float* y = actual.ReadData<float>();
    float hn_diff = 0.0f;
    for (size_t b = 0; b < 4; ++b) {
        for (size_t j = 0; j < 12; ++j) {
            hn_diff = std::max(hn_diff, std::fabs(hn[b * 12 + j] -
                                                  y[(b * 9 + 8) * 12 + j]));
        }
    }
    CHECK(hn_diff == 0.0f);
}

TEST_CASE("NVIDIA provider repeated gru training runs do not leak device memory",
          "[gpu_execution][neural_provider][leak][gru]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest forward;
    forward.target = {cyxwiz::DeviceType::CUDA, 0};
    forward.op = cyxwiz::NeuralOp::GruForward;
    forward.training = true;
    forward.batch = 32;
    forward.seq = 32;
    forward.input = 64;
    forward.hidden = 64;
    auto backward = forward;
    backward.op = cyxwiz::NeuralOp::GruBackward;
    auto provider = registry.FindSupporting(backward);
    if (!provider || !registry.FindSupporting(forward)) {
        WARN("no provider supports gru training on this machine; leak check "
             "not exercised");
        return;
    }
    const size_t gate_width = 3 * forward.hidden;
    const auto x = FilledTensor({forward.batch, forward.seq, forward.input},
                                0.1f, 0.37f);
    const auto dy = FilledTensor(
        {forward.batch, forward.seq, forward.hidden}, 0.2f, 0.91f);
    const auto W_ih = FilledTensor({gate_width, forward.input}, 0.05f, 0.11f);
    const auto W_hh = FilledTensor({gate_width, forward.hidden}, 0.04f, 0.13f);
    const auto b_ih = FilledTensor({gate_width}, 0.01f, 0.02f);
    const auto b_hh = FilledTensor({gate_width}, 0.02f, 0.03f);
    cyxwiz::Tensor y(std::vector<size_t>{forward.batch, forward.seq,
                                         forward.hidden});
    cyxwiz::Tensor dx(std::vector<size_t>{forward.batch, forward.seq,
                                          forward.input});
    cyxwiz::Tensor dW_ih(std::vector<size_t>{gate_width, forward.input});
    cyxwiz::Tensor dW_hh(std::vector<size_t>{gate_width, forward.hidden});
    cyxwiz::Tensor db_ih(std::vector<size_t>{gate_width});
    cyxwiz::Tensor db_hh(std::vector<size_t>{gate_width});
    const auto run_once = [&]() {
        cyxwiz::NeuralOpBuffers fwd_buffers;
        fwd_buffers.inputs = {&x};
        fwd_buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        fwd_buffers.outputs = {&y};
        REQUIRE(provider->Execute(forward, fwd_buffers).ok);
        cyxwiz::NeuralOpBuffers bwd_buffers;
        bwd_buffers.inputs = {&x, &dy};
        bwd_buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        bwd_buffers.outputs = {&dx};
        bwd_buffers.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};
        REQUIRE(provider->Execute(backward, bwd_buffers).ok);
    };
    constexpr int kIterations = 32;
    constexpr size_t kToleranceBytes = 16u * 1024u * 1024u;
    const auto bwd_estimate = provider->EstimateResources(backward);
    CHECK(bwd_estimate.reserve_bytes == 0);
    REQUIRE(static_cast<size_t>(kIterations) * bwd_estimate.workspace_bytes >=
            8 * kToleranceBytes);
    for (int i = 0; i < 3; ++i) {
        run_once();
    }
    size_t free_before = 0, total_before = 0;
    REQUIRE(cyxwiz::NvidiaProviderDeviceMemoryForTesting(free_before,
                                                          total_before));
    for (int i = 0; i < kIterations; ++i) {
        run_once();
    }
    size_t free_after = 0, total_after = 0;
    REQUIRE(cyxwiz::NvidiaProviderDeviceMemoryForTesting(free_after,
                                                          total_after));
    CHECK(total_after == total_before);
    const size_t drop =
        free_before > free_after ? free_before - free_after : 0;
    INFO("free before=" << free_before << " after=" << free_after
                        << " drop=" << drop << " bytes over " << kIterations
                        << " gru forward+backward runs");
    CHECK(drop <= kToleranceBytes);
}


// ---------------------------------------------------------- stacked (0.6.0)

TEST_CASE("Stacked LSTM and GRU layers route training through the provider with parity",
          "[gpu_execution][neural_provider][parity][stacked]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest probe;
    probe.target = {cyxwiz::DeviceType::CUDA, 0};
    probe.op = cyxwiz::NeuralOp::LstmBackward;
    probe.training = true;
    probe.batch = 3;
    probe.seq = 7;
    probe.input = 5;
    probe.hidden = 10;
    probe.layers = 3;
    if (!registry.FindSupporting(probe)) {
        WARN("no provider supports stacked lstm training on this machine; "
             "routing not exercised");
        return;
    }
    struct SelectedBackendGuard {
        af::Backend previous = AF_BACKEND_CPU;
        int previous_device = 0;
        bool active = false;
        ~SelectedBackendGuard() {
            if (active) {
                try {
                    af::setBackend(previous);
                    af::setDevice(previous_device);
                } catch (...) {
                }
            }
        }
    } backend_guard;
    try {
        backend_guard.previous = af::getActiveBackend();
        backend_guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(0);
        backend_guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend cannot be selected in this process; "
             "stacked routing not exercised");
        return;
    }
    const auto input = FilledTensor({3, 7, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({3, 7, 10}, 0.25f, 1.1f);
    const auto compare = [](const cyxwiz::Tensor& a, const cyxwiz::Tensor& e,
                            const char* label) {
        REQUIRE(a.NumElements() == e.NumElements());
        const float* pa = a.ReadData<float>();
        const float* pe = e.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(pa[i] - pe[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    const char* grad_names[] = {
        "layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
        "layer0_grad_b_hh", "layer1_grad_W_ih", "layer1_grad_W_hh",
        "layer1_grad_b_ih", "layer1_grad_b_hh", "layer2_grad_W_ih",
        "layer2_grad_W_hh", "layer2_grad_b_ih", "layer2_grad_b_hh"};

    {
        cyxwiz::LSTMLayer layer(5, 10, 3);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        layer.Forward(input);  // settles weights
        const auto expected = layer.Forward(input);
        const auto expected_h_n = layer.GetHiddenState();
        const auto expected_c_n = layer.GetCellState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);

        const auto actual = layer.Forward(input);
        const auto actual_h_n = layer.GetHiddenState();
        const auto actual_c_n = layer.GetCellState();
        const auto actual_dx = layer.Backward(upstream);
        const auto routed = layer.GetParameters();
        compare(actual, expected, "stacked LSTM forward");
        compare(actual_dx, expected_dx, "stacked LSTM dx");
        for (const char* name : grad_names) {
            compare(routed.at(name), oracle.at(name), name);
        }
        REQUIRE(actual_h_n.Shape() == std::vector<size_t>{3, 3, 10});
        REQUIRE(actual_c_n.Shape() == std::vector<size_t>{3, 3, 10});
        compare(actual_h_n, expected_h_n, "stacked LSTM h_n (all layers)");
        compare(actual_c_n, expected_c_n, "stacked LSTM c_n (all layers)");
    }
    {
        cyxwiz::GRULayer layer(5, 10, 3);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        layer.Forward(input);  // settles weights
        const auto expected = layer.Forward(input);
        const auto expected_h_n = layer.GetHiddenState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);

        const auto actual = layer.Forward(input);
        const auto actual_h_n = layer.GetHiddenState();
        const auto actual_dx = layer.Backward(upstream);
        const auto routed = layer.GetParameters();
        compare(actual, expected, "stacked GRU forward");
        compare(actual_dx, expected_dx, "stacked GRU dx");
        for (const char* name : grad_names) {
            compare(routed.at(name), oracle.at(name), name);
        }
        REQUIRE(actual_h_n.Shape() == std::vector<size_t>{3, 3, 10});
        compare(actual_h_n, expected_h_n, "stacked GRU h_n (all layers)");
    }
}

TEST_CASE("Stacked provider ops validate the per-layer buffer contract",
          "[gpu_execution][neural_provider][stacked]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::LstmForward;
    request.batch = 2;
    request.seq = 3;
    request.input = 4;
    request.hidden = 6;
    request.layers = 2;
    auto provider = registry.FindSupporting(request);
    if (!provider) {
        WARN("no provider on this machine; contract checks not exercised");
        return;
    }
    // Two-layer LSTM: 8 weights, layer 1 W_ih is [4H, hidden].
    const auto x = FilledTensor({2, 3, 4}, 0.1f, 0.3f);
    const auto w0_ih = FilledTensor({24, 4}, 0.1f, 0.1f);
    const auto w1_ih_wrong = FilledTensor({24, 4}, 0.1f, 0.2f);  // should be [24, 6]
    const auto w1_ih = FilledTensor({24, 6}, 0.1f, 0.2f);
    const auto w_hh = FilledTensor({24, 6}, 0.1f, 0.4f);
    const auto b = FilledTensor({24}, 0.01f, 0.5f);
    cyxwiz::Tensor y(std::vector<size_t>{2, 3, 6});

    cyxwiz::NeuralOpBuffers wrong;
    wrong.inputs = {&x};
    wrong.weights = {&w0_ih, &w_hh, &b, &b, &w1_ih_wrong, &w_hh, &b, &b};
    wrong.outputs = {&y};
    const auto refused = provider->Execute(request, wrong);
    CHECK_FALSE(refused.ok);
    CHECK(refused.reason ==
          cyxwiz::BackendFallbackReason::NvidiaProviderUnsupportedContract);

    cyxwiz::NeuralOpBuffers right;
    right.inputs = {&x};
    right.weights = {&w0_ih, &w_hh, &b, &b, &w1_ih, &w_hh, &b, &b};
    right.outputs = {&y};
    CHECK(provider->Execute(request, right).ok);

    // State outputs are [layers, batch, hidden]: a single-layer-sized
    // state buffer on a two-layer request is refused.
    cyxwiz::Tensor h_n_small(std::vector<size_t>{1, 2, 6});
    cyxwiz::Tensor c_n_small(std::vector<size_t>{1, 2, 6});
    right.outputs = {&y, &h_n_small, &c_n_small};
    CHECK_FALSE(provider->Execute(request, right).ok);
    cyxwiz::Tensor h_n(std::vector<size_t>{2, 2, 6});
    cyxwiz::Tensor c_n(std::vector<size_t>{2, 2, 6});
    right.outputs = {&y, &h_n, &c_n};
    CHECK(provider->Execute(request, right).ok);

    // rnn_forward is stacked too since 0.7.0.
    auto rnn = request;
    rnn.op = cyxwiz::NeuralOp::RnnForward;
    rnn.activation = cyxwiz::NeuralActivation::Tanh;
    CHECK(registry.FindSupporting(rnn) != nullptr);
    rnn.layers = 1;
    CHECK(registry.FindSupporting(rnn) != nullptr);
}

TEST_CASE("Stacked RNN layers route training through the provider with parity",
          "[gpu_execution][neural_provider][parity][stacked][rnn]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest probe;
    probe.target = {cyxwiz::DeviceType::CUDA, 0};
    probe.op = cyxwiz::NeuralOp::RnnBackward;
    probe.training = true;
    probe.batch = 3;
    probe.seq = 7;
    probe.input = 5;
    probe.hidden = 10;
    probe.layers = 2;
    probe.activation = cyxwiz::NeuralActivation::Tanh;
    if (!registry.FindSupporting(probe)) {
        WARN("no provider supports stacked rnn training on this machine; "
             "routing not exercised");
        return;
    }
    struct SelectedBackendGuard {
        af::Backend previous = AF_BACKEND_CPU;
        int previous_device = 0;
        bool active = false;
        ~SelectedBackendGuard() {
            if (active) {
                try {
                    af::setBackend(previous);
                    af::setDevice(previous_device);
                } catch (...) {
                }
            }
        }
    } backend_guard;
    try {
        backend_guard.previous = af::getActiveBackend();
        backend_guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(0);
        backend_guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend cannot be selected in this process; "
             "rnn routing not exercised");
        return;
    }
    const auto input = FilledTensor({3, 7, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({3, 7, 10}, 0.25f, 1.1f);
    const auto compare = [](const cyxwiz::Tensor& a, const cyxwiz::Tensor& e,
                            const char* label) {
        REQUIRE(a.NumElements() == e.NumElements());
        const float* pa = a.ReadData<float>();
        const float* pe = e.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(pa[i] - pe[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    const char* grad_names[] = {
        "layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
        "layer0_grad_b_hh", "layer1_grad_W_ih", "layer1_grad_W_hh",
        "layer1_grad_b_ih", "layer1_grad_b_hh"};
    for (const char* nonlinearity : {"tanh", "relu"}) {
        cyxwiz::RNNLayer layer(5, 10, 2, true, false, nonlinearity);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        const auto expected = layer.Forward(input);
        const auto expected_h = layer.GetHiddenState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);

        const auto actual = layer.Forward(input);
        const auto actual_h = layer.GetHiddenState();
        const auto actual_dx = layer.Backward(upstream);
        const auto routed = layer.GetParameters();
        const std::string tag = std::string("stacked RNN ") + nonlinearity;
        compare(actual, expected, (tag + " forward").c_str());
        compare(actual_dx, expected_dx, (tag + " dx").c_str());
        for (const char* name : grad_names) {
            compare(routed.at(name), oracle.at(name), name);
        }
        REQUIRE(actual_h.Shape() == std::vector<size_t>{3, 10});
        compare(actual_h, expected_h, (tag + " h_n").c_str());
    }
}


TEST_CASE("Split bidirectional LSTMModule routes both branches through the provider with parity",
          "[gpu_execution][neural_provider][parity][bidirectional]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest probe;
    probe.target = {cyxwiz::DeviceType::CUDA, 0};
    probe.op = cyxwiz::NeuralOp::LstmBackward;
    probe.training = true;
    probe.batch = 3;
    probe.seq = 6;
    probe.input = 5;
    probe.hidden = 8;
    if (!registry.FindSupporting(probe)) {
        WARN("no provider supports lstm training on this machine; bidirectional "
             "routing not exercised");
        return;
    }
    struct SelectedBackendGuard {
        af::Backend previous = AF_BACKEND_CPU;
        int previous_device = 0;
        bool active = false;
        ~SelectedBackendGuard() {
            if (active) {
                try {
                    af::setBackend(previous);
                    af::setDevice(previous_device);
                } catch (...) {
                }
            }
        }
    } backend_guard;
    try {
        backend_guard.previous = af::getActiveBackend();
        backend_guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        af::setDevice(0);
        backend_guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend cannot be selected; not exercised");
        return;
    }
    const auto input = FilledTensor({3, 6, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({3, 6, 16}, 0.25f, 1.1f);
    cyxwiz::LSTMModule module(5, 8, 2, /*bidirectional=*/true,
                              /*return_sequences=*/true);

    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    module.Forward(input);  // settles weights
    const auto expected = module.Forward(input);
    const auto expected_dx = module.Backward(upstream);
    const auto oracle = module.GetGradients();
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);

    const auto actual = module.Forward(input);
    const auto actual_dx = module.Backward(upstream);
    const auto routed = module.GetGradients();
    const auto compare = [](const cyxwiz::Tensor& a, const cyxwiz::Tensor& e,
                            const std::string& label) {
        REQUIRE(a.NumElements() == e.NumElements());
        const float* pa = a.ReadData<float>();
        const float* pe = e.ReadData<float>();
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.NumElements(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(pa[i] - pe[i]));
        }
        INFO(label << " max_abs_diff=" << max_abs_diff);
        CHECK(max_abs_diff <= 5e-4f);
    };
    REQUIRE(actual.Shape() == std::vector<size_t>{3, 6, 16});
    compare(actual, expected, "split BiLSTM forward");
    compare(actual_dx, expected_dx, "split BiLSTM dx");
    REQUIRE(routed.size() == oracle.size());
    for (const auto& [name, grad] : oracle) {
        compare(routed.at(name), grad, name);
    }
}

TEST_CASE("NVIDIA provider repeated lstm training runs do not leak device memory",
          "[gpu_execution][neural_provider][leak]") {
    // tofix68 validation gate: device buffers, cuBLAS handle, NVRTC module
    // and workspace must not accumulate across runs. Shape chosen so a
    // per-run leak of the backward workspace (~5 MB) is unmistakable over
    // the iteration count, while first-use allocations (module load,
    // cuBLAS workspace) are absorbed by the warm-up before measuring.
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    cyxwiz::NeuralOpRequest forward;
    forward.target = {cyxwiz::DeviceType::CUDA, 0};
    forward.op = cyxwiz::NeuralOp::LstmForward;
    forward.training = true;
    forward.batch = 32;
    forward.seq = 32;
    forward.input = 64;
    forward.hidden = 64;
    auto backward = forward;
    backward.op = cyxwiz::NeuralOp::LstmBackward;
    auto provider = registry.FindSupporting(backward);
    if (!provider || !registry.FindSupporting(forward)) {
        WARN("no provider supports lstm training on this machine; leak "
             "check not exercised");
        return;
    }

    const size_t gate_width = 4 * forward.hidden;
    const auto x = FilledTensor({forward.batch, forward.seq, forward.input},
                                0.1f, 0.37f);
    const auto dy = FilledTensor(
        {forward.batch, forward.seq, forward.hidden}, 0.2f, 0.91f);
    const auto W_ih = FilledTensor({gate_width, forward.input}, 0.05f, 0.11f);
    const auto W_hh = FilledTensor({gate_width, forward.hidden}, 0.04f, 0.13f);
    const auto b_ih = FilledTensor({gate_width}, 0.01f, 0.02f);
    const auto b_hh = FilledTensor({gate_width}, 0.02f, 0.03f);
    cyxwiz::Tensor y(std::vector<size_t>{forward.batch, forward.seq,
                                         forward.hidden});
    cyxwiz::Tensor dx(std::vector<size_t>{forward.batch, forward.seq,
                                          forward.input});
    cyxwiz::Tensor dW_ih(std::vector<size_t>{gate_width, forward.input});
    cyxwiz::Tensor dW_hh(std::vector<size_t>{gate_width, forward.hidden});
    cyxwiz::Tensor db_ih(std::vector<size_t>{gate_width});
    cyxwiz::Tensor db_hh(std::vector<size_t>{gate_width});

    const auto run_once = [&]() {
        cyxwiz::NeuralOpBuffers fwd_buffers;
        fwd_buffers.inputs = {&x};
        fwd_buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        fwd_buffers.outputs = {&y};
        const auto fwd_status = provider->Execute(forward, fwd_buffers);
        REQUIRE(fwd_status.ok);
        cyxwiz::NeuralOpBuffers bwd_buffers;
        bwd_buffers.inputs = {&x, &dy};
        bwd_buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        bwd_buffers.outputs = {&dx};
        bwd_buffers.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};
        const auto bwd_status = provider->Execute(backward, bwd_buffers);
        REQUIRE(bwd_status.ok);
    };

    // Estimates are truthful lower bounds on the per-run workspace and
    // the recompute design keeps no reserve. The leak bound below is only
    // meaningful if a per-run leak of the backward workspace would dwarf
    // the tolerance, so assert that relationship rather than a magic size.
    constexpr int kIterations = 32;
    constexpr size_t kToleranceBytes = 16u * 1024u * 1024u;
    const auto fwd_estimate = provider->EstimateResources(forward);
    const auto bwd_estimate = provider->EstimateResources(backward);
    CHECK(fwd_estimate.reserve_bytes == 0);
    CHECK(bwd_estimate.reserve_bytes == 0);
    CHECK(bwd_estimate.workspace_bytes > fwd_estimate.workspace_bytes);
    REQUIRE(static_cast<size_t>(kIterations) * bwd_estimate.workspace_bytes >=
            8 * kToleranceBytes);

    for (int i = 0; i < 3; ++i) {
        run_once();  // warm-up: module load, cuBLAS workspace
    }
    size_t free_before = 0;
    size_t total_before = 0;
    REQUIRE(cyxwiz::NvidiaProviderDeviceMemoryForTesting(free_before,
                                                          total_before));
    for (int i = 0; i < kIterations; ++i) {
        run_once();
    }
    size_t free_after = 0;
    size_t total_after = 0;
    REQUIRE(cyxwiz::NvidiaProviderDeviceMemoryForTesting(free_after,
                                                          total_after));
    CHECK(total_after == total_before);
    // A per-run leak of the backward workspace alone would cost at least
    // 8x the tolerance (asserted above); the tolerance absorbs unrelated
    // device activity (display, driver caches).
    const size_t drop =
        free_before > free_after ? free_before - free_after : 0;
    INFO("free before=" << free_before << " after=" << free_after
                        << " drop=" << drop << " bytes over " << kIterations
                        << " forward+backward runs (backward workspace "
                        << "estimate " << bwd_estimate.workspace_bytes
                        << " bytes)");
    CHECK(drop <= kToleranceBytes);
}

#endif // CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
