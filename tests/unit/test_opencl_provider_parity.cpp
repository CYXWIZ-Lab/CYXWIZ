// tofix68 device-keyed dispatch tenant #2: the OpenCL neural provider.
// Every test warns and returns when no OpenCL GPU device is available, so
// the suite stays truthful on machines without one.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/recurrent_cuda_placement.h>
#include <cyxwiz/tensor.h>

#ifndef NOMINMAX
#define NOMINMAX  // arrayfire.h pulls windows.h; keep std::max usable
#endif
#include <arrayfire.h>

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER

namespace {

cyxwiz::Tensor FilledTensor(const std::vector<size_t>& shape, float scale,
                            float phase) {
    cyxwiz::Tensor tensor(shape);
    float* data = tensor.Data<float>();
    for (size_t i = 0; i < tensor.NumElements(); ++i) {
        data[i] = scale * std::sin(0.37f * static_cast<float>(i) + phase);
    }
    return tensor;
}

std::shared_ptr<cyxwiz::INeuralNetworkProvider> OpenclProvider() {
    for (const auto& provider :
         cyxwiz::NeuralProviderRegistry::Instance().List()) {
        if (std::string(provider->ProviderId()) == "cyxwiz.opencl-cell") {
            return provider;
        }
    }
    return nullptr;
}

cyxwiz::NeuralOpRequest OpenclRequest(cyxwiz::NeuralOp op, size_t batch,
                                      size_t seq, size_t input, size_t hidden,
                                      size_t layers = 1) {
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::OPENCL, 0};
    request.op = op;
    request.training = op == cyxwiz::NeuralOp::LstmBackward ||
                       op == cyxwiz::NeuralOp::GruBackward;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = batch;
    request.seq = seq;
    request.input = input;
    request.hidden = hidden;
    request.layers = layers;
    request.activation = cyxwiz::NeuralActivation::None;
    return request;
}

void Compare(const cyxwiz::Tensor& actual, const cyxwiz::Tensor& expected,
             const char* label, float tolerance) {
    REQUIRE(actual.NumElements() == expected.NumElements());
    const float* a = actual.ReadData<float>();
    const float* e = expected.ReadData<float>();
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < actual.NumElements(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(a[i] - e[i]));
    }
    INFO(label << " max_abs_diff=" << max_abs_diff);
    CHECK(max_abs_diff <= tolerance);
}

} // namespace

TEST_CASE("OpenCL provider registers as the opencl tenant and serves only opencl targets",
          "[gpu_execution][neural_provider][opencl]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device on this machine; opencl provider not "
             "registered");
        return;
    }
    CHECK(provider->Platform() == cyxwiz::DeviceType::OPENCL);
    CHECK(provider->Version().find("OpenCL GPU devices") != std::string::npos);

    auto request = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 4, 10, 6, 16);
    CHECK(provider->QueryCapability(request).supported);
    // The registry resolves the same tuple to THIS provider for an OpenCL
    // target and never for a CUDA target (device-keyed dispatch).
    const auto picked =
        cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    REQUIRE(picked != nullptr);
    CHECK(std::string(picked->ProviderId()) == "cyxwiz.opencl-cell");
    auto cuda_request = request;
    cuda_request.target = {cyxwiz::DeviceType::CUDA, 0};
    const auto refused = provider->QueryCapability(cuda_request);
    CHECK_FALSE(refused.supported);
    CHECK(refused.detail.find("serves opencl only") != std::string::npos);
    auto bad_device = request;
    bad_device.target = {cyxwiz::DeviceType::OPENCL, 99};
    const auto no_device = provider->QueryCapability(bad_device);
    CHECK_FALSE(no_device.supported);
    CHECK(no_device.reason ==
          cyxwiz::BackendFallbackReason::OpenclProviderUnavailable);
}

TEST_CASE("OpenCL provider rnn/lstm/gru forward match the CPU references",
          "[gpu_execution][neural_provider][opencl][parity]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; parity not exercised");
        return;
    }
    const auto input = FilledTensor({4, 10, 6}, 0.5f, 0.7f);

    {
        cyxwiz::LSTMLayer reference(6, 16);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        reference.Forward(input);
        const auto expected = reference.Forward(input);
        const auto expected_c = reference.GetCellState();
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        const auto p = reference.GetParameters();
        const cyxwiz::Tensor W_ih = p.at("layer0_W_ih"), W_hh = p.at("layer0_W_hh"),
                             b_ih = p.at("layer0_b_ih"), b_hh = p.at("layer0_b_hh");
        cyxwiz::Tensor actual(std::vector<size_t>{4, 10, 16});
        cyxwiz::Tensor h_n(std::vector<size_t>{1, 4, 16});
        cyxwiz::Tensor c_n(std::vector<size_t>{1, 4, 16});
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&actual, &h_n, &c_n};
        auto request = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 4, 10, 6, 16);
        const auto status = provider->Execute(request, buffers);
        INFO("provider: " << provider->Version() << " detail: " << status.detail);
        REQUIRE(status.ok);
        Compare(actual, expected, "opencl lstm_forward", 1e-4f);
        Compare(c_n, expected_c, "opencl lstm_forward c_n", 1e-4f);
    }
    {
        cyxwiz::GRULayer reference(6, 16);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        reference.Forward(input);
        const auto expected = reference.Forward(input);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        const auto p = reference.GetParameters();
        const cyxwiz::Tensor W_ih = p.at("layer0_W_ih"), W_hh = p.at("layer0_W_hh"),
                             b_ih = p.at("layer0_b_ih"), b_hh = p.at("layer0_b_hh");
        cyxwiz::Tensor actual(std::vector<size_t>{4, 10, 16});
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&actual};
        auto request = OpenclRequest(cyxwiz::NeuralOp::GruForward, 4, 10, 6, 16);
        REQUIRE(provider->Execute(request, buffers).ok);
        Compare(actual, expected, "opencl gru_forward", 1e-4f);
    }
    {
        cyxwiz::RNNLayer reference(6, 16, 1, true, false, "tanh");
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        const auto expected = reference.Forward(input);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        const auto p = reference.GetParameters();
        const cyxwiz::Tensor W_ih = p.at("layer0_W_ih"), W_hh = p.at("layer0_W_hh"),
                             b_ih = p.at("layer0_b_ih"), b_hh = p.at("layer0_b_hh");
        cyxwiz::Tensor actual(std::vector<size_t>{4, 10, 16});
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&actual};
        auto request = OpenclRequest(cyxwiz::NeuralOp::RnnForward, 4, 10, 6, 16);
        request.activation = cyxwiz::NeuralActivation::Tanh;
        REQUIRE(provider->Execute(request, buffers).ok);
        Compare(actual, expected, "opencl rnn_forward", 1e-4f);
    }
}

TEST_CASE("OpenCL provider lstm/gru backward match the CPU BPTT references",
          "[gpu_execution][neural_provider][opencl][parity]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; parity not exercised");
        return;
    }
    const auto input = FilledTensor({4, 10, 6}, 0.5f, 0.7f);
    const auto upstream = FilledTensor({4, 10, 16}, 0.3f, 1.3f);
    const char* names[] = {"layer0_grad_W_ih", "layer0_grad_W_hh",
                           "layer0_grad_b_ih", "layer0_grad_b_hh"};
    for (const bool lstm : {true, false}) {
        std::map<std::string, cyxwiz::Tensor> params, grads;
        cyxwiz::Tensor expected_dx;
        if (lstm) {
            cyxwiz::LSTMLayer reference(6, 16);
            cyxwiz::SetNeuralProvidersDisabledForTesting(true);
            cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
            reference.Forward(input);
            params = reference.GetParameters();
            reference.Forward(input);
            expected_dx = reference.Backward(upstream);
            grads = reference.GetParameters();
            cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
            cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        } else {
            cyxwiz::GRULayer reference(6, 16);
            cyxwiz::SetNeuralProvidersDisabledForTesting(true);
            cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
            reference.Forward(input);
            params = reference.GetParameters();
            reference.Forward(input);
            expected_dx = reference.Backward(upstream);
            grads = reference.GetParameters();
            cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
            cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        }
        const size_t gate_width = lstm ? 64 : 48;
        const cyxwiz::Tensor W_ih = params.at("layer0_W_ih"),
                             W_hh = params.at("layer0_W_hh"),
                             b_ih = params.at("layer0_b_ih"),
                             b_hh = params.at("layer0_b_hh");
        cyxwiz::Tensor dx(std::vector<size_t>{4, 10, 6});
        cyxwiz::Tensor dW_ih(std::vector<size_t>{gate_width, 6});
        cyxwiz::Tensor dW_hh(std::vector<size_t>{gate_width, 16});
        cyxwiz::Tensor db_ih(std::vector<size_t>{gate_width});
        cyxwiz::Tensor db_hh(std::vector<size_t>{gate_width});
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input, &upstream};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&dx};
        buffers.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};
        auto request = OpenclRequest(lstm ? cyxwiz::NeuralOp::LstmBackward
                                          : cyxwiz::NeuralOp::GruBackward,
                                     4, 10, 6, 16);
        const auto status = provider->Execute(request, buffers);
        INFO((lstm ? "lstm" : "gru") << " detail: " << status.detail);
        REQUIRE(status.ok);
        const char* tag = lstm ? "opencl lstm_backward" : "opencl gru_backward";
        Compare(dx, expected_dx, tag, 5e-4f);
        const cyxwiz::Tensor* actual[] = {&dW_ih, &dW_hh, &db_ih, &db_hh};
        for (size_t i = 0; i < 4; ++i) {
            Compare(*actual[i], grads.at(names[i]), names[i], 5e-4f);
        }
    }
}

TEST_CASE("Stacked LSTM and GRU layers route through the OpenCL provider on an OpenCL-selected run",
          "[gpu_execution][neural_provider][opencl][parity][stacked]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; routing not exercised");
        return;
    }
    // The harness pins ArrayFire to CPU; select the OpenCL backend for
    // this test so the layers' device-keyed dispatch targets OPENCL.
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
    } guard;
    try {
        guard.previous = af::getActiveBackend();
        guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_OPENCL);
        af::setDevice(0);
        guard.active = true;
    } catch (...) {
        WARN("ArrayFire OpenCL backend cannot be selected in this process; "
             "routing not exercised");
        return;
    }
    REQUIRE(cyxwiz::CaptureCurrentNeuralDeviceTarget().platform ==
            cyxwiz::DeviceType::OPENCL);
    const auto input = FilledTensor({3, 7, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({3, 7, 10}, 0.25f, 1.1f);
    const char* grad_names[] = {
        "layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
        "layer0_grad_b_hh", "layer1_grad_W_ih", "layer1_grad_W_hh",
        "layer1_grad_b_ih", "layer1_grad_b_hh"};
    {
        cyxwiz::LSTMLayer layer(5, 10, 2);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        layer.Forward(input);
        const auto expected = layer.Forward(input);
        const auto expected_h = layer.GetHiddenState();
        const auto expected_c = layer.GetCellState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        const auto actual = layer.Forward(input);
        const auto actual_h = layer.GetHiddenState();
        const auto actual_c = layer.GetCellState();
        const auto actual_dx = layer.Backward(upstream);
        const auto routed = layer.GetParameters();
        Compare(actual, expected, "opencl stacked LSTM forward", 5e-4f);
        Compare(actual_dx, expected_dx, "opencl stacked LSTM dx", 5e-4f);
        for (const char* name : grad_names) {
            Compare(routed.at(name), oracle.at(name), name, 5e-4f);
        }
        REQUIRE(actual_h.Shape() == std::vector<size_t>{2, 3, 10});
        Compare(actual_h, expected_h, "opencl stacked LSTM h_n", 5e-4f);
        Compare(actual_c, expected_c, "opencl stacked LSTM c_n", 5e-4f);
    }
    {
        cyxwiz::GRULayer layer(5, 10, 2);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
        layer.Forward(input);
        const auto expected = layer.Forward(input);
        const auto expected_h = layer.GetHiddenState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        const auto actual = layer.Forward(input);
        const auto actual_h = layer.GetHiddenState();
        const auto actual_dx = layer.Backward(upstream);
        const auto routed = layer.GetParameters();
        Compare(actual, expected, "opencl stacked GRU forward", 5e-4f);
        Compare(actual_dx, expected_dx, "opencl stacked GRU dx", 5e-4f);
        for (const char* name : grad_names) {
            Compare(routed.at(name), oracle.at(name), name, 5e-4f);
        }
        REQUIRE(actual_h.Shape() == std::vector<size_t>{2, 3, 10});
        Compare(actual_h, expected_h, "opencl stacked GRU h_n", 5e-4f);
    }
}

TEST_CASE("OpenCL provider survives repeated training runs and reports its speed vs native CPU",
          "[gpu_execution][neural_provider][opencl][leak]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; repeated-run check not exercised");
        return;
    }
    // OpenCL exposes no portable free-memory query, so this is a
    // repeated-run robustness check (buffers are RAII-released per call);
    // the CUDA tenant carries the byte-exact leak gate.
    auto forward = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 32, 32, 64, 64);
    auto backward = OpenclRequest(cyxwiz::NeuralOp::LstmBackward, 32, 32, 64, 64);
    const auto x = FilledTensor({32, 32, 64}, 0.1f, 0.37f);
    const auto dy = FilledTensor({32, 32, 64}, 0.2f, 0.91f);
    const auto W_ih = FilledTensor({256, 64}, 0.05f, 0.11f);
    const auto W_hh = FilledTensor({256, 64}, 0.04f, 0.13f);
    const auto b_ih = FilledTensor({256}, 0.01f, 0.02f);
    const auto b_hh = FilledTensor({256}, 0.02f, 0.03f);
    cyxwiz::Tensor y(std::vector<size_t>{32, 32, 64});
    cyxwiz::Tensor dx(std::vector<size_t>{32, 32, 64});
    cyxwiz::Tensor dW_ih(std::vector<size_t>{256, 64});
    cyxwiz::Tensor dW_hh(std::vector<size_t>{256, 64});
    cyxwiz::Tensor db_ih(std::vector<size_t>{256});
    cyxwiz::Tensor db_hh(std::vector<size_t>{256});
    std::vector<double> forward_ms;
    for (int i = 0; i < 20; ++i) {
        cyxwiz::NeuralOpBuffers fb;
        fb.inputs = {&x};
        fb.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        fb.outputs = {&y};
        const auto start = std::chrono::steady_clock::now();
        REQUIRE(provider->Execute(forward, fb).ok);
        const auto end = std::chrono::steady_clock::now();
        if (i >= 4) {
            forward_ms.push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
        }
        cyxwiz::NeuralOpBuffers bb;
        bb.inputs = {&x, &dy};
        bb.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        bb.outputs = {&dx};
        bb.gradients = {&dW_ih, &dW_hh, &db_ih, &db_hh};
        REQUIRE(provider->Execute(backward, bb).ok);
    }
    // Native CPU LSTM forward on the same shape for the retention record.
    cyxwiz::LSTMLayer reference(64, 64);
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(x);
    std::vector<double> native_ms;
    for (int i = 0; i < 3; ++i) {
        reference.ResetState();
        const auto start = std::chrono::steady_clock::now();
        reference.Forward(x);
        const auto end = std::chrono::steady_clock::now();
        native_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    std::sort(forward_ms.begin(), forward_ms.end());
    std::sort(native_ms.begin(), native_ms.end());
    const double provider_median = forward_ms[forward_ms.size() / 2];
    const double native_median = native_ms[native_ms.size() / 2];
    WARN("opencl lstm_forward 32x32x64x64: provider "
         << provider_median << " ms vs native CPU " << native_median
         << " ms (" << native_median / provider_median << "x)");
}

#endif // CYXWIZ_HAS_OPENCL_DNN_PROVIDER
