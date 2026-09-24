// tofix67 slice 7: exact-shape benchmark of the recurrent staged ArrayFire
// plan (recurrent_timestep_materialization_v1) vs the native CPU recurrent
// path. Produces the evidence for the native-provider graduation decision
// and for any eval-boundary change to the staged plan.
//
// Usage: test_recurrent_staged_benchmark cpu|cuda|opencl output.json

#include "cyxwiz/layers/recurrent.h"
#include "cyxwiz/device.h"
#include "cyxwiz/neural_provider.h"
#include "cyxwiz/recurrent_cuda_placement.h"
#include "cyxwiz/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
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

struct BenchShape {
    size_t batch;
    size_t seq;
    size_t input;
    int hidden;
};

cyxwiz::Tensor MakeInput(const BenchShape& shape) {
    cyxwiz::Tensor input(
        std::vector<size_t>{shape.batch, shape.seq, shape.input});
    float* data = input.Data<float>();
    for (size_t i = 0; i < input.NumElements(); ++i) {
        data[i] = 0.2f * std::sin(0.31f * static_cast<float>(i));
    }
    return input;
}

double MedianMs(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * (values[mid - 1] + values[mid])
        : values[mid];
}

struct PathTiming {
    double forward_ms = 0.0;
    double backward_ms = 0.0;
};

PathTiming MeasurePath(const BenchShape& shape,
                       const cyxwiz::Tensor& input,
                       bool force_native,
                       int warmup_runs,
                       int measured_runs) {
    // Staged/native legs measure the ArrayFire and CPU paths; the neural
    // provider is measured by its own leg and must not hijack these.
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    if (force_native) {
        cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    }
    std::vector<double> forward_ms;
    std::vector<double> backward_ms;
    for (int run = 0; run < warmup_runs + measured_runs; ++run) {
        cyxwiz::LSTMLayer lstm(static_cast<int>(shape.input), shape.hidden);
        lstm.ResetState();
        const auto f0 = std::chrono::steady_clock::now();
        cyxwiz::Tensor output = lstm.Forward(input);
        const auto f1 = std::chrono::steady_clock::now();
        cyxwiz::Tensor grad_output(output.Shape());
        float* g = grad_output.Data<float>();
        for (size_t i = 0; i < grad_output.NumElements(); ++i) {
            g[i] = 1.0f;
        }
        const auto b0 = std::chrono::steady_clock::now();
        cyxwiz::Tensor grad_input = lstm.Backward(grad_output);
        const auto b1 = std::chrono::steady_clock::now();
        Check(grad_input.NumElements() == input.NumElements(),
              "backward gradient shape must match input");
        if (run >= warmup_runs) {
            forward_ms.push_back(
                std::chrono::duration<double, std::milli>(f1 - f0).count());
            backward_ms.push_back(
                std::chrono::duration<double, std::milli>(b1 - b0).count());
        }
    }
    if (force_native) {
        cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    PathTiming timing;
    timing.forward_ms = MedianMs(forward_ms);
    timing.backward_ms = MedianMs(backward_ms);
    return timing;
}

// tofix68 phase 5: native-provider leg. Forward-only (the provider's P1
// contract is inference), timed INCLUDING host<->device boundary copies —
// the v1 contract a caller actually experiences. Returns median ms, or a
// negative value when no provider supports the tuple.
// OpenCL provider leg device: CYXWIZ_OPENCL_BENCH_DEVICE (default 0, the
// first enumerated OpenCL GPU); 1 targets the second GPU (Intel UHD 630 on
// the dev box) for the AMD/Intel platform benchmark gate.
int OpenclBenchDeviceIndex() {
    static const int index = [] {
        const char* value = std::getenv("CYXWIZ_OPENCL_BENCH_DEVICE");
        if (value == nullptr || value[0] == '\0') {
            return 0;
        }
        return std::atoi(value);
    }();
    return index;
}

double MeasureProviderForward(const BenchShape& shape,
                              const cyxwiz::Tensor& input,
                              int warmup_runs,
                              int measured_runs,
                              std::string& provider_version,
                              cyxwiz::DeviceType platform) {
    cyxwiz::NeuralOpRequest request;
    request.target = {platform,
                      platform == cyxwiz::DeviceType::OPENCL
                          ? OpenclBenchDeviceIndex()
                          : 0};
    request.op = cyxwiz::NeuralOp::LstmForward;
    request.training = false;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = shape.batch;
    request.seq = shape.seq;
    request.input = shape.input;
    request.hidden = static_cast<size_t>(shape.hidden);
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    auto provider = registry.FindSupporting(request);
    if (!provider) {
        // A tenant that serves the platform but declines this tuple (e.g.
        // the OpenCL retention floor) is reported by reason, so the row
        // is not mistaken for "no provider on this machine".
        for (const auto& serving : registry.ListServing(request.target)) {
            const auto capability = serving->QueryCapability(request);
            provider_version = std::string("declined:") +
                               cyxwiz::BackendFallbackReasonName(capability.reason);
        }
        return -1.0;
    }
    provider_version = provider->Version();

    // Real weights from the CPU reference; one forced-native forward
    // settles lazily initialized weight storage before GetParameters.
    cyxwiz::LSTMLayer reference(static_cast<int>(shape.input), shape.hidden);
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    reference.Forward(input);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    auto params = reference.GetParameters();
    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    cyxwiz::Tensor output(std::vector<size_t>{
        shape.batch, shape.seq, static_cast<size_t>(shape.hidden)});

    std::vector<double> forward_ms;
    for (int run = 0; run < warmup_runs + measured_runs; ++run) {
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&output};
        const auto start = std::chrono::steady_clock::now();
        const auto status = provider->Execute(request, buffers);
        const auto end = std::chrono::steady_clock::now();
        Check(status.ok, "provider lstm_forward failed during benchmark: " +
                             status.detail);
        if (run >= warmup_runs) {
            forward_ms.push_back(
                std::chrono::duration<double, std::milli>(end - start)
                    .count());
        }
    }
    return MedianMs(forward_ms);
}

// tofix68 P3: GRU leg — native CPU GRU forward vs provider gru_forward on
// identical weights, forward-only, provider timed INCLUDING host<->device
// boundary copies. provider_ms is negative when no provider supports the
// tuple.
void MeasureGruForwardLeg(const BenchShape& shape,
                          const cyxwiz::Tensor& input,
                          int warmup_runs,
                          int measured_runs,
                          double& native_ms,
                          double& provider_ms) {
    cyxwiz::GRULayer reference(static_cast<int>(shape.input), shape.hidden);
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    cyxwiz::SetForceNativeRecurrentForwardForTesting(true);
    std::vector<double> native;
    for (int run = 0; run < warmup_runs + measured_runs; ++run) {
        reference.ResetState();
        const auto start = std::chrono::steady_clock::now();
        reference.Forward(input);
        const auto end = std::chrono::steady_clock::now();
        if (run >= warmup_runs) {
            native.push_back(
                std::chrono::duration<double, std::milli>(end - start)
                    .count());
        }
    }
    cyxwiz::SetForceNativeRecurrentForwardForTesting(false);
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    native_ms = MedianMs(native);

    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::GruForward;
    request.training = false;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = shape.batch;
    request.seq = shape.seq;
    request.input = shape.input;
    request.hidden = static_cast<size_t>(shape.hidden);
    auto provider =
        cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    if (!provider) {
        provider_ms = -1.0;
        return;
    }
    auto params = reference.GetParameters();
    const cyxwiz::Tensor W_ih = params.at("layer0_W_ih");
    const cyxwiz::Tensor W_hh = params.at("layer0_W_hh");
    const cyxwiz::Tensor b_ih = params.at("layer0_b_ih");
    const cyxwiz::Tensor b_hh = params.at("layer0_b_hh");
    cyxwiz::Tensor output(std::vector<size_t>{
        shape.batch, shape.seq, static_cast<size_t>(shape.hidden)});
    std::vector<double> forward_ms;
    for (int run = 0; run < warmup_runs + measured_runs; ++run) {
        cyxwiz::NeuralOpBuffers buffers;
        buffers.inputs = {&input};
        buffers.weights = {&W_ih, &W_hh, &b_ih, &b_hh};
        buffers.outputs = {&output};
        const auto start = std::chrono::steady_clock::now();
        const auto status = provider->Execute(request, buffers);
        const auto end = std::chrono::steady_clock::now();
        Check(status.ok, "provider gru_forward failed during benchmark: " +
                             status.detail);
        if (run >= warmup_runs) {
            forward_ms.push_back(
                std::chrono::duration<double, std::milli>(end - start)
                    .count());
        }
    }
    provider_ms = MedianMs(forward_ms);
}

} // namespace

int main(int argc, char** argv) {
    Check(argc == 3,
          "Usage: test_recurrent_staged_benchmark cpu|cuda|opencl output.json");
    const std::string backend = argv[1];
    Check(backend == "cpu" || backend == "cuda" || backend == "opencl",
          "backend name");
    auto activation =
        cyxwiz::Device(backend == "cpu" ? cyxwiz::DeviceType::CPU
                       : backend == "cuda" ? cyxwiz::DeviceType::CUDA
                                           : cyxwiz::DeviceType::OPENCL,
                       0)
            .ActivateExact(true);
    Check(activation.success && activation.execution_validated,
          activation.message);

    const std::vector<BenchShape> shapes = {
        {32, 16, 32, 8},     // CUDA-eligible by estimator (3000+8*56=3448)
        {32, 16, 32, 16},    // CUDA-eligible boundary-ish (3896)
        {32, 32, 64, 64},    // policy-CPU on CUDA (6584)
        {32, 64, 128, 128},  // policy-CPU on CUDA
        {64, 64, 128, 256},  // policy-CPU on CUDA, training-realistic
    };
    constexpr int kWarmup = 2;
    constexpr int kMeasured = 5;

    std::ofstream json(argv[2], std::ios::trunc);
    Check(static_cast<bool>(json), "cannot open output json");
    json << "{\n  \"backend\": \"" << backend << "\",\n"
         << "  \"staged_plan\": \""
         << cyxwiz::RecurrentStagedArrayFirePlanName << "\",\n"
         << "  \"shapes\": [\n";

    std::cout << "backend=" << backend
              << " plan=" << cyxwiz::RecurrentStagedArrayFirePlanName << "\n";
    for (size_t i = 0; i < shapes.size(); ++i) {
        const auto& shape = shapes[i];
        cyxwiz::RecurrentCudaPlacementRequest request;
        request.kind = cyxwiz::RecurrentLayerKind::LSTM;
        request.batch_size = shape.batch;
        request.seq_len = shape.seq;
        request.input_size = shape.input;
        request.hidden_size = static_cast<size_t>(shape.hidden);
        request.num_layers = 1;
        const auto decision =
            cyxwiz::EvaluateRecurrentCudaPlacement(request);
        // On the CUDA backend, shapes the estimator rejects run native even
        // on the "staged" leg — the staged measurement is only meaningful
        // when the policy allows the ArrayFire path (always true on
        // non-CUDA ArrayFire backends).
        const bool staged_is_arrayfire =
            backend != "cuda" || decision.should_attempt_arrayfire_cuda;

        const auto input = MakeInput(shape);
        const auto staged =
            MeasurePath(shape, input, false, kWarmup, kMeasured);
        const auto native =
            MeasurePath(shape, input, true, kWarmup, kMeasured);
        std::string provider_version;
        const double provider_fwd_ms = MeasureProviderForward(
            shape, input, kWarmup, kMeasured, provider_version,
            cyxwiz::DeviceType::CUDA);
        std::string opencl_provider_version;
        const double opencl_provider_fwd_ms = MeasureProviderForward(
            shape, input, kWarmup, kMeasured, opencl_provider_version,
            cyxwiz::DeviceType::OPENCL);
        double gru_native_fwd_ms = 0.0;
        double gru_provider_fwd_ms = -1.0;
        MeasureGruForwardLeg(shape, input, kWarmup, kMeasured,
                             gru_native_fwd_ms, gru_provider_fwd_ms);

        const double forward_speedup =
            staged.forward_ms > 0.0 ? native.forward_ms / staged.forward_ms
                                    : 0.0;
        std::cout << "shape batch=" << shape.batch << " seq=" << shape.seq
                  << " input=" << shape.input << " hidden=" << shape.hidden
                  << " est_bytes="
                  << decision.estimated_formal_parameter_bytes
                  << " staged_is_arrayfire="
                  << (staged_is_arrayfire ? "true" : "false")
                  << " staged_fwd_ms=" << staged.forward_ms
                  << " native_fwd_ms=" << native.forward_ms
                  << " staged_bwd_ms=" << staged.backward_ms
                  << " native_bwd_ms=" << native.backward_ms
                  << " fwd_speedup_native_over_staged="
                  << (forward_speedup > 0.0 ? 1.0 / forward_speedup : 0.0)
                  << " provider_fwd_ms=" << provider_fwd_ms
                  << (provider_fwd_ms > 0.0
                          ? " provider_speedup_vs_native=" +
                                std::to_string(native.forward_ms /
                                               provider_fwd_ms)
                          : " provider=" + (provider_version.empty()
                                                ? std::string("unavailable")
                                                : provider_version))
                  << " opencl_device=" << OpenclBenchDeviceIndex()
                  << " opencl_provider_fwd_ms=" << opencl_provider_fwd_ms
                  << (opencl_provider_fwd_ms > 0.0
                          ? " opencl_provider_speedup_vs_native=" +
                                std::to_string(native.forward_ms /
                                               opencl_provider_fwd_ms)
                          : " opencl_provider=" +
                                (opencl_provider_version.empty()
                                     ? std::string("unavailable")
                                     : opencl_provider_version))
                  << " gru_native_fwd_ms=" << gru_native_fwd_ms
                  << " gru_provider_fwd_ms=" << gru_provider_fwd_ms
                  << (gru_provider_fwd_ms > 0.0
                          ? " gru_provider_speedup_vs_native=" +
                                std::to_string(gru_native_fwd_ms /
                                               gru_provider_fwd_ms)
                          : std::string(" gru_provider=unavailable"))
                  << "\n";

        json << "    {\"batch\": " << shape.batch
             << ", \"seq\": " << shape.seq
             << ", \"input\": " << shape.input
             << ", \"hidden\": " << shape.hidden
             << ", \"estimated_formal_parameter_bytes\": "
             << decision.estimated_formal_parameter_bytes
             << ", \"cuda_policy_allows\": "
             << (decision.should_attempt_arrayfire_cuda ? "true" : "false")
             << ", \"staged_is_arrayfire\": "
             << (staged_is_arrayfire ? "true" : "false")
             << ", \"staged_forward_ms\": " << staged.forward_ms
             << ", \"staged_backward_ms\": " << staged.backward_ms
             << ", \"native_forward_ms\": " << native.forward_ms
             << ", \"native_backward_ms\": " << native.backward_ms
             << ", \"provider_forward_ms\": " << provider_fwd_ms
             << ", \"opencl_device\": " << OpenclBenchDeviceIndex()
             << ", \"opencl_provider_forward_ms\": " << opencl_provider_fwd_ms
             << ", \"opencl_provider\": \"" << opencl_provider_version << "\""
             << ", \"gru_native_forward_ms\": " << gru_native_fwd_ms
             << ", \"gru_provider_forward_ms\": " << gru_provider_fwd_ms
             << ", \"provider\": \"" << provider_version << "\"}"
             << (i + 1 < shapes.size() ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    std::cout << "Recurrent staged benchmark complete\n";
    return 0;
}
