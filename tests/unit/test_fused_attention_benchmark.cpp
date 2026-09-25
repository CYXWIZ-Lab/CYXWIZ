// tofix112 phase 5b, package 6: fused attention vs the ArrayFire path.
// Hidden benchmark (run with "[attention_benchmark]"): one decoder block,
// d_model 256, 8 heads (head_dim 32), batch 4, causal, on ArrayFire CUDA.
// For each context length it reports forward+backward time and the device
// memory the layer holds after forward (what backward needs), for the fused
// provider path and the ArrayFire path, and marks out-of-memory runs.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/transformer.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/tensor.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <arrayfire.h>

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Measurement {
    bool ok = false;
    std::string error;
    double milliseconds = 0.0;
    double held_mb = 0.0;
    bool fused = false;
};

// Device memory in use, after returning ArrayFire's cached free blocks, so
// only arrays that are still referenced (the layer's saved activations and
// the inputs) count.
size_t UsedDeviceBytes() {
    af::sync();
    af::deviceGC();
#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
    size_t free_bytes = 0, total_bytes = 0;
    if (cyxwiz::NvidiaProviderDeviceMemoryForTesting(free_bytes, total_bytes)) return total_bytes - free_bytes;
#endif
    return 0;
}

Measurement Run(size_t seq, bool fused) {
    Measurement result;
    cyxwiz::SetNeuralProvidersDisabledForTesting(!fused);
    try {
        af::deviceGC();
        cyxwiz::TransformerBlockOptions options;
        const cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(
            af::randu(4, static_cast<dim_t>(seq), 256) - 0.5f, {4, seq, 256});
        const cyxwiz::Tensor upstream = cyxwiz::Tensor::FromSemanticArray(
            af::randu(4, static_cast<dim_t>(seq), 256) - 0.5f, {4, seq, 256});
        {
            // Warm-up on a separate layer (kernel compilation, allocator);
            // destroyed so its saved activations do not count below.
            cyxwiz::TransformerDecoderLayer warm(256, 8, 512, 0.0f, true, 0.0f, options);
            warm.SetTraining(true);
            warm.Forward(input);
            warm.Backward(upstream);
            af::sync();
        }
        cyxwiz::TransformerDecoderLayer layer(256, 8, 512, 0.0f, true, 0.0f, options);
        layer.SetTraining(true);
        const size_t before = UsedDeviceBytes();
        auto start = std::chrono::steady_clock::now();
        const cyxwiz::Tensor out = layer.Forward(input);
        af::sync();
        result.milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const size_t after_forward = UsedDeviceBytes();
        start = std::chrono::steady_clock::now();
        layer.Backward(upstream);
        af::sync();
        result.milliseconds +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        result.held_mb = (static_cast<double>(after_forward) - static_cast<double>(before)) / (1024.0 * 1024.0);
        result.ok = true;
        (void)out;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    af::deviceGC();
    return result;
}

}  // namespace

TEST_CASE("Fused attention memory and speed vs ArrayFire", "[.][attention_benchmark]") {
    struct Route {
        af::Backend backend;
        int device;
        std::vector<size_t> contexts;
    };
    std::vector<Route> routes = {
        {AF_BACKEND_CUDA, 0, {256, 512, 1024, 2048, 4096, 8192}},
        {AF_BACKEND_OPENCL, 0, {256, 512, 1024, 2048, 4096}},
        {AF_BACKEND_OPENCL, 1, {256, 512, 1024, 2048}},
    };
    // oneAPI is opt-in (selecting an unqualified device can crash inside
    // ArrayFire): CYXWIZ_ONEAPI_TEST_DEVICE=<index>, as in the unit tests.
    if (const char* oneapi = std::getenv("CYXWIZ_ONEAPI_TEST_DEVICE"); oneapi && *oneapi) {
        routes.push_back({AF_BACKEND_ONEAPI, std::atoi(oneapi), {256, 512, 1024, 2048}});
    }
    for (const auto& route : routes) {
        try {
            af::setBackend(route.backend);
            if (route.device >= af::getDeviceCount()) continue;
            af::setDevice(route.device);
        } catch (...) {
            continue;
        }
        char name[64] = "?";
        if (route.backend != AF_BACKEND_ONEAPI) af::deviceInfo(name, nullptr, nullptr, nullptr);  // unsupported there
        std::printf("\n%s device %d (%s); held MB is exact on CUDA only\n",
                    route.backend == AF_BACKEND_CUDA ? "CUDA" : route.backend == AF_BACKEND_ONEAPI ? "oneAPI" : "OpenCL",
                    route.device, name);
        std::printf("%-8s | %-28s | %-28s\n", "context", "fused: ms / held MB", "ArrayFire: ms / held MB");
        for (const size_t seq : route.contexts) {
            const Measurement fused = Run(seq, true);
            const Measurement plain = Run(seq, false);
            auto cell = [](const Measurement& m) {
                char text[64];
                if (m.ok) std::snprintf(text, sizeof(text), "%9.1f / %8.1f", m.milliseconds, m.held_mb);
                else std::snprintf(text, sizeof(text), "failed (%.18s)", m.error.c_str());
                return std::string(text);
            };
            std::printf("%-8zu | %-28s | %-28s\n", seq, cell(fused).c_str(), cell(plain).c_str());
            std::fflush(stdout);
        }
    }
    af::setBackend(AF_BACKEND_CPU);
}

// Kernel-level timing for tuning: fused forward and backward separately
// (median of 5), CUDA, batch 4, 8 heads, causal.
TEST_CASE("Fused attention kernel timing", "[.][attention_tuning]") {
    try {
        af::setBackend(AF_BACKEND_CUDA);
    } catch (...) {
        WARN("ArrayFire CUDA backend not available");
        return;
    }
    std::printf("\n%-10s %-8s | %-12s %-12s\n", "head_dim", "context", "forward ms", "backward ms");
    for (const int head_dim : {32, 64}) {
        for (const size_t seq : {1024u, 4096u}) {
            cyxwiz::NeuralOpRequest request;
            request.op = cyxwiz::NeuralOp::AttentionForward;
            request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
            request.device_resident = true;
            request.batch = 4;
            request.heads = 8;
            request.kv_heads = 8;
            request.seq = seq;
            request.kv_seq = seq;
            request.head_dim = head_dim;
            request.causal = true;
            request.softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
            if (!provider) {
                WARN("no provider");
                return;
            }
            const size_t d = head_dim;
            const cyxwiz::Tensor q = cyxwiz::Tensor::FromSemanticArray(af::randu(head_dim, seq, 4, 8) - 0.5f, {d, seq, 4, 8});
            const cyxwiz::Tensor k = cyxwiz::Tensor::FromSemanticArray(af::randu(head_dim, seq, 4, 8) - 0.5f, {d, seq, 4, 8});
            const cyxwiz::Tensor v = cyxwiz::Tensor::FromSemanticArray(af::randu(head_dim, seq, 4, 8) - 0.5f, {d, seq, 4, 8});
            const cyxwiz::Tensor go = cyxwiz::Tensor::FromSemanticArray(af::randu(head_dim, seq, 4, 8) - 0.5f, {d, seq, 4, 8});
            std::vector<double> fwd, bwd;
            for (int rep = 0; rep < 6; ++rep) {
                std::vector<cyxwiz::Tensor> out, grads;
                request.op = cyxwiz::NeuralOp::AttentionForward;
                af::sync();
                auto start = std::chrono::steady_clock::now();
                REQUIRE(cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&q, &k, &v}, {{d, seq, 4, 8}, {seq, 4, 8}}, out).ok);
                af::sync();
                const double f_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                request.op = cyxwiz::NeuralOp::AttentionBackward;
                start = std::chrono::steady_clock::now();
                REQUIRE(cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&q, &k, &v, &out[0], &go, &out[1]},
                                                        {{d, seq, 4, 8}, {d, seq, 4, 8}, {d, seq, 4, 8}, {seq, 4, 8}}, grads).ok);
                af::sync();
                const double b_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if (rep > 0) {  // first run compiles
                    fwd.push_back(f_ms);
                    bwd.push_back(b_ms);
                }
            }
            std::sort(fwd.begin(), fwd.end());
            std::sort(bwd.begin(), bwd.end());
            std::printf("%-10d %-8zu | %-12.2f %-12.2f\n", head_dim, seq, fwd[2], bwd[2]);
            std::fflush(stdout);
        }
    }
    af::setBackend(AF_BACKEND_CPU);
}
