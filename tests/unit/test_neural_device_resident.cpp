// tofix112 phase 5b, package 1: device-resident provider execution.
// A provider runs device_probe (y = 2x) directly on ArrayFire's device memory
// and queue: correct values, and no host synchronization during the call.
// Each backend section warns and returns when its backend or tenant is absent,
// so the suite stays truthful on machines without the hardware.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/attention.h>
#include <cyxwiz/layers/transformer.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/tensor.h>
#include "algorithms/arrayfire_backend_utils.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <arrayfire.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int g_host_syncs = 0;
void CountHostSync(const cyxwiz::ArrayFireHostSyncEvent&) { ++g_host_syncs; }

struct BackendGuard {
    af::Backend previous = AF_BACKEND_CPU;
    int previous_device = 0;
    bool active = false;
    ~BackendGuard() {
        if (!active) return;
        try {
            af::setBackend(previous);
            af::setDevice(previous_device);
        } catch (...) {
        }
    }
};

void RunProbe(af::Backend backend, int device, const char* label) {
    BackendGuard guard;
    try {
        guard.previous = af::getActiveBackend();
        guard.previous_device = af::getDevice();
        af::setBackend(backend);
        af::setDevice(device);
        guard.active = true;
    } catch (...) {
        WARN(label << ": ArrayFire backend not available in this process; device probe not exercised");
        return;
    }
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::DeviceProbe;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    request.elements = 100003;  // not a multiple of the work-group size
    auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    if (!provider) {
        WARN(label << ": no provider serves device-resident execution here; not exercised");
        return;
    }
    const af::array values = af::randu(static_cast<dim_t>(request.elements)) - 0.5f;
    const cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(values, {request.elements});

    std::vector<cyxwiz::Tensor> outputs;
    g_host_syncs = 0;
    cyxwiz::NeuralOpStatus status;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(&CountHostSync);
        status = cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&input}, {{request.elements}}, outputs);
    }
    INFO(label << " provider " << provider->ProviderId() << ": " << status.detail);
    REQUIRE(status.ok);
    CHECK(g_host_syncs == 0);
    REQUIRE(outputs.size() == 1);
    REQUIRE(outputs[0].Shape() == std::vector<size_t>{request.elements});

    // Ordered after the provider kernel on the same queue; compared on device.
    const af::array expected = 2.0f * values;
    const float max_error = af::max<float>(af::abs(outputs[0].GetSemanticArray() - expected));
    CHECK(max_error == 0.0f);

    // The output keeps working as an ordinary ArrayFire array afterwards.
    const float sum = af::sum<float>(outputs[0].GetSemanticArray() - 2.0f * values);
    CHECK(sum == 0.0f);

    // A v1 (host-copy) call with a device-resident request is refused.
    cyxwiz::NeuralOpBuffers host_buffers;
    CHECK_FALSE(provider->Execute(request, host_buffers).ok);
}

}  // namespace

TEST_CASE("Device-resident provider execution on ArrayFire CUDA memory",
          "[neural_provider][device_resident][cuda]") {
    RunProbe(AF_BACKEND_CUDA, 0, "cuda");
}

TEST_CASE("Device-resident provider execution on ArrayFire OpenCL memory",
          "[neural_provider][device_resident][opencl]") {
    RunProbe(AF_BACKEND_OPENCL, 0, "opencl device 0");
    int count = 0;
    try {
        af::Backend previous = af::getActiveBackend();
        af::setBackend(AF_BACKEND_OPENCL);
        count = af::getDeviceCount();
        af::setBackend(previous);
    } catch (...) {
    }
    if (count > 1) {
        RunProbe(AF_BACKEND_OPENCL, 1, "opencl device 1");
    }
}

namespace {

// True when a provider actually runs device-resident work on the active
// device (the OpenCL tenant declines non-GPU OpenCL devices at execution).
bool ActiveDeviceServed() {
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::DeviceProbe;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    request.elements = 4;
    auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    if (!provider) return false;
    const cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(af::constant(1.0f, 4), {4});
    std::vector<cyxwiz::Tensor> outputs;
    return cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&input}, {{4}}, outputs).ok;
}

struct AttentionCase {
    const char* label;
    int batch, heads, kv_heads, sq, sk, head_dim, offset;
    bool causal;
    int window;
    float softcap;
    bool alibi;
};

// Materialized ArrayFire reference in the provider layouts:
// Q [D, Sq, B, H], K/V [D, Sk, B, KVH] -> O [D, Sq, B, H], LSE [Sq, B, H].
void ReferenceAttention(const AttentionCase& c, const af::array& Q, const af::array& K, const af::array& V,
                        const af::array& slopes, float scale, af::array& O, af::array& LSE) {
    const int groups = c.heads / c.kv_heads;
    af::array q = af::reorder(Q, 1, 0, 2, 3);  // [Sq, D, B, H]
    af::array k = af::reorder(K, 1, 0, 2, 3);  // [Sk, D, B, KVH]
    af::array v = af::reorder(V, 1, 0, 2, 3);
    if (groups > 1) {  // head h uses kv head h / groups
        af::array idx = af::floor(af::range(af::dim4(c.heads)) / groups).as(s32);
        k = af::lookup(k, idx, 3);
        v = af::lookup(v, idx, 3);
    }
    af::array scores = af::matmul(q, k, AF_MAT_NONE, AF_MAT_TRANS) * scale;  // [Sq, Sk, B, H]
    if (c.softcap > 0.0f) scores = c.softcap * af::tanh(scores / c.softcap);
    const af::dim4 grid(c.sq, c.sk);
    const af::array qpos = af::range(grid, 0, f32) + static_cast<float>(c.offset);
    const af::array kpos = af::range(grid, 1, f32);
    if (c.alibi) {
        scores = scores + af::tile(kpos - qpos, af::dim4(1, 1, c.batch, c.heads)) *
                              af::tile(af::moddims(slopes, af::dim4(1, 1, 1, c.heads)), af::dim4(c.sq, c.sk, c.batch, 1));
    }
    af::array hidden = af::constant(0, grid, b8);
    if (c.causal) hidden = hidden || (kpos > qpos);
    if (c.window > 0) hidden = hidden || (qpos - kpos >= static_cast<float>(c.window));
    scores = af::select(af::tile(hidden, af::dim4(1, 1, c.batch, c.heads)), -INFINITY, scores);
    const af::array row_max = af::max(scores, 1);
    const af::array e = af::exp(scores - af::tile(row_max, af::dim4(1, c.sk)));
    const af::array sum = af::sum(e, 1);
    const af::array p = e / af::tile(sum, af::dim4(1, c.sk));
    O = af::reorder(af::matmul(p, v), 1, 0, 2, 3);                       // [D, Sq, B, H]
    LSE = af::moddims(row_max + af::log(sum), af::dim4(c.sq, c.batch, c.heads));
}

void RunAttentionForward(const AttentionCase& c) {
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::AttentionForward;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    request.batch = c.batch;
    request.heads = c.heads;
    request.kv_heads = c.kv_heads;
    request.seq = c.sq;
    request.kv_seq = c.sk;
    request.head_dim = c.head_dim;
    request.query_offset = c.offset;
    request.causal = c.causal;
    request.sliding_window = c.window;
    request.logit_softcap = c.softcap;
    request.softmax_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    request.position_strategy = c.alibi ? cyxwiz::NeuralPositionStrategy::Alibi : cyxwiz::NeuralPositionStrategy::None;
    auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    REQUIRE(provider);
    af::setSeed(52);
    const af::array Q = af::randu(c.head_dim, c.sq, c.batch, c.heads) - 0.5f;
    const af::array K = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array V = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array slopes = af::randu(c.heads) * 0.5f;
    const cyxwiz::Tensor q = cyxwiz::Tensor::FromSemanticArray(Q, {size_t(c.head_dim), size_t(c.sq), size_t(c.batch), size_t(c.heads)});
    const cyxwiz::Tensor k = cyxwiz::Tensor::FromSemanticArray(K, {size_t(c.head_dim), size_t(c.sk), size_t(c.batch), size_t(c.kv_heads)});
    const cyxwiz::Tensor v = cyxwiz::Tensor::FromSemanticArray(V, {size_t(c.head_dim), size_t(c.sk), size_t(c.batch), size_t(c.kv_heads)});
    const cyxwiz::Tensor s = cyxwiz::Tensor::FromSemanticArray(slopes, {size_t(c.heads)});
    std::vector<const cyxwiz::Tensor*> inputs{&q, &k, &v};
    if (c.alibi) inputs.push_back(&s);
    std::vector<cyxwiz::Tensor> outputs;
    g_host_syncs = 0;
    cyxwiz::NeuralOpStatus status;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(&CountHostSync);
        status = cyxwiz::ExecuteNeuralOpOnDevice(
            *provider, request, inputs,
            {{size_t(c.head_dim), size_t(c.sq), size_t(c.batch), size_t(c.heads)},
             {size_t(c.sq), size_t(c.batch), size_t(c.heads)}},
            outputs);
    }
    INFO(c.label << ": " << status.detail);
    REQUIRE(status.ok);
    CHECK(g_host_syncs == 0);
    af::array O, LSE;
    ReferenceAttention(c, Q, K, V, slopes, request.softmax_scale, O, LSE);
    const float o_error = af::max<float>(af::abs(outputs[0].GetSemanticArray() - O));
    const float lse_error = af::max<float>(af::abs(outputs[1].GetSemanticArray() - LSE));
    INFO(c.label << ": max |O - ref| = " << o_error << ", max |LSE - ref| = " << lse_error);
    CHECK(o_error < 2e-5f);
    CHECK(lse_error < 2e-5f);
}

// Analytic reference gradients from the materialized softmax (same math as
// MultiHeadAttentionLayer's ArrayFire backward, which is PyTorch-checked).
void ReferenceAttentionBackward(const AttentionCase& c, const af::array& Q, const af::array& K, const af::array& V,
                                const af::array& dO, const af::array& slopes, float scale, af::array& dQ,
                                af::array& dK, af::array& dV) {
    const int groups = c.heads / c.kv_heads;
    af::array q = af::reorder(Q, 1, 0, 2, 3);
    af::array k = af::reorder(K, 1, 0, 2, 3);
    af::array v = af::reorder(V, 1, 0, 2, 3);
    const af::array go = af::reorder(dO, 1, 0, 2, 3);  // [Sq, D, B, H]
    if (groups > 1) {
        af::array idx = af::floor(af::range(af::dim4(c.heads)) / groups).as(s32);
        k = af::lookup(k, idx, 3);
        v = af::lookup(v, idx, 3);
    }
    const af::array raw = af::matmul(q, k, AF_MAT_NONE, AF_MAT_TRANS) * scale;
    af::array t = af::constant(0.0f, raw.dims());
    af::array scores = raw;
    if (c.softcap > 0.0f) {
        t = af::tanh(raw / c.softcap);
        scores = c.softcap * t;
    }
    const af::dim4 grid(c.sq, c.sk);
    const af::array qpos = af::range(grid, 0, f32) + static_cast<float>(c.offset);
    const af::array kpos = af::range(grid, 1, f32);
    if (c.alibi) {
        scores = scores + af::tile(kpos - qpos, af::dim4(1, 1, c.batch, c.heads)) *
                              af::tile(af::moddims(slopes, af::dim4(1, 1, 1, c.heads)), af::dim4(c.sq, c.sk, c.batch, 1));
    }
    af::array hidden = af::constant(0, grid, b8);
    if (c.causal) hidden = hidden || (kpos > qpos);
    if (c.window > 0) hidden = hidden || (qpos - kpos >= static_cast<float>(c.window));
    scores = af::select(af::tile(hidden, af::dim4(1, 1, c.batch, c.heads)), -INFINITY, scores);
    const af::array e = af::exp(scores - af::tile(af::max(scores, 1), af::dim4(1, c.sk)));
    const af::array p = e / af::tile(af::sum(e, 1), af::dim4(1, c.sk));
    const af::array dp = af::matmul(go, v, AF_MAT_NONE, AF_MAT_TRANS);            // [Sq, Sk, B, H]
    af::array ds = p * (dp - af::tile(af::sum(p * dp, 1), af::dim4(1, c.sk)));
    if (c.softcap > 0.0f) ds = ds * (1.0f - t * t);
    ds = ds * scale;
    af::array dq = af::matmul(ds, k);                                             // [Sq, D, B, H]
    af::array dk = af::matmul(ds, q, AF_MAT_TRANS);                               // [Sk, D, B, H]
    af::array dv = af::matmul(p, go, AF_MAT_TRANS);
    if (groups > 1) {  // sum the query heads of each kv head
        dk = af::sum(af::moddims(dk, af::dim4(c.sk * c.head_dim * c.batch, groups, c.kv_heads)), 1);
        dv = af::sum(af::moddims(dv, af::dim4(c.sk * c.head_dim * c.batch, groups, c.kv_heads)), 1);
        dk = af::moddims(dk, af::dim4(c.sk, c.head_dim, c.batch, c.kv_heads));
        dv = af::moddims(dv, af::dim4(c.sk, c.head_dim, c.batch, c.kv_heads));
    }
    dQ = af::reorder(dq, 1, 0, 2, 3);
    dK = af::reorder(dk, 1, 0, 2, 3);
    dV = af::reorder(dv, 1, 0, 2, 3);
}

void RunAttentionBackward(const AttentionCase& c) {
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::AttentionForward;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    request.batch = c.batch;
    request.heads = c.heads;
    request.kv_heads = c.kv_heads;
    request.seq = c.sq;
    request.kv_seq = c.sk;
    request.head_dim = c.head_dim;
    request.query_offset = c.offset;
    request.causal = c.causal;
    request.sliding_window = c.window;
    request.logit_softcap = c.softcap;
    request.softmax_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    request.position_strategy = c.alibi ? cyxwiz::NeuralPositionStrategy::Alibi : cyxwiz::NeuralPositionStrategy::None;
    auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    REQUIRE(provider);
    af::setSeed(97);
    const size_t d = c.head_dim, sq = c.sq, sk = c.sk, b = c.batch, hq = c.heads, hk = c.kv_heads;
    const af::array Q = af::randu(c.head_dim, c.sq, c.batch, c.heads) - 0.5f;
    const af::array K = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array V = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array dO = af::randu(c.head_dim, c.sq, c.batch, c.heads) - 0.5f;
    const af::array slopes = af::randu(c.heads) * 0.5f;
    const cyxwiz::Tensor q = cyxwiz::Tensor::FromSemanticArray(Q, {d, sq, b, hq});
    const cyxwiz::Tensor k = cyxwiz::Tensor::FromSemanticArray(K, {d, sk, b, hk});
    const cyxwiz::Tensor v = cyxwiz::Tensor::FromSemanticArray(V, {d, sk, b, hk});
    const cyxwiz::Tensor go = cyxwiz::Tensor::FromSemanticArray(dO, {d, sq, b, hq});
    const cyxwiz::Tensor s = cyxwiz::Tensor::FromSemanticArray(slopes, {hq});
    std::vector<const cyxwiz::Tensor*> fwd_inputs{&q, &k, &v};
    if (c.alibi) fwd_inputs.push_back(&s);
    std::vector<cyxwiz::Tensor> forward;
    REQUIRE(cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, fwd_inputs, {{d, sq, b, hq}, {sq, b, hq}}, forward).ok);

    request.op = cyxwiz::NeuralOp::AttentionBackward;
    std::vector<const cyxwiz::Tensor*> inputs{&q, &k, &v, &forward[0], &go, &forward[1]};
    if (c.alibi) inputs.push_back(&s);
    std::vector<cyxwiz::Tensor> grads;
    g_host_syncs = 0;
    cyxwiz::NeuralOpStatus status;
    {
        const cyxwiz::ScopedArrayFireHostSyncObserver observer(&CountHostSync);
        status = cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, inputs,
                                                 {{d, sq, b, hq}, {d, sk, b, hk}, {d, sk, b, hk}, {sq, b, hq}}, grads);
    }
    INFO(c.label << ": " << status.detail);
    REQUIRE(status.ok);
    CHECK(g_host_syncs == 0);
    af::array dQ, dK, dV;
    ReferenceAttentionBackward(c, Q, K, V, dO, slopes, request.softmax_scale, dQ, dK, dV);
    const float eq = af::max<float>(af::abs(grads[0].GetSemanticArray() - dQ));
    const float ek = af::max<float>(af::abs(grads[1].GetSemanticArray() - dK));
    const float ev = af::max<float>(af::abs(grads[2].GetSemanticArray() - dV));
    INFO(c.label << ": max |dQ| err " << eq << ", |dK| err " << ek << ", |dV| err " << ev);
    CHECK(eq < 5e-5f);
    CHECK(ek < 5e-5f);
    CHECK(ev < 5e-5f);
}

}  // namespace

TEST_CASE("Fused attention forward matches the materialized reference on CUDA",
          "[neural_provider][device_resident][attention][cuda]") {
    BackendGuard guard;
    try {
        guard.previous = af::getActiveBackend();
        guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_CUDA);
        guard.active = true;
    } catch (...) {
        WARN("ArrayFire CUDA backend not available; fused attention not exercised");
        return;
    }
    cyxwiz::NeuralOpRequest probe;
    probe.op = cyxwiz::NeuralOp::AttentionForward;
    probe.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    if (cyxwiz::NeuralProviderRegistry::Instance().ListServing(probe.target).empty()) {
        WARN("no CUDA provider; fused attention not exercised");
        return;
    }
    const AttentionCase cases[] = {
        {"causal d64", 2, 4, 4, 100, 100, 64, 0, true, 0, 0.0f, false},
        {"causal d32 70 (dropout-test shape)", 2, 2, 2, 70, 70, 32, 0, true, 0, 0.0f, false},
        {"bidirectional d32", 2, 2, 2, 70, 90, 32, 0, false, 0, 0.0f, false},
        {"gqa 4q/2kv", 2, 4, 2, 65, 65, 32, 0, true, 0, 0.0f, false},
        {"mqa 4q/1kv", 1, 4, 1, 40, 40, 16, 0, true, 0, 0.0f, false},
        {"sliding window 17", 2, 2, 2, 130, 130, 32, 0, true, 17, 0.0f, false},
        {"softcap 3", 2, 2, 2, 50, 50, 32, 0, true, 0, 3.0f, false},
        {"alibi", 2, 4, 4, 60, 60, 32, 0, true, 0, 0.0f, true},
        {"kv-cache step (offset 100, 1 query)", 2, 4, 2, 1, 101, 32, 100, true, 0, 0.0f, true},
        {"kv-cache prefill chunk (offset 64)", 1, 2, 2, 7, 71, 64, 64, true, 9, 2.0f, false},
        {"head_dim 128", 1, 2, 2, 33, 33, 128, 0, true, 0, 0.0f, false},
        {"head_dim 8", 3, 2, 2, 257, 257, 8, 0, true, 0, 0.0f, false},
    };
    for (const auto& c : cases) RunAttentionForward(c);
    for (const auto& c : cases) RunAttentionBackward(c);
}

TEST_CASE("Fused attention matches the materialized reference on OpenCL",
          "[neural_provider][device_resident][attention][opencl]") {
    int count = 0;
    try {
        const af::Backend previous = af::getActiveBackend();
        af::setBackend(AF_BACKEND_OPENCL);
        count = af::getDeviceCount();
        af::setBackend(previous);
    } catch (...) {
        WARN("ArrayFire OpenCL backend not available; fused attention not exercised");
        return;
    }
    const AttentionCase cases[] = {
        {"causal d64", 2, 4, 4, 100, 100, 64, 0, true, 0, 0.0f, false},
        {"bidirectional d32", 2, 2, 2, 70, 90, 32, 0, false, 0, 0.0f, false},
        {"gqa 4q/2kv", 2, 4, 2, 65, 65, 32, 0, true, 0, 0.0f, false},
        {"sliding window 17 + softcap", 2, 2, 2, 130, 130, 32, 0, true, 17, 3.0f, false},
        {"alibi + kv-cache step", 2, 4, 2, 1, 101, 32, 100, true, 0, 0.0f, true},
        {"head_dim 128", 1, 2, 2, 33, 33, 128, 0, true, 0, 0.0f, false},
    };
    for (int device = 0; device < count; ++device) {
        BackendGuard guard;
        guard.previous = af::getActiveBackend();
        guard.previous_device = af::getDevice();
        af::setBackend(AF_BACKEND_OPENCL);
        af::setDevice(device);
        guard.active = true;
        cyxwiz::NeuralOpRequest probe;
        probe.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
        if (cyxwiz::NeuralProviderRegistry::Instance().ListServing(probe.target).empty()) {
            WARN("no OpenCL provider; fused attention not exercised");
            return;
        }
        if (!ActiveDeviceServed()) {
            WARN("OpenCL device " << device << " is not served (not a GPU); fused attention not exercised there");
            continue;
        }
        INFO("OpenCL device " << device << ": " << af::infoString());
        for (const auto& c : cases) RunAttentionForward(c);
        for (const auto& c : cases) RunAttentionBackward(c);
    }
}

namespace {

// Host copy of the kernels' counter-based dropout decision.
float HostKeep(uint64_t seed, uint64_t index, float p) {
    uint64_t z = seed + index * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
    return u >= p ? 1.0f / (1.0f - p) : 0.0f;
}

// Mask in the reference score layout [Sq, Sk, B, H] (i fastest).
af::array DropoutMask(int batch, int heads, int sq, int sk, uint64_t seed, float p, double& kept_fraction) {
    std::vector<float> host(static_cast<size_t>(sq) * sk * batch * heads);
    size_t kept = 0;
    for (int h = 0; h < heads; ++h)
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < sk; ++j)
                for (int i = 0; i < sq; ++i) {
                    const uint64_t index = ((static_cast<uint64_t>(b) * heads + h) * sq + i) * sk + j;
                    const float keep = HostKeep(seed, index, p);
                    kept += keep > 0.0f;
                    host[i + static_cast<size_t>(sq) * (j + static_cast<size_t>(sk) * (b + static_cast<size_t>(batch) * h))] = keep;
                }
    kept_fraction = static_cast<double>(kept) / host.size();
    return af::array(sq, sk, batch, heads, host.data());
}

void RunAttentionDropout(const AttentionCase& c, float p, uint64_t seed) {
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::AttentionForward;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    request.training = true;
    request.batch = c.batch;
    request.heads = c.heads;
    request.kv_heads = c.kv_heads;
    request.seq = c.sq;
    request.kv_seq = c.sk;
    request.head_dim = c.head_dim;
    request.causal = c.causal;
    request.softmax_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    request.attention_dropout = p;
    request.dropout_seed = seed;
    auto provider = cyxwiz::NeuralProviderRegistry::Instance().FindSupporting(request);
    REQUIRE(provider);
    af::setSeed(11);
    const size_t d = c.head_dim, sq = c.sq, sk = c.sk, b = c.batch, hq = c.heads, hk = c.kv_heads;
    const af::array Q = af::randu(c.head_dim, c.sq, c.batch, c.heads) - 0.5f;
    const af::array K = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array V = af::randu(c.head_dim, c.sk, c.batch, c.kv_heads) - 0.5f;
    const af::array dO = af::randu(c.head_dim, c.sq, c.batch, c.heads) - 0.5f;
    const cyxwiz::Tensor q = cyxwiz::Tensor::FromSemanticArray(Q, {d, sq, b, hq});
    const cyxwiz::Tensor k = cyxwiz::Tensor::FromSemanticArray(K, {d, sk, b, hk});
    const cyxwiz::Tensor v = cyxwiz::Tensor::FromSemanticArray(V, {d, sk, b, hk});
    const cyxwiz::Tensor go = cyxwiz::Tensor::FromSemanticArray(dO, {d, sq, b, hq});
    std::vector<cyxwiz::Tensor> forward;
    REQUIRE(cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&q, &k, &v}, {{d, sq, b, hq}, {sq, b, hq}}, forward).ok);
    request.op = cyxwiz::NeuralOp::AttentionBackward;
    std::vector<cyxwiz::Tensor> grads;
    REQUIRE(cyxwiz::ExecuteNeuralOpOnDevice(*provider, request, {&q, &k, &v, &forward[0], &go, &forward[1]},
                                            {{d, sq, b, hq}, {d, sk, b, hk}, {d, sk, b, hk}, {sq, b, hq}}, grads).ok);

    // Reference with the same mask.
    double kept_fraction = 0.0;
    const af::array mask = DropoutMask(c.batch, c.heads, c.sq, c.sk, seed, p, kept_fraction);
    const int groups = c.heads / c.kv_heads;
    af::array qh = af::reorder(Q, 1, 0, 2, 3), kh = af::reorder(K, 1, 0, 2, 3), vh = af::reorder(V, 1, 0, 2, 3);
    const af::array goh = af::reorder(dO, 1, 0, 2, 3);
    if (groups > 1) {
        af::array idx = af::floor(af::range(af::dim4(c.heads)) / groups).as(s32);
        kh = af::lookup(kh, idx, 3);
        vh = af::lookup(vh, idx, 3);
    }
    af::array scores = af::matmul(qh, kh, AF_MAT_NONE, AF_MAT_TRANS) * request.softmax_scale;
    const af::dim4 grid(c.sq, c.sk);
    if (c.causal) {
        const af::array hidden = af::range(grid, 1, f32) > af::range(grid, 0, f32);
        scores = af::select(af::tile(hidden, af::dim4(1, 1, c.batch, c.heads)), -INFINITY, scores);
    }
    const af::array e = af::exp(scores - af::tile(af::max(scores, 1), af::dim4(1, c.sk)));
    const af::array prob = e / af::tile(af::sum(e, 1), af::dim4(1, c.sk));
    const af::array dropped = prob * mask;
    const af::array o_ref = af::matmul(dropped, vh);                                  // [Sq, D, B, H]
    const af::array delta = af::sum(o_ref * goh, 1);                                  // [Sq, 1, B, H]
    const af::array dp = af::matmul(goh, vh, AF_MAT_NONE, AF_MAT_TRANS) * mask;
    const af::array ds = prob * (dp - af::tile(delta, af::dim4(1, c.sk))) * request.softmax_scale;
    af::array dq = af::matmul(ds, kh), dk = af::matmul(ds, qh, AF_MAT_TRANS), dv = af::matmul(dropped, goh, AF_MAT_TRANS);
    if (groups > 1) {
        dk = af::moddims(af::sum(af::moddims(dk, af::dim4(c.sk * c.head_dim * c.batch, groups, c.kv_heads)), 1),
                         af::dim4(c.sk, c.head_dim, c.batch, c.kv_heads));
        dv = af::moddims(af::sum(af::moddims(dv, af::dim4(c.sk * c.head_dim * c.batch, groups, c.kv_heads)), 1),
                         af::dim4(c.sk, c.head_dim, c.batch, c.kv_heads));
    }
    const float eo = af::max<float>(af::abs(forward[0].GetSemanticArray() - af::reorder(o_ref, 1, 0, 2, 3)));
    const float eq = af::max<float>(af::abs(grads[0].GetSemanticArray() - af::reorder(dq, 1, 0, 2, 3)));
    const float ek = af::max<float>(af::abs(grads[1].GetSemanticArray() - af::reorder(dk, 1, 0, 2, 3)));
    const float ev = af::max<float>(af::abs(grads[2].GetSemanticArray() - af::reorder(dv, 1, 0, 2, 3)));
    INFO(c.label << " p=" << p << ": kept " << kept_fraction << ", errors O " << eo << " dQ " << eq << " dK " << ek
                 << " dV " << ev);
    CHECK(std::abs(kept_fraction - (1.0 - p)) < 0.02);
    CHECK(eo < 5e-5f);
    CHECK(eq < 1e-4f);
    CHECK(ek < 1e-4f);
    CHECK(ev < 1e-4f);
}

void RunDropoutCasesOn(af::Backend backend, int device) {
    BackendGuard guard;
    guard.previous = af::getActiveBackend();
    guard.previous_device = af::getDevice();
    af::setBackend(backend);
    af::setDevice(device);
    guard.active = true;
    cyxwiz::NeuralOpRequest probe;
    probe.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    if (cyxwiz::NeuralProviderRegistry::Instance().ListServing(probe.target).empty() || !ActiveDeviceServed()) {
        WARN("no provider serves this device; attention dropout not exercised");
        return;
    }
    const AttentionCase cases[] = {
        {"dropout causal d32", 2, 2, 2, 70, 70, 32, 0, true, 0, 0.0f, false},
        {"dropout gqa bidirectional d16", 1, 4, 2, 45, 60, 16, 0, false, 0, 0.0f, false},
    };
    for (const auto& c : cases) {
        RunAttentionDropout(c, 0.0f, 7ull);
        RunAttentionDropout(c, 0.1f, 1234567ull);
        RunAttentionDropout(c, 0.5f, 42ull);
    }
}

}  // namespace

TEST_CASE("Fused attention dropout matches a reference with the same mask",
          "[neural_provider][device_resident][attention][dropout]") {
    try {
        af::setBackend(AF_BACKEND_CUDA);
        RunDropoutCasesOn(AF_BACKEND_CUDA, 0);
    } catch (const af::exception&) {
        WARN("ArrayFire CUDA backend not available");
    }
    int count = 0;
    try {
        af::setBackend(AF_BACKEND_OPENCL);
        count = af::getDeviceCount();
    } catch (const af::exception&) {
    }
    for (int device = 0; device < count; ++device) RunDropoutCasesOn(AF_BACKEND_OPENCL, device);
    af::setBackend(AF_BACKEND_CPU);
}

TEST_CASE("Attention layers with dropout train through the fused kernels",
          "[neural_provider][device_resident][attention][dropout]") {
    try {
        af::setBackend(AF_BACKEND_CUDA);
    } catch (const af::exception&) {
        WARN("ArrayFire CUDA backend not available");
        return;
    }
    if (!ActiveDeviceServed()) {
        WARN("no CUDA provider; layer dropout path not exercised");
        af::setBackend(AF_BACKEND_CPU);
        return;
    }
    cyxwiz::MultiHeadAttentionLayer layer(32, 4, 0.25f, true);
    layer.DeclareStandardMask(true, 0);
    const cyxwiz::Tensor input = cyxwiz::Tensor::FromSemanticArray(af::randu(2, 40, 32) - 0.5f, {2, 40, 32});
    const cyxwiz::Tensor mask = cyxwiz::TransformerDecoderLayer::GenerateCausalMask(40);
    layer.SetTraining(false);
    const af::array eval_out = layer.Forward(input, input, input, &mask).GetSemanticArray();
    CHECK(layer.LastForwardUsedFusedAttention());
    layer.SetTraining(true);
    const af::array train_out = layer.Forward(input, input, input, &mask).GetSemanticArray();
    CHECK(layer.LastForwardUsedFusedAttention());
    CHECK(af::max<float>(af::abs(train_out - eval_out)) > 1e-3f);  // dropout was applied
    const cyxwiz::Tensor grad = layer.Backward(cyxwiz::Tensor::FromSemanticArray(af::constant(0.01f, 2, 40, 32), {2, 40, 32}));
    CHECK(af::allTrue<bool>(af::isNaN(grad.GetSemanticArray()) == 0));
    const auto params = layer.GetParameters();
    CHECK(af::sum<float>(af::abs(params.at("grad_W_q").GetSemanticArray())) > 0.0f);
    af::setBackend(AF_BACKEND_CPU);
}

TEST_CASE("Device-resident requests fail closed without a serving provider",
          "[neural_provider][device_resident]") {
    cyxwiz::NeuralOpRequest request;
    request.op = cyxwiz::NeuralOp::AttentionForward;
    request.target = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    request.device_resident = true;
    for (const auto& provider : cyxwiz::NeuralProviderRegistry::Instance().List()) {
        const auto capability = provider->QueryCapability(request);
        // Backward and the OpenCL tenant are not implemented yet; the CUDA
        // tenant accepts only complete forward requests.
        CHECK_FALSE(capability.supported);
    }
    CHECK(std::string(cyxwiz::NeuralOpName(cyxwiz::NeuralOp::DeviceProbe)) == "device_probe");
}
