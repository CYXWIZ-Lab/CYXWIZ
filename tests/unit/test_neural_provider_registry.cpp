// tofix68 phase 2: the provider registry and the capability-reporting-only
// NVIDIA skeleton. Valid in both build configurations — a portable build
// has an empty registry (the normal state); a provider-enabled build with
// a CUDA device registers the skeleton, which truthfully supports nothing.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/neural_provider.h>

#include <memory>
#include <string>

namespace {

// Canonical unsupported probe: attention is the next op family on the
// track68 roadmap and no provider claims it yet (RNN, LSTM and GRU are the
// executed recurrent ops as of P3).
cyxwiz::NeuralOpRequest MakeUnsupportedProbe() {
    cyxwiz::NeuralOpRequest request;
    request.target = {cyxwiz::DeviceType::CUDA, 0};
    request.op = cyxwiz::NeuralOp::AttentionForward;
    request.dtype = cyxwiz::DataType::Float32;
    request.batch = 8;
    request.seq = 16;
    request.input = 32;
    request.hidden = 64;
    request.heads = 4;
    return request;
}

// Device-keyed dispatch fixtures: one stub per device family, both
// claiming the SAME deliberately absurd tuple (Float64 keeps it outside
// the real provider's Float32 contract; the stubs ignore dtype). They obey
// the registry-wide truths the first test asserts for every provider.
class PlatformStubProvider final : public cyxwiz::INeuralNetworkProvider {
public:
    PlatformStubProvider(const char* id, cyxwiz::DeviceType platform)
        : id_(id), platform_(platform) {}
    const char* ProviderId() const override { return id_; }
    cyxwiz::DeviceType Platform() const override { return platform_; }
    std::string Version() const override {
        return std::string(id_) + " 9.9.9 / dispatch fixture";
    }
    cyxwiz::NeuralCapability QueryCapability(
        const cyxwiz::NeuralOpRequest& request) const override {
        cyxwiz::NeuralCapability capability;
        capability.supported =
            request.target.platform == platform_ &&
            request.op == cyxwiz::NeuralOp::LstmForward &&
            request.batch == 7777 && request.hidden == 7777 &&
            request.layers == 3;
        if (!capability.supported) {
            capability.detail = "stub supports only the fixture tuple";
        }
        return capability;
    }
    cyxwiz::NeuralResourceEstimate EstimateResources(
        const cyxwiz::NeuralOpRequest&) const override {
        return {};
    }
    cyxwiz::NeuralOpStatus Execute(const cyxwiz::NeuralOpRequest& request,
                                   cyxwiz::NeuralOpBuffers&) override {
        // Contract-shaped like the real provider: an unsupported tuple
        // fails typed as unsupported_contract, never as execution failure.
        cyxwiz::NeuralOpStatus status;
        status.ok = false;
        status.reason = cyxwiz::BackendFallbackReason::
            NvidiaProviderUnsupportedContract;
        status.detail = QueryCapability(request).detail;
        return status;
    }

private:
    const char* id_;
    cyxwiz::DeviceType platform_;
};

cyxwiz::NeuralOpRequest MakeFixtureRequest(cyxwiz::DeviceType platform) {
    cyxwiz::NeuralOpRequest request;
    request.target = {platform, 0};
    request.dtype = cyxwiz::DataType::Float64;  // real provider refuses
    request.op = cyxwiz::NeuralOp::LstmForward;
    request.batch = 7777;
    request.seq = 4;
    request.input = 8;
    request.hidden = 7777;
    request.layers = 3;
    return request;
}

} // namespace

TEST_CASE("Neural provider registry answers truthfully",
          "[gpu_execution][neural_provider]") {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    const auto providers = registry.List();

    // Attention is not implemented yet, so no provider may claim it in
    // ANY build configuration.
    CHECK(registry.FindSupporting(MakeUnsupportedProbe()) == nullptr);

    for (const auto& provider : providers) {
        const auto capability =
            provider->QueryCapability(MakeUnsupportedProbe());
        CHECK_FALSE(capability.supported);
        // Each tenant answers with ITS OWN unsupported-contract code.
        CHECK(std::string(cyxwiz::BackendFallbackReasonName(capability.reason))
                  .find("_provider_unsupported_contract") != std::string::npos);
        CHECK_FALSE(capability.detail.empty());

        auto request = MakeUnsupportedProbe();
        cyxwiz::NeuralOpBuffers buffers;
        const auto status = provider->Execute(request, buffers);
        CHECK_FALSE(status.ok);
        CHECK(std::string(cyxwiz::BackendFallbackReasonName(status.reason))
                  .find("_provider_unsupported_contract") != std::string::npos);

        CHECK_FALSE(provider->Version().empty());
        CHECK(std::string(provider->ProviderId()).find("cyxwiz.") == 0);
    }

#ifdef CYXWIZ_HAS_NVIDIA_DNN_PROVIDER
    // Provider-enabled build on a machine with a CUDA device: the
    // skeleton must be registered and identify itself. (On a
    // provider-enabled build WITHOUT a device the probe declines and the
    // registry stays empty, which the loop above already tolerates.)
    bool found_nvidia = false;
    for (const auto& provider : providers) {
        if (std::string(provider->ProviderId()) ==
            "cyxwiz.nvidia-cublas-cell") {
            found_nvidia = true;
            CHECK(provider->Version().find("CUDA runtime") !=
                  std::string::npos);
        }
    }
    INFO("provider-enabled build: nvidia skeleton expected when a CUDA "
         "device is present");
    CHECK(found_nvidia);
#else
    CHECK(providers.empty());
#endif
}

TEST_CASE("Neural provider registry dispatches on the selected device",
          "[gpu_execution][neural_provider]") {
    using cyxwiz::DeviceType;
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    registry.Register(std::make_shared<PlatformStubProvider>(
        "cyxwiz.test-stub-cuda", DeviceType::CUDA));
    registry.Register(std::make_shared<PlatformStubProvider>(
        "cyxwiz.test-stub-opencl", DeviceType::OPENCL));

    // Same tuple, different selected device -> different provider; the
    // CUDA stub must never answer an OpenCL-targeted run and vice versa.
    const auto cuda_pick = registry.FindSupporting(
        MakeFixtureRequest(DeviceType::CUDA));
    REQUIRE(cuda_pick != nullptr);
    CHECK(std::string(cuda_pick->ProviderId()) == "cyxwiz.test-stub-cuda");
    CHECK(cuda_pick->Platform() == DeviceType::CUDA);

    const auto opencl_pick = registry.FindSupporting(
        MakeFixtureRequest(DeviceType::OPENCL));
    REQUIRE(opencl_pick != nullptr);
    CHECK(std::string(opencl_pick->ProviderId()) ==
          "cyxwiz.test-stub-opencl");
    CHECK(opencl_pick->Platform() == DeviceType::OPENCL);

    // No provider serves the CPU family, and the fixture's cell request is
    // unserved on oneAPI too (the oneAPI tenant, when built and installed,
    // covers device-resident attention only): fail closed, the portable
    // path decides. A default-constructed request (CPU target) therefore
    // never selects a provider.
    CHECK(registry.FindSupporting(MakeFixtureRequest(DeviceType::CPU)) ==
          nullptr);
    CHECK(registry.FindSupporting(MakeFixtureRequest(DeviceType::ONEAPI)) ==
          nullptr);
#ifndef CYXWIZ_HAS_ONEAPI_DNN_PROVIDER
    CHECK(registry.ListServing({DeviceType::ONEAPI, 0}).empty());
#endif
    cyxwiz::NeuralOpRequest unfilled = MakeFixtureRequest(DeviceType::CUDA);
    unfilled.target = {};
    CHECK(unfilled.target.platform == DeviceType::CPU);
    CHECK(registry.FindSupporting(unfilled) == nullptr);

    // Stage-1 filter lists exactly the providers of that family.
    for (const auto& provider : registry.ListServing({DeviceType::OPENCL, 0})) {
        CHECK(provider->Platform() == DeviceType::OPENCL);
    }
    for (const auto& provider : registry.ListServing({DeviceType::CUDA, 0})) {
        CHECK(provider->Platform() == DeviceType::CUDA);
    }

    // The capture helper reports a stable name for every family.
    CHECK(std::string(cyxwiz::NeuralDevicePlatformName(DeviceType::CUDA)) ==
          "cuda");
    CHECK(std::string(cyxwiz::NeuralDevicePlatformName(DeviceType::OPENCL)) ==
          "opencl");
    CHECK(std::string(cyxwiz::NeuralDevicePlatformName(DeviceType::ONEAPI)) ==
          "oneapi");
    CHECK(std::string(cyxwiz::NeuralDevicePlatformName(DeviceType::CPU)) ==
          "cpu");
    const auto current = cyxwiz::CaptureCurrentNeuralDeviceTarget();
    CHECK(std::string(cyxwiz::NeuralDevicePlatformName(current.platform)) !=
          "unknown");
    CHECK(current.device_id >= 0);
}

TEST_CASE("Neural op names are stable identifiers",
          "[gpu_execution][neural_provider][taxonomy]") {
    using cyxwiz::NeuralOp;
    using cyxwiz::NeuralOpName;
    CHECK(std::string(NeuralOpName(NeuralOp::RnnForward)) == "rnn_forward");
    CHECK(std::string(NeuralOpName(NeuralOp::LstmForward)) ==
          "lstm_forward");
    CHECK(std::string(NeuralOpName(NeuralOp::LstmBackward)) ==
          "lstm_backward");
    CHECK(std::string(NeuralOpName(NeuralOp::GruForward)) == "gru_forward");
    CHECK(std::string(NeuralOpName(NeuralOp::AttentionForward)) ==
          "attention_forward");
}
