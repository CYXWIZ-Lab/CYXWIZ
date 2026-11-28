// tofix68 device-keyed dispatch tenant #2: the OpenCL neural provider.
// Every test warns and returns when no OpenCL GPU device is available, so
// the suite stays truthful on machines without one.

#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/tensor.h>

#include "core/loaded_library.h"

#ifndef NOMINMAX
#define NOMINMAX  // arrayfire.h pulls windows.h; keep std::max usable
#endif
#include <arrayfire.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "algorithms/providers/opencl_device_match.h"

// How the OpenCL provider finds ArrayFire's device among its GPUs, with
// injected descriptors (no OpenCL device needed). A cl_device_id from another
// OpenCL loader or a layered driver is not pointer-equal to the provider's,
// and a GPU that worked before must not be declined for the process.
TEST_CASE("OpenCL provider matches ArrayFire's device by identity, then hardware, then index",
          "[neural_provider][opencl]") {
    using cyxwiz::opencl_match::DeviceDescriptor;
    using cyxwiz::opencl_match::DeviceKind;
    using cyxwiz::opencl_match::MatchArrayFireDevice;
    using cyxwiz::opencl_match::MatchedBy;
    int a = 0, b = 0, other = 0;
    const std::vector<DeviceDescriptor> gpus = {
        {&a, "NVIDIA GeForce GTX 1050 Ti", "NVIDIA Corporation", "0:1:0.0", ""},
        {&b, "Intel(R) UHD Graphics 630", "Intel(R) Corporation", "", ""},
    };

    // The same cl_device_id
    auto match = MatchArrayFireDevice(1, DeviceKind::Gpu, {&b, "renamed", "", "", ""}, gpus);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::Identity);

    // Another loader's handle for the same hardware: PCI address, then name and vendor
    match = MatchArrayFireDevice(1, DeviceKind::Gpu, {&other, "NVIDIA GeForce GTX 1050 Ti", "NVIDIA Corporation", "0:1:0.0", ""},
                                 gpus);
    CHECK(match.index == 0);
    CHECK(match.by == MatchedBy::PciOrUuid);
    match = MatchArrayFireDevice(0, DeviceKind::Gpu, {&other, "Intel(R) UHD Graphics 630", "Intel(R) Corporation", "", ""},
                                 gpus);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::NameAndVendor);

    // A device ArrayFire says is not a GPU is the only decline by kind
    match = MatchArrayFireDevice(2, DeviceKind::NotGpu, {&other, "Intel(R) Core(TM) i7", "Intel(R) Corporation", "", ""},
                                 gpus);
    CHECK(match.index == -1);
    CHECK(match.by == MatchedBy::NotGpu);
    CHECK(match.why.find("not one of the provider's GPU devices") != std::string::npos);

    // Read, and it matches no provider GPU: declined with why, never the GPU at its position
    match = MatchArrayFireDevice(1, DeviceKind::Gpu, {&other, "Unknown GPU", "Vendor", "", ""}, gpus);
    CHECK(match.index == -1);
    CHECK(match.by == MatchedBy::None);
    CHECK(match.why.find("matches none of the provider's 2 GPU devices") != std::string::npos);
    // A name that matches but a PCI address that contradicts it is another device
    match = MatchArrayFireDevice(0, DeviceKind::Gpu, {&other, "NVIDIA GeForce GTX 1050 Ti", "NVIDIA Corporation", "0:2:0.0", ""},
                                 gpus);
    CHECK(match.index == -1);
    CHECK(match.by == MatchedBy::None);

    // ArrayFire could not report the kind (afcl_get_device_type missing) and
    // the handle could not be read: it may not be a GPU, so no position is taken
    match = MatchArrayFireDevice(1, DeviceKind::Unknown, {nullptr, "", "", "", ""}, gpus);
    CHECK(match.index == -1);
    CHECK(match.why.find("may not be a GPU") != std::string::npos);
    // Unknown kind, read, but no name and vendor match: declined too
    match = MatchArrayFireDevice(1, DeviceKind::Unknown, {&other, "Intel(R) Core(TM) i7", "Intel(R) Corporation", "", ""},
                                 gpus);
    CHECK(match.index == -1);
    CHECK(match.by == MatchedBy::None);
    // Unknown kind with a name and vendor match: that GPU
    match = MatchArrayFireDevice(0, DeviceKind::Unknown, {&other, "Intel(R) UHD Graphics 630", "Intel(R) Corporation", "", ""},
                                 gpus);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::NameAndVendor);

    // A GPU whose handle could not be read: its position, marked uncertain
    match = MatchArrayFireDevice(1, DeviceKind::Gpu, {nullptr, "", "", "", ""}, gpus);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::Index);
    CHECK(match.uncertain.find("could not be read") != std::string::npos);
    // ... and past the provider's GPUs: none
    match = MatchArrayFireDevice(5, DeviceKind::Gpu, {nullptr, "", "", "", ""}, gpus);
    CHECK(match.index == -1);

    // Identical twins with no PCI address or UUID: the position decides, uncertain
    const std::vector<DeviceDescriptor> twins = {{&a, "Same GPU", "Vendor", "", ""}, {&b, "Same GPU", "Vendor", "", ""}};
    match = MatchArrayFireDevice(1, DeviceKind::Gpu, {&other, "Same GPU", "Vendor", "", ""}, twins);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::Index);
    CHECK(match.uncertain.find("2 GPUs are named Same GPU") != std::string::npos);
    const std::string warning = cyxwiz::opencl_match::UncertainMatchWarning(1, match, twins);
    CHECK(warning.find("GPU 1 (Same GPU) by its position") != std::string::npos);
    CHECK(warning.find("which physical GPU runs it is uncertain") != std::string::npos);
    // ... past them: none
    match = MatchArrayFireDevice(3, DeviceKind::Gpu, {&other, "Same GPU", "Vendor", "", ""}, twins);
    CHECK(match.index == -1);
    // Twins with distinct PCI addresses are told apart, whatever the positions
    const std::vector<DeviceDescriptor> pci_twins = {{&a, "Same GPU", "Vendor", "0:3:0.0", ""},
                                                     {&b, "Same GPU", "Vendor", "0:4:0.0", ""}};
    match = MatchArrayFireDevice(0, DeviceKind::Gpu, {&other, "Same GPU", "Vendor", "0:4:0.0", ""}, pci_twins);
    CHECK(match.index == 1);
    CHECK(match.by == MatchedBy::PciOrUuid);
    CHECK(match.uncertain.empty());
    // ... and a twin whose PCI address is neither of theirs is declined
    match = MatchArrayFireDevice(0, DeviceKind::Gpu, {&other, "Same GPU", "Vendor", "0:5:0.0", ""}, pci_twins);
    CHECK(match.index == -1);
}

// The provider's memory of its answers (IdentifiedDevices, what
// ProbeIndexFor keeps): a device is identified once and then looked up, and an
// uncertain match is warned about once, by the call that keeps it. No GPU
// needed: the provider's wiring is this class.
TEST_CASE("OpenCL provider keeps each device's match and warns once about an uncertain one",
          "[neural_provider][opencl]") {
    using cyxwiz::opencl_match::DeviceDescriptor;
    using cyxwiz::opencl_match::DeviceKind;
    using cyxwiz::opencl_match::IdentifiedDevices;
    using cyxwiz::opencl_match::MatchArrayFireDevice;
    int a = 0, b = 0, other = 0;
    const std::vector<DeviceDescriptor> twins = {{&a, "Same GPU", "Vendor", "", ""}, {&b, "Same GPU", "Vendor", "", ""}};
    IdentifiedDevices identified;
    std::vector<std::string> warnings;
    const auto warn = [&](const std::string& warning) { warnings.push_back(warning); };
    CHECK_FALSE(identified.Find(1).has_value());

    const auto uncertain = MatchArrayFireDevice(1, DeviceKind::Gpu, {&other, "Same GPU", "Vendor", "", ""}, twins);
    auto kept = identified.Keep(1, uncertain, twins, warn);
    CHECK(kept.index == 1);
    REQUIRE(warnings.size() == 1);
    CHECK(warnings[0].find("which physical GPU runs it is uncertain") != std::string::npos);
    // Another call that identified it meanwhile: the first answer stays, no second warning
    kept = identified.Keep(1, MatchArrayFireDevice(1, DeviceKind::Gpu, {&a, "", "", "", ""}, twins), twins, warn);
    CHECK(kept.index == 1);
    CHECK_FALSE(kept.uncertain.empty());
    CHECK(warnings.size() == 1);
    const auto found = identified.Find(1);
    REQUIRE(found.has_value());
    CHECK(found->index == 1);

    // A certain match and a decline are kept without a warning
    CHECK(identified.Keep(0, MatchArrayFireDevice(0, DeviceKind::Gpu, {&a, "", "", "", ""}, twins), twins, warn).index == 0);
    const auto declined = identified.Keep(2, MatchArrayFireDevice(2, DeviceKind::Unknown, {}, twins), twins, warn);
    CHECK(declined.index == -1);
    CHECK_FALSE(declined.why.empty());
    REQUIRE(identified.Find(2).has_value());
    CHECK(identified.Find(2)->why == declined.why);
    CHECK(warnings.size() == 1);
}

#ifdef CYXWIZ_HAS_OPENCL_DNN_PROVIDER

namespace {

// Which enumerated OpenCL GPU the suite targets. Default 0 (the first GPU,
// NVIDIA on the dev box); CYXWIZ_OPENCL_TEST_DEVICE=1 runs the same gates
// on the second GPU (the Intel UHD 630 on the dev box) for the AMD/Intel
// platform gate. The index is ArrayFire's device id; the provider maps it
// to its own GPU by identity (on the dev box the two orders agree).
int OpenclTestDeviceIndex() {
    static const int index = [] {
        const char* value = std::getenv("CYXWIZ_OPENCL_TEST_DEVICE");
        if (value == nullptr || value[0] == '\0') {
            return 0;
        }
        return std::atoi(value);
    }();
    return index;
}

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
    request.target = {cyxwiz::DeviceType::OPENCL, OpenclTestDeviceIndex()};
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

// Small-tuple parity cases (hidden=10) sit below the retention floor; they
// exercise the provider math, not the floor, so they lift it for their scope.
struct RetentionFloorOverride {
    size_t previous = cyxwiz::OpenclProviderRetentionFloorHidden();
    explicit RetentionFloorOverride(size_t floor) {
        cyxwiz::SetOpenclProviderRetentionFloorForTesting(floor);
    }
    ~RetentionFloorOverride() {
        cyxwiz::SetOpenclProviderRetentionFloorForTesting(previous);
    }
};

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
    WARN("opencl suite targets device " << OpenclTestDeviceIndex()
         << "; provider: " << provider->Version());
    CHECK(provider->Version().find(
              "device" + std::to_string(OpenclTestDeviceIndex()) + " '") !=
          std::string::npos);

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

// ArrayFire orders its OpenCL devices itself (GPUs first, CPU devices after,
// its own ranking among GPUs); the provider lists GPUs in platform order. A
// request's device id is ArrayFire's, so the provider maps it by identity:
// every ArrayFire GPU is served, an OpenCL CPU device is declined with why.
TEST_CASE("OpenCL provider maps ArrayFire's device to its own GPU by identity",
          "[gpu_execution][neural_provider][opencl]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device on this machine; opencl provider not registered");
        return;
    }
    const af::Backend backend = af::getActiveBackend();
    const int device = af::getDevice();
    try {
        af::setBackend(AF_BACKEND_OPENCL);
    } catch (const af::exception&) {
        WARN("ArrayFire's OpenCL backend does not start here");
        return;
    }
    // Loaded once ArrayFire's OpenCL backend is
    using GetType = af_err (*)(int*);
    const auto get_type =
        reinterpret_cast<GetType>(cyxwiz::detail::FindLoadedSymbol("afopencl", "afcl_get_device_type"));
    REQUIRE(get_type != nullptr);
    int gpus = 0;
    for (int k = 0; k < af::getDeviceCount(); ++k) {
        af::setDevice(k);
        int type = 0;
        REQUIRE(get_type(&type) == AF_SUCCESS);
        auto request = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 4, 10, 6, 16);
        request.target = {cyxwiz::DeviceType::OPENCL, k};
        const auto capability = provider->QueryCapability(request);
        char name[256] = {}, platform[256] = {}, toolkit[256] = {}, compute[256] = {};
        af::deviceInfo(name, platform, toolkit, compute);
        INFO("ArrayFire OpenCL device " << k << ": " << name << " (" << capability.detail << ")");
        if (type == 4) {  // AFCL_DEVICE_TYPE_GPU
            ++gpus;
            CHECK(capability.supported);
        } else {
            CHECK_FALSE(capability.supported);
            CHECK(capability.detail.find("not one of the provider's GPU devices") != std::string::npos);
        }
    }
    CHECK(gpus > 0);
    af::setBackend(backend);
    af::setDevice(device);
}

TEST_CASE("OpenCL provider rnn/lstm/gru forward match the reference layers",
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
        reference.Forward(input);
        const auto expected = reference.Forward(input);
        const auto expected_c = reference.GetCellState();
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
        reference.Forward(input);
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

TEST_CASE("OpenCL provider lstm/gru backward match the ArrayFire BPTT",
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
            reference.Forward(input);
            params = reference.GetParameters();
            reference.Forward(input);
            expected_dx = reference.Backward(upstream);
            grads = reference.GetParameters();
            cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        } else {
            cyxwiz::GRULayer reference(6, 16);
            cyxwiz::SetNeuralProvidersDisabledForTesting(true);
            reference.Forward(input);
            params = reference.GetParameters();
            reference.Forward(input);
            expected_dx = reference.Backward(upstream);
            grads = reference.GetParameters();
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

TEST_CASE("OpenCL provider rnn_backward matches the CPU RNN BPTT reference (stacked)",
          "[gpu_execution][neural_provider][opencl][parity][rnn]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; parity not exercised");
        return;
    }
    RetentionFloorOverride no_floor(0);
    const auto input = FilledTensor({3, 7, 5}, 0.5f, 0.7f);
    const auto upstream = FilledTensor({3, 7, 10}, 0.3f, 1.3f);
    cyxwiz::RNNLayer reference(5, 10, 2, true, false, "tanh");
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    reference.Forward(input);
    const auto expected_dx = reference.Backward(upstream);
    const auto grads = reference.GetParameters();
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    std::vector<cyxwiz::Tensor> weights_storage = {
        grads.at("layer0_W_ih"), grads.at("layer0_W_hh"), grads.at("layer0_b_ih"),
        grads.at("layer0_b_hh"), grads.at("layer1_W_ih"), grads.at("layer1_W_hh"),
        grads.at("layer1_b_ih"), grads.at("layer1_b_hh")};
    cyxwiz::Tensor dx(std::vector<size_t>{3, 7, 5});
    std::vector<cyxwiz::Tensor> grad_storage = {
        cyxwiz::Tensor(std::vector<size_t>{10, 5}), cyxwiz::Tensor(std::vector<size_t>{10, 10}),
        cyxwiz::Tensor(std::vector<size_t>{10}), cyxwiz::Tensor(std::vector<size_t>{10}),
        cyxwiz::Tensor(std::vector<size_t>{10, 10}), cyxwiz::Tensor(std::vector<size_t>{10, 10}),
        cyxwiz::Tensor(std::vector<size_t>{10}), cyxwiz::Tensor(std::vector<size_t>{10})};
    cyxwiz::NeuralOpBuffers buffers;
    buffers.inputs = {&input, &upstream};
    buffers.outputs = {&dx};
    for (auto& w : weights_storage) buffers.weights.push_back(&w);
    for (auto& g : grad_storage) buffers.gradients.push_back(&g);
    auto request = OpenclRequest(cyxwiz::NeuralOp::RnnBackward, 3, 7, 5, 10, 2);
    request.activation = cyxwiz::NeuralActivation::Tanh;
    const auto status = provider->Execute(request, buffers);
    INFO("detail: " << status.detail);
    REQUIRE(status.ok);
    Compare(dx, expected_dx, "opencl stacked rnn_backward dx", 5e-4f);
    const char* names[] = {"layer0_grad_W_ih", "layer0_grad_W_hh",
                           "layer0_grad_b_ih", "layer0_grad_b_hh",
                           "layer1_grad_W_ih", "layer1_grad_W_hh",
                           "layer1_grad_b_ih", "layer1_grad_b_hh"};
    for (size_t i = 0; i < 8; ++i) {
        Compare(grad_storage[i], grads.at(names[i]), names[i], 5e-4f);
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
        af::setDevice(OpenclTestDeviceIndex());
        guard.active = true;
    } catch (...) {
        WARN("ArrayFire OpenCL backend cannot be selected in this process; "
             "routing not exercised");
        return;
    }
    REQUIRE(cyxwiz::CaptureCurrentNeuralDeviceTarget().platform ==
            cyxwiz::DeviceType::OPENCL);
    RetentionFloorOverride no_floor(0);
    const auto input = FilledTensor({3, 7, 5}, 0.5f, 0.2f);
    const auto upstream = FilledTensor({3, 7, 10}, 0.25f, 1.1f);
    const char* grad_names[] = {
        "layer0_grad_W_ih", "layer0_grad_W_hh", "layer0_grad_b_ih",
        "layer0_grad_b_hh", "layer1_grad_W_ih", "layer1_grad_W_hh",
        "layer1_grad_b_ih", "layer1_grad_b_hh"};
    {
        cyxwiz::LSTMLayer layer(5, 10, 2);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        layer.Forward(input);
        const auto expected = layer.Forward(input);
        const auto expected_h = layer.GetHiddenState();
        const auto expected_c = layer.GetCellState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
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
        layer.Forward(input);
        const auto expected = layer.Forward(input);
        const auto expected_h = layer.GetHiddenState();
        const auto expected_dx = layer.Backward(upstream);
        const auto oracle = layer.GetParameters();
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

TEST_CASE("OpenCL provider declines tuples below the retention floor and serves them from it",
          "[gpu_execution][neural_provider][opencl][retention_floor]") {
    auto provider = OpenclProvider();
    if (!provider) {
        WARN("no OpenCL GPU device; retention floor not exercised");
        return;
    }
    // Owner ruling 2026-09-23: floor at hidden=16 (0.82x at 8, 2.4x at 16).
    REQUIRE(cyxwiz::OpenclProviderRetentionFloorHidden() == 16);
    const auto below = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 32, 16, 32, 8);
    const auto at_floor = OpenclRequest(cyxwiz::NeuralOp::LstmForward, 32, 16, 32, 16);

    const auto declined = provider->QueryCapability(below);
    REQUIRE_FALSE(declined.supported);
    REQUIRE(declined.reason ==
            cyxwiz::BackendFallbackReason::OpenclProviderBelowRetentionFloor);
    REQUIRE(declined.detail.find("retention floor") != std::string::npos);
    REQUIRE(declined.detail.find("hidden=8") != std::string::npos);
    REQUIRE(provider->QueryCapability(at_floor).supported);

    // The registry answers the same way, so no layer selects the tenant for
    // a tiny tuple while the tuple at the floor still routes to it.
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    REQUIRE(registry.FindSupporting(below) == nullptr);
    REQUIRE(registry.FindSupporting(at_floor) != nullptr);

    // The floor is a policy knob, not a contract limit: lifting it (test
    // hook) makes the same tuple executable, and it comes back on scope exit.
    {
        RetentionFloorOverride no_floor(0);
        REQUIRE(provider->QueryCapability(below).supported);
        REQUIRE(registry.FindSupporting(below) != nullptr);
    }
    REQUIRE_FALSE(provider->QueryCapability(below).supported);
}

TEST_CASE("OpenCL provider survives repeated training runs and reports its speed vs ArrayFire",
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
    // ArrayFire LSTM forward on the same shape for the retention record.
    cyxwiz::LSTMLayer reference(64, 64);
    cyxwiz::SetNeuralProvidersDisabledForTesting(true);
    reference.Forward(x);
    std::vector<double> arrayfire_ms;
    for (int i = 0; i < 3; ++i) {
        reference.ResetState();
        const auto start = std::chrono::steady_clock::now();
        reference.Forward(x);
        const auto end = std::chrono::steady_clock::now();
        arrayfire_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    cyxwiz::SetNeuralProvidersDisabledForTesting(false);
    std::sort(forward_ms.begin(), forward_ms.end());
    std::sort(arrayfire_ms.begin(), arrayfire_ms.end());
    const double provider_median = forward_ms[forward_ms.size() / 2];
    const double arrayfire_median = arrayfire_ms[arrayfire_ms.size() / 2];
    WARN("opencl lstm_forward 32x32x64x64: provider "
         << provider_median << " ms vs ArrayFire " << arrayfire_median
         << " ms (" << arrayfire_median / provider_median << "x)");
}

#endif // CYXWIZ_HAS_OPENCL_DNN_PROVIDER
