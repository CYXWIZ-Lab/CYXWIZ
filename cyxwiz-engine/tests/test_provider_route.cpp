// Provider routes under Strict and Compatibility (TOFIX232): the placement
// asks the bound device's providers with the layer's own requests, Strict
// accepts a provider on that device, the trace records every dispatch, and a
// provider failure stops a Strict run instead of changing its route.
//
// Stub providers serve the CPU platform (no real provider does), so the test
// runs on ArrayFire CPU and never meets a real provider.

#include "../src/core/execution_placement_plan.h"
#include "../src/core/provider_placement.h"
#include "../src/core/training_trace_collector.h"
#include "../src/gui/loaders/data_loader.h"
#include "oracle/oracle_test.h"

#include <cyxwiz/layers/recurrent.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/neural_provider_dispatch.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

constexpr size_t kHidden = 5;
constexpr size_t kInput = 3;
constexpr size_t kSeq = 4;
constexpr size_t kBatch = 2;

enum class StubMode { Serve, DeclineBackward, Fail };

// Serves the CPU platform for the LSTM tuple above; Execute fails in Fail mode
// (Serve is only asked about capability, never executed).
class StubCpuProvider final : public cyxwiz::INeuralNetworkProvider {
public:
    explicit StubCpuProvider(StubMode mode) : mode_(mode) {}
    const char* ProviderId() const override { return "cyxwiz.test-cpu-stub"; }
    cyxwiz::DeviceType Platform() const override { return cyxwiz::DeviceType::CPU; }
    std::string Version() const override { return "cyxwiz.test-cpu-stub 0.0.1"; }
    cyxwiz::NeuralCapability QueryCapability(const cyxwiz::NeuralOpRequest& request) const override {
        cyxwiz::NeuralCapability capability;
        const bool tuple = request.hidden == kHidden && request.input == kInput && request.layers == 1;
        const bool op = request.op == cyxwiz::NeuralOp::LstmForward ||
                        (request.op == cyxwiz::NeuralOp::LstmBackward && mode_ != StubMode::DeclineBackward);
        capability.supported = tuple && op;
        if (!capability.supported) capability.detail = "the stub serves one LSTM tuple";
        return capability;
    }
    cyxwiz::NeuralResourceEstimate EstimateResources(const cyxwiz::NeuralOpRequest&) const override { return {}; }
    cyxwiz::NeuralOpStatus Execute(const cyxwiz::NeuralOpRequest&, cyxwiz::NeuralOpBuffers&) override {
        ++executions;
        cyxwiz::NeuralOpStatus status;
        status.ok = false;
        status.reason = cyxwiz::BackendFallbackReason::NvidiaProviderExecutionFailed;
        status.detail = "the stub failed on purpose";
        return status;
    }
    int executions = 0;

private:
    StubMode mode_;
};

std::vector<cyxwiz::NeuralDispatchEvent> g_events;
void CollectEvent(const cyxwiz::NeuralDispatchEvent& event) { g_events.push_back(event); }

std::shared_ptr<StubCpuProvider> RegisterStub(StubMode mode) {
    auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
    registry.UnregisterForTesting("cyxwiz.test-cpu-stub");
    auto stub = std::make_shared<StubCpuProvider>(mode);
    registry.Register(stub);
    return stub;
}

// One LSTM layer as the compiler writes it, with its compiler placement.
cyxwiz::TrainingConfiguration LstmConfig() {
    cyxwiz::TrainingConfiguration config;
    config.batch_size = static_cast<int>(kBatch);
    cyxwiz::CompiledLayer layer;
    layer.node_id = 40;
    layer.name = "LSTM";
    layer.type = gui::NodeType::LSTM;
    layer.input_shape = {kSeq, kInput};
    layer.output_shape = {kHidden};
    layer.parameters["hidden_size"] = std::to_string(kHidden);
    layer.parameters["num_layers"] = "1";
    config.layers.push_back(layer);
    cyxwiz::BackendPlacementEntry placement;
    placement.node_id = layer.node_id;
    placement.node_name = layer.name;
    placement.node_type = "LSTM";
    placement.status = cyxwiz::BackendPlacementStatus::Gpu;
    placement.reason_code = cyxwiz::BackendPlacementReason::ArrayFireTensorOpCapable;
    config.backend_placements.push_back(placement);
    return config;
}

cyxwiz::ExecutionDeviceContext CpuContext() {
    cyxwiz::ExecutionDeviceContext context;
    context.requested_backend = "arrayfire_cpu";
    context.effective_backend = "arrayfire_cpu";
    context.device_name = "Contract CPU";
    context.valid = true;
    context.execution_policy = cyxwiz::ExecutionPolicy::Strict;
    return context;
}

const cyxwiz::BackendPlacementEntry* LstmEntry(const cyxwiz::ExecutionPlacementPlan& plan) {
    for (const auto& entry : plan.entries)
        if (entry.node_id == 40) return &entry;
    return nullptr;
}

void TestRequestsMatchTheLayer() {
    const auto config = LstmConfig();
    const cyxwiz::NeuralDeviceTarget cpu{cyxwiz::DeviceType::CPU, 0};
    const auto requests = cyxwiz::ProviderRequestsForLayer(config, 0, cpu);
    Check(requests.size() == 2, "an LSTM asks for its forward and its backward");
    if (requests.size() == 2) {
        Check(requests[0].op == cyxwiz::NeuralOp::LstmForward && !requests[0].training,
              "the forward request is the layer's (training=false: the forward math is the same)");
        Check(requests[1].op == cyxwiz::NeuralOp::LstmBackward && requests[1].training,
              "the backward request is a training request");
        Check(requests[0].batch == kBatch && requests[0].seq == kSeq && requests[0].input == kInput &&
                  requests[0].hidden == kHidden && requests[0].layers == 1,
              "the request carries the compiled shapes");
    }
    auto bidirectional = config;
    bidirectional.layers[0].parameters["bidirectional"] = "true";
    bidirectional.layers[0].parameters["num_layers"] = "2";
    const auto bi = cyxwiz::ProviderRequestsForLayer(bidirectional, 0, cpu);
    Check(bi.size() == 4 && bi[0].layers == 1 && bi[2].input == 2 * kHidden,
          "a bidirectional layer asks for each direction's single-layer child, level 2 reads 2 x hidden");
    const cyxwiz::ExecutionDeviceContext context = CpuContext();
    Check(cyxwiz::NeuralTargetOfExecutionContext(context).platform == cyxwiz::DeviceType::CPU,
          "an ArrayFire CPU run targets the CPU platform");
}

void TestPlacement() {
    const auto config = LstmConfig();
    const auto context = CpuContext();

    RegisterStub(StubMode::Serve);
    auto plan = cyxwiz::BuildExecutionPlacementPlan(config, context);
    const auto* entry = LstmEntry(plan);
    Check(entry && entry->status == cyxwiz::ExecutionPlacementStatus::Provider &&
              entry->expected_backend == "cyxwiz.test-cpu-stub",
          "a provider serving the bound device for both ops places the layer on it");
    Check(plan.IsStrictlyExecutable(), "Strict accepts a provider on the bound device: " + plan.StrictBlockerSummary());

    RegisterStub(StubMode::DeclineBackward);
    plan = cyxwiz::BuildExecutionPlacementPlan(config, context);
    entry = LstmEntry(plan);
    Check(entry && entry->status == cyxwiz::ExecutionPlacementStatus::ArrayFire &&
              entry->explanation.find("declined lstm_backward") != std::string::npos,
          "a provider that declines the backward leaves the layer on ArrayFire and says why");
    Check(plan.IsStrictlyExecutable(), "an ArrayFire layer passes Strict");

    // A compiler entry naming a provider (the device active at compile time)
    // does not count: the bound device's answer does.
    auto compiled_on_gpu = config;
    compiled_on_gpu.backend_placements[0].reason_code = cyxwiz::BackendPlacementReason::NativeProviderSelected;
    compiled_on_gpu.backend_placements[0].expected_backend = "native provider cyxwiz.nvidia-cublas-cell";
    plan = cyxwiz::BuildExecutionPlacementPlan(compiled_on_gpu, context);
    entry = LstmEntry(plan);
    Check(entry && entry->status == cyxwiz::ExecutionPlacementStatus::ArrayFire &&
              entry->expected_backend == "arrayfire_cpu",
          "the compile-time provider verdict is replaced by the bound device's");
    Check(plan.IsStrictlyExecutable(), "and passes Strict: " + plan.StrictBlockerSummary());
    cyxwiz::NeuralProviderRegistry::Instance().UnregisterForTesting("cyxwiz.test-cpu-stub");
}

void TestDispatch() {
    const cyxwiz::ScopedNeuralDispatchObserver observer(&CollectEvent);
    const cyxwiz::NeuralDeviceTarget cpu{cyxwiz::DeviceType::CPU, 0};
    const auto request = cyxwiz::RecurrentProviderRequest(cyxwiz::NeuralOp::LstmForward, cpu,
                                                          {kBatch, kSeq, kInput, kHidden, 1});
    const auto run = [&](cyxwiz::INeuralNetworkProvider& provider) {
        cyxwiz::NeuralOpBuffers buffers;
        return provider.Execute(request, buffers);
    };

    // No provider serves the device: nothing runs, and the observer hears it
    // once for the op, however often it is asked for
    g_events.clear();
    bool ran = false;
    for (int call = 0; call < 3; ++call) ran = ran || cyxwiz::DispatchNeuralProviderOp(request, "LSTM", run);
    Check(!ran && g_events.size() == 1 && g_events[0].outcome == cyxwiz::NeuralDispatchOutcome::NotServed &&
              g_events[0].provider_id.empty() && g_events[0].op == "lstm_forward" && g_events[0].layer == "LSTM" &&
              g_events[0].device == "cpu",
          "no provider serves the device: nothing runs, and it is reported once in three calls");
    Check(!g_events.empty() &&
              g_events[0].reason == cyxwiz::BackendFallbackReasonName(cyxwiz::BackendFallbackReason::ProviderNotOffered),
          "the CPU has no provider by design: " + (g_events.empty() ? std::string("no event") : g_events[0].reason));

    auto stub = RegisterStub(StubMode::DeclineBackward);
    auto declined = cyxwiz::RecurrentProviderRequest(cyxwiz::NeuralOp::LstmBackward, cpu,
                                                     {kBatch, kSeq, kInput, kHidden, 1});
    g_events.clear();
    Check(!cyxwiz::DispatchNeuralProviderOp(declined, "LSTM", run) && g_events.size() == 1 &&
              g_events[0].outcome == cyxwiz::NeuralDispatchOutcome::Declined && stub->executions == 0,
          "a serving provider that declines is reported as declined, not run");

    stub = RegisterStub(StubMode::Fail);
    g_events.clear();
    Check(!cyxwiz::DispatchNeuralProviderOp(request, "LSTM", run) && g_events.size() == 1 &&
              g_events[0].outcome == cyxwiz::NeuralDispatchOutcome::Failed &&
              g_events[0].provider_id == "cyxwiz.test-cpu-stub" && g_events[0].op == "lstm_forward",
          "Compatibility: a failure is reported and the caller falls back");
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        bool threw = false;
        try {
            cyxwiz::DispatchNeuralProviderOp(request, "LSTM", run);
        } catch (const cyxwiz::NeuralProviderRouteError& error) {
            threw = std::string(error.what()).find("cyxwiz.test-cpu-stub lstm_forward failed in LSTM") !=
                    std::string::npos;
        }
        Check(threw, "Strict: a failure stops the run, naming the provider, the op and the layer");
    }
    Check(cyxwiz::CurrentNeuralProviderFailurePolicy() == cyxwiz::NeuralProviderFailurePolicy::FallBack,
          "the scoped policy is restored");
    cyxwiz::NeuralProviderRegistry::Instance().UnregisterForTesting("cyxwiz.test-cpu-stub");
}

// The layer itself: a failing provider under Compatibility gives the same
// output as the ArrayFire path; under Strict its Forward throws.
void TestLayer() {
    // The stub provider serves the CPU only: on another device the layer never asks it.
    if (cyxwiz::CaptureCurrentNeuralDeviceTarget().platform != cyxwiz::DeviceType::CPU) {
        std::cout << "SKIP: the layer checks need the CPU device, the platform the stub provider serves\n";
        return;
    }
    cyxwiz::LSTMLayer reference(static_cast<int>(kInput), static_cast<int>(kHidden));
    cyxwiz::LSTMLayer layer(static_cast<int>(kInput), static_cast<int>(kHidden));
    layer.SetParameters(reference.GetParameters());
    std::vector<float> values(kBatch * kSeq * kInput);
    for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;
    const cyxwiz::Tensor input({kBatch, kSeq, kInput}, values.data());
    const cyxwiz::Tensor expected = reference.Forward(input);

    auto stub = RegisterStub(StubMode::Fail);
    const cyxwiz::ScopedNeuralDispatchObserver observer(&CollectEvent);
    g_events.clear();
    const cyxwiz::Tensor fallback = layer.Forward(input);
    const float* a = expected.ReadData<float>();
    const float* b = fallback.ReadData<float>();
    float worst = 0.0f;
    for (size_t i = 0; i < expected.NumElements(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    Check(stub->executions == 1 && worst == 0.0f,
          "Compatibility: the failed provider's batch runs on ArrayFire with the same numbers");
    Check(g_events.size() == 1 && g_events[0].outcome == cyxwiz::NeuralDispatchOutcome::Failed,
          "and the failure is reported");

    cyxwiz::LSTMLayer strict_layer(static_cast<int>(kInput), static_cast<int>(kHidden));
    bool threw = false;
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        try {
            strict_layer.Forward(input);
        } catch (const cyxwiz::NeuralProviderRouteError&) {
            threw = true;
        }
    }
    Check(threw, "Strict: the layer's Forward stops on the provider failure");

    // A provider that runs the forward but declines the backward is not asked
    // for a training forward: the batch would otherwise be recomputed on
    // ArrayFire for its backward, a route the placement did not declare.
    stub = RegisterStub(StubMode::DeclineBackward);
    cyxwiz::LSTMLayer training_layer(static_cast<int>(kInput), static_cast<int>(kHidden));
    training_layer.SetTraining(true);
    (void)training_layer.Forward(input);
    Check(stub->executions == 0, "a training forward stays on ArrayFire when the backward is not served");
    cyxwiz::NeuralProviderRegistry::Instance().UnregisterForTesting("cyxwiz.test-cpu-stub");
}

// An op no provider serves: one event per (layer, op, device family) and
// observer scope, with the reason and what to do, and nothing unobserved.
void TestNotServed() {
    using cyxwiz::BackendFallbackReason;
    using cyxwiz::BackendFallbackReasonName;
    using cyxwiz::NeuralDispatchOutcome;
    const cyxwiz::NeuralDeviceTarget cpu{cyxwiz::DeviceType::CPU, 0};
    const cyxwiz::NeuralDeviceTarget cuda{cyxwiz::DeviceType::CUDA, 0};
    const cyxwiz::RecurrentProviderShape shape{kBatch, kSeq, kInput, kHidden, 1};
    const auto forward = cyxwiz::RecurrentProviderRequest(cyxwiz::NeuralOp::LstmForward, cpu, shape);
    const auto backward = cyxwiz::RecurrentProviderRequest(cyxwiz::NeuralOp::LstmBackward, cpu, shape);
    int executions = 0;
    const auto run = [&](cyxwiz::INeuralNetworkProvider&) {
        ++executions;
        return cyxwiz::NeuralOpStatus{};
    };
    const auto count = [](NeuralDispatchOutcome outcome, const std::string& op) {
        size_t n = 0;
        for (const auto& e : g_events) n += e.outcome == outcome && e.op == op ? 1 : 0;
        return n;
    };

    // Unobserved: nothing is reported (and nothing built)
    g_events.clear();
    cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
    Check(g_events.empty() && executions == 0, "without an observer an unserved op reports nothing");

    {
        const cyxwiz::ScopedNeuralDispatchObserver first_run(&CollectEvent);
        for (int call = 0; call < 5; ++call) {
            cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
            cyxwiz::DispatchNeuralProviderOp(backward, "LSTM", run);
        }
        Check(count(NeuralDispatchOutcome::NotServed, "lstm_forward") == 1 &&
                  count(NeuralDispatchOutcome::NotServed, "lstm_backward") == 1 && g_events.size() == 2,
              "once per op in a run: two ops, ten calls, two events");
        // The same op from another layer family is another line of the path
        cyxwiz::DispatchNeuralProviderOp(forward, "Bidirectional LSTM", run);
        Check(g_events.size() == 3, "another layer asking for the same op is reported once too");
    }
    {
        // The next run hears it again: "once" is per run, not per process
        g_events.clear();
        const cyxwiz::ScopedNeuralDispatchObserver second_run(&CollectEvent);
        cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
        cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
        Check(g_events.size() == 1, "the next run is told once again");

        // A tally inside the run is a scope of its own, and passes on what it hears
        g_events.clear();
        {
            const cyxwiz::NeuralDispatchTally tally;
            cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
            cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
            const auto* entry = tally.Find("LSTM", cyxwiz::NeuralOp::LstmForward);
            Check(entry && entry->not_served == 1 && entry->ran == 0 && entry->provider_id.empty() &&
                      entry->reason == BackendFallbackReasonName(BackendFallbackReason::ProviderNotOffered),
                  "a tally counts the unserved op once, with its reason");
            Check(tally.Find("LSTM", cyxwiz::NeuralOp::LstmBackward) == nullptr, "and knows nothing of an op never asked");
        }
        Check(g_events.size() == 1 && g_events[0].outcome == NeuralDispatchOutcome::NotServed,
              "the tally passes the event on to the run's observer");
        // The run's own scope is as it was: it has heard of this op already
        cyxwiz::DispatchNeuralProviderOp(forward, "LSTM", run);
        Check(g_events.size() == 1, "after the tally the run's scope still knows what it heard");
    }
    {
        // A provider that is missing and can be supplied: the reason and what to do
        g_events.clear();
        const cyxwiz::ScopedNeuralDispatchObserver third_run(&CollectEvent);
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        const auto on_cuda = cyxwiz::RecurrentProviderRequest(cyxwiz::NeuralOp::LstmForward, cuda, shape);
        cyxwiz::DispatchNeuralProviderOp(on_cuda, "LSTM", run);
        cyxwiz::DispatchNeuralProviderOp(on_cuda, "LSTM", run);
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        Check(g_events.size() == 1 && g_events[0].outcome == NeuralDispatchOutcome::NotServed &&
                  g_events[0].device == "cuda" &&
                  g_events[0].reason == BackendFallbackReasonName(BackendFallbackReason::ProviderDisabled) &&
                  g_events[0].detail == "Neural providers turned off. Unset CYXWIZ_DISABLE_NEURAL_PROVIDERS and start "
                                        "again to use them.",
              "providers turned off: " + (g_events.empty() ? std::string("no event") : g_events[0].detail));

        // A probe that will never dispatch reports the same way, and once
        g_events.clear();
        cyxwiz::SetNeuralProvidersDisabledForTesting(true);
        auto probe = on_cuda;
        probe.op = cyxwiz::NeuralOp::MlpForward;
        cyxwiz::ReportNeuralProviderProbe(probe, "fused MLP");
        cyxwiz::ReportNeuralProviderProbe(probe, "fused MLP");
        cyxwiz::SetNeuralProvidersDisabledForTesting(false);
        Check(g_events.size() == 1 && g_events[0].layer == "fused MLP" && g_events[0].op == "mlp_forward",
              "a probe that finds no provider is reported once");
    }
    Check(executions == 0, "nothing ran on a provider");
}

void TestTrace() {
    auto& trace = cyxwiz::TrainingTraceCollector::Instance();
    trace.StartRun("provider-route-test");
    cyxwiz::NeuralDispatchEvent event;
    event.provider_id = "cyxwiz.test-cpu-stub";
    event.op = "lstm_forward";
    event.layer = "LSTM";
    event.device = "cpu";
    event.outcome = cyxwiz::NeuralDispatchOutcome::Ran;
    cyxwiz::RecordNeuralDispatchForActiveTrace(event);
    cyxwiz::RecordNeuralDispatchForActiveTrace(event);
    event.outcome = cyxwiz::NeuralDispatchOutcome::Declined;
    event.reason = "unsupported";
    event.detail = "a shorter last batch";
    cyxwiz::RecordNeuralDispatchForActiveTrace(event);
    cyxwiz::RecordNeuralDispatchForActiveTrace(event);
    const auto summary = trace.Snapshot();
    trace.FinishRun("completed");
    Check(summary.provider_dispatch_count == 4 && summary.provider_groups.size() == 2,
          "the trace groups the dispatches by provider, op and outcome");
    size_t ran = 0;
    for (const auto& group : summary.provider_groups)
        if (group.reason == "ran") ran = group.event_count;
    Check(ran == 2, "two runs counted in one group");
    size_t warnings = 0;
    for (const auto& warning : summary.warnings)
        if (warning.find("a shorter last batch") != std::string::npos) ++warnings;
    Check(warnings == 1, "a declined group is warned about once");
    // A finished run keeps its id; a later dispatch (a Run Test) is not its.
    cyxwiz::RecordNeuralDispatchForActiveTrace(event);
    Check(trace.Snapshot().provider_dispatch_count == 4, "a finished run takes no more dispatches");

    // An op no provider serves: in the trace, and no false alarm.
    //  - a provider that is missing and can be supplied: one "ok" event with
    //    the reason and what to do, no warning;
    //  - nothing offered for the op on the device (the CPU): counted only.
    // Neither counts as a dispatch a provider answered.
    trace.StartRun("provider-not-served-test");
    cyxwiz::NeuralDispatchEvent missing;
    missing.op = "ppo_sample";
    missing.layer = "PPO";
    missing.device = "cuda";
    missing.outcome = cyxwiz::NeuralDispatchOutcome::NotServed;
    missing.reason = cyxwiz::BackendFallbackReasonName(cyxwiz::BackendFallbackReason::ProviderNotInstalled);
    missing.detail = "NVIDIA provider not installed. Install the CUDA GPU pack.";
    cyxwiz::RecordNeuralDispatchForActiveTrace(missing);
    cyxwiz::RecordNeuralDispatchForActiveTrace(missing);  // a second thread of the run reporting it
    cyxwiz::NeuralDispatchEvent designed = missing;
    designed.op = "lstm_forward";
    designed.layer = "LSTM";
    designed.device = "cpu";
    designed.reason = cyxwiz::BackendFallbackReasonName(cyxwiz::BackendFallbackReason::ProviderNotOffered);
    designed.detail = "No provider is offered for this on the CPU";
    cyxwiz::RecordNeuralDispatchForActiveTrace(designed);
    const auto not_served = trace.Snapshot();
    trace.FinishRun("completed");
    Check(not_served.provider_dispatch_count == 0, "an unserved op is no dispatch a provider answered");
    size_t groups = 0;
    for (const auto& group : not_served.provider_groups)
        if (group.reason == "not_served" && group.category == "no provider") ++groups;
    Check(groups == 2 && not_served.provider_groups.size() == 2, "each unserved op is a group of the trace");
    Check(not_served.warnings.empty(), "an unserved op is not a warning: " +
                                           (not_served.warnings.empty() ? std::string() : not_served.warnings.front()));
    size_t said = 0, said_designed = 0;
    for (const auto& e : not_served.recent_events) {
        if (e.stage != "NeuralProvider.NotServed") continue;
        if (e.message.find("Install the CUDA GPU pack.") != std::string::npos && e.status == "ok") ++said;
        if (e.message.find("lstm_forward") != std::string::npos) ++said_designed;
    }
    Check(said == 1, "a missing provider is said once in the run's events, with what to do");
    Check(said_designed == 0, "the designed ArrayFire path of the CPU is not an event");
}

}  // namespace

int main() {
    if (!cyxwiz::test::BeginOracleTest("test_provider_route")) return 1;
    // With no device asked for, run where the stub provider serves (the
    // default device here may be a GPU).
    std::optional<cyxwiz::test::RequestedTestDevice> requested;
    if (!cyxwiz::test::ReadRequestedTestDevice(requested)) return 1;
    if (!requested && !cyxwiz::test::ActivateTestDevice({cyxwiz::DeviceType::CPU, 0, "cpu"})) return 1;
    TestRequestsMatchTheLayer();
    TestPlacement();
    TestDispatch();
    TestNotServed();
    TestLayer();
    TestTrace();
    if (g_failures > 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "Provider route tests passed\n";
    return 0;
}
