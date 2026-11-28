// The RL provider calls go through the one provider dispatch (TOFIX233 B):
// the fused small-network kernels (FusedMlp), the joint Adam step
// (AdamOptimizer::StepTogether) and PPO's kernels (sampling, GAE) report each
// call to the run's dispatch observer and follow its failure policy, as the
// graph layers do (TOFIX232).
//
// On the CPU route a stub provider stands in for a GPU provider. Its calls
// fail (the CPU route has no device-resident queue), which is the case under
// test: with FallBack the call reports Failed and the caller runs its
// ArrayFire path; with Fail (a Strict run) it stops with
// NeuralProviderRouteError. On a GPU route whose provider runs the fused MLP,
// a forward is reported as Ran.

#include <cyxwiz/fused_mlp.h>
#include <cyxwiz/neural_provider.h>
#include <cyxwiz/neural_provider_dispatch.h>
#include <cyxwiz/optimizers/adam.h>
#include <cyxwiz/rl/ppo.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include "oracle/oracle_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr const char* kStub = "cyxwiz.test-rl-stub";

// Serves every RL op on the CPU route; declines a fused forward of 5 rows
constexpr size_t kDeclinedRows = 5;
constexpr const char* kDeclineReason = "the stub takes no 5-row forward";

class StubRlProvider final : public cyxwiz::INeuralNetworkProvider {
public:
    const char* ProviderId() const override { return kStub; }
    cyxwiz::DeviceType Platform() const override { return cyxwiz::DeviceType::CPU; }
    std::string Version() const override { return std::string(kStub) + " 0.0.1"; }
    cyxwiz::NeuralCapability QueryCapability(const cyxwiz::NeuralOpRequest& request) const override {
        cyxwiz::NeuralCapability capability;
        if (request.op == cyxwiz::NeuralOp::MlpForward && request.batch == kDeclinedRows) {
            capability.detail = kDeclineReason;
            return capability;
        }
        switch (request.op) {
            case cyxwiz::NeuralOp::MlpForward:
            case cyxwiz::NeuralOp::MlpBackward:
            case cyxwiz::NeuralOp::AdamStep:
            case cyxwiz::NeuralOp::PpoSample:
            case cyxwiz::NeuralOp::PpoLoss:
            case cyxwiz::NeuralOp::PpoGae:
                capability.supported = true;
                break;
            default:
                capability.detail = "the stub serves the RL ops";
        }
        return capability;
    }
    cyxwiz::NeuralResourceEstimate EstimateResources(const cyxwiz::NeuralOpRequest&) const override { return {}; }
    cyxwiz::NeuralOpStatus Execute(const cyxwiz::NeuralOpRequest&, cyxwiz::NeuralOpBuffers&) override {
        cyxwiz::NeuralOpStatus status;
        status.detail = "the stub runs nothing on the host";
        return status;
    }
};

std::vector<cyxwiz::NeuralDispatchEvent> events;
void Collect(const cyxwiz::NeuralDispatchEvent& event) { events.push_back(event); }

bool OneEvent(const std::string& layer, const std::string& op, cyxwiz::NeuralDispatchOutcome outcome,
              const std::string& provider) {
    return events.size() == 1 && events[0].layer == layer && events[0].op == op && events[0].outcome == outcome &&
           events[0].provider_id == provider;
}

std::string Describe() {
    std::string out = std::to_string(events.size()) + " event(s)";
    for (const auto& e : events)
        out += "; " + e.provider_id + " " + e.layer + " " + e.op + " " + cyxwiz::NeuralDispatchOutcomeName(e.outcome);
    return out;
}

cyxwiz::Tensor Filled(const std::vector<size_t>& shape, float start) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + 0.01f * static_cast<float>(i % 17);
    return cyxwiz::Tensor(shape, v.data(), cyxwiz::DataType::Float32);
}

cyxwiz::SequentialModel Network() {
    cyxwiz::SequentialModel m;
    m.Add<cyxwiz::LinearModule>(4, 8, true);
    m.Add<cyxwiz::TanhModule>();
    m.Add<cyxwiz::LinearModule>(8, 2, true);
    return m;
}

enum class Thrown { Nothing, Other, Route };

template <typename F>
Thrown Run(F&& call) {
    try {
        call();
    } catch (const cyxwiz::NeuralProviderRouteError&) {
        return Thrown::Route;
    } catch (const std::exception&) {
        return Thrown::Other;
    }
    return Thrown::Nothing;
}

void CheckFusedMlp() {
    auto model = Network();
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    Check(fused != nullptr, "the stub serves the network (" + why + ")");
    if (!fused) return;
    Check(fused->ProviderId() == kStub, "the fused network names its provider: " + fused->ProviderId());
    const auto x = Filled({3, 4}, 0.1f);

    events.clear();
    Check(Run([&] { fused->Forward(x, true); }) == Thrown::Other,
          "FallBack: the fused forward fails as itself, so the trainer runs the layers");
    Check(OneEvent("fused MLP", "mlp_forward", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "FallBack: the fused forward reports Failed: " + Describe());

    events.clear();
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        Check(Run([&] { fused->Forward(x, false); }) == Thrown::Route,
              "Fail: the fused forward stops with the provider route error");
    }
    Check(OneEvent("fused MLP", "mlp_forward", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "Fail: the fused forward reports Failed: " + Describe());

    // A declined call: the trainer's message carries the provider's reason
    events.clear();
    std::string declined;
    try {
        fused->Forward(Filled({kDeclinedRows, 4}, 0.1f), false);
    } catch (const std::exception& e) {
        declined = e.what();
    }
    Check(declined.find(kStub) != std::string::npos && declined.find(kDeclineReason) != std::string::npos,
          "a declined fused forward names the provider and its reason: " + declined);
    Check(OneEvent("fused MLP", "mlp_forward", cyxwiz::NeuralDispatchOutcome::Declined, kStub),
          "a declined fused forward reports Declined: " + Describe());

    // Two networks in one launch
    auto other = Network();
    auto fused_other = cyxwiz::FusedMlp::For(other, why);
    Check(fused_other != nullptr, "the stub serves the second network (" + why + ")");
    if (!fused_other) return;
    events.clear();
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        Check(Run([&] { cyxwiz::FusedMlp::ForwardPair(*fused, *fused_other, x, true); }) == Thrown::Route,
              "Fail: the paired forward stops with the provider route error");
    }
    Check(OneEvent("fused MLP", "mlp_forward", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "Fail: the paired forward reports Failed: " + Describe());
}

void CheckAdam() {
    std::map<std::string, cyxwiz::Tensor> parameters{{"w", Filled({3, 2}, 0.5f)}};
    std::map<std::string, cyxwiz::Tensor> gradients{{"w", Filled({3, 2}, -0.2f)}};
    cyxwiz::AdamOptimizer optimizer(0.01);
    const std::vector<cyxwiz::AdamOptimizer::Group> groups{{&optimizer, &parameters, &gradients}};

    events.clear();
    std::string why;
    bool together = true;
    Check(Run([&] { together = cyxwiz::AdamOptimizer::StepTogether(groups, 0.0f, why); }) == Thrown::Nothing,
          "FallBack: the joint Adam step does not throw");
    Check(!together && why.find(kStub) != std::string::npos,
          "FallBack: the joint Adam step says the provider failed, so each optimizer steps: " + why);
    Check(OneEvent("Adam", "adam_step", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "FallBack: the joint Adam step reports Failed: " + Describe());

    events.clear();
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        Check(Run([&] { cyxwiz::AdamOptimizer::StepTogether(groups, 0.0f, why); }) == Thrown::Route,
              "Fail: the joint Adam step stops with the provider route error");
    }
    Check(OneEvent("Adam", "adam_step", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "Fail: the joint Adam step reports Failed: " + Describe());
}

double MaxDifference(const cyxwiz::Tensor& a, const cyxwiz::Tensor& b) {
    if (a.Shape() != b.Shape()) return 1e30;
    const float* x = a.ReadData<float>();
    const float* y = b.ReadData<float>();
    double d = 0.0;
    for (size_t i = 0; i < a.NumElements(); ++i) d = std::max(d, static_cast<double>(std::fabs(x[i] - y[i])));
    return d;
}

void CheckPpo(const cyxwiz::Tensor& want_advantages, const cyxwiz::Tensor& want_returns) {
    const auto rewards = Filled({5, 2}, 1.0f);
    const auto values = Filled({5, 2}, 0.3f);
    const auto dones = cyxwiz::Tensor::Zeros({5, 2}, cyxwiz::DataType::Float32);
    const auto last = Filled({2}, 0.2f);

    cyxwiz::Tensor advantages, returns;
    const auto gae = [&] { cyxwiz::rl::ComputeGae(rewards, values, dones, last, 0.99f, 0.95f, advantages, returns); };
    {
        // Run 1, FallBack: the ArrayFire GAE, the same as with no provider
        const cyxwiz::rl::ScopedPpoProviderRun first_run;
        events.clear();
        Check(Run(gae) == Thrown::Nothing, "FallBack: GAE does not throw");
        Check(MaxDifference(advantages, want_advantages) < 1e-6 && MaxDifference(returns, want_returns) < 1e-6,
              "FallBack: GAE equals the ArrayFire path");
        Check(OneEvent("PPO", "ppo_gae", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
              "FallBack: GAE reports Failed: " + Describe());
        Check(first_run.IsOff(2) && !first_run.IsOff(0) && !first_run.IsOff(1),
              "the run turns off the op that failed, and only that one");
        // The op stays on ArrayFire for the rest of this run: no provider call again
        events.clear();
        gae();
        gae();
        Check(events.empty(), "after a failure GAE stays on ArrayFire for the rest of the run: " + Describe());
        // ... but not for a Strict run, which asks the provider whatever was
        // met before, and stops at its failure (without setting the latch)
        events.clear();
        {
            const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
            Check(Run(gae) == Thrown::Route, "Fail: GAE after an earlier failure still asks the provider and stops");
        }
        Check(OneEvent("PPO", "ppo_gae", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
              "Fail: GAE after an earlier failure reports Failed: " + Describe());
    }
    {
        // Run 2 in the same process: run 1's failure did not change its path.
        // It asks the provider again (the latch was run 1's, not the process's).
        const cyxwiz::rl::ScopedPpoProviderRun second_run;
        Check(!second_run.IsOff(2), "a new run starts with nothing turned off");
        events.clear();
        Check(Run(gae) == Thrown::Nothing, "run 2: GAE does not throw");
        Check(OneEvent("PPO", "ppo_gae", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
              "run 2 asks the provider again after run 1's failure: " + Describe());
        Check(MaxDifference(advantages, want_advantages) < 1e-6, "run 2: GAE equals the ArrayFire path");
    }
    // Outside any run nothing is remembered: every call asks the provider
    events.clear();
    gae();
    gae();
    Check(events.size() == 2, "outside a run every call asks the provider: " + Describe());

    // Fail: sampling stops with the provider route error
    const auto head = Filled({3, 2}, 0.0f);
    const auto value = Filled({3}, 0.5f);
    events.clear();
    {
        const cyxwiz::ScopedNeuralProviderFailurePolicy strict(cyxwiz::NeuralProviderFailurePolicy::Fail);
        Check(Run([&] { cyxwiz::rl::PpoSampleStep(head, value, cyxwiz::Tensor(), true, 7); }) == Thrown::Route,
              "Fail: PPO sampling stops with the provider route error");
    }
    Check(OneEvent("PPO", "ppo_sample", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "Fail: PPO sampling reports Failed: " + Describe());
    // A Strict failure sets no latch: a later Compatibility call still asks
    events.clear();
    cyxwiz::rl::PpoSampleStep(head, value, cyxwiz::Tensor(), true, 7);
    Check(OneEvent("PPO", "ppo_sample", cyxwiz::NeuralDispatchOutcome::Failed, kStub),
          "a Strict failure leaves the op on the provider for a later run: " + Describe());
}

// On a GPU route with a provider that runs the fused MLP: Ran
void CheckRanOnDevice() {
    auto model = Network();
    std::string why;
    auto fused = cyxwiz::FusedMlp::For(model, why);
    if (!fused) {
        std::printf("No provider runs the fused MLP on this device (%s): the Ran check was not run\n", why.c_str());
        return;
    }
    events.clear();
    Check(Run([&] { fused->Forward(Filled({3, 4}, 0.1f), false); }) == Thrown::Nothing, "the fused forward runs");
    Check(OneEvent("fused MLP", "mlp_forward", cyxwiz::NeuralDispatchOutcome::Ran, fused->ProviderId()),
          "the fused forward reports Ran: " + Describe());
}

}  // namespace

int main() {
    if (!cyxwiz::test::BeginOracleTest("test_rl_provider_dispatch")) return 1;
    std::optional<cyxwiz::test::RequestedTestDevice> requested;
    if (!cyxwiz::test::ReadRequestedTestDevice(requested)) return 1;
    // With no device asked for, run where the stub serves (the default device
    // here may be a GPU)
    if (!requested && !cyxwiz::test::ActivateTestDevice({cyxwiz::DeviceType::CPU, 0, "cpu"})) return 1;

    const cyxwiz::ScopedNeuralDispatchObserver observer(&Collect);
    if (cyxwiz::CaptureCurrentNeuralDeviceTarget().platform == cyxwiz::DeviceType::CPU) {
        // The reference GAE before any provider is registered
        cyxwiz::Tensor want_advantages, want_returns;
        cyxwiz::rl::ComputeGae(Filled({5, 2}, 1.0f), Filled({5, 2}, 0.3f),
                               cyxwiz::Tensor::Zeros({5, 2}, cyxwiz::DataType::Float32), Filled({2}, 0.2f), 0.99f,
                               0.95f, want_advantages, want_returns);
        auto& registry = cyxwiz::NeuralProviderRegistry::Instance();
        registry.UnregisterForTesting(kStub);
        registry.Register(std::make_shared<StubRlProvider>());
        CheckFusedMlp();
        CheckAdam();
        CheckPpo(want_advantages, want_returns);
        registry.UnregisterForTesting(kStub);
    } else {
        CheckRanOnDevice();
    }

    if (failures > 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("RL provider dispatch tests passed\n");
    return 0;
}
