#include "core/language_model_generation.h"
#include "core/execution_device_context.h"
#include "gui/panels/language_model_generation_panel_metadata.h"

#include <cyxwiz/sequential.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class ScriptedLogitModule : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        const auto& shape = input.Shape();
        if (shape.size() != 2 || shape[0] != 1) {
            throw std::runtime_error(
                "ScriptedLogitModule expects [1, seq] token IDs");
        }

        const size_t seq = shape[1];
        const size_t vocab = 5;
        std::vector<float> logits(seq * vocab, -10.0f);
        const size_t next_token = seq == 2 ? 3 : 4;
        logits[(seq - 1) * vocab + next_token] = 10.0f;
        return cyxwiz::Tensor({1, seq, vocab}, logits.data());
    }

    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad_output) override {
        return grad_output;
    }

    std::string GetName() const override {
        return "ScriptedLogitModule";
    }
};

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(float actual,
               float expected,
               float tolerance,
               const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message
                  << " actual=" << actual
                  << " expected=" << expected << "\n";
        std::exit(1);
    }
}

void TestConfigValidation() {
    cyxwiz::LanguageModelGenerationConfig config;
    Check(cyxwiz::ValidateLanguageModelGenerationConfig(config).empty(),
          "default generation config should be valid");

    config.max_new_tokens = 0;
    config.temperature = 0.0f;
    config.top_p = 1.5f;
    const auto issues = cyxwiz::ValidateLanguageModelGenerationConfig(config);
    Check(issues.size() == 3,
          "invalid generation config should report each bad field");
}

void TestStopReasonNames() {
    Check(cyxwiz::LanguageModelGenerationStopReasonName(
              cyxwiz::LanguageModelGenerationStopReason::MaxTokens) ==
              "max_tokens",
          "max-token stop reason should have a stable UI name");
    Check(cyxwiz::LanguageModelGenerationStopReasonName(
              cyxwiz::LanguageModelGenerationStopReason::EosToken) == "eos",
          "EOS stop reason should have a stable UI name");
    Check(cyxwiz::LanguageModelGenerationStopReasonName(
              cyxwiz::LanguageModelGenerationStopReason::Error) == "error",
          "error stop reason should have a stable UI name");
    Check(cyxwiz::LanguageModelGenerationStopReasonName(
              cyxwiz::LanguageModelGenerationStopReason::UserCancelled) ==
              "user_cancel",
          "user-cancel stop reason should have a stable UI name");
}

void TestGreedyTopKDistribution() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.top_k = 2;

    const std::vector<float> logits = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 1.0f, 0.0f, 2.0f,
    };

    const auto candidates = cyxwiz::BuildNextTokenDistribution(
        logits, 1, 2, 4, config);

    Check(candidates.size() == 2,
          "top_k should keep only the requested number of candidates");
    Check(candidates[0].token_id == 0,
          "distribution should use logits from the last sequence position");
    Check(candidates[1].token_id == 3,
          "top_k should keep the second-best last-position token");
    CheckNear(candidates[0].probability + candidates[1].probability,
              1.0f,
              1e-6f,
              "filtered candidate probabilities should renormalize");

    std::mt19937 rng(123);
    const auto selected = cyxwiz::SelectNextTokenFromDistribution(
        candidates, config, rng);
    Check(selected.token_id == 0,
          "greedy selection should choose the highest-probability token");
}

void TestTopPFiltering() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.top_p = 0.70f;

    const std::vector<float> logits = {4.0f, 3.0f, 2.0f, 1.0f};
    const auto candidates = cyxwiz::BuildNextTokenDistribution(
        logits, 1, 1, 4, config);

    Check(candidates.size() == 2,
          "top_p should keep candidates until cumulative probability crosses threshold");
    Check(candidates[0].token_id == 0 && candidates[1].token_id == 1,
          "top_p should retain the highest-probability nucleus tokens");
    CheckNear(candidates[0].probability + candidates[1].probability,
              1.0f,
              1e-6f,
              "top_p probabilities should renormalize after filtering");
}

void TestMultinomialSelectionIsSeeded() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.sampling_mode = cyxwiz::LanguageModelSamplingMode::Multinomial;
    config.top_k = 3;
    config.temperature = 0.7f;

    const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};

    std::mt19937 rng_a(7);
    std::mt19937 rng_b(7);
    const auto selected_a = cyxwiz::SelectNextTokenFromLogits(
        logits, 1, 1, 4, config, rng_a);
    const auto selected_b = cyxwiz::SelectNextTokenFromLogits(
        logits, 1, 1, 4, config, rng_b);

    Check(selected_a.token_id == selected_b.token_id,
          "multinomial sampling should be reproducible with the same seed");
    Check(selected_a.candidates.size() == 3,
          "multinomial sampling should expose the filtered candidate set");
}

void TestShapeValidation() {
    cyxwiz::LanguageModelGenerationConfig config;
    bool threw = false;
    try {
        (void)cyxwiz::BuildNextTokenDistribution(
            {1.0f, 2.0f}, 1, 1, 3, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "generation distribution should reject logits shape mismatch");
}

void TestGenerationPanelRunMetadataSmoke() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 4;
    config.temperature = 0.75f;
    config.top_k = 3;
    config.top_p = 0.8f;
    config.eos_token_id = 4;
    config.include_prompt = false;
    config.sampling_mode = cyxwiz::LanguageModelSamplingMode::Multinomial;

    cyxwiz::LanguageModelGenerationResult report;
    report.prompt_length = 2;
    report.remaining_budget = 1;
    report.stop_reason = cyxwiz::LanguageModelGenerationStopReason::EosToken;
    report.steps.push_back({
        0,
        2,
        3,
        0.6f,
        {{3, 0.6f}, {1, 0.4f}}
    });

    const auto metadata = cyxwiz::BuildLanguageModelGenerationPanelRunMetadata(
        report,
        config,
        12,
        99);

    Check(metadata.stop_reason == "eos",
          "panel metadata should expose stable stop reason text");
    Check(metadata.prompt_length == 2,
          "panel metadata should expose prompt length");
    Check(metadata.max_context_length == 12,
          "panel metadata should expose max context length");
    Check(metadata.remaining_budget == 1,
          "panel metadata should expose remaining generation budget");
    Check(metadata.last_candidates.size() == 2,
          "panel metadata should expose last-token candidates");
    Check(metadata.last_candidates[0].token_id == 3,
          "panel metadata should preserve candidate token order");
    Check(metadata.sampling_settings ==
              "multinomial, max_new_tokens=4, temperature=0.75, top_k=3, "
              "top_p=0.8, eos=4, seed=99, include_prompt=false",
          "panel metadata should expose the sampling settings used for the run");
}
void TestGenerateTokenIdsWithReportStopsAtEos() {
    cyxwiz::SequentialModel model;
    model.Add<ScriptedLogitModule>();

    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 4;
    config.eos_token_id = 4;
    config.include_prompt = false;

    const auto result = cyxwiz::GenerateTokenIdsWithReport(
        model,
        {1, 2},
        config);

    Check(result.stop_reason ==
              cyxwiz::LanguageModelGenerationStopReason::EosToken,
          "runtime generation report should stop at EOS");
    Check(result.token_ids == std::vector<int64_t>({3, 4}),
          "runtime generation report should return generated-only token IDs");
    Check(result.new_token_ids == std::vector<int64_t>({3, 4}),
          "runtime generation report should expose new token IDs separately");
    Check(result.prompt_length == 2,
          "runtime generation report should record prompt length");
    Check(result.max_new_tokens == 4,
          "runtime generation report should record max new tokens");
    Check(result.remaining_budget == 2,
          "runtime generation report should record remaining budget after EOS");
    Check(result.steps.size() == 2,
          "runtime generation report should record one step per generated token");
    Check(result.steps[0].step_index == 0,
          "first generation step should record step index");
    Check(result.steps[0].input_length == 2,
          "first generation step should record model input length");
    Check(result.steps[0].token_id == 3,
          "first generation step should record selected token");
    Check(!result.steps[0].candidates.empty(),
          "generation report should expose candidate distribution diagnostics");
}

void TestGenerateTokenIdsWithReportStopsAtMaxTokens() {
    cyxwiz::SequentialModel model;
    model.Add<ScriptedLogitModule>();

    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 1;
    config.eos_token_id = -1;
    config.include_prompt = true;

    const auto result = cyxwiz::GenerateTokenIdsWithReport(
        model,
        {1, 2},
        config);

    Check(result.stop_reason ==
              cyxwiz::LanguageModelGenerationStopReason::MaxTokens,
          "generation report should use max-token stop reason when budget ends");
    Check(result.token_ids == std::vector<int64_t>({1, 2, 3}),
          "generation report should include prompt when requested");
    Check(result.new_token_ids == std::vector<int64_t>({3}),
          "generation report should record only generated IDs separately");
    Check(result.remaining_budget == 0,
          "generation report should have no remaining budget at max tokens");

    config.max_new_tokens = 4;
    config.max_context_tokens = 3;
    config.include_prompt = false;
    const auto context_result = cyxwiz::GenerateTokenIdsWithReport(
        model,
        {1, 2},
        config);
    Check(context_result.stop_reason ==
              cyxwiz::LanguageModelGenerationStopReason::MaxTokens,
          "context budget exhaustion should use the max-token stop reason");
    Check(context_result.new_token_ids == std::vector<int64_t>({3}),
          "context budget should stop generation before exceeding max length");
    Check(context_result.remaining_budget == 0,
          "context budget exhaustion should report zero remaining budget");
}

void TestGenerateTokenIdsWithConfigCompatibilityWrapper() {
    cyxwiz::SequentialModel model;
    model.Add<ScriptedLogitModule>();

    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 4;
    config.eos_token_id = 4;
    config.include_prompt = false;

    const auto generated = cyxwiz::GenerateTokenIdsWithConfig(
        model,
        {1, 2},
        config);

    Check(generated == std::vector<int64_t>({3, 4}),
          "legacy runtime generation wrapper should preserve token output");
}

void TestGenerateTokenIdsWithReportRejectsBadPrompt() {
    cyxwiz::SequentialModel model;
    model.Add<ScriptedLogitModule>();

    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 2;

    bool empty_prompt_threw = false;
    try {
        (void)cyxwiz::GenerateTokenIdsWithReport(model, {}, config);
    } catch (const std::invalid_argument&) {
        empty_prompt_threw = true;
    }
    Check(empty_prompt_threw,
          "generation report should reject empty prompts before inference");

    config.max_context_tokens = 2;
    bool long_prompt_threw = false;
    try {
        (void)cyxwiz::GenerateTokenIdsWithReport(model, {1, 2}, config);
    } catch (const std::invalid_argument&) {
        long_prompt_threw = true;
    }
    Check(long_prompt_threw,
          "generation report should reject prompts without context budget");
}

// Deterministic full-prefix logits provide an independent reference for both
// selection and every candidate probability, with deliberately different rows.
std::vector<float> PrefixLogits(size_t seq, size_t vocab) {
    std::vector<float> values(seq * vocab);
    for (size_t s = 0; s < seq; ++s)
        for (size_t v = 0; v < vocab; ++v)
            values[s * vocab + v] = std::sin(static_cast<float>(s * 11 + v) * .37f);
    return values;
}

class DeviceLogitModule : public cyxwiz::Module {
public:
    explicit DeviceLogitModule(size_t vocab) : vocab_(vocab) {}
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        const size_t seq = input.Shape()[1];
        const auto values = PrefixLogits(seq, vocab_);
        cyxwiz::Tensor host({1, seq, vocab_}, values.data());
        cyxwiz::Tensor device;
        device.SetFromSemanticArray(host.GetSemanticArray(), host.Shape());
        device.GetSemanticArray().eval();
        return device;
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override { return grad; }
    std::string GetName() const override { return "DeviceLogitFixture"; }
private:
    size_t vocab_;
};

uint64_t readback_bytes = 0, readback_count = 0, fallback_count = 0;
void ObserveReadback(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++readback_count;
    readback_bytes += event.bytes;
    Check(event.attribution_category == "output_materialization" &&
          event.attribution_operation == "GenerateTokenIdsWithReport::NextTokenLogits",
          "generation must name its host output boundary");
    Check(event.tensor_shape.size() == 3 && event.tensor_shape[0] == 1 &&
          event.tensor_shape[1] == 1, "only final-position logits may reach host");
}
void ObserveFallback(const cyxwiz::ArrayFireNativeCpuFallbackEvent&) { ++fallback_count; }

void TestDeviceGenerationReadback() {
    size_t cases = 0;
    for (const size_t vocab : {size_t(7), size_t(2540)}) {
        for (const size_t prefix : {size_t(1), size_t(2), size_t(24)}) {
            for (int mode = 0; mode < 3; ++mode) {
                for (const uint32_t seed : {52u, 97u}) {
                    cyxwiz::SequentialModel model;
                    model.Add<DeviceLogitModule>(vocab);
                    cyxwiz::LanguageModelGenerationConfig config;
                    config.max_new_tokens = 4;
                    config.max_context_tokens = prefix + 3;
                    config.include_prompt = mode == 0;
                    config.temperature = .7f;
                    if (mode != 0) config.sampling_mode = cyxwiz::LanguageModelSamplingMode::Multinomial;
                    if (mode == 1) config.top_k = 5;
                    if (mode == 2) config.top_p = .72f;
                    const std::vector<int64_t> prompt(prefix, 1);
                    cyxwiz::LanguageModelGenerationResult report;
                    readback_bytes = readback_count = fallback_count = 0;
                    {
                        cyxwiz::ScopedArrayFireHostSyncObserver reads(ObserveReadback);
                        cyxwiz::ScopedArrayFireNativeCpuFallbackObserver fallbacks(ObserveFallback);
                        report = cyxwiz::GenerateTokenIdsWithReport(model, prompt, config, seed);
                    }
                    Check(report.steps.size() == 3 && report.remaining_budget == 0,
                          "context stopping metadata must remain unchanged");
                    Check(report.prompt_length == prefix && report.max_new_tokens == 4 &&
                          report.include_prompt == config.include_prompt &&
                          report.stop_reason == cyxwiz::LanguageModelGenerationStopReason::MaxTokens,
                          "generation report contract");
                    Check(readback_count == 3 && readback_bytes == 3 * vocab * sizeof(float) &&
                          fallback_count == 0, "one vocabulary row per token, zero native fallback");
                    std::mt19937 rng(seed);
                    std::vector<int64_t> expected;
                    for (size_t step = 0; step < report.steps.size(); ++step) {
                        const auto reference = cyxwiz::SelectNextTokenFromLogits(
                            PrefixLogits(prefix + step, vocab), 1, prefix + step, vocab, config, rng);
                        const auto& actual = report.steps[step];
                        Check(actual.step_index == step && actual.input_length == prefix + step &&
                              actual.token_id == reference.token_id, "seeded token/step parity");
                        CheckNear(actual.probability, reference.probability, 0, "selected probability parity");
                        Check(actual.candidates.size() == reference.candidates.size(), "candidate count parity");
                        for (size_t i = 0; i < actual.candidates.size(); ++i) {
                            Check(actual.candidates[i].token_id == reference.candidates[i].token_id,
                                  "candidate ID parity");
                            CheckNear(actual.candidates[i].probability, reference.candidates[i].probability,
                                      0, "candidate probability parity");
                        }
                        expected.push_back(reference.token_id);
                    }
                    Check(report.new_token_ids == expected, "new token sequence parity");
                    if (config.include_prompt) expected.insert(expected.begin(), prompt.begin(), prompt.end());
                    Check(report.token_ids == expected, "returned token sequence parity");
                    ++cases;
                }
            }
        }
    }
    std::cout << cases << " device readback/candidate/seed/context cases passed; zero native fallback\n";
}

class InvalidOutputModule : public cyxwiz::Module {
public:
    explicit InvalidOutputModule(cyxwiz::Tensor output) : output_(std::move(output)) {}
    cyxwiz::Tensor Forward(const cyxwiz::Tensor&) override { return output_; }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override { return grad; }
    std::string GetName() const override { return "InvalidOutputFixture"; }
private:
    cyxwiz::Tensor output_;
};

void TestInvalidModelOutput() {
    for (const auto& shape : std::vector<std::vector<size_t>>{{2, 2, 7}, {1, 1, 7}, {2, 7}, {1, 2, 7}}) {
        const auto dtype = shape == std::vector<size_t>{1, 2, 7}
            ? cyxwiz::DataType::Int64 : cyxwiz::DataType::Float32;
        cyxwiz::SequentialModel model;
        model.Add<InvalidOutputModule>(cyxwiz::Tensor(shape, dtype));
        bool rejected = false;
        try { (void)cyxwiz::GenerateTokenIdsWithReport(model, {1, 2}, {}); }
        catch (const std::invalid_argument&) { rejected = true; }
        Check(rejected, "invalid model output must be rejected before slicing");
    }
}

} // namespace

void TestTrainingGenerationPreview();

int main(int argc, char** argv) {
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    Check(argc <= 2 && (backend == "cpu" || backend == "cuda" || backend == "opencl"),
          "Usage: test_language_model_generation [cpu|cuda|opencl]");
    const auto activation = cyxwiz::Device(
        backend == "cpu" ? cyxwiz::DeviceType::CPU :
        backend == "cuda" ? cyxwiz::DeviceType::CUDA : cyxwiz::DeviceType::OPENCL, 0).ActivateExact(true);
    Check(activation.success && activation.execution_validated, activation.message);
    const auto policy = cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
    const auto context = cyxwiz::CaptureCurrentExecutionDeviceContext(policy);
    cyxwiz::ScopedActiveExecutionDeviceContext active;
    cyxwiz::ScopedExecutionDeviceContext binding(context);
    cyxwiz::ScopedArrayFireFallbackPolicy strict(policy);
    std::cout << "Exact activation validated: " << backend << "\n" << context.Describe() << "\n";
    TestDeviceGenerationReadback();
    TestInvalidModelOutput();
    TestTrainingGenerationPreview();
    TestConfigValidation();
    TestStopReasonNames();
    TestGreedyTopKDistribution();
    TestTopPFiltering();
    TestMultinomialSelectionIsSeeded();
    TestShapeValidation();
    TestGenerationPanelRunMetadataSmoke();
    TestGenerateTokenIdsWithReportStopsAtEos();
    TestGenerateTokenIdsWithReportStopsAtMaxTokens();
    TestGenerateTokenIdsWithConfigCompatibilityWrapper();
    TestGenerateTokenIdsWithReportRejectsBadPrompt();
    std::cout << "Language model generation controls test passed\n";
    return 0;
}
