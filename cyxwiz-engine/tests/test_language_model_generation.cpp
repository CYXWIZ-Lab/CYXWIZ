#include "core/language_model_generation.h"
#include "core/execution_device_context.h"
#include "core/executable_model.h"

#include <cyxwiz/sequential.h>
#include <cyxwiz/layers/transformer.h>

#include <algorithm>
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

// A model that is not a layer stack (like a graph with branches): generation
// recomputes the whole sequence through Forward each token.
class ScriptedGraphModel final : public cyxwiz::IExecutableModel {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override { ++forwards; return module.Forward(input); }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad_output) override { return grad_output; }
    void SetTraining(bool) override {}
    std::map<std::string, cyxwiz::Tensor> GetParameters() override { return {}; }
    void SetParameters(const std::map<std::string, cyxwiz::Tensor>&) override {}
    std::map<std::string, cyxwiz::Tensor> GetGradients() override { return {}; }
    void UpdateParameters(cyxwiz::Optimizer*) override {}
    size_t LayerCount() const override { return 1; }
    ScriptedLogitModule module;
    int forwards = 0;
};

void TestGenerationOnAnyExecutableModelStreamsSteps() {
    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 3;
    config.include_prompt = false;
    std::vector<int64_t> streamed;
    config.on_step = [&streamed](const cyxwiz::LanguageModelGenerationStep& step) {
        streamed.push_back(step.token_id);
    };

    ScriptedGraphModel graph;
    const auto from_graph = cyxwiz::GenerateTokenIdsWithReport(
        static_cast<cyxwiz::IExecutableModel&>(graph), {1, 2}, config);
    Check(from_graph.new_token_ids == std::vector<int64_t>({3, 4, 4}),
          "a non-layer-stack executable model should generate through Forward");
    Check(graph.forwards == 3 && !from_graph.used_kv_cache,
          "a non-layer-stack model recomputes the sequence for every token");
    Check(streamed == from_graph.new_token_ids, "on_step should report every new token in order");

    streamed.clear();
    auto stack = std::make_shared<cyxwiz::SequentialModel>();
    stack->Add<ScriptedLogitModule>();
    cyxwiz::SequentialExecutableModel wrapped(stack);
    config.stop_check = [](const std::vector<int64_t>& ids) { return ids.size() == 2; };
    const auto from_stack = cyxwiz::GenerateTokenIdsWithReport(
        static_cast<cyxwiz::IExecutableModel&>(wrapped), {1, 2}, config);
    Check(from_stack.new_token_ids == std::vector<int64_t>({3, 4}) &&
              from_stack.stop_reason == cyxwiz::LanguageModelGenerationStopReason::StopText,
          "a wrapped layer stack should take the layer-stack path and honour the stop check");
    Check(streamed == from_stack.new_token_ids, "on_step should run before the stop check ends generation");
}

// Next-token logits from the last token alone: after 1, token 2 (p 0.6) or 3
// (p 0.4); after 2, all five alike; after 3, almost surely 4; after 4,
// token 4 at 2.0 and token 3 at 1.9.
class BranchingLogits : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& input) override {
        const size_t seq = input.Shape()[1];
        const int64_t last = input.ReadData<int64_t>()[seq - 1];
        std::vector<float> logits(seq * 5, 0.0f);
        float* row = logits.data() + (seq - 1) * 5;
        for (int v = 0; v < 5; ++v) row[v] = -20.0f;
        if (last == 1) { row[2] = std::log(0.6f); row[3] = std::log(0.4f); }
        else if (last == 2) { for (int v = 0; v < 5; ++v) row[v] = 0.0f; }
        else if (last == 3) { row[4] = 10.0f; }
        else { row[4] = 2.0f; row[3] = 1.9f; }
        return cyxwiz::Tensor({1, seq, 5}, logits.data());
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& g) override { return g; }
    std::string GetName() const override { return "BranchingLogits"; }
};

void TestDecodingOptions() {
    // Min-p keeps tokens at least that fraction as likely as the best one.
    cyxwiz::LanguageModelGenerationConfig config;
    const std::vector<float> logits = {2.0f, 1.0f, 0.0f, -1.0f};
    config.min_p = 0.5f;
    Check(cyxwiz::BuildNextTokenDistribution(logits, 1, 1, 4, config).size() == 1, "min-p 0.5 keeps the best token");
    config.min_p = 0.3f;
    Check(cyxwiz::BuildNextTokenDistribution(logits, 1, 1, 4, config).size() == 2, "min-p 0.3 keeps two tokens");
    config.min_p = 1.0f;
    Check(!cyxwiz::ValidateLanguageModelGenerationConfig(config).empty(), "min-p 1 is refused");

    cyxwiz::SequentialModel model;
    model.Add<BranchingLogits>();
    cyxwiz::LanguageModelGenerationConfig greedy;
    greedy.max_new_tokens = 3;
    greedy.include_prompt = false;
    Check(cyxwiz::GenerateTokenIdsWithReport(model, {4}, greedy).new_token_ids == std::vector<int64_t>({4, 4, 4}),
          "greedy repeats the most likely token");
    auto frequency = greedy;
    frequency.frequency_penalty = 0.5f;
    Check(cyxwiz::GenerateTokenIdsWithReport(model, {4}, frequency).new_token_ids[1] == 3,
          "a frequency penalty lowers a token the output already used");
    auto presence = greedy;
    presence.presence_penalty = 0.2f;
    Check(cyxwiz::GenerateTokenIdsWithReport(model, {4}, presence).new_token_ids[1] == 3,
          "a presence penalty lowers a token the output already used");
    auto ngram = greedy;
    ngram.no_repeat_ngram_size = 2;
    const auto banned = cyxwiz::GenerateTokenIdsWithReport(model, {4}, ngram).new_token_ids;
    Check(banned == std::vector<int64_t>({4, 3, 4}), "no repeated 2-gram: the pair 4 4 is written once, never again");

    // Beam search finds the sequence greedy misses: 3 then 4 (0.4) beats
    // 2 then anything (0.6 x 0.2).
    greedy.max_new_tokens = 2;
    Check(cyxwiz::GenerateTokenIdsWithReport(model, {1}, greedy).new_token_ids[0] == 2, "greedy takes 2 first");
    auto beam = greedy;
    beam.num_beams = 2;
    const auto best = cyxwiz::GenerateTokenIdsWithReport(model, {1}, beam);
    Check(best.new_token_ids == std::vector<int64_t>({3, 4}) && !best.used_kv_cache,
          "beam search returns the most likely sequence");
    beam.eos_token_id = 4;
    const auto stopped = cyxwiz::GenerateTokenIdsWithReport(model, {1}, beam);
    Check(stopped.stop_reason == cyxwiz::LanguageModelGenerationStopReason::EosToken &&
              stopped.new_token_ids == std::vector<int64_t>({3, 4}),
          "a beam that writes end-of-sequence finishes");
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

uint64_t readback_bytes = 0, readback_count = 0;
void ObserveReadback(const cyxwiz::ArrayFireHostSyncEvent& event) {
    ++readback_count;
    readback_bytes += event.bytes;
    Check(event.attribution_category == "output_materialization" &&
          event.attribution_operation == "GenerateTokenIdsWithReport::NextTokenLogits",
          "generation must name its host output boundary");
    Check(event.tensor_shape.size() == 3 && event.tensor_shape[0] == 1 &&
          event.tensor_shape[1] == 1, "only final-position logits may reach host");
}

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
                    readback_bytes = readback_count = 0;
                    {
                        cyxwiz::ScopedArrayFireHostSyncObserver reads(ObserveReadback);
                        report = cyxwiz::GenerateTokenIdsWithReport(model, prompt, config, seed);
                    }
                    Check(report.steps.size() == 3 && report.remaining_budget == 0,
                          "context stopping metadata must remain unchanged");
                    Check(report.prompt_length == prefix && report.max_new_tokens == 4 &&
                          report.include_prompt == config.include_prompt &&
                          report.stop_reason == cyxwiz::LanguageModelGenerationStopReason::MaxTokens,
                          "generation report contract");
                    Check(readback_count == 3 && readback_bytes == 3 * vocab * sizeof(float),
                          "one vocabulary row per token");
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
    std::cout << cases << " device readback/candidate/seed/context cases passed\n";
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

// KV cache (tofix112 phase 5): for every block design, the cached path must
// give the same next-token logits as full recomputation at every step, and
// greedy generation must return the same tokens.
void TestKvCacheMatchesFullRecompute() {
    using cyxwiz::TransformerBlockOptions;
    constexpr size_t kVocab = 11, kWidth = 16, kHeads = 4, kFeedForward = 24;
    struct Variant {
        std::string label;
        TransformerBlockOptions options;
        bool pre_norm;
        int positions;  // 0 none, 1 sinusoidal, 2 learned
    };
    std::vector<Variant> variants;
    variants.push_back({"classic + sinusoidal", {}, false, 1});
    TransformerBlockOptions llama;
    llama.norm_type = cyxwiz::TransformerNormType::RMSNorm;
    llama.ffn_type = cyxwiz::TransformerFeedForwardType::Gated;
    llama.ffn_activation = cyxwiz::ActivationType::SiLU;
    llama.ffn_bias = false;
    llama.attention_bias = false;
    llama.position_encoding = cyxwiz::TransformerPositionEncoding::Rope;
    variants.push_back({"llama_style", llama, true, 0});
    TransformerBlockOptions gemma = llama;
    gemma.position_encoding = cyxwiz::TransformerPositionEncoding::Alibi;
    gemma.sandwich_norm = true;
    gemma.qk_norm = true;
    gemma.num_kv_heads = 2;
    gemma.sliding_window = 4;
    gemma.attn_logit_softcap = 3.0f;
    variants.push_back({"alibi + gqa + window + softcap + sandwich", gemma, true, 0});
    TransformerBlockOptions parallel = llama;
    parallel.block_layout = cyxwiz::TransformerBlockLayout::Parallel;
    parallel.num_kv_heads = 1;
    parallel.rope_fraction = 0.5f;
    variants.push_back({"parallel + mqa + partial rope + learned positions + tied head", parallel, true, 2});

    for (const auto& variant : variants) {
        cyxwiz::SequentialModel model;
        model.Add<cyxwiz::EmbeddingModule>(kVocab, kWidth);
        if (variant.positions == 1) model.Add<cyxwiz::PositionalEncodingModule>(kWidth, 64);
        if (variant.positions == 2) model.Add<cyxwiz::LearnedPositionalEmbeddingModule>(kWidth, 64);
        for (int block = 0; block < 2; ++block) {
            model.Add<cyxwiz::TransformerDecoderModule>(kWidth, kHeads, kFeedForward, 0.0f, variant.pre_norm, 0.0f,
                                                        variant.options);
        }
        model.Add<cyxwiz::LayerNormModule>(std::vector<int>{static_cast<int>(kWidth)});
        if (variant.positions == 2) {  // also exercise the tied output head
            model.Add<cyxwiz::TiedOutputProjectionModule>(
                *dynamic_cast<cyxwiz::EmbeddingModule*>(model.GetModule(0)), false);
        } else {
            model.Add<cyxwiz::TimeDistributedDenseModule>(kWidth, kVocab);
        }
        model.SetTraining(false);
        Check(model.SupportsIncrementalDecoding(), variant.label + ": model should support KV-cached decoding");

        // Teacher-forced logits: prefill 3 tokens, then one token at a time.
        const std::vector<int64_t> tokens = {1, 4, 7, 2, 9, 3, 3, 5, 10, 0, 6, 8};
        const size_t prefill = 3;
        float worst = 0.0f;
        for (size_t length = prefill; length <= tokens.size(); ++length) {
            const size_t fed = length == prefill ? prefill : 1;
            const cyxwiz::Tensor step_input({1, fed}, tokens.data() + (length - fed), cyxwiz::DataType::Int64);
            const cyxwiz::Tensor cached = model.ForwardIncremental(step_input, length - fed);
            const cyxwiz::Tensor full_input({1, length}, tokens.data(), cyxwiz::DataType::Int64);
            const cyxwiz::Tensor full = model.Forward(full_input);
            Check(cached.Shape() == std::vector<size_t>{1, fed, kVocab}, variant.label + ": cached logits shape");
            const cyxwiz::Tensor cached_last = cached.Slice(1, -1);
            const cyxwiz::Tensor full_last = full.Slice(1, -1);
            const float* a = cached_last.ReadData<float>();
            const float* b = full_last.ReadData<float>();
            for (size_t v = 0; v < kVocab; ++v) {
                CheckNear(a[v], b[v], 2e-4f, variant.label + ": cached vs full logits at length " +
                                                 std::to_string(length));
                worst = std::max(worst, std::fabs(a[v] - b[v]));
            }
        }
        model.ResetIncrementalState();

        cyxwiz::LanguageModelGenerationConfig config;
        config.max_new_tokens = 10;
        config.use_kv_cache = true;
        const auto cached_report = cyxwiz::GenerateTokenIdsWithReport(model, {1, 4, 7}, config, 52);
        config.use_kv_cache = false;
        const auto full_report = cyxwiz::GenerateTokenIdsWithReport(model, {1, 4, 7}, config, 52);
        Check(cached_report.used_kv_cache && !full_report.used_kv_cache, variant.label + ": cache path selection");
        Check(cached_report.token_ids == full_report.token_ids,
              variant.label + ": greedy tokens must match with and without the KV cache");

        // A chat's next turn: the cache kept from the first reply, only the
        // new part fed, gives the tokens a fresh run gives.
        std::vector<int64_t> kept;
        config.use_kv_cache = true;
        config.max_new_tokens = 4;
        config.include_prompt = false;
        config.reuse_cache = &kept;
        const auto first = cyxwiz::GenerateTokenIdsWithReport(model, {1, 4, 7}, config, 52);
        Check(kept.size() == 3 + first.new_token_ids.size() - 1 && first.reused_cache_tokens == 0,
              variant.label + ": the kept cache covers the prompt and every new token but the last");
        std::vector<int64_t> next = {1, 4, 7};
        next.insert(next.end(), first.new_token_ids.begin(), first.new_token_ids.end());
        next.insert(next.end(), {2, 9});
        const size_t held = kept.size();
        const auto reused = cyxwiz::GenerateTokenIdsWithReport(model, next, config, 52);
        Check(reused.reused_cache_tokens == held, variant.label + ": the next turn feeds only what is new");
        config.reuse_cache = nullptr;
        const auto fresh = cyxwiz::GenerateTokenIdsWithReport(model, next, config, 52);
        Check(reused.new_token_ids == fresh.new_token_ids,
              variant.label + ": a reused cache gives the tokens of a fresh run");
        config.reuse_cache = &kept;
        const auto other = cyxwiz::GenerateTokenIdsWithReport(model, {2, 2, 2, 5}, config, 52);
        Check(other.reused_cache_tokens == 0, variant.label + ": a prompt that differs from the cache starts over");
        model.ResetIncrementalState();
        config.reuse_cache = nullptr;
        std::cout << "KV cache parity: " << variant.label << " max |logit diff| " << worst << "\n";
    }

    // Modules without incremental support fall back to full recomputation.
    cyxwiz::SequentialModel scripted;
    scripted.Add<ScriptedLogitModule>();
    Check(!scripted.SupportsIncrementalDecoding(), "scripted module has no incremental support");
    cyxwiz::LanguageModelGenerationConfig config;
    config.max_new_tokens = 2;
    Check(!cyxwiz::GenerateTokenIdsWithReport(scripted, {1}, config).used_kv_cache,
          "unsupported models must use full recomputation");
}

void TestTrainingGenerationPreview();

int main(int argc, char** argv) {
    const std::string backend = argc > 1 ? argv[1] : "cpu";
    Check(argc <= 2 && (backend == "cpu" || backend == "cuda" || backend == "opencl"),
          "Usage: test_language_model_generation [cpu|cuda|opencl]");
    const auto activation = cyxwiz::Device(
        backend == "cpu" ? cyxwiz::DeviceType::CPU :
        backend == "cuda" ? cyxwiz::DeviceType::CUDA : cyxwiz::DeviceType::OPENCL, 0).ActivateExact(true);
    Check(activation.success && activation.execution_validated, activation.message);
    const auto policy = cyxwiz::ExecutionPolicy::Strict;
    const auto context = cyxwiz::CaptureCurrentExecutionDeviceContext(policy);
    cyxwiz::ScopedActiveExecutionDeviceContext active;
    cyxwiz::ScopedExecutionDeviceContext binding(context);
    std::cout << "Exact activation validated: " << backend << "\n" << context.Describe() << "\n";
    TestDeviceGenerationReadback();
    TestInvalidModelOutput();
    TestTrainingGenerationPreview();
    TestKvCacheMatchesFullRecompute();
    TestConfigValidation();
    TestStopReasonNames();
    TestGreedyTopKDistribution();
    TestTopPFiltering();
    TestMultinomialSelectionIsSeeded();
    TestShapeValidation();
    TestGenerateTokenIdsWithReportStopsAtEos();
    TestGenerateTokenIdsWithReportStopsAtMaxTokens();
    TestGenerateTokenIdsWithConfigCompatibilityWrapper();
    TestGenerateTokenIdsWithReportRejectsBadPrompt();
    TestGenerationOnAnyExecutableModelStreamsSteps();
    TestDecodingOptions();
    std::cout << "Language model generation controls test passed\n";
    return 0;
}
