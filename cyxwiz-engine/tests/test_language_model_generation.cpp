#include "core/language_model_generation.h"

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
    if (std::fabs(actual - expected) > tolerance) {
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

void TestGenerateTokenIdsWithConfigRuntime() {
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
          "runtime generation should emit generated-only tokens and stop at EOS");
}

} // namespace

int main() {
    TestConfigValidation();
    TestGreedyTopKDistribution();
    TestTopPFiltering();
    TestMultinomialSelectionIsSeeded();
    TestShapeValidation();
    TestGenerateTokenIdsWithConfigRuntime();
    std::cout << "Language model generation controls test passed\n";
    return 0;
}
