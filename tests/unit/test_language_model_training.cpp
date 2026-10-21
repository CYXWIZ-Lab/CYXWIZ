#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cyxwiz/loss.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>

#include "../../cyxwiz-engine/src/core/language_model_training.h"

#include <stdexcept>
#include <map>
#include <string>
#include <vector>

namespace {

void RequireApproxVector(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float tolerance = 1.0e-5f) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(tolerance));
    }
}

} // namespace

TEST_CASE("Causal LM batch shifts next-token targets",
          "[language_model][training]") {
    const auto batch = cyxwiz::BuildCausalLmBatch(
        {{10, 11, 12, 13}},
        3,
        0,
        -100);

    REQUIRE(batch.batch_size == 1);
    REQUIRE(batch.sequence_length == 3);
    REQUIRE(batch.input_ids == std::vector<int>{10, 11, 12});
    REQUIRE(batch.target_ids == std::vector<int>{11, 12, 13});
    REQUIRE(batch.attention_mask == std::vector<int>{1, 1, 1});
}

TEST_CASE("Causal LM batch pads short sequences and ignores missing targets",
          "[language_model][training]") {
    const auto batch = cyxwiz::BuildCausalLmBatch(
        {{7, 8}},
        4,
        0,
        -100);

    REQUIRE(batch.input_ids == std::vector<int>{7, 8, 0, 0});
    REQUIRE(batch.target_ids == std::vector<int>{8, -100, -100, -100});
    REQUIRE(batch.attention_mask == std::vector<int>{1, 1, 0, 0});
}

TEST_CASE("Causal LM batch ignores padding-token targets",
          "[language_model][training]") {
    const auto batch = cyxwiz::BuildCausalLmBatch(
        {{4, 0, 5}},
        3,
        0,
        -100);

    REQUIRE(batch.input_ids == std::vector<int>{4, 0, 5});
    REQUIRE(batch.target_ids == std::vector<int>{-100, -100, -100});
    REQUIRE(batch.attention_mask == std::vector<int>{1, 0, 1});
}

TEST_CASE("Causal LM batch validates inputs",
          "[language_model][training]") {
    REQUIRE_THROWS_AS(
        cyxwiz::BuildCausalLmBatch({}, 3, 0, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildCausalLmBatch({{1, 2}}, 0, 0, -100),
        std::invalid_argument);
}

TEST_CASE("Causal LM token packing creates fixed windows",
          "[language_model][training]") {
    const auto windows = cyxwiz::PackTokenStreamForCausalLm(
        {1, 2, 3, 4, 5, 6},
        4,
        2);

    REQUIRE(windows == std::vector<std::vector<int>>{
        {1, 2, 3, 4},
        {3, 4, 5, 6}
    });
}

TEST_CASE("Causal LM token packing supports overlapping windows",
          "[language_model][training]") {
    const auto windows = cyxwiz::PackTokenStreamForCausalLm(
        {9, 8, 7, 6},
        3,
        1);

    REQUIRE(windows == std::vector<std::vector<int>>{
        {9, 8, 7},
        {8, 7, 6}
    });
}

TEST_CASE("Causal LM token packing validates window settings",
          "[language_model][training]") {
    REQUIRE(cyxwiz::PackTokenStreamForCausalLm({1, 2}, 3, 1).empty());
    REQUIRE_THROWS_AS(
        cyxwiz::PackTokenStreamForCausalLm({1, 2}, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::PackTokenStreamForCausalLm({1, 2}, 2, 0),
        std::invalid_argument);
}

TEST_CASE("Instruction records format into causal LM text",
          "[language_model][instruction_dataset]") {
    const cyxwiz::InstructionTextRecord record{
        "What is CyxWiz?",
        "A graph-driven ML engine.",
        "Answer briefly."
    };

    const std::string text = cyxwiz::FormatInstructionTextRecord(record);

    REQUIRE(text ==
            "System: Answer briefly.\n"
            "User: What is CyxWiz?\n"
            "Assistant: A graph-driven ML engine.\n"
            "[EOS]");
}

TEST_CASE("Instruction records support custom query answer format",
          "[language_model][instruction_dataset]") {
    const cyxwiz::InstructionTextRecord record{
        "query text",
        "answer text",
        ""
    };
    cyxwiz::InstructionTextFormatOptions options;
    options.prompt_prefix = "Query: ";
    options.response_prefix = "Answer: ";
    options.append_eos_text = false;

    const std::string text =
        cyxwiz::FormatInstructionTextRecord(record, options);

    REQUIRE(text == "Query: query text\nAnswer: answer text");
}

TEST_CASE("Instruction records pack through the causal LM window packer",
          "[language_model][instruction_dataset]") {
    const std::vector<cyxwiz::InstructionTextRecord> records = {
        {"p", "r", ""}
    };

    const auto windows = cyxwiz::PackInstructionRecordsForCausalLm(
        records,
        [](const std::string&) {
            return std::vector<int>{1, 2, 3, 4, 5};
        },
        3,
        2);

    REQUIRE(windows == std::vector<std::vector<int>>{
        {1, 2, 3},
        {3, 4, 5}
    });
}

TEST_CASE("Instruction records map query answer table rows",
          "[language_model][instruction_dataset]") {
    const std::vector<std::map<std::string, std::string>> rows = {
        {{"query", "What is CyxWiz?"}, {"answer", "A graph-driven ML engine."}}
    };
    cyxwiz::InstructionRecordColumnMapping mapping;
    mapping.prompt_column = "query";
    mapping.response_column = "answer";

    const auto records = cyxwiz::BuildInstructionRecordsFromRows(rows, mapping);

    REQUIRE(records.size() == 1);
    REQUIRE(records[0].prompt == "What is CyxWiz?");
    REQUIRE(records[0].response == "A graph-driven ML engine.");
    REQUIRE(records[0].system.empty());
}

TEST_CASE("Instruction records map optional system column",
          "[language_model][instruction_dataset]") {
    const std::vector<std::map<std::string, std::string>> rows = {
        {{"prompt", "Explain."},
         {"response", "Short answer."},
         {"system", "Be concise."}}
    };
    cyxwiz::InstructionRecordColumnMapping mapping;
    mapping.system_column = "system";

    const auto records = cyxwiz::BuildInstructionRecordsFromRows(rows, mapping);

    REQUIRE(records.size() == 1);
    REQUIRE(records[0].system == "Be concise.");
}

TEST_CASE("Instruction records can skip empty rows when requested",
          "[language_model][instruction_dataset]") {
    const std::vector<std::map<std::string, std::string>> rows = {
        {{"prompt", ""}, {"response", "skip"}},
        {{"prompt", "keep"}, {"response", "answer"}}
    };
    cyxwiz::InstructionRecordColumnMapping mapping;
    mapping.skip_empty_records = true;

    const auto records = cyxwiz::BuildInstructionRecordsFromRows(rows, mapping);

    REQUIRE(records.size() == 1);
    REQUIRE(records[0].prompt == "keep");
    REQUIRE(records[0].response == "answer");
}

TEST_CASE("Instruction record formatting validates required fields",
          "[language_model][instruction_dataset]") {
    REQUIRE_THROWS_AS(
        cyxwiz::FormatInstructionTextRecord({"", "answer", ""}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::FormatInstructionTextRecord({"prompt", "", ""}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::FormatInstructionTextRecords({}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::PackInstructionRecordsForCausalLm(
            {{"prompt", "answer", ""}},
            {},
            3,
            1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildInstructionRecordsFromRows({}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildInstructionRecordsFromRows(
            {{{"prompt", "p"}}}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildInstructionRecordsFromRows(
            {{{"prompt", ""}, {"response", "r"}}}),
        std::invalid_argument);
}

TEST_CASE("Causal attention mask blocks future positions",
          "[language_model][training][attention]") {
    const auto mask = cyxwiz::BuildCausalAttentionMask(4);

    REQUIRE(mask == std::vector<float>{
        0.0f, -1.0e9f, -1.0e9f, -1.0e9f,
        0.0f, 0.0f, -1.0e9f, -1.0e9f,
        0.0f, 0.0f, 0.0f, -1.0e9f,
        0.0f, 0.0f, 0.0f, 0.0f
    });
}

TEST_CASE("Causal attention mask supports explicit mask values",
          "[language_model][training][attention]") {
    const auto mask = cyxwiz::BuildCausalAttentionMask(3, 1.0f, -5.0f);

    REQUIRE(mask == std::vector<float>{
        1.0f, -5.0f, -5.0f,
        1.0f, 1.0f, -5.0f,
        1.0f, 1.0f, 1.0f
    });
}

TEST_CASE("Causal attention mask validates sequence length",
          "[language_model][training][attention]") {
    REQUIRE_THROWS_AS(
        cyxwiz::BuildCausalAttentionMask(0),
        std::invalid_argument);
}

TEST_CASE("Batched causal key-padding mask blocks future and padded keys",
          "[language_model][training][attention]") {
    const auto mask = cyxwiz::BuildBatchedCausalKeyPaddingMask(
        {1, 1, 0},
        1,
        3);

    REQUIRE(mask == std::vector<float>{
        0.0f, -1.0e9f, -1.0e9f,
        0.0f, 0.0f, -1.0e9f,
        0.0f, 0.0f, -1.0e9f
    });
}

TEST_CASE("Batched causal key-padding mask handles multiple batches",
          "[language_model][training][attention]") {
    const auto mask = cyxwiz::BuildBatchedCausalKeyPaddingMask(
        {1, 0, 1, 1},
        2,
        2,
        1.0f,
        -5.0f);

    REQUIRE(mask == std::vector<float>{
        1.0f, -5.0f,
        1.0f, -5.0f,
        1.0f, -5.0f,
        1.0f, 1.0f
    });
}

TEST_CASE("Batched causal key-padding mask validates shape",
          "[language_model][training][attention]") {
    REQUIRE_THROWS_AS(
        cyxwiz::BuildBatchedCausalKeyPaddingMask({}, 0, 2),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildBatchedCausalKeyPaddingMask({1, 1}, 1, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::BuildBatchedCausalKeyPaddingMask({1, 1}, 2, 2),
        std::invalid_argument);
}

TEST_CASE("Scaled dot-product attention CPU computes unmasked output",
          "[language_model][training][attention]") {
    const std::vector<float> query = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const std::vector<float> key = query;
    const std::vector<float> value = {
        10.0f, 0.0f,
        0.0f, 20.0f
    };

    const auto output = cyxwiz::ScaledDotProductAttentionCpu(
        query,
        key,
        value,
        {},
        2,
        2);

    RequireApproxVector(output, {
        6.6976166f, 6.6047668f,
        3.3023834f, 13.3952332f
    });
}

TEST_CASE("Scaled dot-product attention CPU applies additive mask",
          "[language_model][training][attention]") {
    const std::vector<float> query = {
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const std::vector<float> key = query;
    const std::vector<float> value = {
        10.0f, 0.0f,
        0.0f, 20.0f
    };
    const std::vector<float> mask = {
        0.0f, -1.0e9f,
        0.0f, 0.0f
    };

    const auto output = cyxwiz::ScaledDotProductAttentionCpu(
        query,
        key,
        value,
        mask,
        2,
        2);

    RequireApproxVector(output, {
        10.0f, 0.0f,
        3.3023834f, 13.3952332f
    });
}

TEST_CASE("Scaled dot-product attention CPU validates shapes",
          "[language_model][training][attention]") {
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionCpu({}, {}, {}, {}, 0, 2),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionCpu({1.0f}, {1.0f}, {1.0f}, {}, 1, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionCpu({1.0f}, {}, {1.0f}, {}, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionCpu({1.0f}, {1.0f}, {1.0f}, {0.0f, 0.0f}, 1, 1),
        std::invalid_argument);
}

TEST_CASE("Batched scaled dot-product attention CPU computes per batch outputs",
          "[language_model][training][attention]") {
    const std::vector<float> query = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const std::vector<float> key = query;
    const std::vector<float> value = {
        10.0f, 0.0f,
        0.0f, 20.0f,
        30.0f, 0.0f,
        0.0f, 40.0f
    };
    const std::vector<float> mask = {
        0.0f, -1.0e9f,
        0.0f, 0.0f,
        0.0f, -1.0e9f,
        0.0f, 0.0f
    };

    const auto output = cyxwiz::ScaledDotProductAttentionBatchedCpu(
        query,
        key,
        value,
        mask,
        2,
        2,
        2);

    RequireApproxVector(output, {
        10.0f, 0.0f,
        3.3023834f, 13.3952332f,
        30.0f, 0.0f,
        9.9071503f, 26.7904663f
    });
}

TEST_CASE("Batched scaled dot-product attention CPU validates shapes",
          "[language_model][training][attention]") {
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionBatchedCpu({}, {}, {}, {}, 0, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionBatchedCpu({1.0f}, {1.0f}, {1.0f}, {}, 1, 0, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionBatchedCpu({1.0f}, {1.0f}, {1.0f}, {}, 1, 1, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionBatchedCpu({1.0f}, {}, {1.0f}, {}, 1, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionBatchedCpu({1.0f}, {1.0f}, {1.0f}, {0.0f, 0.0f}, 1, 1, 1),
        std::invalid_argument);
}

TEST_CASE("Multi-head scaled dot-product attention CPU computes per-head outputs",
          "[language_model][training][attention]") {
    const std::vector<float> query = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const std::vector<float> key = query;
    const std::vector<float> value = {
        10.0f, 0.0f,
        0.0f, 20.0f,
        30.0f, 0.0f,
        0.0f, 40.0f
    };
    const std::vector<float> mask = {
        0.0f, -1.0e9f,
        0.0f, 0.0f,
        0.0f, -1.0e9f,
        0.0f, 0.0f
    };

    const auto output = cyxwiz::ScaledDotProductAttentionMultiHeadCpu(
        query,
        key,
        value,
        mask,
        1,
        2,
        2,
        2);

    RequireApproxVector(output, {
        10.0f, 0.0f,
        3.3023834f, 13.3952332f,
        30.0f, 0.0f,
        9.9071503f, 26.7904663f
    });
}

TEST_CASE("Multi-head scaled dot-product attention CPU validates shapes",
          "[language_model][training][attention]") {
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionMultiHeadCpu({}, {}, {}, {}, 0, 1, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionMultiHeadCpu({}, {}, {}, {}, 1, 0, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionMultiHeadCpu({1.0f}, {}, {1.0f}, {}, 1, 1, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::ScaledDotProductAttentionMultiHeadCpu({1.0f}, {1.0f}, {1.0f}, {0.0f, 0.0f}, 1, 1, 1, 1),
        std::invalid_argument);
}

TEST_CASE("Token cross entropy CPU matches toy reference loss",
          "[language_model][training][loss]") {
    const std::vector<float> logits = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    const std::vector<int> targets = {0, 2};

    const auto result = cyxwiz::TokenCrossEntropyLossCpu(
        logits,
        targets,
        1,
        2,
        3,
        -100);

    REQUIRE(result.valid_token_count == 2);
    REQUIRE(result.loss == Catch::Approx(0.40760595f).margin(1e-6f));
}

TEST_CASE("Token cross entropy CPU ignores padding targets",
          "[language_model][training][loss]") {
    const std::vector<float> logits = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    const std::vector<int> targets = {0, -100};

    const auto result = cyxwiz::TokenCrossEntropyLossCpu(
        logits,
        targets,
        1,
        2,
        3,
        -100);

    REQUIRE(result.valid_token_count == 1);
    REQUIRE(result.loss == Catch::Approx(0.40760595f).margin(1e-6f));
}

TEST_CASE("Token cross entropy CPU gradient matches softmax reference",
          "[language_model][training][loss]") {
    const std::vector<float> logits = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    const std::vector<int> targets = {0, -100};

    const auto gradient = cyxwiz::TokenCrossEntropyGradientCpu(
        logits,
        targets,
        1,
        2,
        3,
        -100);

    RequireApproxVector(gradient, {
        -0.3347590f, 0.2447285f, 0.0900306f,
        0.0f, 0.0f, 0.0f
    });
}

TEST_CASE("Token cross entropy CPU validates shapes and labels",
          "[language_model][training][loss]") {
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({}, {}, 0, 1, 1, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({1.0f}, {0}, 1, 0, 1, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({1.0f}, {0}, 1, 1, 0, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({1.0f}, {}, 1, 1, 1, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({1.0f}, {1}, 1, 1, 1, -100),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::TokenCrossEntropyLossCpu({1.0f}, {-100}, 1, 1, 1, -100),
        std::invalid_argument);
}

TEST_CASE("Token cross entropy CPU reference matches backend CrossEntropyLoss",
          "[language_model][training][loss]") {
    const std::vector<float> logits = {
        2.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 2.0f
    };
    const std::vector<int> targets = {0, -100};

    const auto reference_loss = cyxwiz::TokenCrossEntropyLossCpu(
        logits,
        targets,
        1,
        2,
        3,
        -100);
    const auto reference_gradient = cyxwiz::TokenCrossEntropyGradientCpu(
        logits,
        targets,
        1,
        2,
        3,
        -100);

    cyxwiz::Tensor logits_tensor({1, 2, 3}, logits.data(), cyxwiz::DataType::Float32);
    int64_t target_values[] = {0, -100};
    cyxwiz::Tensor target_tensor({1, 2}, target_values, cyxwiz::DataType::Int64);

    cyxwiz::CrossEntropyLoss backend_loss(cyxwiz::Reduction::Mean, -100);
    cyxwiz::Tensor backend_loss_value = backend_loss.Forward(logits_tensor, target_tensor);
    cyxwiz::Tensor backend_gradient = backend_loss.Backward(logits_tensor, target_tensor);

    REQUIRE(backend_loss_value.Data<float>()[0] ==
            Catch::Approx(reference_loss.loss).margin(1.0e-6f));

    const float* gradient_data = backend_gradient.Data<float>();
    REQUIRE(backend_gradient.NumElements() == reference_gradient.size());
    for (size_t i = 0; i < reference_gradient.size(); ++i) {
        REQUIRE(gradient_data[i] ==
                Catch::Approx(reference_gradient[i]).margin(1.0e-6f));
    }
}

TEST_CASE("Greedy generation selects the last-position argmax token",
          "[language_model][generation]") {
    const std::vector<float> logits = {
        9.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 5.0f
    };

    REQUIRE(cyxwiz::GreedyNextTokenFromLogits(logits, 1, 2, 3) == 2);
}

TEST_CASE("Greedy generation validates logits shape",
          "[language_model][generation]") {
    REQUIRE_THROWS_AS(
        cyxwiz::GreedyNextTokenFromLogits({}, 0, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::GreedyNextTokenFromLogits({1.0f}, 1, 0, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::GreedyNextTokenFromLogits({1.0f}, 1, 1, 0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::GreedyNextTokenFromLogits({1.0f}, 1, 1, 1, 1),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        cyxwiz::GreedyNextTokenFromLogits({1.0f}, 1, 2, 1),
        std::invalid_argument);
}

TEST_CASE("Greedy generation appends tokens from a sequential LM head",
          "[language_model][generation]") {
    cyxwiz::SequentialModel model;
    model.Add<cyxwiz::EmbeddingModule>(3, 2, -1);
    model.Add<cyxwiz::TimeDistributedDenseModule>(2, 3, true);

    const auto generated = cyxwiz::GenerateGreedyTokenIds(
        model,
        {0},
        2);

    REQUIRE(generated.size() == 3);
    REQUIRE(generated[0] == 0);
    for (int64_t token : generated) {
        REQUIRE(token >= 0);
        REQUIRE(token < 3);
    }
}

TEST_CASE("Greedy generation validates prompt input",
          "[language_model][generation]") {
    cyxwiz::SequentialModel model;
    REQUIRE_THROWS_AS(
        cyxwiz::GenerateGreedyTokenIds(model, {}, 1),
        std::invalid_argument);
}
