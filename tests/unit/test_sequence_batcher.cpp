#include <catch2/catch_test_macros.hpp>

#include "../../cyxwiz-engine/src/core/sequence_batcher.h"

#include <cstdint>
#include <vector>

namespace {

std::vector<int64_t> TensorToInt64Vector(const cyxwiz::Tensor& tensor) {
    std::vector<int64_t> values(tensor.NumElements());
    const int64_t* data = tensor.Data<int64_t>();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = data[i];
    }
    return values;
}

} // namespace

TEST_CASE("SequenceBatcher builds causal LM shifted targets",
          "[sequence_batcher][language_model]") {
    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 1;
    config.max_sequence_length = 3;
    config.create_causal_lm_targets = true;
    config.word_pad_id = 0;
    config.target_ignore_index = -100;

    cyxwiz::SequenceBatcher batcher({{{10, 11, 12, 13}, {}, {}}}, config);
    const cyxwiz::SequenceBatch batch = batcher.GetNextSequenceBatch();

    REQUIRE(batch.IsValid());
    REQUIRE(batch.IsLanguageModeling());
    REQUIRE_FALSE(batch.HasTagIds());
    REQUIRE(batch.word_ids.Shape() == std::vector<size_t>{1, 3});
    REQUIRE(batch.target_ids.Shape() == std::vector<size_t>{1, 3});
    REQUIRE(TensorToInt64Vector(batch.word_ids) == std::vector<int64_t>{10, 11, 12});
    REQUIRE(TensorToInt64Vector(batch.target_ids) == std::vector<int64_t>{11, 12, 13});
    REQUIRE(TensorToInt64Vector(batch.attention_mask) == std::vector<int64_t>{1, 1, 1});
}

TEST_CASE("SequenceBatcher causal LM targets ignore missing and padded tokens",
          "[sequence_batcher][language_model]") {
    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.max_sequence_length = 4;
    config.create_causal_lm_targets = true;
    config.word_pad_id = 0;
    config.target_ignore_index = -100;

    cyxwiz::SequenceBatcher batcher({
        {{7, 8}, {}, {}},
        {{4, 0, 5}, {}, {}}
    }, config);
    const cyxwiz::SequenceBatch batch = batcher.GetNextSequenceBatch();

    REQUIRE(batch.size == 2);
    REQUIRE(batch.sequence_length == 4);
    REQUIRE(TensorToInt64Vector(batch.word_ids) == std::vector<int64_t>{
        7, 8, 0, 0,
        4, 0, 5, 0
    });
    REQUIRE(TensorToInt64Vector(batch.target_ids) == std::vector<int64_t>{
        8, -100, -100, -100,
        -100, -100, -100, -100
    });
    REQUIRE(TensorToInt64Vector(batch.attention_mask) == std::vector<int64_t>{
        1, 1, 0, 0,
        1, 0, 1, 0
    });
}

TEST_CASE("SequenceBatcher preserves token-tagging batches by default",
          "[sequence_batcher][ner]") {
    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 1;
    config.max_sequence_length = 3;
    config.create_causal_lm_targets = false;
    config.tag_ignore_index = -100;

    cyxwiz::SequenceBatcher batcher({{{1, 2}, {}, {5, 6}}}, config);
    const cyxwiz::SequenceBatch batch = batcher.GetNextSequenceBatch();

    REQUIRE(batch.IsSupervised());
    REQUIRE_FALSE(batch.IsLanguageModeling());
    REQUIRE(batch.HasTagIds());
    REQUIRE_FALSE(batch.HasTargetIds());
    REQUIRE(TensorToInt64Vector(batch.word_ids) == std::vector<int64_t>{1, 2, 0});
    REQUIRE(TensorToInt64Vector(batch.tag_ids) == std::vector<int64_t>{5, 6, -100});
}
