#include "../src/core/arrow_dataset.h"
#include "../src/core/ner_sequence_builder.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/preprocessing_state.h"
#include "../src/core/sequence_batcher.h"
#include "../src/core/sequence_tag_metrics.h"
#include "../src/core/sequence_training_step.h"
#include "../src/core/sequence_vocabulary.h"
#include "../src/core/test_executor.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_trace_collector.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void CheckNear(double actual,
               double expected,
               double tolerance,
               const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

#ifndef NDEBUG
constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value == nullptr) {
            ClearEnvVar(name_);
        } else {
            SetEnvVar(name_, value);
        }
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};
#endif

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTrainingTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.1f, 0.9f, 1.0f, 0.2f, 0.8f}),
            FinishFloatArray({0.0f, 0.2f, 0.8f, 1.0f, 0.1f, 0.9f}),
            FinishFloatArray({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        },
        6);
}

std::shared_ptr<arrow::Table> MakeRegressionTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("target", arrow::float32()),
        arrow::field("target_1", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}),
            FinishFloatArray({1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f}),
            FinishFloatArray({1.5f, 1.7f, 1.9f, 2.1f, 2.3f, 2.5f}),
            FinishFloatArray({-0.5f, -0.1f, 0.3f, 0.7f, 1.1f, 1.5f}),
        },
        6);
}

void WriteParquetWithRowGroupSize(const cyxwiz::ArrowDataset& dataset,
                                  const std::string& path,
                                  int64_t row_group_size) {
    auto table = dataset.GetArrowTable();
    Check(table != nullptr, "source table should exist for row-group parquet write");
    auto output = arrow::io::FileOutputStream::Open(path);
    Check(output.ok(), output.status().ToString());
    auto status = parquet::arrow::WriteTable(*table,
                                             arrow::default_memory_pool(),
                                             *output,
                                             row_group_size);
    Check(status.ok(), status.ToString());
}

cyxwiz::TrainingConfiguration MakeConfig(const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "training_executor_parity";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.train_ratio = 0.67f;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

cyxwiz::TrainingConfiguration MakeRegressionConfig(
    const std::filesystem::path& checkpoint_dir) {
    auto config = MakeConfig(checkpoint_dir);
    config.dataset_name = "training_executor_regression";
    config.output_size = 2;
    config.loss_type = gui::NodeType::MSELoss;
    config.target.required_by_objective = true;
    config.target.origin = cyxwiz::TargetOrigin::DatasetColumn;
    config.target.value_kind = cyxwiz::TargetValueKind::Continuous;
    config.target.primary_column = "target";
    config.target.width = 2;
    config.layers.front().units = 2;
    return config;
}

void TestSequenceBatchContract() {
    cyxwiz::SequenceBatch empty;
    Check(!empty.IsValid(), "empty SequenceBatch should be invalid");
    Check(!empty.IsSupervised(),
          "empty SequenceBatch should not be supervised");

    const std::vector<int64_t> word_ids = {1, 2, 0, 3, 4, 0};
    const std::vector<int64_t> mask = {1, 1, 0, 1, 1, 0};
    cyxwiz::SequenceBatch inference;
    inference.word_ids =
        cyxwiz::Tensor({2, 3}, word_ids.data(), cyxwiz::DataType::Int64);
    inference.attention_mask =
        cyxwiz::Tensor({2, 3}, mask.data(), cyxwiz::DataType::Int64);
    inference.size = 2;
    inference.sequence_length = 3;
    Check(inference.IsValid(),
          "SequenceBatch with word ids should be valid");
    Check(inference.HasAttentionMask(),
          "SequenceBatch should report attention mask");
    Check(!inference.IsSupervised(),
          "SequenceBatch without tag ids should not be supervised");

    const std::vector<int64_t> tag_ids = {5, 6, -100, 7, 8, -100};
    inference.tag_ids =
        cyxwiz::Tensor({2, 3}, tag_ids.data(), cyxwiz::DataType::Int64);
    Check(inference.IsSupervised(),
          "SequenceBatch with tag ids should be supervised");
}

void TestSequenceBatcherPadsNamedPayloads() {
    std::vector<cyxwiz::SequenceSample> samples = {
        {{11, 12, 13}, {1, 2, 3}, {5, 6, 7}},
        {{21}, {4}, {8}},
        {{31, 32, 33, 34}, {9, 10, 11, 12}, {1, 2, 3, 4}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.max_sequence_length = 3;
    config.shuffle = false;
    config.create_attention_mask = true;
    config.tag_ignore_index = -100;

    cyxwiz::SequenceBatcher batcher(samples, config);
    Check(batcher.GetNumSamples() == 3,
          "SequenceBatcher should report sample count");
    Check(batcher.GetNumBatches() == 2,
          "SequenceBatcher should ceil partial final batch");

    auto batch = batcher.GetNextSequenceBatch();
    Check(batch.IsSupervised(),
          "first sequence batch should be supervised");
    Check(batch.HasPosIds(), "first sequence batch should include POS ids");
    Check(batch.HasAttentionMask(),
          "first sequence batch should include attention mask");
    Check(batch.word_ids.Shape() == std::vector<size_t>({2, 3}),
          "word_ids should be [batch, seq]");
    Check(batch.tag_ids.Shape() == std::vector<size_t>({2, 3}),
          "tag_ids should be [batch, seq]");

    const auto* words = batch.word_ids.Data<int64_t>();
    const auto* mask = batch.attention_mask.Data<int64_t>();
    const auto* tags = batch.tag_ids.Data<int64_t>();
    Check(words[0] == 11 && words[1] == 12 && words[2] == 13,
          "first row words should copy exactly");
    Check(words[3] == 21 && words[4] == 0 && words[5] == 0,
          "short row words should pad with word_pad_id");
    Check(mask[0] == 1 && mask[1] == 1 && mask[2] == 1,
          "full row mask should be all ones");
    Check(mask[3] == 1 && mask[4] == 0 && mask[5] == 0,
          "short row mask should mark padding");
    Check(tags[3] == 8 && tags[4] == -100 && tags[5] == -100,
          "short row tags should pad with ignore_index");

    auto final_batch = batcher.GetNextSequenceBatch();
    Check(final_batch.size == 1,
          "final sequence batch should keep partial batch by default");
    const auto* final_words = final_batch.word_ids.Data<int64_t>();
    const auto* final_tags = final_batch.tag_ids.Data<int64_t>();
    Check(final_words[0] == 31 && final_words[1] == 32 &&
              final_words[2] == 33,
          "long row words should truncate to max_sequence_length");
    Check(final_tags[0] == 1 && final_tags[1] == 2 && final_tags[2] == 3,
          "long row tags should truncate with words");
    Check(batcher.IsEpochComplete(),
          "sequence batcher should complete after final batch");
}

void TestSequenceBatcherDropLast() {
    std::vector<cyxwiz::SequenceSample> samples = {
        {{1}, {}, {2}},
        {{3}, {}, {4}},
        {{5}, {}, {6}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.drop_last = true;
    cyxwiz::SequenceBatcher batcher(samples, config);
    Check(batcher.GetNumBatches() == 1,
          "drop_last should floor sequence batch count");
    Check(batcher.GetNextSequenceBatch().size == 2,
          "drop_last first batch should be full");
    Check(!batcher.GetNextSequenceBatch().IsValid(),
          "drop_last should suppress partial final batch");
}

void TestSequenceTagMetrics() {
    const std::vector<std::string> labels = {
        "O",
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
    };

    const std::vector<float> logits = {
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
    };
    const std::vector<int64_t> gold = {
        1, 2, 0, -100,
        3, 4, 0, -100,
    };

    cyxwiz::Tensor logits_tensor(
        {2, 4, labels.size()}, logits.data(), cyxwiz::DataType::Float32);
    cyxwiz::Tensor gold_tensor({2, 4}, gold.data(), cyxwiz::DataType::Int64);
    const auto metrics = cyxwiz::ComputeSequenceTagMetricsFromLogits(
        logits_tensor, gold_tensor, labels, -100);

    Check(metrics.correct_tokens == 5,
          "token metrics should count correct non-ignored tokens");
    Check(metrics.total_tokens == 6,
          "token metrics should skip ignored padding labels");
    CheckNear(metrics.token_accuracy, 5.0 / 6.0, 1e-9,
              "token accuracy should use non-ignored denominator");
    Check(metrics.predicted_entities == 2,
          "BIO metrics should ignore entity predictions on padding labels");
    Check(metrics.gold_entities == 2,
          "BIO metrics should count gold PER and LOC entities");
    Check(metrics.matched_entities == 1,
          "BIO metrics should require exact span/type match");
    CheckNear(metrics.entity_precision, 0.5, 1e-9,
              "BIO precision should match exact entities over predictions");
    CheckNear(metrics.entity_recall, 0.5, 1e-9,
              "BIO recall should match exact entities over gold spans");
    CheckNear(metrics.entity_f1, 0.5, 1e-9,
              "BIO F1 should combine exact precision and recall");
}

void TestSequenceVocabulary() {
    std::vector<std::vector<std::string>> token_sequences = {
        {"John", "lives", "in", "Berlin"},
        {"john", "works", "in", "Berlin"},
        {"Mary", "lives", "there"},
    };

    cyxwiz::SequenceVocabularyConfig token_config;
    token_config.kind = cyxwiz::SequenceVocabularyKind::Token;
    token_config.lowercase = true;
    token_config.min_frequency = 2;
    token_config.max_size = 5;

    const auto token_vocab =
        cyxwiz::BuildSequenceVocabulary(token_sequences, token_config);
    Check(token_vocab.Size() == 5,
          "token vocabulary should honor max_size including PAD/UNK");
    Check(token_vocab.PadId() == 0,
          "token vocabulary should reserve PAD id first");
    Check(token_vocab.UnkId() == 1,
          "token vocabulary should reserve UNK id second");
    Check(token_vocab.IdFor("berlin") == 2,
          "token vocabulary should sort by frequency then lexical order");
    Check(token_vocab.IdFor("in") == 3,
          "token vocabulary should keep frequent tokens");
    Check(token_vocab.IdFor("john") == 4,
          "token vocabulary should lowercase before counting");
    Check(token_vocab.IdFor("mary") == token_vocab.UnkId(),
          "token vocabulary should map filtered tokens to UNK");

    std::vector<std::vector<std::string>> tag_sequences = {
        {"B-PER", "I-PER", "O"},
        {"B-LOC", "O"},
    };
    cyxwiz::SequenceVocabularyConfig tag_config;
    tag_config.kind = cyxwiz::SequenceVocabularyKind::Tag;
    const auto tag_vocab =
        cyxwiz::BuildSequenceVocabulary(tag_sequences, tag_config);
    Check(!tag_vocab.HasPad() && !tag_vocab.HasUnk(),
          "tag vocabulary should not reserve PAD/UNK ids");
    Check(tag_vocab.ValueFor(0) == "O",
          "tag vocabulary should keep O at id zero when present");
    Check(tag_vocab.Contains("B-PER") && tag_vocab.Contains("I-PER"),
          "tag vocabulary should contain BIO labels");
    bool unknown_tag_failed = false;
    try {
        (void)tag_vocab.IdFor("B-ORG");
    } catch (const std::runtime_error&) {
        unknown_tag_failed = true;
    }
    Check(unknown_tag_failed,
          "tag vocabulary should reject unknown labels instead of using UNK");
}

void TestNERSequenceBuilder() {
    std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {"NNP", "VBZ", "IN", "NNP"},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"john", "works"},
         {"NNP", "VBZ"},
         {"B-PER", "O"}},
    };

    cyxwiz::NERSequenceBuilderConfig config;
    config.token_vocabulary.lowercase = true;
    config.batcher.batch_size = 2;
    config.batcher.max_sequence_length = 5;
    config.batcher.shuffle = false;

    const auto built = cyxwiz::BuildNERSequenceData(rows, config);
    Check(built.has_pos_tags,
          "NERSequenceBuilder should detect POS payloads");
    Check(built.has_tags,
          "NERSequenceBuilder should detect supervised NER tags");
    Check(built.samples.size() == 2,
          "NERSequenceBuilder should produce one sample per row");
    Check(built.token_vocabulary.PadId() == 0,
          "NER token vocabulary should reserve PAD id");
    Check(built.token_vocabulary.UnkId() == 1,
          "NER token vocabulary should reserve UNK id");
    Check(built.pos_vocabulary.PadId() == 0,
          "NER POS vocabulary should reserve PAD id");
    Check(built.tag_vocabulary.ValueFor(0) == "O",
          "NER tag vocabulary should keep O at id zero");
    Check(built.samples[0].word_ids[0] ==
              built.token_vocabulary.IdFor("john"),
          "NERSequenceBuilder should lowercase tokens during encoding");
    Check(built.samples[0].pos_ids[0] ==
              built.pos_vocabulary.IdFor("NNP"),
          "NERSequenceBuilder should encode POS tags");
    Check(built.samples[0].tag_ids[0] ==
              built.tag_vocabulary.IdFor("B-PER"),
          "NERSequenceBuilder should encode BIO tags");

    auto batcher = built.CreateBatcher();
    const auto batch = batcher.GetNextSequenceBatch();
    Check(batch.IsSupervised(),
          "NERSequenceBuilder batch should be supervised");
    Check(batch.HasPosIds(),
          "NERSequenceBuilder batch should include POS ids");
    Check(batch.HasAttentionMask(),
          "NERSequenceBuilder batch should include attention mask");
    Check(batch.word_ids.Shape() == std::vector<size_t>({2, 5}),
          "NERSequenceBuilder batch word ids should use configured length");

    const auto* words = batch.word_ids.Data<int64_t>();
    const auto* pos = batch.pos_ids.Data<int64_t>();
    const auto* tags = batch.tag_ids.Data<int64_t>();
    const auto* mask = batch.attention_mask.Data<int64_t>();
    Check(words[4] == built.token_vocabulary.PadId(),
          "NERSequenceBuilder should pad word ids with token PAD id");
    Check(pos[4] == built.pos_vocabulary.PadId(),
          "NERSequenceBuilder should pad POS ids with POS PAD id");
    Check(tags[4] == -100,
          "NERSequenceBuilder should pad tags with ignore_index");
    Check(mask[0] == 1 && mask[3] == 1 && mask[4] == 0,
          "NERSequenceBuilder should build attention masks from token length");

    bool mismatch_failed = false;
    try {
        (void)cyxwiz::BuildNERSequenceData({
            {{"bad", "row"}, {}, {"O"}},
        });
    } catch (const std::runtime_error&) {
        mismatch_failed = true;
    }
    Check(mismatch_failed,
          "NERSequenceBuilder should reject mismatched tag lengths");
}

void TestSequenceTrainingStep() {
    std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
    };

    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 2;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = false;
    builder_config.batcher.tag_ignore_index = -100;

    const auto built = cyxwiz::BuildNERSequenceData(rows, builder_config);
    auto batcher = built.CreateBatcher();

    cyxwiz::TrainingConfiguration config;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = built.tag_vocabulary.Size();
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.loss_params["ignore_index"] = "-100";
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.sequence_batch.enabled = true;
    config.sequence_batch.ignore_index = -100;

    cyxwiz::CompiledLayer embedding;
    embedding.type = gui::NodeType::Embedding;
    embedding.parameters["num_embeddings"] =
        std::to_string(built.token_vocabulary.Size());
    embedding.parameters["embedding_dim"] = "6";
    config.layers.push_back(embedding);

    cyxwiz::CompiledLayer token_head;
    token_head.type = gui::NodeType::TimeDistributed;
    token_head.units = static_cast<int>(built.tag_vocabulary.Size());
    config.layers.push_back(token_head);

    const auto result = cyxwiz::TrainSequenceTaggerEpoch(
        config, batcher, built.tag_vocabulary.Values());

    Check(result.success,
          "sequence training step should succeed: " + result.error);
    Check(result.batches == 1,
          "sequence training step should consume one batch");
    Check(result.samples == 2,
          "sequence training step should report trained samples");
    Check(std::isfinite(result.mean_loss),
          "sequence training step should produce finite loss");
    Check(result.metrics.total_tokens == 8,
          "sequence training step should score non-padding tokens");
    Check(result.metrics.token_accuracy >= 0.0 &&
              result.metrics.token_accuracy <= 1.0,
          "sequence training step token accuracy should be a probability");
}

void TestSequenceTrainingExecutor() {
    const std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
    };

    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 2;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = false;
    builder_config.batcher.tag_ignore_index = -100;

    const auto built = cyxwiz::BuildNERSequenceData(rows, builder_config);

    cyxwiz::TrainingConfiguration config;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = built.tag_vocabulary.Size();
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.loss_params["ignore_index"] = "-100";
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.sequence_batch.enabled = true;
    config.sequence_batch.ignore_index = -100;
    config.save_best_checkpoint = false;
    const auto checkpoint_dir =
        std::filesystem::temp_directory_path() /
        "cyxwiz_sequence_training_executor";
    std::filesystem::remove_all(checkpoint_dir);
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer embedding;
    embedding.type = gui::NodeType::Embedding;
    embedding.parameters["num_embeddings"] =
        std::to_string(built.token_vocabulary.Size());
    embedding.parameters["embedding_dim"] = "6";
    config.layers.push_back(embedding);

    cyxwiz::CompiledLayer token_head;
    token_head.type = gui::NodeType::TimeDistributed;
    token_head.units = static_cast<int>(built.tag_vocabulary.Size());
    config.layers.push_back(token_head);

    auto batcher = std::make_unique<cyxwiz::SequenceBatcher>(
        built.samples, built.batcher_config);
    cyxwiz::TrainingExecutor executor(
        config, std::move(batcher), built.tag_vocabulary.Values());

    bool saw_batch = false;
    bool saw_epoch = false;
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;

    executor.Train(
        1,
        2,
        [&](int epoch, int batch, int total_batches, float loss, float acc) {
            Check(epoch == 1,
                  "sequence executor batch callback should report epoch 1");
            Check(batch == 1,
                  "sequence executor batch callback should report batch 1");
            Check(total_batches == 1,
                  "sequence executor should report one batch");
            Check(std::isfinite(loss),
                  "sequence executor batch loss should be finite");
            Check(acc >= 0.0f && acc <= 1.0f,
                  "sequence executor batch accuracy should be a probability");
            saw_batch = true;
        },
        [&](int epoch,
            float train_loss,
            float train_acc,
            float val_loss,
            float val_acc,
            float) {
            Check(epoch == 1,
                  "sequence executor epoch callback should report epoch 1");
            Check(std::isfinite(train_loss),
                  "sequence executor train loss should be finite");
            Check(std::isfinite(val_loss),
                  "sequence executor val loss should be finite");
            Check(train_acc >= 0.0f && train_acc <= 1.0f,
                  "sequence executor train token accuracy should be a probability");
            Check(val_acc >= 0.0f && val_acc <= 1.0f,
                  "sequence executor val token accuracy should be a probability");
            saw_epoch = true;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_batch, "sequence executor should run a batch callback");
    Check(saw_epoch, "sequence executor should run an epoch callback");
    Check(completed, "sequence executor should run completion callback");
    Check(final_metrics.is_complete,
          "sequence executor should mark training complete");
    Check(final_metrics.total_batches == 1,
          "sequence executor should report one training batch");
    Check(final_metrics.train_token_count == 8,
          "sequence executor should score train tokens");
    Check(final_metrics.val_token_count == 8,
          "sequence executor should score validation tokens");
    Check(final_metrics.train_token_accuracy == final_metrics.train_accuracy,
          "sequence executor should mirror token accuracy to train_accuracy");
    Check(final_metrics.val_token_accuracy == final_metrics.val_accuracy,
          "sequence executor should mirror val token accuracy to val_accuracy");
    Check(final_metrics.train_entity_f1 >= 0.0f &&
              final_metrics.train_entity_f1 <= 1.0f,
          "sequence executor train entity F1 should be a probability");
    Check(final_metrics.val_entity_f1 >= 0.0f &&
              final_metrics.val_entity_f1 <= 1.0f,
          "sequence executor val entity F1 should be a probability");

    std::filesystem::remove_all(checkpoint_dir);
}

void TestArrowDataLoaderSeedDeterminism(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset) {
    cyxwiz::ArrowDatasetBatcher first(
        dataset,
        "label",
        2,
        true,
        1.0f,
        true,
        "",
        0,
        0,
        cyxwiz::BatcherPhase::Train,
        0.0f,
        1234);
    cyxwiz::ArrowDatasetBatcher second(
        dataset,
        "label",
        2,
        true,
        1.0f,
        true,
        "",
        0,
        0,
        cyxwiz::BatcherPhase::Train,
        0.0f,
        1234);
    first.SetOneHotEncoding(2);
    second.SetOneHotEncoding(2);

    const cyxwiz::Batch first_batch = first.GetNextBatch();
    const cyxwiz::Batch second_batch = second.GetNextBatch();
    Check(first_batch.IsValid(),
          "seeded Arrow batcher should produce a first batch");
    Check(second_batch.IsValid(),
          "matching seeded Arrow batcher should produce a first batch");
    Check(first_batch.data.NumElements() == second_batch.data.NumElements(),
          "matching seeds should produce same-sized data batches");
    Check(first_batch.labels.NumElements() == second_batch.labels.NumElements(),
          "matching seeds should produce same-sized label batches");

    const float* first_data = first_batch.data.Data<float>();
    const float* second_data = second_batch.data.Data<float>();
    for (size_t i = 0; i < first_batch.data.NumElements(); ++i) {
        CheckNear(first_data[i],
                  second_data[i],
                  0.0,
                  "matching seeds should produce identical data order");
    }

    const float* first_labels = first_batch.labels.Data<float>();
    const float* second_labels = second_batch.labels.Data<float>();
    for (size_t i = 0; i < first_batch.labels.NumElements(); ++i) {
        CheckNear(first_labels[i],
                  second_labels[i],
                  0.0,
                  "matching seeds should produce identical label order");
    }
}

void RunExecutor(cyxwiz::TrainingExecutor& executor,
                 const std::string& label,
                 int expected_epochs = 1,
                 int expected_validation_points = 1,
                 int expected_optimizer_steps = -1) {
    int saw_epochs = 0;
    bool completed = false;
    bool saw_active_execution_context = false;
    cyxwiz::TrainingMetrics final_metrics;

    Check(!cyxwiz::HasActiveExecutionDeviceContext(),
          label + " should start without an active execution context");
    executor.Train(
        expected_epochs,
        2,
        [&](int, int, int, float, float) {
            Check(cyxwiz::HasActiveExecutionDeviceContext(),
                  label + " should hold an active execution context during batches");
            Check(cyxwiz::CurrentExecutionDeviceContext() != nullptr,
                  label + " should expose the current execution context during batches");
            saw_active_execution_context = true;
        },
        [&](int epoch,
            float train_loss,
            float,
            float val_loss,
            float,
            float) {
            Check(epoch >= 1 && epoch <= expected_epochs,
                  label + " epoch callback should report a valid epoch");
            Check(std::isfinite(train_loss), label + " train loss should be finite");
            if (val_loss >= 0.0f) {
                Check(std::isfinite(val_loss), label + " val loss should be finite");
            }
            ++saw_epochs;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_epochs == expected_epochs, label + " should run each epoch callback");
    Check(saw_active_execution_context,
          label + " should run a batch under an active execution context");
    Check(!cyxwiz::HasActiveExecutionDeviceContext(),
          label + " should clear the active execution context after training");
    Check(completed, label + " should run completion callback");
    Check(final_metrics.is_complete, label + " should mark training complete");
    Check(!final_metrics.is_training, label + " should clear training state");
    Check(final_metrics.current_epoch == expected_epochs, label + " should finish expected epoch");
    Check(final_metrics.total_epochs == expected_epochs, label + " should keep total epochs");
    Check(final_metrics.total_batches == 2, label + " should train two batches");
    const int expected_steps = expected_optimizer_steps >= 0
        ? expected_optimizer_steps
        : expected_epochs * 2;
    Check(final_metrics.optimizer_step_count == expected_steps,
          label + " should report expected optimizer step count");
    Check(final_metrics.loss_history.size() == static_cast<size_t>(expected_epochs),
          label + " should store one train loss history entry per epoch");
    Check(final_metrics.val_loss_history.size() == static_cast<size_t>(expected_validation_points),
          label + " should store validation history only for validation epochs");
    Check(std::isfinite(final_metrics.train_loss),
          label + " final train loss should be finite");
    Check(std::isfinite(final_metrics.val_loss),
          label + " final validation loss should be finite");
}

#ifndef NDEBUG
void TestAllowedTrainingRecordsForcedLinearFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    ScopedEnvVar env(kForceFallbackEnv, "LinearLayer::Forward");
    auto allowed_config = config;
    allowed_config.forbid_native_cpu_fallback = false;
    cyxwiz::TrainingExecutor executor(allowed_config, dataset, "label");

    bool completed = false;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "allowed fallback training should finish with native CPU fallback");
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.available, "allowed fallback training should leave a trace");
    Check(trace.native_cpu_fallback_count > 0,
          "allowed fallback training should record native CPU fallback count");
    Check(trace.residency_verdict == "native_cpu_fallback_observed",
          "allowed fallback training should record fallback residency verdict");

    bool saw_linear_forward = false;
    int execution_context_bind_count = 0;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ExecutionDeviceContext.Bind") {
            ++execution_context_bind_count;
            Check(event.status == "ok",
                  "execution context bind should be valid");
            Check(event.execution_platform == "arrayfire",
                  "execution context should record ArrayFire platform");
            Check(!event.requested_backend.empty(),
                  "execution context should record requested backend");
            Check(!event.effective_backend.empty(),
                  "execution context should record effective backend");
            Check(event.requested_backend == event.effective_backend,
                  "current run context should bind requested/effective backend");
            Check(!event.execution_context_id.empty(),
                  "execution context should record stable identity");
            Check(event.capability_generation > 0,
                  "execution context should record capability generation");
            Check(event.fallback_policy == "allow_native_cpu_fallback",
                  "execution context should record fallback policy");
        }
        if (!event.native_cpu_fallback) {
            continue;
        }
        if (event.fallback_operation == "LinearLayer::Forward") {
            saw_linear_forward = true;
            Check(event.status == "warning",
                  "allowed fallback event should be a warning");
            Check(event.fallback_target == "native_cpu",
                  "fallback target should distinguish native CPU");
            Check(event.fallback_policy == "allow_native_cpu_fallback",
                  "allowed fallback event should record allow policy");
            Check(!event.compute_backend.empty(),
                  "fallback event should record selected ArrayFire backend");
        }
    }
    Check(execution_context_bind_count == 1,
          "training trace should record one execution device context bind");
    Check(saw_linear_forward,
          "allowed fallback trace should name Linear forward");
}

void TestStrictTrainingRejectsForcedLinearFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    ScopedEnvVar env(kForceFallbackEnv, "LinearLayer::Forward");
    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool threw = false;
    try {
        executor.Train(1, 2);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        Check(message.find("LinearLayer::Forward") != std::string::npos,
              "strict training fallback error should name Linear forward");
        Check(message.find("native CPU fallback is forbidden") !=
                  std::string::npos,
              "strict training fallback error should forbid native CPU fallback");
    }

    Check(threw, "strict training should reject forced Linear fallback");
    Check(!executor.IsTraining(),
          "strict training fallback failure should clear training state");
}

void TestStrictTrainingSkipsFirstBatchDebugHostDump(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool completed = false;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "strict training should complete when supported operations stay ArrayFire-backed");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    bool saw_skip_event = false;
    bool saw_loss_boundary = false;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "TrainingExecutor.DebugSampleDump") {
            saw_skip_event = true;
            Check(event.status == "ok",
                  "debug sample dump skip should be recorded as an ok runtime event");
            Check(event.message.find("strict ArrayFire residency") !=
                      std::string::npos,
                  "debug sample dump skip should explain strict residency");
        }
        if (event.stage == "TrainingExecutor.OutputBoundary" &&
            event.message.find("loss_scalar_readback") != std::string::npos) {
            saw_loss_boundary = true;
            Check(event.status == "ok",
                  "loss scalar output boundary should be an ok runtime event");
            Check(event.message.find("not native CPU compute fallback") !=
                      std::string::npos,
                  "loss scalar output boundary should distinguish reporting from fallback");
        }
    }
    Check(saw_skip_event,
          "strict training should record skipped first-batch debug host dump");
    Check(saw_loss_boundary,
          "strict training should declare scalar loss readback boundary");
}
#endif

void TestPendingExecutionDeviceSelectionAppliesAndClears(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    auto* current_device = cyxwiz::Device::GetCurrentDevice();
    if (current_device == nullptr) {
        return;
    }

    cyxwiz::SetPendingExecutionDeviceSelection(
        current_device->GetType(),
        current_device->GetDeviceId());
    Check(cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "pending execution device selection should be queued before training");

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    bool saw_batch = false;
    executor.Train(
        1,
        2,
        [&](int, int, int, float, float) {
            Check(cyxwiz::HasActiveExecutionDeviceContext(),
                  "pending device training should bind active context");
            saw_batch = true;
        },
        nullptr,
        nullptr);

    Check(saw_batch,
          "pending execution device selection should still allow training");
    Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "pending execution device selection should clear after apply");
}

void TestPendingExecutionDeviceSelectionRejectsNonArrayFireBackends() {
    bool rejected_metal = false;
    try {
        cyxwiz::SetPendingExecutionDeviceSelection(
            cyxwiz::DeviceType::METAL,
            0);
    } catch (const std::invalid_argument&) {
        rejected_metal = true;
    }
    Check(rejected_metal,
          "pending execution device selection should reject Metal as non-ArrayFire");

    bool rejected_vulkan = false;
    try {
        cyxwiz::SetPendingExecutionDeviceSelection(
            cyxwiz::DeviceType::VULKAN,
            0);
    } catch (const std::invalid_argument&) {
        rejected_vulkan = true;
    }
    Check(rejected_vulkan,
          "pending execution device selection should reject Vulkan as non-ArrayFire");
}

std::optional<cyxwiz::DeviceInfo> FirstDeviceOfType(
    const std::vector<cyxwiz::DeviceInfo>& devices,
    cyxwiz::DeviceType type) {
    for (const auto& device : devices) {
        if (device.type == type) {
            return device;
        }
    }
    return std::nullopt;
}

#ifndef NDEBUG
void TestStrictArrayFireCpuDenseTrainingDoesNotFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = FirstDeviceOfType(devices, cyxwiz::DeviceType::CPU);
    Check(cpu.has_value(),
          "strict ArrayFire CPU regression requires ArrayFire CPU discovery");

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::CPU,
        cpu->device_id);

    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool completed = false;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "strict ArrayFire CPU dense training should complete");
    Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "strict ArrayFire CPU run should consume pending selection");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.native_cpu_fallback_count == 0,
          "strict ArrayFire CPU dense training should record zero native CPU fallback events");
    Check(trace.residency_verdict ==
              "strict_arrayfire_declared_boundaries",
          "strict ArrayFire CPU run should record strict residency verdict");
    Check(trace.execution_platform == "arrayfire",
          "strict ArrayFire CPU summary should record ArrayFire platform");
    Check(trace.requested_backend == "arrayfire_cpu",
          "strict ArrayFire CPU summary should record requested CPU backend");
    Check(trace.effective_backend == "arrayfire_cpu",
          "strict ArrayFire CPU summary should record effective CPU backend");
    Check(trace.fallback_policy == "forbid_native_cpu_fallback",
          "strict ArrayFire CPU summary should record strict fallback policy");
    Check(trace.declared_output_boundary_count > 0,
          "strict ArrayFire CPU summary should count declared output boundaries");
    Check(trace.arrayfire_host_sync_count > 0,
          "strict ArrayFire CPU summary should count ArrayFire host synchronizations");
    Check(trace.arrayfire_host_sync_bytes > 0,
          "strict ArrayFire CPU summary should count ArrayFire host synchronization bytes");
    Check(trace.transfer_event_count >= trace.arrayfire_host_sync_count,
          "strict ArrayFire CPU summary should count transfer events");
    Check(trace.transfer_known_bytes >= trace.arrayfire_host_sync_bytes,
          "strict ArrayFire CPU summary should count known transfer bytes");
    Check(trace.transfer_summary.find("arrayfire_to_host") !=
              std::string::npos,
          "strict ArrayFire CPU summary should explain transfer modes and reasons");
    Check(trace.synchronization_event_count == trace.arrayfire_host_sync_count,
          "strict ArrayFire CPU summary should count synchronization events");
    Check(trace.synchronization_known_bytes == trace.arrayfire_host_sync_bytes,
          "strict ArrayFire CPU summary should count synchronization bytes");
    Check(trace.synchronization_summary.find("tensor_host_materialization") !=
              std::string::npos,
          "strict ArrayFire CPU summary should explain synchronization reasons");
    Check(!trace.placement_fingerprint.empty(),
          "strict ArrayFire CPU summary should record placement fingerprint");
    Check(trace.placement_entry_count >
              static_cast<uint64_t>(strict_config.backend_placements.size()),
          "strict ArrayFire CPU summary should add dense runtime placement entries");
    Check(!trace.placement_summary.empty(),
          "strict ArrayFire CPU summary should record placement summary");
    Check(trace.placement_summary.find("=gpu(") == std::string::npos,
          "strict ArrayFire CPU placement must not retain stale GPU entries");
    Check(trace.placement_summary.find("=arrayfire_cpu(") !=
              std::string::npos,
          "strict ArrayFire CPU placement should resolve compiler entries to the bound backend");
    Check(trace.placement_summary.find("dataset_ingress") !=
              std::string::npos,
          "placement summary should include dataset ingress");
    Check(trace.placement_summary.find("ModelForward.Dense") !=
              std::string::npos,
          "placement summary should include dense forward");
    Check(trace.placement_summary.find("Loss.") != std::string::npos,
          "placement summary should include loss stage");
    Check(trace.placement_summary.find("metrics") != std::string::npos,
          "placement summary should include metrics stage");
    Check(trace.placement_summary.find("optimizer") != std::string::npos,
          "placement summary should include optimizer stage");
    Check(trace.placement_summary.find("loss_scalar_readback") !=
              std::string::npos,
          "placement summary should include declared scalar output boundary");

    bool saw_cpu_bind = false;
    bool saw_placement_plan = false;
    bool saw_cpu_stage = false;
    bool saw_host_sync = false;
    for (const auto& event : trace.recent_events) {
        if (!event.stage_backend.empty()) {
            Check(event.stage_backend == "arrayfire_cpu",
                  "active run events should record ArrayFire CPU stage backend");
            Check(event.stage_device_id == cpu->device_id,
                  "active run events should record ArrayFire CPU stage device id");
            if (event.stage != "ExecutionDeviceContext.Bind") {
                saw_cpu_stage = true;
            }
        }
        if (event.stage == "TrainingExecutor.PlacementPlan") {
            saw_placement_plan = true;
            Check(event.placement_fingerprint == trace.placement_fingerprint,
                  "placement plan event should match summary fingerprint");
            Check(event.placement_entry_count ==
                      trace.placement_entry_count,
                  "placement plan event should match summary entry count");
            Check(event.placement_summary == trace.placement_summary,
                  "placement plan event should match summary placement text");
        }
        if (event.stage == "ArrayFire.HostSync") {
            saw_host_sync = true;
            Check(event.transfer_mode == "arrayfire_to_host",
                  "host sync event should identify ArrayFire-to-host transfer mode");
            Check(event.arrayfire_host_sync_bytes > 0,
                  "host sync event should record byte count");
        }
        if (event.stage != "ExecutionDeviceContext.Bind") {
            continue;
        }
        saw_cpu_bind = true;
        Check(event.execution_platform == "arrayfire",
              "strict CPU run should bind the ArrayFire platform");
        Check(event.requested_backend == "arrayfire_cpu",
              "strict CPU run should record requested ArrayFire CPU backend");
        Check(event.effective_backend == "arrayfire_cpu",
              "strict CPU run should activate ArrayFire CPU backend");
        Check(event.requested_device_id == cpu->device_id,
              "strict CPU run should record requested CPU device id");
        Check(event.effective_device_id == cpu->device_id,
              "strict CPU run should record effective CPU device id");
        Check(event.fallback_policy == "forbid_native_cpu_fallback",
              "strict CPU run should record forbidden native CPU fallback policy");
    }
    Check(saw_cpu_bind,
          "strict ArrayFire CPU run should record execution context bind");
    Check(saw_placement_plan,
          "strict ArrayFire CPU run should record placement plan fingerprint");
    Check(saw_cpu_stage,
          "strict ArrayFire CPU run should record stage backend/device fields");
    Check(saw_host_sync,
          "strict ArrayFire CPU run should record at least one host sync event");

    const auto persisted_trace =
        cyxwiz::TrainingTraceCollector::LoadLastTrace();
    Check(persisted_trace.has_value(),
          "strict ArrayFire CPU run should persist the training trace");
    if (persisted_trace.has_value()) {
        Check(persisted_trace->transfer_event_count ==
                  trace.transfer_event_count,
              "persisted trace should preserve transfer event count");
        Check(persisted_trace->transfer_known_bytes ==
                  trace.transfer_known_bytes,
              "persisted trace should preserve known transfer bytes");
        Check(persisted_trace->transfer_summary == trace.transfer_summary,
              "persisted trace should preserve transfer summary");
        Check(persisted_trace->synchronization_event_count ==
                  trace.synchronization_event_count,
              "persisted trace should preserve synchronization event count");
        Check(persisted_trace->synchronization_known_bytes ==
                  trace.synchronization_known_bytes,
              "persisted trace should preserve synchronization bytes");
        Check(persisted_trace->synchronization_summary ==
                  trace.synchronization_summary,
              "persisted trace should preserve synchronization summary");
    }

    auto* active_device = cyxwiz::Device::GetCurrentDevice();
    Check(active_device != nullptr,
          "strict ArrayFire CPU run should leave runtime device queryable");
    Check(active_device->GetType() == cyxwiz::DeviceType::CPU,
          "strict ArrayFire CPU run should leave ArrayFire CPU active");
    Check(active_device->GetDeviceId() == cpu->device_id,
          "strict ArrayFire CPU run should leave selected CPU device active");
}
#endif

void TestTrainingDeviceSelectionSwitchesBetweenRuns(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = FirstDeviceOfType(devices, cyxwiz::DeviceType::CPU);
    Check(cpu.has_value(),
          "device switching regression requires ArrayFire CPU discovery");

    std::vector<cyxwiz::DeviceInfo> run_order;
    run_order.push_back(*cpu);

    const auto cuda = FirstDeviceOfType(devices, cyxwiz::DeviceType::CUDA);
    if (cuda.has_value()) {
        run_order.push_back(*cuda);
        run_order.push_back(*cpu);
    }

    const auto oneapi = FirstDeviceOfType(devices, cyxwiz::DeviceType::ONEAPI);
    if (oneapi.has_value()) {
        run_order.push_back(*oneapi);
        run_order.push_back(*cpu);
    }

    const auto opencl = FirstDeviceOfType(devices, cyxwiz::DeviceType::OPENCL);
    if (opencl.has_value()) {
        run_order.push_back(*opencl);
        run_order.push_back(*cpu);
    }

    if (run_order.size() <= 1) {
        return;
    }

    for (const auto& selection : run_order) {
        cyxwiz::ClearPendingExecutionDeviceSelection();
        cyxwiz::SetPendingExecutionDeviceSelection(
            selection.type,
            selection.device_id);

        cyxwiz::TrainingExecutor executor(config, dataset, "label");
        bool saw_batch = false;
        executor.Train(
            1,
            2,
            [&](int, int, int, float, float) {
                Check(cyxwiz::HasActiveExecutionDeviceContext(),
                      "device switch run should bind active context");
                saw_batch = true;
            },
            nullptr,
            nullptr);
        Check(saw_batch,
              "device switch run should execute at least one training batch");

        const std::string expected_backend =
            cyxwiz::ExecutionDeviceSelectionBackendName(selection.type);
        const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
        bool saw_bind = false;
        for (const auto& event : trace.recent_events) {
            if (event.stage != "ExecutionDeviceContext.Bind") {
                continue;
            }
            saw_bind = true;
            Check(event.requested_backend == expected_backend,
                  "device switch bind should record requested backend");
            Check(event.requested_device_id == selection.device_id,
                  "device switch bind should record requested device id");
            Check(event.effective_backend == expected_backend,
                  "device switch bind should activate requested backend");
            Check(event.effective_device_id == selection.device_id,
                  "device switch bind should activate requested device id");
        }
        Check(saw_bind,
              "device switch run should record execution context bind");

        auto* active_device = cyxwiz::Device::GetCurrentDevice();
        Check(active_device != nullptr,
              "device switch run should leave ArrayFire runtime queryable");
        Check(active_device->GetType() == selection.type,
              "device switch run should leave selected backend active");
        Check(active_device->GetDeviceId() == selection.device_id,
              "device switch run should leave selected device active");
        Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
              "device switch run should clear pending selection");
    }
}

void TestObjectiveAwareRegressionMetrics(
    const std::filesystem::path& checkpoint_dir) {
    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeRegressionTable(), "training_executor_regression");
    const auto config = MakeRegressionConfig(checkpoint_dir);
    Check(cyxwiz::UsesContinuousTargetMetrics(config),
          "continuous target contract should select regression metrics");

    cyxwiz::TrainingExecutor executor(config, dataset, "target");
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(completed, "regression executor should complete");
    Check(std::isfinite(final_metrics.train_mae),
          "regression train MAE should be finite");
    Check(std::isfinite(final_metrics.train_rmse),
          "regression train RMSE should be finite");
    Check(std::isfinite(final_metrics.val_mae),
          "regression validation MAE should be finite");
    Check(std::isfinite(final_metrics.val_rmse),
          "regression validation RMSE should be finite");
    Check(final_metrics.mae_history.size() == 1,
          "regression should store one MAE point per epoch");
    Check(final_metrics.rmse_history.size() == 1,
          "regression should store one RMSE point per epoch");
    Check(final_metrics.val_mae_history.size() == 1,
          "regression should store validation MAE history");
    Check(final_metrics.val_rmse_history.size() == 1,
          "regression should store validation RMSE history");
    Check(final_metrics.accuracy_history.empty(),
          "regression must not manufacture classification accuracy history");
    Check(final_metrics.val_accuracy_history.empty(),
          "regression must not manufacture validation accuracy history");
    CheckNear(final_metrics.train_accuracy, 0.0, 0.0,
              "regression train accuracy should remain unset");
    CheckNear(final_metrics.val_accuracy, 0.0, 0.0,
              "regression validation accuracy should remain unset");

#ifndef NDEBUG
    if (cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        auto strict_config = MakeRegressionConfig(
            checkpoint_dir / "strict_residency");
        strict_config.forbid_native_cpu_fallback = true;

        cyxwiz::TrainingExecutor strict_executor(
            strict_config, dataset, "target");
        bool strict_completed = false;
        cyxwiz::TrainingMetrics strict_metrics;
        strict_executor.Train(
            1,
            2,
            nullptr,
            nullptr,
            [&](const cyxwiz::TrainingMetrics& metrics) {
                strict_metrics = metrics;
                strict_completed = true;
            });

        Check(strict_completed,
              "strict regression executor should complete with ArrayFire metrics");
        Check(std::isfinite(strict_metrics.train_mae),
              "strict regression train MAE should be finite");
        Check(std::isfinite(strict_metrics.val_rmse),
              "strict regression validation RMSE should be finite");

        const auto trace =
            cyxwiz::TrainingTraceCollector::Instance().Snapshot();
        Check(trace.native_cpu_fallback_count == 0,
              "strict regression metrics should not use native CPU fallback");
    }
#endif
}

void TestRegressionMetricAccumulator(
    const std::filesystem::path&) {
    cyxwiz::RegressionMetricAccumulator metrics;
    const float predictions[] = {2.0f, -1.0f, 4.0f, 6.0f};
    const float targets[] = {1.0f, 1.0f, 5.0f, 3.0f};
    metrics.Add(predictions, targets, 4);

    Check(metrics.value_count == 4,
          "regression metrics should count every target horizon");
    CheckNear(metrics.Mae(), 1.75, 1e-6,
              "regression metrics should compute elementwise MAE");
    CheckNear(metrics.Rmse(), std::sqrt(3.75), 1e-6,
              "regression metrics should compute elementwise RMSE");

    metrics.Reset();
    Check(metrics.value_count == 0,
          "regression metrics reset should clear the target count");
    CheckNear(metrics.Mae(), 0.0, 0.0,
              "empty regression metrics should have zero MAE");
    CheckNear(metrics.Rmse(), 0.0, 0.0,
              "empty regression metrics should have zero RMSE");
}

void TestRegressionTargetTransform(
    const std::filesystem::path& work_dir) {
    cyxwiz::FittedPreprocessingState state;
    state.operator_name = "StandardScaler";
    state.fit_rows = 4;
    state.input_schema_fingerprint = "fixture";
    state.configuration["with_mean"] = "true";
    state.configuration["with_std"] = "true";

    cyxwiz::PreprocessingFeatureState first;
    first.name = "target";
    first.data_type = "float";
    first.numeric_values["mean"] = 100.0;
    first.numeric_values["scale"] = 10.0;
    state.features.push_back(first);

    cyxwiz::PreprocessingFeatureState second;
    second.name = "target_1";
    second.data_type = "float";
    second.numeric_values["mean"] = 200.0;
    second.numeric_values["scale"] = 2.0;
    state.features.push_back(second);

    const auto path = work_dir / "target_scaler.cyxstate.json";
    std::string error;
    Check(cyxwiz::SaveFittedPreprocessingState(
              path.string(), state, false, error),
          "target scaler fixture should save: " + error);

    cyxwiz::RegressionTargetTransform transform;
    transform.enabled = true;
    transform.operator_name = "StandardScaler";
    transform.state_path = path.string();
    transform.target_columns = {"target", "target_1"};
    Check(cyxwiz::ResolveRegressionTargetTransform(transform, error),
          "target scaler fixture should resolve: " + error);
    Check(transform.IsResolvedForWidth(2),
          "resolved target scaler should match output width");
    CheckNear(transform.InverseValue(1.5, 0), 115.0, 1e-9,
              "first horizon should inverse-transform with its state");
    CheckNear(transform.InverseValue(-2.0, 1), 196.0, 1e-9,
              "second horizon should inverse-transform with its state");

    cyxwiz::RegressionMetricAccumulator metrics(&transform);
    const float predictions[] = {1.0f, 1.0f, 2.0f, 2.0f};
    const float targets[] = {0.0f, 0.0f, 1.0f, 1.0f};
    metrics.Add(predictions, targets, 4, 2);
    CheckNear(metrics.Mae(), 6.0, 1e-6,
              "target-scaled MAE should be reported in original units");
    CheckNear(metrics.Rmse(), std::sqrt(52.0), 1e-6,
              "target-scaled RMSE should be reported in original units");

    auto wrong_order = transform;
    wrong_order.target_columns = {"target_1", "target"};
    Check(!cyxwiz::ResolveRegressionTargetTransform(wrong_order, error),
          "target scaler should reject reordered target columns");
    Check(error.find("expected") != std::string::npos,
          "target scaler order error should explain the mismatch");
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    TestSequenceBatchContract();
    TestSequenceBatcherPadsNamedPayloads();
    TestSequenceBatcherDropLast();
    TestSequenceTagMetrics();
    TestSequenceVocabulary();
    TestNERSequenceBuilder();
    TestSequenceTrainingStep();
    TestSequenceTrainingExecutor();

    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_training_executor_arrow_parquet";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);

    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeTrainingTable(), "training_executor_arrow");
    const auto config = MakeConfig(work_dir / "checkpoints");
    TestArrowDataLoaderSeedDeterminism(dataset);

#ifndef NDEBUG
    TestAllowedTrainingRecordsForcedLinearFallback(dataset, config);
    TestStrictTrainingRejectsForcedLinearFallback(dataset, config);
    TestStrictTrainingSkipsFirstBatchDebugHostDump(dataset, config);
    TestStrictArrayFireCpuDenseTrainingDoesNotFallback(dataset, config);
#endif
    TestPendingExecutionDeviceSelectionAppliesAndClears(dataset, config);
    TestPendingExecutionDeviceSelectionRejectsNonArrayFireBackends();
    TestTrainingDeviceSelectionSwitchesBetweenRuns(dataset, config);

    {
        auto sequence_config = config;
        sequence_config.sequence_batch.enabled = true;
        sequence_config.sequence_batch.token_column = "tokens";
        sequence_config.sequence_batch.tag_column = "ner_tags";
        cyxwiz::TrainingExecutor sequence_executor(
            sequence_config, dataset, "label");
        bool saw_batch = false;
        bool saw_epoch = false;
        bool completed = false;
        sequence_executor.Train(
            1,
            sequence_config.batch_size,
            [&](int, int, int, float, float) { saw_batch = true; },
            [&](int, float, float, float, float, float) { saw_epoch = true; },
            [&](const cyxwiz::TrainingMetrics&) { completed = true; });
        Check(!saw_batch,
              "sequence batch guard should reject before any training batch");
        Check(!saw_epoch,
              "sequence batch guard should reject before epoch callback");
        Check(!completed,
              "sequence batch guard should reject before completion callback");
        Check(!sequence_executor.IsTraining(),
              "sequence batch guard should clear executor training state");
    }

    {
        cyxwiz::TrainingExecutor arrow_executor(config, dataset, "label");
        RunExecutor(arrow_executor, "Arrow TrainingExecutor");
    }

    TestObjectiveAwareRegressionMetrics(work_dir / "regression_checkpoints");
    TestRegressionMetricAccumulator(
        work_dir / "regression_test_checkpoints");
    TestRegressionTargetTransform(work_dir);

    {
        auto scheduled_validation_config = config;
        scheduled_validation_config.epochs = 3;
        scheduled_validation_config.validation_freq = 2;
        scheduled_validation_config.log_interval = 1;
        cyxwiz::TrainingExecutor scheduled_executor(
            scheduled_validation_config, dataset, "label");
        RunExecutor(scheduled_executor,
                    "Arrow scheduled validation TrainingExecutor",
                    3,
                    2);
    }

    {
        auto grad_accum_config = config;
        grad_accum_config.epochs = 3;
        grad_accum_config.grad_accum_steps = 2;
        cyxwiz::TrainingExecutor grad_accum_executor(
            grad_accum_config, dataset, "label");
        RunExecutor(grad_accum_executor,
                    "Arrow gradient accumulation TrainingExecutor",
                    3,
                    3,
                    3);
    }

    const fs::path parquet_path = work_dir / "training_executor.parquet";
    WriteParquetWithRowGroupSize(*dataset, parquet_path.string(), 2);
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "training_executor_parquet");
    Check(parquet_dataset != nullptr, "Parquet fixture should open");

    {
        cyxwiz::TrainingExecutor parquet_executor(config, parquet_dataset, "label");
        RunExecutor(parquet_executor, "Parquet TrainingExecutor");
    }

    parquet_dataset.reset();
    fs::remove_all(work_dir);

    std::cout << "TrainingExecutor Arrow/Parquet parity passed\n";
    return 0;
}
