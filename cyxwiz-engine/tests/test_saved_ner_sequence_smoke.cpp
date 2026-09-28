#include "../src/core/arrow_dataset.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/model_builder.h"
#include "../src/core/sequence_arrow_batcher.h"
#include "../src/core/sequence_inference_response.h"
#include "../src/core/sequence_model_input.h"
#include "../src/core/sequence_tag_metrics.h"
#include "../src/core/training_executor.h"
#include "route_qualification_test_fixture.h"
#include "../src/gui/loaders/data_loader.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path FindRepoRoot() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        if (std::filesystem::exists(dir / "cyxwiz-engine" / "CMakeLists.txt") &&
            std::filesystem::exists(dir / "examples" / "cyxgraph")) {
            return dir;
        }
        const auto parent = dir.parent_path();
        if (parent == dir) {
            break;
        }
        dir = parent;
    }
    return std::filesystem::current_path();
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    Check(in.is_open(), "could not open " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// The Engine's own loader (factory pins, current contracts, links resolved by
// pin index); strict, so a link that no longer resolves fails the smoke.
std::pair<std::vector<gui::MLNode>, std::vector<gui::NodeLink>>
LoadCyxGraphForTest(const std::filesystem::path& path) {
    const auto root = json::parse(ReadFile(path));
    cyxwiz::GraphDocumentLoadOptions options;
    options.strict_links = true;
    cyxwiz::GraphDocument document;
    std::string error;
    Check(cyxwiz::BuildGraphDocument(root, root, options, document, error),
          "saved NER graph should load: " + error);
    return {document.nodes, document.links};
}

std::vector<std::string> SplitSimpleCsvRow(const std::string& line) {
    std::vector<std::string> values;
    std::string value;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                value.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }
        if (ch == ',' && !in_quotes) {
            values.push_back(value);
            value.clear();
            continue;
        }
        value.push_back(ch);
    }

    if (!value.empty() && value.back() == '\r') {
        value.pop_back();
    }
    values.push_back(value);
    return values;
}

std::shared_ptr<arrow::Array> FinishStringArray(
    const std::vector<std::string>& values) {
    arrow::StringBuilder builder;
    for (const auto& value : values) {
        const auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    const auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> LoadNerSentencesCsvTable(
    const std::filesystem::path& csv_path) {
    std::ifstream in(csv_path);
    Check(in.is_open(), "NER sentence CSV should open: " + csv_path.string());

    std::string line;
    Check(static_cast<bool>(std::getline(in, line)),
          "NER sentence CSV should include header");
    const auto header = SplitSimpleCsvRow(line);

    auto find_index = [&](const std::string& name) {
        for (int i = 0; i < static_cast<int>(header.size()); ++i) {
            if (header[i] == name) {
                return i;
            }
        }
        return -1;
    };

    const int sentence_idx = find_index("sentence_id");
    const int token_idx = find_index("tokens");
    const int pos_idx = find_index("pos_tags");
    const int tag_idx = find_index("ner_tags");
    Check(sentence_idx >= 0, "NER CSV must include sentence_id column");
    Check(token_idx >= 0, "NER CSV must include tokens column");
    Check(pos_idx >= 0, "NER CSV must include pos_tags column");
    Check(tag_idx >= 0, "NER CSV must include ner_tags column");

    std::vector<std::string> sentence_values;
    std::vector<std::string> token_values;
    std::vector<std::string> pos_values;
    std::vector<std::string> tag_values;
    const int max_index =
        std::max({sentence_idx, token_idx, pos_idx, tag_idx});

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = SplitSimpleCsvRow(line);
        Check(static_cast<int>(values.size()) > max_index,
              "NER CSV row should include all required fields");
        sentence_values.push_back(values[sentence_idx]);
        token_values.push_back(values[token_idx]);
        pos_values.push_back(values[pos_idx]);
        tag_values.push_back(values[tag_idx]);
    }

    auto schema = arrow::schema({
        arrow::field("sentence_id", arrow::utf8()),
        arrow::field("tokens", arrow::utf8()),
        arrow::field("pos_tags", arrow::utf8()),
        arrow::field("ner_tags", arrow::utf8()),
    });
    return arrow::Table::Make(schema,
                              {FinishStringArray(sentence_values),
                               FinishStringArray(token_values),
                               FinishStringArray(pos_values),
                               FinishStringArray(tag_values)},
                              static_cast<int64_t>(token_values.size()));
}

bool HasIssueText(const cyxwiz::TrainingConfiguration& config,
                  const std::string& needle) {
    for (const auto& issue : config.issues) {
        if (issue.message.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

cyxwiz::TrainingConfiguration MakePosFusedSequenceTrainingConfig(
    cyxwiz::TrainingConfiguration compiled,
    const cyxwiz::SequenceArrowBatcherBuildResult& build,
    const std::filesystem::path& checkpoint_dir) {
    compiled.is_valid = true;
    compiled.issues.clear();
    compiled.batch_size = 2;
    compiled.epochs = 1;
    compiled.shuffle = false;
    compiled.drop_last = false;
    compiled.has_data_split = false;
    compiled.train_ratio = 1.0f;
    compiled.val_ratio = 0.0f;
    compiled.test_ratio = 0.0f;
    compiled.num_workers = 0;
    compiled.prefetch_factor = 0;
    compiled.log_interval = 1;
    compiled.validation_freq = 1;
    compiled.dataloader_seed = 7;
    compiled.grad_accum_steps = 1;
    compiled.save_best_checkpoint = false;
    compiled.early_stopping_patience = 0;
    compiled.checkpoint_dir = checkpoint_dir.string();
    compiled.loss_type = gui::NodeType::CrossEntropyLoss;
    compiled.optimizer_type = gui::NodeType::SGD;
    compiled.learning_rate = 0.01f;

    cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(build, compiled);
    return compiled;
}

int NodeIdNamed(const std::vector<gui::MLNode>& nodes, const std::string& name) {
    for (const auto& node : nodes) {
        if (node.name == name) return node.id;
    }
    Check(false, "saved NER graph should contain node '" + name + "'");
    return -1;
}

gui::MLNode& NodeWithId(std::vector<gui::MLNode>& nodes, int id) {
    for (auto& node : nodes) {
        if (node.id == id) return node;
    }
    Check(false, "missing node id " + std::to_string(id));
    return nodes.front();
}

void CheckWordPosFusionRejections(const std::vector<gui::MLNode>& nodes,
                                  const std::vector<gui::NodeLink>& links) {
    cyxwiz::GraphCompiler compiler;
    const int data = NodeIdNamed(nodes, "NER Sentence CSV");
    const int word = NodeIdNamed(nodes, "Word Embedding");
    const int concat = NodeIdNamed(nodes, "Concat Word + POS");
    const int dropout = NodeIdNamed(nodes, "Dropout");
    const auto rejected = [&](std::vector<gui::MLNode> n, std::vector<gui::NodeLink> l,
                              const std::string& reason, const std::string& what) {
        const auto config = compiler.Compile(n, l, true);
        Check(!config.is_valid && HasIssueText(config, "Word + POS fusion: " + reason),
              "word + POS fusion should reject " + what);
    };

    auto n = nodes;
    NodeWithId(n, concat).parameters["dim"] = "1";
    rejected(n, links, "Concatenate dim is 1, the sequence axis.", "a sequence-axis concat");

    n = nodes;
    NodeWithId(n, data).parameters["pos_column"] = "";
    rejected(n, links, "The sequence data declares no POS column.", "a missing POS column");

    auto l = links;
    const auto& input2 = NodeWithId(n, concat).inputs.at(1);
    l.erase(std::remove_if(l.begin(), l.end(), [&](const gui::NodeLink& link) {
        return link.to_node == concat && link.to_pin == input2.id;
    }), l.end());
    rejected(nodes, l, "Connect exactly Input 1 and Input 2.", "a Concatenate with one Embedding input");

    l = links;
    gui::NodeLink extra;
    extra.id = 9000;
    extra.from_node = word;
    extra.from_pin = NodeWithId(n, word).outputs.at(0).id;
    extra.to_node = dropout;
    extra.to_pin = NodeWithId(n, dropout).inputs.at(0).id;
    l.push_back(extra);
    rejected(nodes, l, "Word Embedding may feed only the Concatenate.", "a shared embedding output");
}

void CheckDecodedGoldLogits(const cyxwiz::SequenceBatch& batch,
                            const std::vector<std::string>& id_to_label,
                            int64_t ignore_index) {
    const auto shape = batch.word_ids.Shape();
    Check(shape.size() == 2, "decode smoke should use [batch, seq] word ids");
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t tag_count = id_to_label.size();
    Check(tag_count > 0, "decode smoke should have tag labels");

    std::vector<float> logits(batch_size * seq_len * tag_count, -1.0f);
    const auto* gold_ids = batch.tag_ids.Data<int64_t>();
    for (size_t i = 0; i < batch.tag_ids.NumElements(); ++i) {
        const int64_t gold = gold_ids[i];
        if (gold < 0 || static_cast<size_t>(gold) >= tag_count) {
            continue;
        }
        logits[i * tag_count + static_cast<size_t>(gold)] = 1.0f;
    }

    const cyxwiz::Tensor predictions(
        {batch_size, seq_len, tag_count}, logits.data(), cyxwiz::DataType::Float32);
    const auto predicted_ids = cyxwiz::ArgmaxSequenceTagLogits(predictions);
    Check(predicted_ids.Shape() == batch.tag_ids.Shape(),
          "decoded tag ids should preserve [batch, seq] shape");

    const auto metrics = cyxwiz::ComputeSequenceTagMetricsFromLogits(
        predictions, batch.tag_ids, id_to_label, ignore_index);
    Check(metrics.total_tokens > 0,
          "decode metrics should count non-ignored tokens");
    Check(std::isfinite(metrics.token_accuracy),
          "decode metrics should produce finite token accuracy");

    std::vector<int64_t> lengths(batch_size, 0);
    const auto* mask = batch.attention_mask.Data<int64_t>();
    for (size_t row = 0; row < batch_size; ++row) {
        for (size_t col = 0; col < seq_len; ++col) {
            if (mask[row * seq_len + col] != 0) {
                ++lengths[row];
            }
        }
    }

    const auto decoded = cyxwiz::DecodeSequenceTagIdsForInference(
        predicted_ids, id_to_label, lengths);
    Check(decoded.tag_labels.size() == batch_size,
          "decode response should preserve batch rows");
    Check(!decoded.tag_labels.empty() && !decoded.tag_labels[0].empty(),
          "decode response should expose readable tag labels");
}

bool FloatTensorsDiffer(const cyxwiz::Tensor& lhs,
                        const cyxwiz::Tensor& rhs) {
    if (lhs.Shape() != rhs.Shape() ||
        lhs.GetDataType() != cyxwiz::DataType::Float32 ||
        rhs.GetDataType() != cyxwiz::DataType::Float32) {
        return true;
    }
    const float* a = lhs.Data<float>();
    const float* b = rhs.Data<float>();
    for (size_t i = 0; i < lhs.NumElements(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1.0e-6f) {
            return true;
        }
    }
    return false;
}

void CheckSequenceFusionInputGuards(
    const cyxwiz::SequenceBatch& batch,
    const cyxwiz::TrainingConfiguration& config) {
    Check(cyxwiz::UsesSequenceFeatureFusion(config),
          "test config should declare sequence feature fusion");

    const auto packed = cyxwiz::BuildSequenceModelInput(batch, config);
    Check(packed.Shape() ==
              std::vector<size_t>({batch.size, batch.sequence_length, 2}),
          "sequence fusion input should pack word and POS ids");

    cyxwiz::SequenceBatch missing_pos = batch;
    missing_pos.pos_ids = cyxwiz::Tensor();
    bool missing_threw = false;
    try {
        (void)cyxwiz::BuildSequenceModelInput(missing_pos, config);
    } catch (const std::exception& e) {
        missing_threw = std::string(e.what()).find("POS ids") !=
                        std::string::npos;
    }
    Check(missing_threw,
          "sequence fusion should reject missing POS ids before forward");

    std::vector<int64_t> mismatched_pos(
        batch.size * (batch.sequence_length + 1), 0);
    cyxwiz::SequenceBatch mismatched = batch;
    mismatched.pos_ids = cyxwiz::Tensor(
        {batch.size, batch.sequence_length + 1},
        mismatched_pos.data(),
        cyxwiz::DataType::Int64);
    bool mismatch_threw = false;
    try {
        (void)cyxwiz::BuildSequenceModelInput(mismatched, config);
    } catch (const std::exception& e) {
        mismatch_threw = std::string(e.what()).find("shape") !=
                         std::string::npos;
    }
    Check(mismatch_threw,
          "sequence fusion should reject POS shape mismatches before forward");
}

void CheckPosIdsAffectModelPath(
    const cyxwiz::SequenceBatch& batch,
    const cyxwiz::TrainingConfiguration& config,
    size_t pos_vocabulary_size) {
    Check(pos_vocabulary_size > 1,
          "POS influence check requires more than one POS id");
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok() && built.model,
          "sequence fusion model should build for POS influence check");

    cyxwiz::Tensor original_input =
        cyxwiz::BuildSequenceModelInput(batch, config);
    cyxwiz::Tensor original_logits = built.model->Forward(original_input);

    const auto& pos_shape = batch.pos_ids.Shape();
    std::vector<int64_t> altered_pos(batch.pos_ids.NumElements(), 0);
    const auto* pos_data = batch.pos_ids.Data<int64_t>();
    const int64_t vocab = static_cast<int64_t>(pos_vocabulary_size);
    for (size_t i = 0; i < altered_pos.size(); ++i) {
        altered_pos[i] = pos_data[i] == 0 ? 1 : ((pos_data[i] % (vocab - 1)) + 1);
    }

    cyxwiz::SequenceBatch altered = batch;
    altered.pos_ids = cyxwiz::Tensor(pos_shape,
                                     altered_pos.data(),
                                     cyxwiz::DataType::Int64);
    cyxwiz::Tensor altered_input =
        cyxwiz::BuildSequenceModelInput(altered, config);
    cyxwiz::Tensor altered_logits = built.model->Forward(altered_input);

    Check(FloatTensorsDiffer(original_logits, altered_logits),
          "changing POS ids should change the fused model logits");
}

void CheckPackagedSequenceAssets(const std::filesystem::path& package_path,
                                 const std::filesystem::path& graph_path,
                                 const std::filesystem::path& generated_dir,
                                 const cyxwiz::TrainingConfiguration& config,
                                 const cyxwiz::TrainingMetrics& metrics) {
    cyxwiz::ModelManifest manifest;
    manifest.model_name = "Saved NER sequence smoke";
    manifest.model_type = "SequenceTagger";
    manifest.epochs_trained = metrics.current_epoch;
    manifest.final_accuracy = metrics.train_accuracy;
    manifest.final_loss = metrics.train_loss;
    manifest.has_graph = true;
    manifest.has_sequence = true;
    manifest.has_sequence_token_vocabulary = true;
    manifest.has_sequence_pos_vocabulary = true;
    manifest.has_sequence_tag_vocabulary = true;
    manifest.sequence_batch_first = config.sequence_batch.batch_first;
    manifest.sequence_create_attention_mask =
        config.sequence_batch.create_attention_mask;
    manifest.sequence_create_causal_lm_targets =
        config.sequence_batch.create_causal_lm_targets;
    manifest.sequence_max_sequence_length =
        static_cast<size_t>(config.sequence_batch.max_sequence_length);
    manifest.sequence_word_pad_id = 0;
    manifest.sequence_pos_pad_id = 0;
    manifest.sequence_tag_ignore_index = config.sequence_batch.ignore_index;
    manifest.sequence_target_ignore_index =
        config.sequence_batch.target_ignore_index;
    manifest.sequence_token_vocabulary_path = "sequence/token_vocab.txt";
    manifest.sequence_pos_vocabulary_path = "sequence/pos_vocab.txt";
    manifest.sequence_tag_vocabulary_path = "sequence/tag_vocab.txt";

    cyxwiz::ExportOptions options;
    options.include_graph = true;
    options.include_sequence_assets = true;
    options.include_training_history = false;
    options.include_optimizer_state = false;
    options.sequence_max_sequence_length =
        static_cast<size_t>(config.sequence_batch.max_sequence_length);
    options.sequence_tag_ignore_index = config.sequence_batch.ignore_index;
    options.sequence_target_ignore_index =
        config.sequence_batch.target_ignore_index;
    options.sequence_token_vocabulary_path =
        (generated_dir / "ner_word_vocab.txt").string();
    options.sequence_pos_vocabulary_path =
        (generated_dir / "ner_pos_vocab.txt").string();
    options.sequence_tag_vocabulary_path =
        (generated_dir / "ner_tag_vocab.txt").string();

    cyxwiz::TrainingConfig package_config;
    package_config.optimizer_type = config.GetOptimizerName();
    package_config.learning_rate = config.learning_rate;
    package_config.batch_size = config.batch_size;
    package_config.epochs = metrics.current_epoch;
    package_config.loss_function = config.GetLossName();
    package_config.dataset_name = config.dataset_name;
    package_config.num_classes = static_cast<int>(config.output_size);
    package_config.input_shape = {
        static_cast<int64_t>(config.input_shape.empty() ? 0 : config.input_shape[0])};

    cyxwiz::formats::CyxModelFormat format;
    const bool created = format.Create(package_path.string(),
                                       manifest,
                                       ReadFile(graph_path),
                                       package_config,
                                       nullptr,
                                       {},
                                       {},
                                       nullptr,
                                       options);
    Check(created, "failed to create sequence package: " + format.GetLastError());

    const auto probe = format.Probe(package_path.string());
    Check(probe.valid, "sequence package should probe: " + probe.error_message);
    Check(probe.has_sequence, "sequence package should advertise sequence content");
    Check(probe.has_sequence_token_vocabulary,
          "sequence package should include token vocabulary");
    Check(probe.has_sequence_pos_vocabulary,
          "sequence package should include POS vocabulary");
    Check(probe.has_sequence_tag_vocabulary,
          "sequence package should include tag vocabulary");
    Check(probe.sequence_max_sequence_length ==
              static_cast<size_t>(config.sequence_batch.max_sequence_length),
          "sequence package should preserve max sequence length");

    std::string token_text;
    std::string pos_text;
    std::string tag_text;
    Check(format.ExtractSequenceVocabularyAssets(
              package_path.string(), token_text, pos_text, tag_text),
          "sequence vocabularies should extract: " + format.GetLastError());
    Check(token_text.find("[PAD]") != std::string::npos,
          "token vocabulary asset should contain PAD");
    Check(pos_text.find("[PAD]") != std::string::npos,
          "POS vocabulary asset should contain PAD");
    Check(tag_text.find("B-geo") != std::string::npos,
          "tag vocabulary asset should contain BIO labels");
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    cyxwiz::test::InstallQualifiedRouteSnapshot();

    const fs::path repo_root = FindRepoRoot();
    const fs::path ner_dir = repo_root / "examples" / "cyxgraph" / "NER";
    const fs::path generated_dir = ner_dir / "generated";
    const fs::path graph_path = ner_dir / "ner_bilstm_sequence_tagger.cyxgraph";
    const fs::path csv_path = generated_dir / "ner_sentences.csv";

    auto [nodes, links] = LoadCyxGraphForTest(graph_path);
    cyxwiz::GraphCompiler compiler;
    auto compiled = compiler.Compile(nodes, links, true);
    for (const auto& issue : compiled.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error) {
            std::cerr << "compile error: " << issue.node_name << ": " << issue.message << '\n';
        }
    }
    Check(compiled.is_valid, "saved NER graph should compile without errors");
    Check(!compiled.layers.empty() &&
              cyxwiz::IsSequenceFeatureFusionLayer(compiled.layers.front()),
          "Word + POS Embeddings joined by Concatenate should compile to one fused first layer");
    Check(compiled.layers.front().parameters.at("word_embedding_dim") == "100" &&
              compiled.layers.front().parameters.at("pos_embedding_dim") == "16",
          "fused layer should take each Embedding node's width");
    Check(compiled.graph_op_node_ids.empty(),
          "the fused Concatenate should not also run as a graph op");
    Check(compiled.input_shape == std::vector<size_t>{96} &&
              compiled.layers.front().output_shape == std::vector<size_t>{96, 116} &&
              compiled.layers.back().output_shape == std::vector<size_t>{96, 19},
          "compiled shapes are per token: [96] ids -> [96, 116] fused -> [96, 19] tags");
    CheckWordPosFusionRejections(nodes, links);
    Check(compiled.sequence_batch.enabled,
          "saved NER graph should compile a sequence batch contract");
    Check(compiled.sequence_batch.token_column == "tokens",
          "saved NER graph should preserve token column");
    Check(compiled.sequence_batch.pos_column == "pos_tags",
          "saved NER graph should preserve POS column");
    Check(compiled.sequence_batch.tag_column == "ner_tags",
          "saved NER graph should preserve tag column");
    Check(compiled.sequence_batch.max_sequence_length == 96,
          "saved NER graph should take max length from the sequence Data Input");
    Check(compiled.sequence_batch.ignore_index == 0,
          "saved NER graph should preserve token loss ignore index");
    Check(!HasIssueText(compiled, "Dense-encoded NER"),
          "saved NER graph should not trip the fake Dense NER guard");

    auto table = LoadNerSentencesCsvTable(csv_path);
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        table, "saved_ner_sequence_smoke");

    auto sequence_build = cyxwiz::BuildSequenceBatcherFromArrowDataset(
        dataset, compiled, 2);
    Check(sequence_build.success(),
          "saved NER sequence batcher should build: " +
              sequence_build.error_message);
    Check(sequence_build.sample_count == static_cast<size_t>(table->num_rows()),
          "saved NER sequence batcher should preserve sentence rows");
    Check(sequence_build.sequence_length == 96,
          "saved NER sequence batcher should honor saved max length");
    Check(sequence_build.token_vocabulary_size > 0,
          "saved NER sequence batcher should infer token vocabulary");
    Check(sequence_build.pos_vocabulary_size > 0,
          "saved NER sequence batcher should infer POS vocabulary");
    Check(sequence_build.tag_vocabulary_size > 0,
          "saved NER sequence batcher should infer tag vocabulary");
    Check(sequence_build.id_to_label.size() ==
              sequence_build.tag_vocabulary_size,
          "saved NER sequence batcher should expose tag labels");

    sequence_build.batcher->SetPhase(cyxwiz::BatcherPhase::Train);
    auto decode_batch = sequence_build.batcher->GetNextSequenceBatch();
    Check(decode_batch.IsSupervised(),
          "saved NER sequence batcher should produce supervised batches");
    Check(decode_batch.HasPosIds(),
          "saved NER sequence batcher should carry POS ids");
    Check(decode_batch.HasAttentionMask(),
          "saved NER sequence batcher should carry attention masks");
    CheckDecodedGoldLogits(decode_batch,
                           sequence_build.id_to_label,
                           compiled.sequence_batch.ignore_index);

    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_saved_ner_sequence_smoke";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");

    auto training_config = MakePosFusedSequenceTrainingConfig(
        compiled, sequence_build, work_dir / "checkpoints");
    Check(training_config.input_size == 96,
          "sequence training config should normalize input length");
    Check(training_config.output_size == sequence_build.tag_vocabulary_size,
          "sequence training config should normalize tag width");
    Check(training_config.layers.front().parameters["word_num_embeddings"] ==
              std::to_string(sequence_build.token_vocabulary_size),
          "sequence training config should normalize word vocabulary");
    Check(training_config.layers.front().parameters["pos_num_embeddings"] ==
              std::to_string(sequence_build.pos_vocabulary_size),
          "sequence training config should normalize POS vocabulary");
    Check(training_config.layers.back().units ==
              static_cast<int>(sequence_build.tag_vocabulary_size),
          "sequence training config should normalize token head width");
    CheckSequenceFusionInputGuards(decode_batch, training_config);
    CheckPosIdsAffectModelPath(decode_batch,
                               training_config,
                               sequence_build.pos_vocabulary_size);

    auto training_build = cyxwiz::BuildSequenceBatcherFromArrowDataset(
        dataset, training_config, training_config.batch_size);
    Check(training_build.success(),
          "saved NER training sequence batcher should build: " +
              training_build.error_message);
    // Product order (StartTrainingFromGraph): the executor trains with the
    // batcher whose vocabulary sizes were applied to the config. The
    // training build uses the full-data training config, so its
    // vocabularies can be larger than the first (split-configured) build
    // that sized the model above; apply THIS build before training.
    cyxwiz::ApplySequenceBatcherBuildResultToTrainingConfig(training_build,
                                                            training_config);

    cyxwiz::TrainingExecutor executor(
        training_config,
        std::move(training_build.batcher),
        training_build.id_to_label);

    bool saw_batch = false;
    bool saw_epoch = false;
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        1,
        training_config.batch_size,
        [&](int epoch, int batch, int total_batches, float loss, float acc) {
            Check(epoch == 1, "saved NER batch callback should report epoch 1");
            Check(batch >= 1 && total_batches >= 1,
                  "saved NER batch callback should report progress");
            Check(std::isfinite(loss),
                  "saved NER batch loss should be finite");
            Check(acc >= 0.0f && acc <= 1.0f,
                  "saved NER batch accuracy should be a probability");
            saw_batch = true;
        },
        [&](int epoch,
            float train_loss,
            float train_acc,
            float val_loss,
            float val_acc,
            float) {
            Check(epoch == 1, "saved NER epoch callback should report epoch 1");
            Check(std::isfinite(train_loss),
                  "saved NER train loss should be finite");
            Check(std::isfinite(val_loss),
                  "saved NER validation loss should be finite");
            Check(train_acc >= 0.0f && train_acc <= 1.0f,
                  "saved NER train token accuracy should be a probability");
            Check(val_acc >= 0.0f && val_acc <= 1.0f,
                  "saved NER validation token accuracy should be a probability");
            saw_epoch = true;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_batch, "saved NER smoke should train at least one batch");
    Check(saw_epoch, "saved NER smoke should run an epoch callback");
    Check(completed, "saved NER smoke should complete training");
    Check(final_metrics.is_complete,
          "saved NER smoke should mark training complete");
    Check(final_metrics.train_token_count > 0,
          "saved NER smoke should score training tokens");
    Check(final_metrics.val_token_count > 0,
          "saved NER smoke should score validation tokens");
    Check(std::isfinite(final_metrics.train_loss),
          "saved NER final train loss should be finite");
    Check(std::isfinite(final_metrics.val_loss),
          "saved NER final validation loss should be finite");
    Check(final_metrics.train_entity_f1 >= 0.0f &&
              final_metrics.train_entity_f1 <= 1.0f,
          "saved NER train entity F1 should be a probability");
    Check(final_metrics.val_entity_f1 >= 0.0f &&
              final_metrics.val_entity_f1 <= 1.0f,
          "saved NER validation entity F1 should be a probability");

    CheckPackagedSequenceAssets(work_dir / "saved_ner_sequence_smoke.cyxmodel",
                                graph_path,
                                generated_dir,
                                training_config,
                                final_metrics);

    fs::remove_all(work_dir);

    std::cout << "Saved NER sequence smoke passed\n";
    return 0;
}
