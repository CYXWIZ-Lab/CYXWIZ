// Level 2 of the oracle harness (TOFIX228 1.5b): a whole model trained for a
// few steps by the Engine, against PyTorch running the Engine's own PyTorch
// export of the same graph.
//
// For each zoo model:
//   1. the graph's PyTorch export (core/code_export, unstamped) must equal the
//      committed one in fixtures/train_trajectory/<model>.py: the fixture was
//      made from that text, so a changed export needs a new fixture;
//   2. the generator (reference/generate_train_trajectory_fixtures.py) ran
//      that export, wrote its starting weights as a PyTorch state-dict
//      (<model>_init.pth) and recorded every step: the epoch's running loss
//      and every parameter and buffer;
//   3. here the graph's parameter nodes take the export's module names
//      (self.layer3 -> node "layer3"), the run starts from the state-dict the
//      way a user's run does (initialize_from_weights_file: weight file
//      reader, matcher, PyTorch-to-Engine layouts), trains through the
//      TrainingExecutor on the fixture's rows in order, and after each step
//      its parameters are compared with the fixture's, put in the Engine's
//      layouts by the same matcher.
//
//   test_oracle_train                      compare with the committed fixture
//   test_oracle_train --write-exports DIR  write each model's export and spec
//                                          (then run the generator)
#include "training_oracle_test.h"
#include "train_oracle_support.h"
#include "../../src/core/arrow_dataset.h"
#include "../../src/core/code_export.h"
#include "../../src/core/debug_run_paths.h"
#include "../../src/core/graph_compiler.h"
#include "../../src/core/graph_compiler_dataset_hooks.h"
#include "../../src/core/graph_node_factory.h"
#include "../../src/core/sequence_batcher.h"
#include "../../src/core/sha256_digest.h"
#include "../../src/core/training_executor.h"
#include "../../src/core/weight_match.h"
#include "../../src/core/weight_start.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using cyxwiz::test::Compare;
using cyxwiz::test::Difference;
using cyxwiz::test::ExpectedInEngineLayout;
using cyxwiz::test::ReadText;
using cyxwiz::test::Sha256Text;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

fs::path FixtureDir() {
    return fs::path(CYXWIZ_SOURCE_ROOT) / "cyxwiz-engine" / "tests" / "computation_truth" / "fixtures" /
           "train_trajectory";
}

// ---------------------------------------------------------------- the zoo --

struct ZooModel : cyxwiz::test::TrainGraph {
    std::string name;
    std::string summary;
    std::vector<size_t> input_shape;  // one row, as the Data node gives it
    bool image_rows = false;          // rows are [H, W, C] images; torch takes them [C, H, W]
    int token_ids = 0;                // > 0: rows are token ids below it (torch takes them as integers)
    // "classification" (an integer label per row), "regression", or
    // "token_classification" (a label per token; the Engine trains it through
    // the sequence batcher, padded positions carry the ignore index)
    std::string task;
    int classes = 0;                  // classification
    int rows = 0;
    int batch_size = 0;
};

using cyxwiz::test::GraphBuilder;
using gui::NodeType;

ZooModel With(GraphBuilder& g, ZooModel model) {
    static_cast<cyxwiz::test::TrainGraph&>(model) = g.Take();
    return model;
}

// MLP + BatchNorm 1D + Dropout, SGD with momentum, Nesterov, weight decay and
// clipping. Dropout's rate is 0: torch's and the Engine's masks come from
// different generators, so a trajectory with a live mask cannot match (the
// mask itself is level 1's, test_transformer_dropout).
ZooModel MlpBatchNorm() {
    GraphBuilder g;
    const int data = g.Data({12});
    const int last = g.Chain(data, {g.Add(NodeType::Dense, "Hidden", {{"units", "16"}}),
                                    g.Add(NodeType::BatchNorm, "Norm"),
                                    g.Add(NodeType::ReLU, "ReLU"),
                                    g.Add(NodeType::Dropout, "Dropout", {{"rate", "0.0"}}),
                                    g.Add(NodeType::Dense, "Head", {{"units", "4"}})});
    const int loss = g.Add(NodeType::CrossEntropyLoss, "Loss");
    const int sgd = g.Add(NodeType::SGD, "SGD",
                          {{"learning_rate", "0.05"}, {"momentum", "0.9"}, {"nesterov", "true"},
                           {"weight_decay", "0.01"}, {"grad_clip_norm", "1.0"}});
    g.Finish(data, last, loss, sgd);
    ZooModel model;
    model.name = "mlp_batchnorm";
    model.summary = "Dense 16, BatchNorm, ReLU, Dropout 0, Dense 4; cross-entropy; SGD momentum Nesterov decay clip";
    model.input_shape = {12};
    model.task = "classification";
    model.classes = 4;
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

// 18 values reshaped to 6 steps of 3, a GRU over them, its last state to one
// output; mean squared error; Adam with a linear warm-up per update.
ZooModel GruRegressor() {
    GraphBuilder g;
    const int data = g.Data({18});
    const int last = g.Chain(data, {g.Add(NodeType::Reshape, "Steps", {{"shape", "6,3"}}),
                                    g.Add(NodeType::GRU, "Recurrent",
                                          {{"input_size", "3"}, {"hidden_size", "5"}, {"return_sequences", "false"}}),
                                    g.Add(NodeType::Dense, "Head", {{"units", "1"}})});
    const int loss = g.Add(NodeType::MSELoss, "Loss");
    const int adam = g.Add(NodeType::Adam, "Adam",
                           {{"learning_rate", "0.01"}, {"lr_schedule", "warmup_linear"}, {"warmup_ratio", "0.4"},
                            {"min_lr_ratio", "0.1"}});
    g.Finish(data, last, loss, adam);
    ZooModel model;
    model.name = "gru_regressor";
    model.summary = "Reshape 6 x 3, GRU 5 (last state), Dense 1; mean squared error; Adam, linear warm-up per update";
    model.input_shape = {18};
    model.task = "regression";
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

// 48 values reshaped to 6 positions of width 8, one transformer encoder block, flattened to a
// 3-class head; cross-entropy; AdamW with warm-up and cosine decay, clipping.
// Its dropouts are 0 (masks: see MlpBatchNorm).
ZooModel TransformerClassifier() {
    GraphBuilder g;
    const int data = g.Data({48});
    const int last = g.Chain(data, {g.Add(NodeType::Reshape, "Positions", {{"shape", "6,8"}}),
                                    g.Add(NodeType::TransformerEncoder, "Encoder",
                                          {{"d_model", "8"}, {"num_heads", "2"}, {"dim_feedforward", "16"},
                                           {"dropout", "0.0"}, {"ffn_dropout", "0.0"}}),
                                    g.Add(NodeType::Flatten, "Flatten"),
                                    g.Add(NodeType::Dense, "Head", {{"units", "3"}})});
    const int loss = g.Add(NodeType::CrossEntropyLoss, "Loss");
    const int adamw = g.Add(NodeType::AdamW, "AdamW",
                            {{"learning_rate", "0.003"}, {"weight_decay", "0.05"}, {"lr_schedule", "warmup_cosine"},
                             {"warmup_ratio", "0.4"}, {"min_lr_ratio", "0.1"}, {"grad_clip_norm", "0.5"}});
    g.Finish(data, last, loss, adamw);
    ZooModel model;
    model.name = "transformer_classifier";
    model.summary = "Reshape 6 x 8, transformer encoder (d 8, 2 heads, ff 16), Flatten, Dense 3; cross-entropy; AdamW, "
                    "warm-up + cosine per update, clip";
    model.input_shape = {48};
    model.task = "classification";
    model.classes = 3;
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

// A residual block on 6 x 5 images of 2 channels (each row is height,
// width, channel, the Engine's image rows; torch gets them channels first, as
// image data reaches a torch model), a stem convolution, two convolution + BatchNorm
// stages added back to the stem's output, Flatten, Dense 3; cross-entropy;
// RMSprop with momentum and clipping.
ZooModel ResNetBlock() {
    GraphBuilder g;
    const int data = g.Data({6, 5, 2});
    // A convolution feeding a BatchNorm has no bias, as in ResNet: the norm
    // takes the mean of it, so no gradient reaches it and only rounding noise
    // (scaled up by RMSprop) would move it, and the norm's running mean with it.
    const auto conv = [&](const std::string& name, bool bias) {
        return g.Add(NodeType::Conv2D, name, {{"filters", "4"}, {"kernel_size", "3"}, {"padding", "same"},
                                              {"use_bias", bias ? "true" : "false"}});
    };
    const int stem = g.Chain(data, {conv("Stem", true), g.Add(NodeType::ReLU, "StemReLU")});
    const int branch = g.Chain(stem, {conv("ConvA", false), g.Add(NodeType::BatchNorm, "NormA"),
                                      g.Add(NodeType::ReLU, "ReLUA"), conv("ConvB", false),
                                      g.Add(NodeType::BatchNorm, "NormB")});
    const int add = g.Add(NodeType::Add, "Residual");
    g.Connect(branch, add, 0, 0);
    g.Connect(stem, add, 0, 1);
    const int last = g.Chain(add, {g.Add(NodeType::ReLU, "OutReLU"), g.Add(NodeType::Flatten, "Flatten"),
                                   g.Add(NodeType::Dense, "Head", {{"units", "3"}})});
    const int loss = g.Add(NodeType::CrossEntropyLoss, "Loss");
    const int rmsprop = g.Add(NodeType::RMSprop, "RMSprop",
                              {{"learning_rate", "0.01"}, {"alpha", "0.99"}, {"momentum", "0.9"},
                               {"grad_clip_norm", "2.0"}});
    g.Finish(data, last, loss, rmsprop);
    ZooModel model;
    model.name = "resnet_block";
    model.summary = "6 x 5 x 2 images, Conv 4 + ReLU, [Conv 4 no bias, BN, ReLU, Conv 4 no bias, BN] + skip, ReLU, "
                    "Flatten, Dense 3; "
                    "cross-entropy; RMSprop momentum clip";
    model.input_shape = {6, 5, 2};
    model.image_rows = true;
    model.task = "classification";
    model.classes = 3;
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

// Token ids (8 per row, a vocabulary of 20; torch gets them as integers, as
// text reaches a torch embedding), an embedding, a bidirectional LSTM whose
// last time step goes to a 3-class head; cross-entropy; Adam with clipping.
ZooModel BiLstmClassifier() {
    GraphBuilder g;
    const int data = g.Data({8});
    const int last = g.Chain(data, {g.Add(NodeType::Embedding, "Tokens", {{"num_embeddings", "20"}, {"embedding_dim", "6"}}),
                                    g.Add(NodeType::LSTM, "Recurrent",
                                          {{"input_size", "6"}, {"hidden_size", "5"}, {"bidirectional", "true"},
                                           {"return_sequences", "false"}}),
                                    g.Add(NodeType::Dense, "Head", {{"units", "3"}})});
    const int loss = g.Add(NodeType::CrossEntropyLoss, "Loss");
    const int adam = g.Add(NodeType::Adam, "Adam", {{"learning_rate", "0.01"}, {"grad_clip_norm", "1.0"}});
    g.Finish(data, last, loss, adam);
    ZooModel model;
    model.name = "bilstm_classifier";
    model.summary = "Embedding 20 x 6, bidirectional LSTM 5 (last step), Dense 3; cross-entropy; Adam clip";
    model.input_shape = {8};
    model.token_ids = 20;
    model.task = "classification";
    model.classes = 3;
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

// A per-token tagger: token ids (up to 8 per row, the rest padding: token 0,
// label -100), an embedding, a bidirectional LSTM over every step, a Dense
// head per token; cross-entropy over the real tokens; Adam with clipping. The
// Engine reads the rows through its sequence batcher, as a token-tagging
// dataset trains.
ZooModel BiLstmTagger() {
    GraphBuilder g;
    const int data = g.Data({8});
    const int last = g.Chain(data, {g.Add(NodeType::Embedding, "Tokens", {{"num_embeddings", "20"}, {"embedding_dim", "6"}}),
                                    g.Add(NodeType::LSTM, "Recurrent",
                                          {{"input_size", "6"}, {"hidden_size", "5"}, {"bidirectional", "true"},
                                           {"return_sequences", "true"}}),
                                    g.Add(NodeType::TimeDistributed, "Tags", {{"units", "4"}})});
    const int loss = g.Add(NodeType::CrossEntropyLoss, "Loss", {{"ignore_index", "-100"}});
    const int adam = g.Add(NodeType::Adam, "Adam", {{"learning_rate", "0.01"}, {"grad_clip_norm", "1.0"}});
    g.Finish(data, last, loss, adam);
    ZooModel model;
    model.name = "bilstm_tagger";
    model.summary = "Embedding 20 x 6, bidirectional LSTM 5 (every step), Dense 4 per token; token cross-entropy "
                    "(padding ignored); Adam clip";
    model.input_shape = {8};
    model.token_ids = 20;
    model.task = "token_classification";
    model.classes = 4;
    model.rows = 40;
    model.batch_size = 8;
    return With(g, model);
}

std::vector<ZooModel> Zoo() {
    return {MlpBatchNorm(), GruRegressor(), TransformerClassifier(), ResNetBlock(), BiLstmClassifier(),
            BiLstmTagger()};
}

// ------------------------------------------------------------- the export --

cyxwiz::GraphCodeExport Export(const ZooModel& model) { return cyxwiz::test::UnstampedPyTorchExport(model); }

json Spec(const ZooModel& model, const cyxwiz::GraphCodeExport& exported) {
    json modules = json::array();
    for (const auto& module : exported.modules) {
        for (const auto& node : model.nodes)
            if (node.id == module.node_id) modules.push_back({{"node", node.name}, {"module", module.name}});
    }
    return {{"model", model.name},     {"summary", model.summary}, {"input_shape", model.input_shape},
            {"image_rows", model.image_rows},  {"token_ids", model.token_ids},
            {"task", model.task},      {"classes", model.classes}, {"rows", model.rows},
            {"batch_size", model.batch_size}, {"modules", modules}};
}

int WriteExports(const fs::path& dir) {
    fs::create_directories(dir);
    for (const ZooModel& model : Zoo()) {
        const auto exported = Export(model);
        std::ofstream(dir / (model.name + ".py"), std::ios::binary) << exported.code;
        std::ofstream(dir / (model.name + ".json"), std::ios::binary) << Spec(model, exported).dump(2) << "\n";
        std::cout << "wrote " << (dir / (model.name + ".py")).string() << "\n";
    }
    return 0;
}

// ------------------------------------------------------------ the Engine --

std::shared_ptr<arrow::Array> FloatColumn(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> IntColumn(const std::vector<int64_t>& values) {
    arrow::Int64Builder builder;
    (void)builder.AppendValues(values);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

// The fixture's rows as a table: one float column per input value, then "label".
std::shared_ptr<cyxwiz::ArrowDataset> Rows(const json& c) {
    const auto x = c.at("x").at("values").get<std::vector<float>>();
    const size_t rows = c.at("rows").get<size_t>();
    const size_t width = x.size() / rows;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t f = 0; f < width; ++f) {
        std::vector<float> column(rows);
        for (size_t r = 0; r < rows; ++r) column[r] = x[r * width + f];
        fields.push_back(arrow::field("x" + std::to_string(f), arrow::float32()));
        columns.push_back(FloatColumn(column));
    }
    if (c.at("task") == "classification") {
        fields.push_back(arrow::field("label", arrow::int64()));
        columns.push_back(IntColumn(c.at("y").get<std::vector<int64_t>>()));
    } else {
        fields.push_back(arrow::field("label", arrow::float32()));
        columns.push_back(FloatColumn(c.at("y").get<std::vector<float>>()));
    }
    auto table = arrow::Table::Make(arrow::schema(fields), columns, static_cast<int64_t>(rows));
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "oracle_train");
}

// A token-tagging fixture's rows as the sequence batcher's samples: each row's
// tokens up to its first padded position (label -100), in order, unshuffled.
std::unique_ptr<cyxwiz::ISequenceBatcher> TokenRows(const json& c, int batch_size) {
    const auto x = c.at("x").at("values").get<std::vector<float>>();
    const auto y = c.at("y").get<std::vector<std::vector<int64_t>>>();
    const size_t width = x.size() / y.size();
    std::vector<cyxwiz::SequenceSample> samples;
    for (size_t r = 0; r < y.size(); ++r) {
        cyxwiz::SequenceSample sample;
        for (size_t t = 0; t < width && y[r][t] != -100; ++t) {
            sample.word_ids.push_back(static_cast<int64_t>(x[r * width + t]));
            sample.tag_ids.push_back(y[r][t]);
        }
        samples.push_back(std::move(sample));
    }
    cyxwiz::SequenceBatcherConfig batcher;
    batcher.batch_size = static_cast<size_t>(batch_size);
    batcher.max_sequence_length = width;
    batcher.shuffle = false;
    batcher.word_pad_id = 0;
    batcher.tag_ignore_index = -100;
    return std::make_unique<cyxwiz::SequenceBatcher>(std::move(samples), batcher);
}

std::vector<std::string> TagNames(const json& c) {
    std::vector<std::string> names;
    for (int k = 0; k < c.at("classes").get<int>(); ++k) names.push_back("tag" + std::to_string(k));
    return names;
}

std::unique_ptr<cyxwiz::TrainingExecutor> MakeExecutor(const cyxwiz::TrainingConfiguration& config, const json& c,
                                                       int batch_size) {
    if (c.at("task") == "token_classification")
        return std::make_unique<cyxwiz::TrainingExecutor>(config, TokenRows(c, batch_size), TagNames(c));
    return std::make_unique<cyxwiz::TrainingExecutor>(config, Rows(c), "label");
}

struct Trajectory {
    std::string status, reason, layout_problem;
    size_t steps = 0;   // steps compared
    size_t values = 0;  // values compared, over every step
    Difference worst;
};

// One epoch over the fixture's rows; after each step the running loss and
// every parameter against the fixture's step.
Trajectory Train(const cyxwiz::TrainingConfiguration& config, const json& c, int batch_size, double atol,
                 double rtol, double noise_atol, bool print) {
    const json& steps = c.at("steps");
    cyxwiz::test::PinTrainingDevice();
    const auto made = MakeExecutor(config, c, batch_size);
    cyxwiz::TrainingExecutor& executor = *made;
    Trajectory out;
    std::map<std::string, std::vector<float>> noise;  // Engine key -> 1 where only noise moves the element
    bool noise_ready = false;
    executor.Train(1, batch_size, [&](int, int, int, float loss, float) {
        if (out.steps >= steps.size()) return;
        const json& expected = steps.at(out.steps);
        const std::string at = "step " + std::to_string(out.steps + 1);
        Difference here;
        Compare({loss}, {expected.at("running_loss").get<float>()}, atol, rtol, at + " running loss", here);
        if (cyxwiz::IExecutableModel* built = executor.GetExecutableModel()) {
            // CYXWIZ_ORACLE_TRAIN_KEYS=1: the model's parameter keys and the
            // matcher's slots, once (to see how the two name a layer).
            if (out.steps == 0 && std::getenv("CYXWIZ_ORACLE_TRAIN_KEYS")) {
                for (const auto& [key, tensor] : built->GetParameters())
                    std::cout << "      model key " << key << " " << cyxwiz::weight_match::ShapeText(tensor.Shape()) << "\n";
                for (const auto& slot : cyxwiz::WeightSlots(*built, executor.GetModuleProvenance()))
                    std::cout << "      slot " << slot.node_name << " prefix " << slot.key_prefix << "\n";
            }
            if (!noise_ready) {
                // A whole file for the matcher (it loads a layer only when
                // every tensor of it is there): the mask, zeros elsewhere.
                json full = json::object();
                const json none = json::object();
                const json& masked = c.contains("noise_mask") ? c.at("noise_mask") : none;
                for (const auto& [name, tensor] : expected.at("parameters").items()) {
                    if (masked.contains(name)) {
                        full[name] = masked.at(name);
                    } else {
                        full[name] = {{"shape", tensor.at("shape")},
                                      {"values", std::vector<float>(tensor.at("values").size(), 0.0f)}};
                    }
                }
                std::string ignored;
                noise = ExpectedInEngineLayout(full, *built, executor.GetModuleProvenance(), ignored);
                noise_ready = true;
            }
            std::string problem;
            const auto want =
                ExpectedInEngineLayout(expected.at("parameters"), *built, executor.GetModuleProvenance(), problem);
            if (!problem.empty() && out.layout_problem.empty()) out.layout_problem = problem;
            const auto have = built->GetParameters();
            for (const auto& [key, values] : want) {
                const auto it = have.find(key);
                if (it == have.end()) {
                    here.excess = 1e30;
                    here.where = at + ": the model has no " + key;
                    continue;
                }
                const float* data = it->second.ReadData<float>();
                // Elements no gradient reaches (an attention key bias, a bias
                // a BatchNorm takes the mean of): rounding noise, scaled up by
                // Adam / RMSprop, is all that moves them, on either side; they
                // are held to the fixture's absolute noise bound.
                const auto mask = noise.find(key);
                Compare(std::vector<float>(data, data + it->second.NumElements()), values, atol, rtol, at + " " + key,
                        here, mask == noise.end() ? nullptr : &mask->second, noise_atol);
                out.values += values.size();
            }
        }
        if (print)
            std::cout << "    " << at << ": loss " << loss << " (torch " << expected.at("running_loss").get<float>()
                      << "), worst " << here.excess << " tolerances at " << here.where << "\n";
        if (here.excess > out.worst.excess) out.worst = here;
        ++out.steps;
    });
    const auto metrics = executor.GetMetrics();
    out.status = metrics.terminal_status;
    out.reason = metrics.terminal_reason;
    return out;
}

void RunModel(const ZooModel& model, const json& c, const json& tolerance) {
    std::cout << model.name << ": " << model.summary << "\n";
    const auto exported = Export(model);
    const fs::path committed = FixtureDir() / (model.name + ".py");
    const std::string committed_text = ReadText(committed);
    Check(!committed_text.empty(), model.name + ": the committed export is readable (" + committed.string() + ")");
    Check(exported.code == committed_text,
          model.name + ": the Engine's PyTorch export is the committed one (if the export changed: "
                       "test_oracle_train --write-exports, then the generator)");
    Check(c.at("export_sha256") == Sha256Text(committed_text),
          model.name + ": the fixture was made from the committed export");

    // Parameter nodes take the export's module names: the state-dict's keys.
    const cyxwiz::test::TrainGraph named = cyxwiz::test::NamedAfterModules(model, exported);

    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(named.nodes, named.links, true);
    for (const auto& issue : config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cout << "    compile: " << issue.message << "\n";
    Check(config.is_valid, model.name + ": the graph compiles");
    if (!config.is_valid) return;
    for (const auto& layer : config.layers) {
        std::cout << "    layer " << layer.name << ": " << cyxwiz::weight_match::ShapeText(layer.input_shape) << " -> "
                  << cyxwiz::weight_match::ShapeText(layer.output_shape);
        for (const char* key : {"input_size", "hidden_size", "units"})
            if (const auto it = layer.parameters.find(key); it != layer.parameters.end())
                std::cout << " " << key << "=" << it->second;
        std::cout << "\n";
    }

    const fs::path work = fs::temp_directory_path() / ("cyxwiz_oracle_train_" + model.name);
    fs::remove_all(work);
    fs::create_directories(work);
    config.dataset_name = "oracle_train";
    config.batch_size = model.batch_size;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 1;
    config.checkpoint_dir = (work / "checkpoints").string();
    config.initialize_from_weights_file = (FixtureDir() / c.at("init_weights").get<std::string>()).string();
    if (model.task == "token_classification") {
        // A token-tagging dataset trains through the sequence batcher.
        config.sequence_batch.enabled = true;
        config.sequence_batch.ignore_index = -100;
        config.sequence_batch.max_sequence_length = static_cast<int>(model.input_shape.at(0));
        config.loss_params["ignore_index"] = "-100";
    }

    const double atol = tolerance.at("atol").get<double>();
    const double rtol = tolerance.at("rtol").get<double>();
    const double noise_atol = tolerance.at("noise_atol").get<double>();
    const Trajectory run = Train(config, c, model.batch_size, atol, rtol, noise_atol, true);
    const size_t steps = c.at("steps").size();
    Check(run.status == "completed", model.name + ": the run completes (" + run.status + ": " + run.reason + ")");
    Check(run.layout_problem.empty(), model.name + ": every layer takes the fixture's values: " + run.layout_problem);
    Check(run.steps == steps && run.values > 0,
          model.name + ": " + std::to_string(run.steps) + " of " + std::to_string(steps) + " steps compared, " +
              std::to_string(run.values) + " values");
    Check(run.worst.excess <= 1.0, model.name + ": every step within tolerance (worst " +
                                       std::to_string(run.worst.excess) + " tolerances, " +
                                       std::to_string(run.worst.abs) + " at " + run.worst.where + ")");

    // The comparison can fail: the same run with the learning rate x 1.1
    // must miss by more than ten tolerances.
    cyxwiz::TrainingConfiguration mutated = config;
    mutated.learning_rate *= 1.1f;
    const Trajectory miss = Train(mutated, c, model.batch_size, atol, rtol, noise_atol, false);
    Check(miss.worst.excess > 10.0, model.name + ": learning rate x 1.1 misses (" +
                                        std::to_string(miss.worst.excess) + " tolerances at " + miss.worst.where + ")");
    fs::remove_all(work);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--write-exports") return WriteExports(argv[2]);
    if (!cyxwiz::test::BeginTrainingOracleTest("test_oracle_train")) return 1;
    try {
        cyxwiz::GraphCompilerDatasetHooks hooks;
        hooks.is_dataset_registered = [](const std::string&) { return false; };
        cyxwiz::SetGraphCompilerDatasetHooks(hooks);
        const fs::path debug_root = fs::temp_directory_path() / "cyxwiz_oracle_train_debug";
        const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_override(debug_root);

        std::ifstream in(FixtureDir().parent_path() / "train_trajectory_pytorch.json");
        Check(in.good(), "train_trajectory_pytorch.json is readable");
        if (!in.good()) return 1;
        const json fixture = json::parse(in);
        std::cout << "fixture: torch " << fixture.at("torch_version").get<std::string>() << "\n";
        for (const ZooModel& model : Zoo()) {
            const auto it = std::find_if(fixture.at("models").begin(), fixture.at("models").end(),
                                         [&](const json& c) { return c.at("name") == model.name; });
            Check(it != fixture.at("models").end(), model.name + ": the fixture has it");
            if (it != fixture.at("models").end()) RunModel(model, *it, fixture.at("tolerance"));
        }
        const std::string device_problem = cyxwiz::test::TrainingDeviceProblem();
        Check(device_problem.empty(), "every run on the requested device: " + device_problem);
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
