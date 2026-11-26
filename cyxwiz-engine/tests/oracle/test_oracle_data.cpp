// Level 3 of the oracle harness (TOFIX228 1.6), tabular data end to end: a
// committed CSV goes through the Engine's own data path, and every step of it
// is compared with Python on the same file.
//
//   CSV -> Data Input -> StandardScaler (fitted on the Train rows) with the
//   training split -> the Arrow batchers (BuildArrowTrainingBatchers, as a
//   training run builds them) -> TrainingExecutor, 2 epochs, shuffled
//
// Compared with fixtures/data_parity_tabular_pytorch.json
// (reference/generate_data_parity_fixtures.py):
//   - each row's part (train / validation / test): the generator ports the
//     split rule (SplitAssigner on std::mt19937_64) to Python on its own;
//   - the scaled columns: StandardScaler's formula (scikit-learn's) fitted on the Train rows;
//   - each epoch's order of the Train rows and every batch's values and
//     labels: the generator ports the epoch shuffle (std::mt19937 seeded with
//     TrainingEpochSeed(dataloader seed, epoch), portable Fisher-Yates) and
//     feeds that order to a torch DataLoader as its sampler;
//   - the training: per step the epoch's running loss, per epoch the
//     validation loss and every parameter, PyTorch running the model's own
//     export (as level 2) on those batches.
//
//   test_oracle_data                      compare with the committed fixture
//   test_oracle_data --write-exports DIR  write the model's export and spec
#include "training_oracle_test.h"
#include "train_oracle_support.h"
#include "../../src/core/arrow_dataset.h"
#include "../../src/core/audio_dataset_batcher.h"
#include "../../src/core/batch_uploads.h"
#include "../../src/core/image_dataset_batcher.h"
#include "../../src/core/prepared_image_set.h"
#include "../../src/core/sample_split.h"
#include "../../src/core/data_registry.h"
#include "../../src/core/debug_run_paths.h"
#include "../../src/core/graph_compiler.h"
#include "../../src/core/graph_compiler_dataset_hooks.h"
#include "../../src/core/pipeline_executor.h"
#include "../../src/core/split_partitioning.h"
#include "../../src/core/training_batcher_setup.h"
#include "../../src/core/training_executor.h"
#include "../../src/core/training_resume_checkpoint.h"
#include "../../src/core/training_trace_collector.h"

#include <arrow/api.h>

#include <algorithm>
#include <functional>

#include <iostream>
#include <memory>
#include <optional>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
using cyxwiz::test::Compare;
using cyxwiz::test::Difference;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

fs::path FixtureRoot() {
    return fs::path(CYXWIZ_SOURCE_ROOT) / "cyxwiz-engine" / "tests" / "computation_truth" / "fixtures";
}
fs::path ModelDir() { return FixtureRoot() / "data_parity"; }

constexpr const char* kModel = "tabular_mlp";
constexpr int kBatch = 8;
constexpr int kEpochs = 2;
constexpr int kSplitSeed = 7;
constexpr int kLoaderSeed = 11;

// Data [4] -> Dense 6 -> ReLU -> Dense 3; cross-entropy; plain SGD.
cyxwiz::test::TrainGraph Model() {
    cyxwiz::test::GraphBuilder g;
    const int data = g.Data({4}, "oracle_data");
    const int last = g.Chain(data, {g.Add(gui::NodeType::Dense, "Hidden", {{"units", "6"}}),
                                    g.Add(gui::NodeType::ReLU, "ReLU"),
                                    g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}})});
    const int loss = g.Add(gui::NodeType::CrossEntropyLoss, "Loss");
    const int sgd = g.Add(gui::NodeType::SGD, "SGD", {{"learning_rate", "0.1"}, {"momentum", "0"}});
    g.Finish(data, last, loss, sgd);
    return g.Take();
}

// The sparse model: 10 TF-IDF features straight into a Dense 3 head (the
// Engine routes CSR rows into its first Linear); cross-entropy; plain SGD.
cyxwiz::test::TrainGraph SparseModel() {
    cyxwiz::test::GraphBuilder g;
    const int data = g.Data({10}, "oracle_sparse");
    const int last = g.Chain(data, {g.Add(gui::NodeType::Dense, "Head", {{"units", "3"}})});
    const int loss = g.Add(gui::NodeType::CrossEntropyLoss, "Loss");
    const int sgd = g.Add(gui::NodeType::SGD, "SGD", {{"learning_rate", "0.5"}, {"momentum", "0"}});
    g.Finish(data, last, loss, sgd);
    return g.Take();
}

// A model per other kind, trained on that kind's batchers (the loss
// trajectory of design step 5): small, so the comparison is of the data path
// and the training loop, not of a layer level 1 and 2 cover.
cyxwiz::test::TrainGraph KindModel(const std::vector<size_t>& shape, const std::string& dataset,
                                   std::vector<std::pair<gui::NodeType, std::map<std::string, std::string>>> layers,
                                   gui::NodeType loss_type, const std::string& learning_rate) {
    cyxwiz::test::GraphBuilder g;
    const int data = g.Data(shape, dataset);
    std::vector<int> chain;
    int n = 0;
    for (auto& [type, params] : layers) chain.push_back(g.Add(type, "Layer" + std::to_string(++n), params));
    const int last = g.Chain(data, chain);
    const int loss = g.Add(loss_type, "Loss");
    const int sgd = g.Add(gui::NodeType::SGD, "SGD", {{"learning_rate", learning_rate}, {"momentum", "0"}});
    g.Finish(data, last, loss, sgd);
    return g.Take();
}

// Text: the 6 token ids of a row, an embedding, Flatten, Dense 3.
cyxwiz::test::TrainGraph TextModel() {
    return KindModel({6}, "oracle_text",
                     {{gui::NodeType::Embedding, {{"num_embeddings", "12"}, {"embedding_dim", "3"}}},
                      {gui::NodeType::Flatten, {}},
                      {gui::NodeType::Dense, {{"units", "3"}}}},
                     gui::NodeType::CrossEntropyLoss, "0.5");
}

// Series: a window of 4 values and 4 temperatures to the next value.
cyxwiz::test::TrainGraph SeriesModel() {
    return KindModel({8}, "oracle_series", {{gui::NodeType::Dense, {{"units", "1"}}}}, gui::NodeType::MSELoss,
                     "0.0005");
}

// Image: 4 x 4 RGB in [H, W, C] order, Flatten, Dense 2.
cyxwiz::test::TrainGraph ImageModel() {
    return KindModel({4, 4, 3}, "oracle_image", {{gui::NodeType::Flatten, {}}, {gui::NodeType::Dense, {{"units", "2"}}}},
                     gui::NodeType::CrossEntropyLoss, "0.05");
}

// Audio: the [33 bins, 63 frames] spectrogram, Flatten, Dense 2.
cyxwiz::test::TrainGraph AudioModel() {
    return KindModel({33, 63}, "oracle_audio", {{gui::NodeType::Flatten, {}}, {gui::NodeType::Dense, {{"units", "2"}}}},
                     gui::NodeType::CrossEntropyLoss, "0.005");
}

void WriteExport(const fs::path& dir, const std::string& name, const cyxwiz::test::TrainGraph& graph) {
    const auto exported = cyxwiz::test::UnstampedPyTorchExport(graph);
    std::ofstream(dir / (name + ".py"), std::ios::binary) << exported.code;
    std::cout << "wrote " << (dir / (name + ".py")).string() << "\n";
}

int WriteExports(const fs::path& dir) {
    fs::create_directories(dir);
    const auto graph = Model();
    const auto exported = cyxwiz::test::UnstampedPyTorchExport(graph);
    json modules = json::array();
    for (const auto& module : exported.modules)
        for (const auto& node : graph.nodes)
            if (node.id == module.node_id) modules.push_back({{"node", node.name}, {"module", module.name}});
    std::ofstream(dir / (std::string(kModel) + ".py"), std::ios::binary) << exported.code;
    std::ofstream(dir / (std::string(kModel) + ".json"), std::ios::binary)
        << json{{"model", kModel}, {"modules", modules}, {"batch_size", kBatch}, {"epochs", kEpochs},
                {"split_seed", kSplitSeed}, {"dataloader_seed", kLoaderSeed}}
               .dump(2)
        << "\n";
    std::cout << "wrote " << (dir / (std::string(kModel) + ".py")).string() << "\n";
    WriteExport(dir, "sparse_linear", SparseModel());
    WriteExport(dir, "text_model", TextModel());
    WriteExport(dir, "series_model", SeriesModel());
    WriteExport(dir, "image_model", ImageModel());
    WriteExport(dir, "audio_model", AudioModel());
    return 0;
}

std::vector<double> Values(const cyxwiz::Tensor& tensor) {
    std::vector<double> out(tensor.NumElements());
    switch (tensor.GetDataType()) {
        case cyxwiz::DataType::Float32: {
            const float* v = tensor.ReadData<float>();
            std::copy(v, v + out.size(), out.begin());
            break;
        }
        case cyxwiz::DataType::Float64: {
            const double* v = tensor.ReadData<double>();
            std::copy(v, v + out.size(), out.begin());
            break;
        }
        case cyxwiz::DataType::Int32: {
            const int32_t* v = tensor.ReadData<int32_t>();
            std::copy(v, v + out.size(), out.begin());
            break;
        }
        default:
            throw std::runtime_error("a batch tensor of an unexpected type");
    }
    return out;
}

// The Train batcher as the executor sees it, recording each epoch: the order
// of its rows, every batch's values and labels, and the epoch seed it was
// given. A Reset ends an epoch.
class RecordingBatcher : public cyxwiz::IBatcher {
public:
    struct Epoch {
        std::optional<uint64_t> seed;  // the seed set before the Reset that started it
        std::vector<int64_t> rows;
        std::vector<size_t> batch_sizes;
        std::vector<double> data;
        std::vector<double> labels;
    };

    explicit RecordingBatcher(std::shared_ptr<cyxwiz::IBatcher> inner) : inner_(std::move(inner)) { epochs_.emplace_back(); }

    cyxwiz::Batch GetNextBatch() override {
        cyxwiz::Batch batch = inner_->GetNextBatch();
        if (batch.IsValid()) {
            Epoch& epoch = epochs_.back();
            epoch.rows.insert(epoch.rows.end(), batch.row_ids.begin(), batch.row_ids.end());
            epoch.batch_sizes.push_back(batch.size);
            const auto data = Values(batch.data);
            const auto labels = Values(batch.labels);
            epoch.data.insert(epoch.data.end(), data.begin(), data.end());
            // A class id per row, or one-hot rows (their largest entry).
            const size_t per_row = batch.size > 0 ? labels.size() / batch.size : 0;
            for (size_t r = 0; r < batch.size; ++r) {
                if (per_row <= 1) {
                    epoch.labels.push_back(labels.at(r));
                } else {
                    const auto first = labels.begin() + static_cast<std::ptrdiff_t>(r * per_row);
                    epoch.labels.push_back(static_cast<double>(
                        std::max_element(first, first + static_cast<std::ptrdiff_t>(per_row)) - first));
                }
            }
        }
        return batch;
    }
    void Reset() override {
        inner_->Reset();
        epochs_.emplace_back();
    }
    bool BeginEpoch(uint64_t seed) override {
        const bool exact = inner_->BeginEpoch(seed);
        epochs_.emplace_back();
        epochs_.back().seed = seed;
        return exact;
    }
    bool IsEpochComplete() const override { return inner_->IsEpochComplete(); }
    size_t GetNumBatches() const override { return inner_->GetNumBatches(); }
    size_t GetNumSamples() const override { return inner_->GetNumSamples(); }
    void SetNormalization(float mean, float std_dev) override { inner_->SetNormalization(mean, std_dev); }
    void SetOneHotEncoding(size_t classes) override { inner_->SetOneHotEncoding(classes); }
    void SetScalarLabelMode(bool enable) override { inner_->SetScalarLabelMode(enable); }
    void SetClassIndexLabelMode(bool enable) override { inner_->SetClassIndexLabelMode(enable); }
    void SetFlatten(bool flatten) override { inner_->SetFlatten(flatten); }
    void SetBatchInspectionEnabled(bool enable) override { inner_->SetBatchInspectionEnabled(enable); }
    void SetDropLast(bool drop_last) override { inner_->SetDropLast(drop_last); }
    void SetSparseFeatureOutput(bool enable) override { inner_->SetSparseFeatureOutput(enable); }
    void SetPhase(cyxwiz::BatcherPhase phase) override { inner_->SetPhase(phase); }
    std::optional<cyxwiz::Batch> GetSourceSamples(const std::vector<size_t>& indices) override {
        return inner_->GetSourceSamples(indices);
    }
    size_t SourceSampleCount() const override { return inner_->SourceSampleCount(); }
    std::optional<cyxwiz::BatcherPhase> SourceSamplePart(size_t index) const override {
        return inner_->SourceSamplePart(index);
    }
    std::string SourceSampleName(size_t index) const override { return inner_->SourceSampleName(index); }
    std::vector<std::string> ClassNames() const override { return inner_->ClassNames(); }
    std::optional<std::vector<size_t>> TrainClassCounts() const override { return inner_->TrainClassCounts(); }

    // The epochs that served batches, in order.
    std::vector<Epoch> Epochs() const {
        std::vector<Epoch> out;
        for (const auto& epoch : epochs_)
            if (!epoch.rows.empty()) out.push_back(epoch);
        return out;
    }

private:
    std::shared_ptr<cyxwiz::IBatcher> inner_;
    std::vector<Epoch> epochs_;
};

// A training run against a fixture's "steps" (each step's running loss) and
// "epochs" (each epoch's validation loss and every parameter, torch names and
// layouts, put in the Engine's by its weight matcher).
struct RunCheck {
    Difference loss, epoch;
    std::string layout_problem, status, reason;
    size_t steps = 0;
};

RunCheck TrainAgainst(const cyxwiz::TrainingConfiguration& config, cyxwiz::ResolvedExternalBatchers resolved,
                      const json& steps, const json& epochs, double atol, double rtol) {
    RunCheck out;
    cyxwiz::test::PinTrainingDevice();
    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    int epoch_index = 0;
    executor.Train(
        config.epochs, config.batch_size,
        [&](int, int, int, float loss, float) {
            if (out.steps < steps.size()) {
                const float torch_loss = steps.at(out.steps).at("running_loss").get<float>();
                std::cout << "    step " << (out.steps + 1) << ": running loss " << loss << " (torch " << torch_loss
                          << ")\n";
                Compare({loss}, {torch_loss}, atol, rtol, "step " + std::to_string(out.steps + 1) + " running loss",
                        out.loss);
            }
            ++out.steps;
        },
        [&](int, float, float, float val_loss, float, float) {
            if (epoch_index >= static_cast<int>(epochs.size())) return;
            const json& expected = epochs.at(epoch_index);
            const std::string at = "epoch " + std::to_string(epoch_index + 1);
            Compare({val_loss}, {expected.at("val_loss").get<float>()}, atol, rtol, at + " validation loss",
                    out.epoch);
            if (cyxwiz::IExecutableModel* built = executor.GetExecutableModel()) {
                std::string problem;
                const auto want = cyxwiz::test::ExpectedInEngineLayout(expected.at("parameters"), *built,
                                                                       executor.GetModuleProvenance(), problem);
                if (!problem.empty() && out.layout_problem.empty()) out.layout_problem = problem;
                const auto have = built->GetParameters();
                for (const auto& [key, values] : want) {
                    const auto it = have.find(key);
                    if (it == have.end()) continue;
                    const float* data = it->second.ReadData<float>();
                    Compare(std::vector<float>(data, data + it->second.NumElements()), values, atol, rtol,
                            at + " " + key, out.epoch);
                }
            }
            ++epoch_index;
        });
    const auto metrics = executor.GetMetrics();
    out.status = metrics.terminal_status;
    out.reason = metrics.terminal_reason;
    return out;
}

// The copy budget (design step 5): the same training run with nothing
// reading the model in between, and the copies it made per training batch,
// by kind: device-to-host reads (the trace's host-synchronization groups)
// and host-to-device uploads (the batchers' upload groups, batch_uploads.h).
struct CopyCount {
    uint64_t total = 0, unknown = 0;
    std::map<std::string, uint64_t> by_group;  // "category / operation (...)" -> reads
    uint64_t uploads = 0;
    std::map<std::string, uint64_t> uploads_by_reason;  // "batch", "resident", "index" -> uploads
    std::map<std::string, uint64_t> uploads_by_group;   // "batcher / part (...)" -> uploads
};

std::string GroupName(const cyxwiz::TrainingTraceHostSyncGroup& group) {
    std::string shape;
    for (size_t d : group.shape) shape += (shape.empty() ? "" : "x") + std::to_string(d);
    return group.category + " / " + group.operation + " (" + group.reason + "; " + group.dtype + " " + shape + ")";
}

CopyCount CountCopies(const cyxwiz::TrainingConfiguration& config, const std::shared_ptr<cyxwiz::ArrowDataset>& rows,
                      size_t& batches) {
    auto resolved = cyxwiz::TakeResolvedExternalBatchers(cyxwiz::BuildArrowTrainingBatchers(config, rows, "segment", kBatch));
    cyxwiz::test::PinTrainingDevice();
    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    batches = 0;
    executor.Train(kEpochs, kBatch, [&](int, int, int, float, float) { ++batches; });
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    CopyCount out;
    for (const auto& group : trace.arrayfire_host_sync_groups) {
        out.total += group.event_count;
        if (group.category == "unknown") out.unknown += group.event_count;
        out.by_group[GroupName(group)] += group.event_count;
    }
    for (const auto& group : trace.host_upload_groups) {
        out.uploads += group.event_count;
        out.uploads_by_reason[group.reason] += group.event_count;
        out.uploads_by_group[GroupName(group)] += group.event_count;
    }
    return out;
}

std::string PipelineJson(const fs::path& csv) {
    json j;
    j["nodes"] = json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Rows"},
          {"parameters", {{"source_type", "file"}, {"file_path", csv.string()}, {"type", "csv"}, {"has_header", "true"},
                          {"delimiter", ","}, {"label_column", "segment"}, {"force_disk_backed", "false"}}}},
         {{"id", 2}, {"type", "StandardScaler"}, {"name", "Scale"}, {"parameters", {{"columns", "age,income"}}}}});
    j["links"] = json::array({{{"start_node", 1}, {"end_node", 2}}});
    j["training_split"] = {{"label_column", "segment"}, {"train", 0.7}, {"val", 0.15}, {"test", 0.15},
                           {"seed", kSplitSeed},        {"stratified", true}, {"shuffle", true}};
    return j.dump();
}

std::vector<double> ColumnValues(const std::shared_ptr<arrow::Table>& table, const std::string& name) {
    std::vector<double> out;
    const auto column = table->GetColumnByName(name);
    if (!column) return out;
    for (const auto& chunk : column->chunks()) {
        for (int64_t i = 0; i < chunk->length(); ++i) {
            switch (chunk->type_id()) {
                case arrow::Type::DOUBLE: out.push_back(std::static_pointer_cast<arrow::DoubleArray>(chunk)->Value(i)); break;
                case arrow::Type::FLOAT: out.push_back(std::static_pointer_cast<arrow::FloatArray>(chunk)->Value(i)); break;
                case arrow::Type::INT64: out.push_back(static_cast<double>(std::static_pointer_cast<arrow::Int64Array>(chunk)->Value(i))); break;
                case arrow::Type::INT8: out.push_back(std::static_pointer_cast<arrow::Int8Array>(chunk)->Value(i)); break;
                default: throw std::runtime_error("column " + name + " has type " + chunk->type()->ToString());
            }
        }
    }
    return out;
}

void Run(const json& fixture) {
    const json& tol = fixture.at("tolerance");
    const fs::path csv = ModelDir() / fixture.at("csv").get<std::string>();

    // 1. The pipeline run: the file, the split, the scaler fitted on Train rows.
    cyxwiz::PipelineExecutor pipeline;
    const bool ran = pipeline.ExecutePipeline(PipelineJson(csv));
    Check(ran, "the pipeline runs the CSV, the split and the scaler: " + pipeline.GetLastError());
    if (!ran) return;
    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto scaled = registry.GetArrowDataset(pipeline.NodeResults().at(2));
    Check(scaled && scaled->GetArrowTable(), "the scaled rows are in memory");
    if (!scaled || !scaled->GetArrowTable()) return;
    const auto table = scaled->GetArrowTable();
    const size_t rows = fixture.at("rows").get<size_t>();
    Check(static_cast<size_t>(table->num_rows()) == rows, "the CSV's " + std::to_string(rows) + " rows are read");

    const auto partition = ColumnValues(table, cyxwiz::kSplitPartitionColumn);
    const auto expected_partition = fixture.at("partition").get<std::vector<int>>();
    size_t partition_mismatch = 0;
    for (size_t r = 0; r < std::min(partition.size(), expected_partition.size()); ++r)
        if (static_cast<int>(partition[r]) != expected_partition[r]) ++partition_mismatch;
    Check(partition.size() == expected_partition.size() && partition_mismatch == 0,
          "every row's part (train / validation / test) is the Python port's of the split rule (" +
              std::to_string(partition_mismatch) + " rows differ)");

    for (const auto& [column, expected] : fixture.at("scaled").items()) {
        Difference worst;
        const auto values = ColumnValues(table, column);
        std::vector<float> as_float(values.size());
        std::transform(values.begin(), values.end(), as_float.begin(), [](double v) { return static_cast<float>(v); });
        Compare(as_float, expected.get<std::vector<float>>(), tol.at("data_atol").get<double>(),
                tol.at("data_rtol").get<double>(), column, worst);
        Check(worst.excess <= 1.0, column + ": scaled as StandardScaler (scikit-learn's formula) fitted on the Train rows "
                                            "(worst " + std::to_string(worst.excess) + " tolerances at " + worst.where + ")");
    }

    // 2. The training run: the batchers a run builds, the Train one recorded.
    const auto graph = Model();
    const auto exported = cyxwiz::test::UnstampedPyTorchExport(graph);
    Check(exported.code == cyxwiz::test::ReadText(ModelDir() / (std::string(kModel) + ".py")),
          "the model's PyTorch export is the committed one (if it changed: test_oracle_data --write-exports, then "
          "the generator)");
    Check(fixture.at("export_sha256") ==
              cyxwiz::test::Sha256Text(cyxwiz::test::ReadText(ModelDir() / (std::string(kModel) + ".py"))),
          "the fixture was made from the committed export");
    const auto named = cyxwiz::test::NamedAfterModules(graph, exported);
    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(named.nodes, named.links, true);
    for (const auto& issue : config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cout << "    compile: " << issue.message << "\n";
    Check(config.is_valid, "the model compiles");
    if (!config.is_valid) return;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_oracle_data";
    fs::remove_all(work);
    fs::create_directories(work);
    config.dataset_name = pipeline.NodeResults().at(2);
    config.batch_size = kBatch;
    config.epochs = kEpochs;
    config.shuffle = true;
    config.stratified = true;
    config.split_seed = kSplitSeed;
    config.dataloader_seed = kLoaderSeed;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.has_data_split = true;
    config.validation_freq = 1;
    config.num_workers = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 1;
    config.checkpoint_dir = (work / "checkpoints").string();
    config.initialize_from_weights_file = (ModelDir() / fixture.at("init_weights").get<std::string>()).string();

    auto set = cyxwiz::BuildArrowTrainingBatchers(config, scaled, "segment", kBatch);
    const auto classes = set.label_classes;
    Check(classes == fixture.at("classes").get<std::vector<std::string>>(),
          "the label classes, in index order, are the fixture's");
    auto resolved = cyxwiz::TakeResolvedExternalBatchers(std::move(set));
    Check(resolved.train && resolved.dev, "a Train and a Validation batcher");
    if (!resolved.train || !resolved.dev) return;
    auto recorder = std::make_shared<RecordingBatcher>(resolved.train);
    resolved.train = recorder;

    const json& epochs = fixture.at("epochs");
    const RunCheck run = TrainAgainst(config, std::move(resolved), fixture.at("steps"), epochs,
                                      tol.at("atol").get<double>(), tol.at("rtol").get<double>());
    Check(run.status == "completed", "the run completes (" + run.status + ": " + run.reason + ")");
    const size_t step = run.steps;
    const json& steps = fixture.at("steps");
    const Difference& loss_worst = run.loss;
    const Difference& epoch_worst = run.epoch;
    const std::string& layout_problem = run.layout_problem;

    // 3. What the Train batcher served, epoch by epoch.
    const auto served = recorder->Epochs();
    Check(served.size() == static_cast<size_t>(kEpochs),
          std::to_string(served.size()) + " epochs served (" + std::to_string(kEpochs) + " trained)");
    const json& features = fixture.at("features");
    const auto class_ids = fixture.at("class_ids").get<std::vector<int>>();
    const size_t width = features.at(0).size();
    for (size_t e = 0; e < std::min(served.size(), epochs.size()); ++e) {
        const std::string at = "epoch " + std::to_string(e + 1);
        const auto order = epochs.at(e).at("order").get<std::vector<int64_t>>();
        std::cout << "    " << at << ": seed " << (served[e].seed ? std::to_string(*served[e].seed) : "none")
                  << ", first rows";
        for (size_t i = 0; i < std::min<size_t>(6, served[e].rows.size()); ++i) std::cout << " " << served[e].rows[i];
        std::cout << " (Python:";
        for (size_t i = 0; i < std::min<size_t>(6, order.size()); ++i) std::cout << " " << order[i];
        std::cout << ")\n";
        Check(served[e].rows == order, at + ": the Train rows in the order of the Python port of the epoch shuffle");
        // Every batch's values and labels are the rows it names.
        Difference data_worst;
        size_t label_mismatch = 0;
        for (size_t i = 0; i < served[e].rows.size(); ++i) {
            const auto row = static_cast<size_t>(served[e].rows[i]);
            std::vector<float> got(width);
            for (size_t f = 0; f < width; ++f) got[f] = static_cast<float>(served[e].data.at(i * width + f));
            Compare(got, features.at(row).get<std::vector<float>>(), tol.at("data_atol").get<double>(),
                    tol.at("data_rtol").get<double>(), at + " row " + std::to_string(row), data_worst);
            if (static_cast<int>(served[e].labels.at(i)) != class_ids.at(row)) ++label_mismatch;
        }
        Check(data_worst.excess <= 1.0, at + ": every batch holds its rows' values (worst " +
                                            std::to_string(data_worst.excess) + " tolerances at " + data_worst.where + ")");
        Check(label_mismatch == 0, at + ": every batch holds its rows' class ids (" + std::to_string(label_mismatch) +
                                       " differ)");
        const auto sizes = epochs.at(e).at("batch_sizes").get<std::vector<size_t>>();
        Check(served[e].batch_sizes == sizes, at + ": the batch sizes (the last one partial, not dropped)");
    }

    Check(step == steps.size(), std::to_string(step) + " of " + std::to_string(steps.size()) + " steps");
    Check(loss_worst.excess <= 1.0, "every step's running loss is PyTorch's (worst " +
                                        std::to_string(loss_worst.excess) + " tolerances at " + loss_worst.where + ")");
    Check(layout_problem.empty(), "every layer takes the fixture's values: " + layout_problem);
    Check(epoch_worst.excess <= 1.0, "every epoch's validation loss and parameters are PyTorch's (worst " +
                                         std::to_string(epoch_worst.excess) + " tolerances at " + epoch_worst.where + ")");

    size_t batches = 0;
    const CopyCount copies = CountCopies(config, scaled, batches);
    std::cout << "    copies: " << copies.total << " device-to-host reads over " << batches << " training batches\n";
    for (const auto& [group, reads] : copies.by_group) std::cout << "      " << reads << "  " << group << "\n";
    // The training loop's own reads: the epoch's running loss and accuracy and
    // the report window's weight, one scalar each per batch (log_interval 1);
    // no batch-sized read. The validation / test passes read their scalars too.
    uint64_t per_batch = 0;
    for (const auto& [group, reads] : copies.by_group)
        if (group.find("TrainingExecutor::ReadAccumulated") != std::string::npos ||
            group.find("TrainingExecutor::ReportWindowWeight") != std::string::npos)
            per_batch += reads;
    Check(batches > 0 && per_batch <= 3 * batches,
          "copy budget: at most 3 scalar reads per training batch (" + std::to_string(per_batch) + " over " +
              std::to_string(batches) + " batches)");
    // Uploads: a batch sends at most its features (or, from a set kept on the
    // device, its row positions) and its labels; a whole set goes up at most
    // once per batcher (train, validation, test).
    std::cout << "    copies: " << copies.uploads << " host-to-device uploads\n";
    for (const auto& [group, uploads] : copies.uploads_by_group) std::cout << "      " << uploads << "  " << group << "\n";
    const auto uploads_for = [&](const char* reason) {
        const auto it = copies.uploads_by_reason.find(reason);
        return it == copies.uploads_by_reason.end() ? uint64_t{0} : it->second;
    };
    const auto rows_in = [&](int part) {
        return static_cast<size_t>(std::count(expected_partition.begin(), expected_partition.end(), part));
    };
    const size_t evaluation_batches =
        kEpochs * ((rows_in(1) + kBatch - 1) / kBatch) + (rows_in(2) + kBatch - 1) / kBatch;
    const uint64_t per_batch_uploads =
        uploads_for(cyxwiz::BatchUploadReason::Batch) + uploads_for(cyxwiz::BatchUploadReason::Index);
    Check(copies.uploads > 0, "copy budget: the batchers' uploads are counted");
    Check(per_batch_uploads <= 2 * (batches + evaluation_batches),
          "copy budget: at most 2 uploads per batch (" + std::to_string(per_batch_uploads) + " for " +
              std::to_string(batches) + " training and at most " + std::to_string(evaluation_batches) +
              " validation / test batches)");
    Check(uploads_for(cyxwiz::BatchUploadReason::Resident) <= 3,
          "copy budget: a set kept on the device goes up once per batcher (" +
              std::to_string(uploads_for(cyxwiz::BatchUploadReason::Resident)) + ")");
    // Every read names its reason (ORA-30: the epoch report's weight
    // histograms were read without one).
    Check(copies.unknown == 0, "copy budget: every device-to-host read is named (" + std::to_string(copies.unknown) +
                                   " are not)");
    fs::remove_all(work);
}

// Scenario B: the same CSV with no preparation step, a Data Split that
// neither shuffles nor stratifies: the batchers split the rows themselves.
// Their rows, part by part, against the Python port of the split rule
// unshuffled (contiguous slices with the rule's largest-remainder counts).
void RunUnshuffledSplit(const json& fixture) {
    const fs::path csv = ModelDir() / fixture.at("csv").get<std::string>();
    json j;
    j["nodes"] = json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Rows"},
          {"parameters", {{"source_type", "file"}, {"file_path", csv.string()}, {"type", "csv"}, {"has_header", "true"},
                          {"delimiter", ","}, {"label_column", "segment"}, {"force_disk_backed", "false"}}}}});
    j["links"] = json::array();
    cyxwiz::PipelineExecutor pipeline;
    const bool ran = pipeline.ExecutePipeline(j.dump());
    Check(ran, "unshuffled: the pipeline reads the CSV: " + pipeline.GetLastError());
    if (!ran) return;
    const auto rows = cyxwiz::DataRegistry::Instance().GetArrowDataset(pipeline.NodeResults().at(1));
    Check(rows && rows->GetArrowTable() &&
              rows->GetArrowTable()->schema()->GetFieldIndex(cyxwiz::kSplitPartitionColumn) < 0,
          "unshuffled: the rows are not marked (no preparation step)");
    if (!rows) return;

    const auto graph = Model();
    const auto named = cyxwiz::test::NamedAfterModules(graph, cyxwiz::test::UnstampedPyTorchExport(graph));
    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(named.nodes, named.links, true);
    config.has_data_split = true;
    config.shuffle = false;
    config.stratified = false;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.split_seed = kSplitSeed;
    config.num_workers = 0;
    auto set = cyxwiz::BuildArrowTrainingBatchers(config, rows, "segment", kBatch);
    const auto expected = fixture.at("unshuffled_partition").get<std::vector<int>>();
    const char* names[3] = {"Train", "Validation", "Test"};
    cyxwiz::IBatcher* batchers[3] = {set.train, set.val, set.test};
    for (int part = 0; part < 3; ++part) {
        std::vector<int64_t> want;
        for (size_t r = 0; r < expected.size(); ++r)
            if (expected[r] == part) want.push_back(static_cast<int64_t>(r));
        std::vector<int64_t> got;
        if (batchers[part]) {
            batchers[part]->Reset();
            while (!batchers[part]->IsEpochComplete()) {
                const cyxwiz::Batch batch = batchers[part]->GetNextBatch();
                if (!batch.IsValid()) break;
                got.insert(got.end(), batch.row_ids.begin(), batch.row_ids.end());
            }
        }
        Check(got == want, std::string("unshuffled: the ") + names[part] + " rows are the rule's (" +
                               std::to_string(got.size()) + " rows, the rule " + std::to_string(want.size()) + ")");
    }
}

// ------------------------------------------------- batch-level, other kinds --
// A kind's batchers read as a training run reads them (TOFIX228 1.6, P4):
// before each epoch the epoch's seed (TrainingEpochSeed(dataloader seed,
// epoch)), then Reset, then every batch; Validation and Test once.

struct PartRead {
    std::vector<std::vector<int64_t>> epochs;  // Train: each epoch's rows in order
    std::map<int64_t, std::vector<double>> values;  // row -> its values as served
    std::map<int64_t, double> labels;               // row -> its class id / target
    std::vector<int64_t> rows;                      // Validation / Test: rows in order
};

void ReadInto(cyxwiz::IBatcher& batcher, PartRead& out, std::vector<int64_t>& rows) {
    while (!batcher.IsEpochComplete()) {
        const cyxwiz::Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;
        // A sparse batch serves CSR rows (the first layer reads them as
        // they are); each is compared as its dense row.
        std::vector<double> data;
        if (batch.HasSparseFeatures()) {
            const auto& csr = *batch.sparse_features;
            data.assign(csr.rows * csr.columns, 0.0);
            for (size_t r = 0; r < csr.rows; ++r)
                for (int32_t k = csr.row_offsets[r]; k < csr.row_offsets[r + 1]; ++k)
                    data[r * csr.columns + static_cast<size_t>(csr.column_indices[static_cast<size_t>(k)])] =
                        csr.values[static_cast<size_t>(k)];
        } else {
            data = Values(batch.data);
        }
        const auto labels = Values(batch.labels);
        const size_t width = batch.size ? data.size() / batch.size : 0;
        const size_t per_label = batch.size ? labels.size() / batch.size : 0;
        for (size_t i = 0; i < batch.size && i < batch.row_ids.size(); ++i) {
            const int64_t row = batch.row_ids[i];
            rows.push_back(row);
            out.values[row].assign(data.begin() + static_cast<std::ptrdiff_t>(i * width),
                                   data.begin() + static_cast<std::ptrdiff_t>((i + 1) * width));
            if (per_label <= 1) {
                out.labels[row] = labels.at(i);
            } else {  // one-hot: its largest entry
                const auto first = labels.begin() + static_cast<std::ptrdiff_t>(i * per_label);
                out.labels[row] = static_cast<double>(
                    std::max_element(first, first + static_cast<std::ptrdiff_t>(per_label)) - first);
            }
        }
    }
}

PartRead ReadTrain(cyxwiz::IBatcher& batcher, int epochs, int loader_seed, bool& seeded) {
    PartRead out;
    seeded = true;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        seeded = batcher.BeginEpoch(cyxwiz::TrainingEpochSeed(static_cast<uint64_t>(loader_seed), epoch)) && seeded;
        out.epochs.emplace_back();
        ReadInto(batcher, out, out.epochs.back());
    }
    return out;
}

PartRead ReadOnce(cyxwiz::IBatcher* batcher) {
    PartRead out;
    if (!batcher) return out;
    batcher->Reset();
    ReadInto(*batcher, out, out.rows);
    return out;
}

// The kind's fixture: "epochs" (Train rows per epoch), "val_rows",
// "test_rows", "values" (row -> its values) and "labels" (row -> its label).
void CheckParts(const std::string& kind, const json& f, const PartRead& train, const PartRead& val,
                const PartRead& test, bool seeded) {
    Check(seeded, kind + ": the Train batcher takes the epoch's seed (BeginEpoch), so a resumed run "
                         "replays its order");
    const auto& epochs = f.at("epochs");
    for (size_t e = 0; e < epochs.size(); ++e) {
        const auto want = epochs.at(e).get<std::vector<int64_t>>();
        const auto got = e < train.epochs.size() ? train.epochs[e] : std::vector<int64_t>();
        std::cout << "    " << kind << " epoch " << (e + 1) << ": first rows";
        for (size_t i = 0; i < std::min<size_t>(6, got.size()); ++i) std::cout << " " << got[i];
        std::cout << " (Python:";
        for (size_t i = 0; i < std::min<size_t>(6, want.size()); ++i) std::cout << " " << want[i];
        std::cout << ")\n";
        Check(got == want, kind + ": epoch " + std::to_string(e + 1) + " serves the Train rows in the Python port's "
                                                                       "order (" + std::to_string(got.size()) +
                               " rows, Python " + std::to_string(want.size()) + ")");
    }
    Check(val.rows == f.at("val_rows").get<std::vector<int64_t>>(),
          kind + ": the Validation rows (" + std::to_string(val.rows.size()) + ")");
    Check(test.rows == f.at("test_rows").get<std::vector<int64_t>>(),
          kind + ": the Test rows (" + std::to_string(test.rows.size()) + ")");
    const json& tol = f.at("tolerance");
    Difference worst;
    size_t label_mismatch = 0, missing = 0;
    for (const PartRead* part : {&train, &val, &test}) {
        for (const auto& [row, values] : part->values) {
            const std::string key = std::to_string(row);
            if (!f.at("values").contains(key)) {
                ++missing;
                continue;
            }
            std::vector<float> got(values.size());
            std::transform(values.begin(), values.end(), got.begin(), [](double v) { return static_cast<float>(v); });
            Compare(got, f.at("values").at(key).get<std::vector<float>>(), tol.at("atol").get<double>(),
                    tol.at("rtol").get<double>(), "row " + key, worst);
            if (std::abs(part->labels.at(row) - f.at("labels").at(key).get<double>()) > 1e-6) ++label_mismatch;
        }
    }
    Check(missing == 0, kind + ": every served row is one the Python side has (" + std::to_string(missing) +
                            " are not)");
    Check(worst.excess <= 1.0, kind + ": every row's values as Python reads them (worst " +
                                   std::to_string(worst.excess) + " tolerances at " + worst.where + ")");
    Check(label_mismatch == 0, kind + ": every row's label (" + std::to_string(label_mismatch) + " differ)");
}

// Training on a kind's batchers: its model, started from the fixture's
// state-dict, trained by the TrainingExecutor for two epochs on the batchers
// `make` builds for the model's configuration; every step's running loss and
// each epoch's validation loss and weights against PyTorch running the
// model's export on the same rows in the same order (generate_data_parity_kinds.py).
void TrainKind(const std::string& kind, const json& f, const cyxwiz::test::TrainGraph& graph,
               const std::string& model_name,
               const std::function<void(cyxwiz::TrainingConfiguration&)>& data_settings,
               const std::function<cyxwiz::ResolvedExternalBatchers(const cyxwiz::TrainingConfiguration&)>& make) {
    const json& run = f.at("training");
    const auto exported = cyxwiz::test::UnstampedPyTorchExport(graph);
    const std::string committed = cyxwiz::test::ReadText(ModelDir() / (model_name + ".py"));
    Check(exported.code == committed, kind + ": the model's PyTorch export is the committed one");
    Check(run.at("export_sha256") == cyxwiz::test::Sha256Text(committed), kind + ": the fixture was made from it");
    const auto named = cyxwiz::test::NamedAfterModules(graph, exported);
    cyxwiz::GraphCompiler compiler;
    cyxwiz::TrainingConfiguration config = compiler.Compile(named.nodes, named.links, true);
    for (const auto& issue : config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cout << "    compile: " << issue.message << "\n";
    Check(config.is_valid, kind + ": the model compiles");
    if (!config.is_valid) return;
    const fs::path work = fs::temp_directory_path() / ("cyxwiz_oracle_data_" + kind);
    fs::remove_all(work);
    fs::create_directories(work);
    data_settings(config);
    config.dataloader_seed = kLoaderSeed;
    config.num_workers = 0;
    config.batch_size = f.at("batch_size").get<int>();
    config.epochs = kEpochs;
    config.validation_freq = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 1;
    config.checkpoint_dir = (work / "checkpoints").string();
    config.initialize_from_weights_file = (ModelDir() / run.at("init_weights").get<std::string>()).string();
    const RunCheck check = TrainAgainst(config, make(config), run.at("steps"), run.at("epochs"),
                                        run.at("tolerance").at("atol").get<double>(),
                                        run.at("tolerance").at("rtol").get<double>());
    Check(check.status == "completed", kind + ": the run completes (" + check.status + ": " + check.reason + ")");
    Check(check.steps == run.at("steps").size(), kind + ": " + std::to_string(check.steps) + " steps (PyTorch " +
                                                      std::to_string(run.at("steps").size()) + ")");
    Check(check.layout_problem.empty(), kind + ": the model takes the fixture's weights: " + check.layout_problem);
    Check(check.loss.excess <= 1.0, kind + ": every step's running loss is PyTorch's (worst " +
                                        std::to_string(check.loss.excess) + " tolerances at " + check.loss.where + ")");
    Check(check.epoch.excess <= 1.0, kind + ": every epoch's validation loss and weights are PyTorch's (worst " +
                                         std::to_string(check.epoch.excess) + " tolerances at " + check.epoch.where + ")");
    fs::remove_all(work);
}

// The split a kind's Data Split asks for (ratios, stratified, shuffled).
void ShuffledSplit(cyxwiz::TrainingConfiguration& config, float train, float val, float test) {
    config.has_data_split = true;
    config.shuffle = true;
    config.stratified = true;
    config.train_ratio = train;
    config.val_ratio = val;
    config.test_ratio = test;
}

json LoadKind(const std::string& name) {
    std::ifstream in(FixtureRoot() / ("data_parity_" + name + ".json"));
    if (!in) throw std::runtime_error("data_parity_" + name + ".json is not readable");
    return json::parse(in);
}

// Text: Data Input -> TextTokenizer (Word tokens, vocabulary fitted on the
// Train rows) with the training split -> the Arrow batchers.
void RunText() {
    const json f = LoadKind("text");
    std::cout << "text: " << f.at("summary").get<std::string>() << "\n";
    json j;
    j["nodes"] = json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Reviews"},
          {"parameters", {{"source_type", "file"}, {"file_path", (ModelDir() / "text.csv").string()}, {"type", "csv"},
                          {"has_header", "true"}, {"delimiter", ","}, {"label_column", "label"},
                          {"force_disk_backed", "false"}}}},
         {{"id", 2}, {"type", "TextTokenizer"}, {"name", "Tokens"},
          {"parameters", f.at("tokenizer")}}});
    j["links"] = json::array({{{"start_node", 1}, {"end_node", 2}}});
    j["training_split"] = f.at("training_split");
    cyxwiz::PipelineExecutor pipeline;
    const bool ran = pipeline.ExecutePipeline(j.dump());
    Check(ran, "text: the pipeline tokenizes the CSV: " + pipeline.GetLastError());
    if (!ran) return;
    const auto tokens = cyxwiz::DataRegistry::Instance().GetArrowDataset(pipeline.NodeResults().at(2));
    Check(tokens && tokens->GetArrowTable(), "text: the token rows are in memory");
    if (!tokens) return;
    cyxwiz::TrainingConfiguration config;
    config.has_data_split = true;
    config.shuffle = true;
    config.stratified = true;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.dataloader_seed = kLoaderSeed;
    config.num_workers = 0;
    config.output_size = f.at("classes").get<size_t>();
    auto set = cyxwiz::BuildArrowTrainingBatchers(config, tokens, "y", f.at("batch_size").get<int>());
    bool seeded = false;
    const PartRead train = ReadTrain(*set.train, 2, kLoaderSeed, seeded);
    CheckParts("text", f, train, ReadOnce(set.val), ReadOnce(set.test), seeded);
    TrainKind("text", f, TextModel(), "text_model",
              [](cyxwiz::TrainingConfiguration& c) { ShuffledSplit(c, 0.7f, 0.15f, 0.15f); },
              [&](const cyxwiz::TrainingConfiguration& c) {
                  return cyxwiz::TakeResolvedExternalBatchers(
                      cyxwiz::BuildArrowTrainingBatchers(c, tokens, "y", c.batch_size));
              });
}

// Time series: Data Input -> TimeSeriesWindow -> TimeSeriesSplit
// (chronological) -> the Arrow batchers.
void RunSeries() {
    const json f = LoadKind("series");
    std::cout << "series: " << f.at("summary").get<std::string>() << "\n";
    json j;
    j["nodes"] = json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Readings"},
          {"parameters", {{"source_type", "file"}, {"file_path", (ModelDir() / "series.csv").string()}, {"type", "csv"},
                          {"has_header", "true"}, {"delimiter", ","}, {"force_disk_backed", "false"}}}},
         {{"id", 2}, {"type", "TimeSeriesWindow"}, {"name", "Windows"}, {"parameters", f.at("window")}},
         {{"id", 3}, {"type", "TimeSeriesSplit"}, {"name", "Split"}, {"parameters", f.at("split")}}});
    j["links"] = json::array({{{"start_node", 1}, {"end_node", 2}}, {{"start_node", 2}, {"end_node", 3}}});
    cyxwiz::PipelineExecutor pipeline;
    const bool ran = pipeline.ExecutePipeline(j.dump());
    Check(ran, "series: the pipeline windows and splits the CSV: " + pipeline.GetLastError());
    if (!ran) return;
    const auto windows = cyxwiz::DataRegistry::Instance().GetArrowDataset(pipeline.NodeResults().at(3));
    Check(windows && windows->GetArrowTable(), "series: the windows are in memory");
    if (!windows) return;
    std::cout << "    series columns:";
    for (const auto& field : windows->GetArrowTable()->schema()->fields()) std::cout << " " << field->name();
    std::cout << "\n";
    cyxwiz::TrainingConfiguration config;
    config.is_time_series = true;
    config.loss_type = gui::NodeType::MSELoss;  // a forecast: continuous targets
    config.has_data_split = true;
    config.shuffle = true;
    config.dataloader_seed = kLoaderSeed;
    config.num_workers = 0;
    auto set = cyxwiz::BuildArrowTrainingBatchers(config, windows, "y", f.at("batch_size").get<int>());
    bool seeded = false;
    const PartRead train = ReadTrain(*set.train, 2, kLoaderSeed, seeded);
    CheckParts("series", f, train, ReadOnce(set.val), ReadOnce(set.test), seeded);
    TrainKind("series", f, SeriesModel(), "series_model",
              [](cyxwiz::TrainingConfiguration& c) {
                  c.is_time_series = true;
                  c.has_data_split = true;
                  c.shuffle = true;
              },
              [&](const cyxwiz::TrainingConfiguration& c) {
                  return cyxwiz::TakeResolvedExternalBatchers(
                      cyxwiz::BuildArrowTrainingBatchers(c, windows, "y", c.batch_size));
              });
}

// One batcher serves an image / audio set's three parts (SetPhase), as the
// executor uses it.
void ReadPhases(const std::string& kind, const json& f, cyxwiz::IBatcher& batcher) {
    bool seeded = false;
    batcher.SetPhase(cyxwiz::BatcherPhase::Train);
    const PartRead train = ReadTrain(batcher, 2, kLoaderSeed, seeded);
    batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    const PartRead val = ReadOnce(&batcher);
    batcher.SetPhase(cyxwiz::BatcherPhase::Test);
    const PartRead test = ReadOnce(&batcher);
    CheckParts(kind, f, train, val, test, seeded);
}

cyxwiz::SampleSplitSettings SplitOf(const json& f) {
    const json& s = f.at("split");
    return {s.at("train").get<float>(), s.at("val").get<float>(), s.at("test").get<float>(),
            s.at("stratified").get<bool>(), s.at("seed").get<uint32_t>()};
}

// Images: a class-per-folder set (PNG: RGB, grey, RGBA, one 8 x 8 to shrink)
// prepared at 4 x 4 and served by ImageDatasetBatcher.
void RunImage() {
    const json f = LoadKind("image");
    std::cout << "image: " << f.at("summary").get<std::string>() << "\n";
    cyxwiz::DataRegistry::ImageDatasetEntry entry;
    entry.folder_path = (ModelDir() / "images").string();
    entry.layout = 0;
    entry.num_images = f.at("rows").get<int>();
    entry.num_classes = f.at("classes").get<int>();
    cyxwiz::ImagePreprocessingConfig preprocessing;
    preprocessing.resize_mode = cyxwiz::ResizeMode::Exact;
    preprocessing.target_width = f.at("size").get<int>();
    preprocessing.target_height = f.at("size").get<int>();
    const auto prepared = cyxwiz::PrepareImageSet(entry, cyxwiz::PreparedImageSpecFrom(preprocessing));
    Check(prepared.set != nullptr, "image: the set is prepared");
    if (!prepared.set) return;
    cyxwiz::ImageDatasetBatcher batcher(prepared.set, f.at("batch_size").get<int>(), SplitOf(f), true, kLoaderSeed);
    ReadPhases("image", f, batcher);
    TrainKind("image", f, ImageModel(), "image_model", [](cyxwiz::TrainingConfiguration&) {},
              [&](const cyxwiz::TrainingConfiguration& c) {
                  auto shared = std::make_shared<cyxwiz::ImageDatasetBatcher>(prepared.set, c.batch_size, SplitOf(f),
                                                                              true, kLoaderSeed);
                  // As the Engine's image setup does (image_batchers.cpp): one-hot class labels.
                  shared->SetOneHotEncoding(f.at("classes").get<size_t>());
                  return cyxwiz::ResolvedExternalBatchers{shared, shared, shared};
              });
}

// Audio: a class-per-folder set of 16-bit mono WAVs, the magnitude
// spectrogram (n_fft 64, hop 32), served by AudioDatasetBatcher.
void RunAudio() {
    const json f = LoadKind("audio");
    std::cout << "audio: " << f.at("summary").get<std::string>() << "\n";
    cyxwiz::DataRegistry::AudioDatasetEntry entry;
    entry.folder_path = (ModelDir() / "audio").string();
    entry.labeled_subdirs = true;
    entry.target_sr = f.at("sample_rate").get<int>();
    entry.max_duration = f.at("seconds").get<float>();
    entry.num_samples = f.at("rows").get<int>();
    entry.num_classes = f.at("classes").get<int>();
    cyxwiz::AudioPreprocessingConfig preprocessing;
    preprocessing.has_feature_node = true;
    preprocessing.feature_type = cyxwiz::AudioFeatureType::Spectrogram;
    preprocessing.n_fft = f.at("n_fft").get<int>();
    preprocessing.hop_length = f.at("hop").get<int>();
    cyxwiz::AudioDatasetBatcher batcher(entry, preprocessing, f.at("batch_size").get<int>(), SplitOf(f), true, 0,
                                        kLoaderSeed);
    Check(batcher.GetNumSamples() > 0, "audio: the clips load");
    ReadPhases("audio", f, batcher);
    TrainKind("audio", f, AudioModel(), "audio_model", [](cyxwiz::TrainingConfiguration&) {},
              [&](const cyxwiz::TrainingConfiguration& c) {
                  auto shared = std::make_shared<cyxwiz::AudioDatasetBatcher>(entry, preprocessing, c.batch_size,
                                                                              SplitOf(f), true, 0, kLoaderSeed);
                  // As the Engine's audio setup does (audio_dataset_batcher.cpp): one-hot class labels.
                  shared->SetOneHotEncoding(f.at("classes").get<size_t>());
                  return cyxwiz::ResolvedExternalBatchers{shared, shared, shared};
              });
}

// Sparse: the text CSV through TFIDFVectorizer (sparse output, vocabulary
// and idf from the Train rows) with the training split ->
// BuildSparseTrainingBatchers.
void RunSparse() {
    const json f = LoadKind("sparse");
    std::cout << "sparse: " << f.at("summary").get<std::string>() << "\n";
    json j;
    j["nodes"] = json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Reviews"},
          {"parameters", {{"source_type", "file"}, {"file_path", (ModelDir() / "text.csv").string()}, {"type", "csv"},
                          {"has_header", "true"}, {"delimiter", ","}, {"label_column", "label"},
                          {"force_disk_backed", "false"}}}},
         {{"id", 2}, {"type", "TFIDFVectorizer"}, {"name", "Terms"}, {"parameters", f.at("vectorizer")}}});
    j["links"] = json::array({{{"start_node", 1}, {"end_node", 2}}});
    j["training_split"] = f.at("training_split");
    cyxwiz::PipelineExecutor pipeline;
    const bool ran = pipeline.ExecutePipeline(j.dump());
    Check(ran, "sparse: the pipeline vectorizes the CSV: " + pipeline.GetLastError());
    if (!ran) return;
    const auto features = cyxwiz::DataRegistry::Instance().GetSparseFeatureDataset(pipeline.NodeResults().at(2));
    Check(features != nullptr, "sparse: the sparse rows are registered");
    if (!features) return;
    cyxwiz::TrainingConfiguration config;
    config.has_data_split = true;
    config.shuffle = true;
    config.stratified = true;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.15f;
    config.test_ratio = 0.15f;
    config.dataloader_seed = kLoaderSeed;
    config.num_workers = 0;
    config.output_size = f.at("classes").get<size_t>();
    auto set = cyxwiz::BuildSparseTrainingBatchers(config, features, f.at("batch_size").get<int>());
    bool seeded = false;
    const PartRead train = ReadTrain(*set.train, 2, kLoaderSeed, seeded);
    CheckParts("sparse", f, train, ReadOnce(set.val), ReadOnce(set.test), seeded);

    // Training on those batches: a linear model on the CSR rows against
    // PyTorch running its export on the dense rows in the same order.
    TrainKind("sparse", f, SparseModel(), "sparse_linear",
              [](cyxwiz::TrainingConfiguration& c) { ShuffledSplit(c, 0.7f, 0.15f, 0.15f); },
              [&](const cyxwiz::TrainingConfiguration& c) {
                  return cyxwiz::TakeResolvedExternalBatchers(
                      cyxwiz::BuildSparseTrainingBatchers(c, features, c.batch_size));
              });
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--write-exports") return WriteExports(argv[2]);
    if (!cyxwiz::test::BeginTrainingOracleTest("test_oracle_data")) return 1;
    try {
        cyxwiz::GraphCompilerDatasetHooks hooks;
        hooks.is_dataset_registered = [](const std::string&) { return true; };
        cyxwiz::SetGraphCompilerDatasetHooks(hooks);
        const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_override(fs::temp_directory_path() /
                                                                          "cyxwiz_oracle_data_debug");
        std::ifstream in(FixtureRoot() / "data_parity_tabular_pytorch.json");
        Check(in.good(), "data_parity_tabular_pytorch.json is readable");
        if (!in.good()) return 1;
        const json fixture = json::parse(in);
        std::cout << "fixture: torch " << fixture.at("torch_version").get<std::string>() << ", numpy "
                  << fixture.at("numpy_version").get<std::string>() << "\n";
        Run(fixture);
        RunUnshuffledSplit(fixture);
        for (const auto& kind : {RunText, RunSeries, RunImage, RunAudio, RunSparse}) {
            try {
                kind();
            } catch (const std::exception& e) {
                Check(false, std::string("exception: ") + e.what());
            }
        }
        const std::string device_problem = cyxwiz::test::TrainingDeviceProblem();
        Check(device_problem.empty(), "every run on the requested device: " + device_problem);
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
