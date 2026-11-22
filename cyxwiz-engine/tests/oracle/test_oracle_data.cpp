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
#include "../../src/core/data_registry.h"
#include "../../src/core/debug_run_paths.h"
#include "../../src/core/graph_compiler.h"
#include "../../src/core/graph_compiler_dataset_hooks.h"
#include "../../src/core/pipeline_executor.h"
#include "../../src/core/split_partitioning.h"
#include "../../src/core/training_batcher_setup.h"
#include "../../src/core/training_executor.h"

#include <arrow/api.h>

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
        epochs_.back().seed = pending_seed_;
        pending_seed_.reset();
    }
    bool SetEpochShuffleSeed(uint64_t seed) override {
        pending_seed_ = seed;
        return inner_->SetEpochShuffleSeed(seed);
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
    std::optional<uint64_t> pending_seed_;
};

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

    const json& steps = fixture.at("steps");
    const json& epochs = fixture.at("epochs");
    const double atol = tol.at("atol").get<double>();
    const double rtol = tol.at("rtol").get<double>();
    cyxwiz::test::PinTrainingDevice();
    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    size_t step = 0;
    int epoch_index = 0;
    Difference loss_worst, epoch_worst;
    std::string layout_problem;
    executor.Train(
        kEpochs, kBatch,
        [&](int, int, int, float loss, float) {
            if (step < steps.size())
                Compare({loss}, {steps.at(step).at("running_loss").get<float>()}, atol, rtol,
                        "step " + std::to_string(step + 1) + " running loss", loss_worst);
            ++step;
        },
        [&](int, float, float, float val_loss, float, float) {
            if (epoch_index >= static_cast<int>(epochs.size())) return;
            const json& expected = epochs.at(epoch_index);
            const std::string at = "epoch " + std::to_string(epoch_index + 1);
            Compare({val_loss}, {expected.at("val_loss").get<float>()}, atol, rtol, at + " validation loss",
                    epoch_worst);
            if (cyxwiz::IExecutableModel* built = executor.GetExecutableModel()) {
                std::string problem;
                const auto want = cyxwiz::test::ExpectedInEngineLayout(expected.at("parameters"), *built,
                                                                       executor.GetModuleProvenance(), problem);
                if (!problem.empty() && layout_problem.empty()) layout_problem = problem;
                const auto have = built->GetParameters();
                for (const auto& [key, values] : want) {
                    const auto it = have.find(key);
                    if (it == have.end()) continue;
                    const float* data = it->second.ReadData<float>();
                    Compare(std::vector<float>(data, data + it->second.NumElements()), values, atol, rtol,
                            at + " " + key, epoch_worst);
                }
            }
            ++epoch_index;
        });
    const auto metrics = executor.GetMetrics();
    Check(metrics.terminal_status == "completed",
          "the run completes (" + metrics.terminal_status + ": " + metrics.terminal_reason + ")");

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
        const std::string device_problem = cyxwiz::test::TrainingDeviceProblem();
        Check(device_problem.empty(), "every run on the requested device: " + device_problem);
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
