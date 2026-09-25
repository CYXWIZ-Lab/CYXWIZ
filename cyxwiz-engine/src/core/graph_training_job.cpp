#include "graph_training_job.h"

#include "arrow_dataset.h"
#include "graph_compiler.h"
#include "graph_compiler_dataset_hooks.h"
#include "graph_training_prep.h"
#include "node_metadata_registry.h"
#include "pipeline_runtime_capabilities.h"
#include "sequence_arrow_batcher.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cctype>
#include <filesystem>
#include <mutex>
#include <set>

namespace cyxwiz {

namespace {

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string Parameter(const gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

gui::NodeCategory CategoryOf(const gui::MLNode& node) {
    if (node.category != gui::NodeCategory::Unknown) return node.category;
    auto& registry = NodeMetadataRegistry::Instance();
    if (!registry.IsInitialized()) registry.Initialize();
    const NodeMetadata* metadata = registry.GetMetadata(node.type);
    return metadata ? metadata->category : gui::NodeCategory::Unknown;
}

// Data-path steps a headless run cannot perform yet: the Engine applies them
// through its preparation pipeline (every operator-backed node) before
// training. Inspection nodes (SampleRows, DescribeStats...) do not change the
// training data and pass.
bool NeedsPreparation(const gui::MLNode& node) {
    if (node.type == gui::NodeType::Augmentation) return true;
    if (ResolvePipelineRuntimeSupport(node.type).mode == PipelineRuntimeSupportMode::OperatorBacked) return true;
    switch (CategoryOf(node)) {
    case gui::NodeCategory::DataTransform:
    case gui::NodeCategory::Preprocessing:
    case gui::NodeCategory::TextProcessing:
    case gui::NodeCategory::TimeSeries:
    case gui::NodeCategory::Audio:
    case gui::NodeCategory::Database:
    case gui::NodeCategory::CloudStorage:
    case gui::NodeCategory::JsonXml:
    case gui::NodeCategory::BigData:
        return true;
    default:
        return false;
    }
}

GraphTrainingJobResult Fail(std::string error) {
    GraphTrainingJobResult result;
    result.error = std::move(error);
    spdlog::error("Graph training job: {}", result.error);
    return result;
}

// Installs the job's datasets as the catalog for compile and preparation, and
// restores the host's (one job at a time holds it).
std::mutex& CatalogMutex() {
    static std::mutex mutex;
    return mutex;
}

class ScopedJobCatalog {
public:
    ScopedJobCatalog(const std::map<std::string, std::shared_ptr<ArrowDataset>>& datasets,
                     const std::map<std::string, std::string>& source_paths)
        : lock_(CatalogMutex()), previous_(GetGraphDatasetCatalog()) {
        GraphDatasetCatalog catalog;
        catalog.arrow_dataset = [&datasets](const std::string& name) -> std::shared_ptr<ArrowDataset> {
            const auto it = datasets.find(name);
            return it == datasets.end() ? nullptr : it->second;
        };
        catalog.source_path = [&source_paths](const std::string& name) -> std::optional<std::string> {
            const auto it = source_paths.find(name);
            if (it == source_paths.end()) return std::nullopt;
            return it->second;
        };
        SetGraphDatasetCatalog(std::move(catalog));
    }
    ~ScopedJobCatalog() { SetGraphDatasetCatalog(previous_); }
    ScopedJobCatalog(const ScopedJobCatalog&) = delete;
    ScopedJobCatalog& operator=(const ScopedJobCatalog&) = delete;

private:
    std::lock_guard<std::mutex> lock_;
    GraphDatasetCatalog previous_;
};

}  // namespace

GraphTrainingJobResult RunGraphTrainingJob(const GraphTrainingJobRequest& request,
                                           const GraphTrainingJobCallbacks& callbacks) {
    GraphDocument graph;
    std::string error;
    if (!ParseGraphDocument(request.graph_json, graph, error)) return Fail(error);

    // Refuse data-path steps this runner cannot perform.
    for (const auto& node : graph.nodes) {
        if (NeedsPreparation(node)) {
            return Fail("node '" + node.name + "' is a data preparation step; this host trains graphs whose "
                        "Data Inputs feed the Data Loader directly (prepare the data in the Engine first)");
        }
    }

    // Load every Data Input file.
    std::map<std::string, std::shared_ptr<ArrowDataset>> datasets;
    std::map<std::string, std::string> source_paths;
    for (const auto& node : graph.nodes) {
        if (node.type != gui::NodeType::DataInput) continue;
        const std::string name = Parameter(node, "dataset_name");
        if (name.empty()) return Fail("Data Input '" + node.name + "' has no dataset_name");
        std::string path = Parameter(node, "file_path");
        if (const auto mapped = request.dataset_files.find(name); mapped != request.dataset_files.end()) {
            path = mapped->second;
        }
        if (path.empty()) return Fail("Data Input '" + node.name + "' has no file");
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) {
            return Fail("Data Input '" + node.name + "': file not found: " + path);
        }
        const std::string extension = Lower(std::filesystem::path(path).extension().string());
        std::shared_ptr<ArrowDataset> dataset;
        if (extension == ".parquet") {
            dataset = ArrowDataset::FromParquet(path, name);
        } else if (extension == ".arrow" || extension == ".feather" || extension == ".ipc") {
            dataset = ArrowDataset::FromFile(path, name);
        } else {
            return Fail("Data Input '" + node.name + "': " + extension +
                        " files are not supported by this host (use Parquet or Arrow IPC)");
        }
        if (!dataset) return Fail("Data Input '" + node.name + "': could not read " + path);
        datasets[name] = dataset;
        source_paths[name] = path;
        spdlog::info("Graph training job: loaded '{}' ({} rows) from {}", name, dataset->GetNumRows(), path);
    }
    if (datasets.empty()) return Fail("the graph has no Data Input with a file");

    // Compile and prepare against the job's datasets (the Engine launcher's
    // steps, for inputs that are already materialized).
    TrainingConfiguration config;
    {
        ScopedJobCatalog catalog(datasets, source_paths);
        GraphCompiler compiler;
        config = compiler.Compile(graph.nodes, graph.links, true);
        if (!config.is_valid) {
            return Fail("the graph does not compile: " +
                        (config.error_message.empty() ? std::string("see the compiler issues")
                                                      : config.error_message));
        }

        const std::string dataset_name =
            !config.dataset_name.empty() ? config.dataset_name : FindGraphDatasetName(graph.nodes);
        if (dataset_name.empty()) return Fail("no Data Input feeds the Data Loader");
        if (datasets.find(dataset_name) == datasets.end()) {
            return Fail("the training dataset '" + dataset_name + "' is not a loaded input");
        }
        std::string label_column = FindGraphLabelColumn(graph.nodes, dataset_name, config.data_source_node_id);
        label_column = ResolveRuntimeArrowLabelColumn(dataset_name, label_column);
        ReconcileRuntimeDatasetTarget(config, label_column, dataset_name);
        ReconcileRuntimeTabularFeatureWidth(dataset_name, label_column, config);

        auto roles = config.dataset_roles;
        roles.train.dataset_name = dataset_name;
        roles.train.label_column = label_column;
        for (auto* supplied : {&roles.dev, &roles.test}) {
            if (!supplied->IsSupplied()) continue;
            if (datasets.find(supplied->dataset_name) == datasets.end()) {
                return Fail("the supplied dataset '" + supplied->dataset_name + "' is not a loaded input");
            }
            supplied->label_column =
                FindGraphLabelColumn(graph.nodes, supplied->dataset_name, supplied->source_node_id);
        }
        LaunchBlock block;
        if (!ValidateSuppliedRolePreflight(roles, config, block)) {
            return Fail(block.error_message.empty() ? block.title + ": " + block.detail : block.error_message);
        }
        config.dataset_roles = roles;
        if (config.sequence_batch.enabled) {
            std::string column_error;
            if (!ValidateSequenceLaunchColumns(dataset_name, config, column_error)) return Fail(column_error);
        }
        config.dataset_name = dataset_name;
    }
    if (request.epochs_override > 0) config.epochs = request.epochs_override;
    if (request.batch_size_override > 0) config.batch_size = request.batch_size_override;
    if (!request.checkpoint_dir_override.empty()) config.checkpoint_dir = request.checkpoint_dir_override;
    const int epochs = std::max(1, config.epochs);
    const int batch_size = std::max(1, config.batch_size);

    const auto role = [&datasets](const DatasetSourceRef& source) -> std::shared_ptr<ArrowDataset> {
        if (!source.IsSupplied()) return nullptr;
        const auto it = datasets.find(source.dataset_name);
        return it == datasets.end() ? nullptr : it->second;
    };
    const auto train_it = datasets.find(config.dataset_name);

    std::unique_ptr<TrainingExecutor> executor;
    if (config.sequence_batch.enabled) {
        auto dev = role(config.dataset_roles.dev);
        auto test = role(config.dataset_roles.test);
        auto sequence = BuildSequenceBatcherFromArrowDataset(train_it->second, config, batch_size, dev, test);
        if (!sequence.success()) return Fail("sequence batcher: " + sequence.error_message);
        ApplySequenceBatcherBuildResultToTrainingConfig(sequence, config);
        executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(sequence.batcher),
                                                      std::move(sequence.id_to_label));
    } else {
        const std::string label = config.dataset_roles.train.label_column;
        executor = std::make_unique<TrainingExecutor>(std::move(config), train_it->second, label);
    }

    // Cancellation: a watcher asks the executor to stop cooperatively.
    GraphTrainingJobResult result;
    std::atomic<bool> finished{false};
    std::thread watcher;
    if (callbacks.should_cancel) {
        watcher = std::thread([&] {
            while (!finished.load()) {
                if (callbacks.should_cancel()) {
                    result.cancelled = true;
                    executor->Stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }
    if (callbacks.on_start) callbacks.on_start(epochs, batch_size);
    try {
        executor->Train(epochs, batch_size, callbacks.on_batch, callbacks.on_epoch);
    } catch (const std::exception& e) {
        finished.store(true);
        if (watcher.joinable()) watcher.join();
        return Fail(std::string("training failed: ") + e.what());
    }
    finished.store(true);
    if (watcher.joinable()) watcher.join();

    result.metrics = executor->GetMetrics();
    if (result.metrics.terminal_status == "failed") {
        result.error = result.metrics.terminal_reason.empty() ? "training failed" : result.metrics.terminal_reason;
        return result;
    }
    result.ok = !result.cancelled;
    result.model = executor->ReleaseModel();
    return result;
}

}  // namespace cyxwiz
