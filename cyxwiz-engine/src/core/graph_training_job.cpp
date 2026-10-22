#include "graph_training_job.h"

#include "arrow_dataset.h"
#include "data_budget.h"
#include "graph_compiler.h"
#include "graph_compiler_dataset_hooks.h"
#include "graph_node_factory.h"
#include "graph_training_prep.h"
#include "node_metadata_registry.h"
#include "parquet_backed_dataset.h"
#include "pipeline_runtime_capabilities.h"
#include "sequence_arrow_batcher.h"
#include "sha256_digest.h"
#include "training_batcher_setup.h"
#include "training_resume_checkpoint.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cctype>
#include <cstdlib>
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

// Data-path steps a headless run cannot perform yet: the Engine applies them
// through its preparation pipeline (every operator-backed node) before
// training. Inspection nodes (SampleRows, DescribeStats...) do not change the
// training data and pass. Image transforms (Resize, crops, flips,
// augmentation) compile into the trainer's image batcher, which applies them
// here too.
bool NeedsPreparation(const gui::MLNode& node) {
    if (ResolvePipelineRuntimeSupport(node.type).mode == PipelineRuntimeSupportMode::OperatorBacked) return true;
    if (GraphCompiler::GetPreprocessingNodeDomain(node.type) == PreprocessingDomain::Image) return false;
    switch (gui::GetNodeCategoryForType(node.type)) {
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

GraphTrainingJobResult Fail(TrainingFailureKind kind, std::string error) {
    GraphTrainingJobResult result;
    result.failure = kind;
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
                     const std::map<std::string, std::shared_ptr<ParquetBackedDataset>>& disk_datasets,
                     const std::map<std::string, std::string>& source_paths)
        : lock_(CatalogMutex()), previous_(GetGraphDatasetCatalog()) {
        GraphDatasetCatalog catalog;
        catalog.arrow_dataset = [&datasets](const std::string& name) -> std::shared_ptr<ArrowDataset> {
            const auto it = datasets.find(name);
            return it == datasets.end() ? nullptr : it->second;
        };
        catalog.parquet_dataset = [&disk_datasets](const std::string& name) -> std::shared_ptr<ParquetBackedDataset> {
            const auto it = disk_datasets.find(name);
            return it == disk_datasets.end() ? nullptr : it->second;
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

// The first node RunGraphTrainingJob cannot run, with the reason.
bool FindPreparationNode(const GraphDocument& graph, std::string& error) {
    for (const auto& node : graph.nodes) {
        if (NeedsPreparation(node)) {
            error = "node '" + node.name + "' is a data preparation step; this host trains graphs whose "
                    "Data Inputs feed the Data Loader directly (prepare the data in the Engine first)";
            return true;
        }
    }
    return false;
}

}  // namespace

GraphTrainingJobResult PrepareGraphJob(const GraphTrainingJobRequest& request, PreparedGraphJob& prepared) {
    GraphDocument graph;
    std::string error;
    if (!ParseGraphDocument(request.graph_json, graph, error)) return Fail(TrainingFailureKind::Refused, error);

    // Refuse data-path steps this runner cannot perform.
    if (FindPreparationNode(graph, error)) return Fail(TrainingFailureKind::Refused, error);

    auto& datasets = prepared.datasets;
    auto& disk_datasets = prepared.disk_datasets;
    auto& source_paths = prepared.source_paths;
    // Load every Data Input file. A Parquet file larger than a quarter of the
    // data budget stays on disk and is read a row group at a time (TOFIX154),
    // as in the Engine.
    for (const auto& node : graph.nodes) {
        if (node.type != gui::NodeType::DataInput) continue;
        const std::string name = Parameter(node, "dataset_name");
        if (name.empty()) return Fail(TrainingFailureKind::Refused, "Data Input '" + node.name + "' has no dataset_name");
        std::string path = Parameter(node, "file_path");
        if (const auto mapped = request.dataset_files.find(name); mapped != request.dataset_files.end()) {
            path = mapped->second;
        }
        if (path.empty()) return Fail(TrainingFailureKind::DataError, "Data Input '" + node.name + "' has no file");
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) {
            return Fail(TrainingFailureKind::DataError, "Data Input '" + node.name + "': file not found: " + path);
        }
        const std::string extension = Lower(std::filesystem::path(path).extension().string());
        std::shared_ptr<ArrowDataset> dataset;
        if (extension == ".parquet") {
            auto file = ParquetBackedDataset::Open(path, name);
            if (!file) return Fail(TrainingFailureKind::DataError, "Data Input '" + node.name + "': could not read " + path);
            if (file->GetInMemoryBytesEstimate() > DataResultThresholdBytes()) {
                if (file->GetNumRows() == 0) {
                    return Fail(TrainingFailureKind::DataError,
                                "Data Input '" + node.name + "': the dataset is empty (0 rows) in " + path);
                }
                spdlog::info("Graph training job: '{}' ({} rows, about {} in memory) stays on disk, over {} "
                             "(a quarter of the data budget)", name, file->GetNumRows(),
                             FormatDataBytes(file->GetInMemoryBytesEstimate()),
                             FormatDataBytes(DataResultThresholdBytes()));
                disk_datasets[name] = std::move(file);
                source_paths[name] = path;
                continue;
            }
            file.reset();
            dataset = ArrowDataset::FromParquet(path, name);
        } else if (extension == ".arrow" || extension == ".feather" || extension == ".ipc") {
            dataset = ArrowDataset::FromFile(path, name);
        } else {
            return Fail(TrainingFailureKind::Refused, "Data Input '" + node.name + "': " + extension +
                        " files are not supported by this host (use Parquet or Arrow IPC)");
        }
        if (!dataset) return Fail(TrainingFailureKind::DataError, "Data Input '" + node.name + "': could not read " + path);
        if (dataset->GetNumRows() == 0) {
            return Fail(TrainingFailureKind::DataError,
                        "Data Input '" + node.name + "': the dataset is empty (0 rows) in " + path);
        }
        datasets[name] = dataset;
        source_paths[name] = path;
        spdlog::info("Graph training job: loaded '{}' ({} rows) from {}", name, dataset->GetNumRows(), path);
    }
    if (datasets.empty() && disk_datasets.empty()) {
        return Fail(TrainingFailureKind::Refused, "the graph has no Data Input with a file");
    }
    const auto is_input = [&](const std::string& name) {
        return datasets.count(name) > 0 || disk_datasets.count(name) > 0;
    };

    // Compile and prepare against the job's datasets (the Engine launcher's
    // steps, for inputs that are already materialized).
    auto& config = prepared.config;
    {
        ScopedJobCatalog catalog(datasets, disk_datasets, source_paths);
        GraphCompiler compiler;
        config = compiler.Compile(graph.nodes, graph.links, true);
        if (!config.is_valid) {
            return Fail(TrainingFailureKind::Refused, "the graph does not compile: " +
                        (config.error_message.empty() ? std::string("see the compiler issues")
                                                      : config.error_message));
        }

        const std::string dataset_name =
            !config.dataset_name.empty() ? config.dataset_name : FindGraphDatasetName(graph.nodes);
        if (dataset_name.empty()) return Fail(TrainingFailureKind::Refused, "no Data Input feeds the Data Loader");
        if (!is_input(dataset_name)) {
            return Fail(TrainingFailureKind::DataError, "the training dataset '" + dataset_name + "' is not a loaded input");
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
            if (!is_input(supplied->dataset_name)) {
                return Fail(TrainingFailureKind::DataError, "the supplied dataset '" + supplied->dataset_name + "' is not a loaded input");
            }
            supplied->label_column =
                FindGraphLabelColumn(graph.nodes, supplied->dataset_name, supplied->source_node_id);
        }
        LaunchBlock block;
        if (!ValidateSuppliedRolePreflight(roles, config, block)) {
            return Fail(TrainingFailureKind::DataError, block.error_message.empty() ? block.title + ": " + block.detail : block.error_message);
        }
        config.dataset_roles = roles;
        if (config.sequence_batch.enabled) {
            std::string column_error;
            if (!ValidateSequenceLaunchColumns(dataset_name, config, column_error)) return Fail(TrainingFailureKind::DataError, column_error);
        }
        config.dataset_name = dataset_name;
    }
    return {};
}

GraphTrainingJobResult RunGraphTrainingJob(const GraphTrainingJobRequest& request,
                                           const GraphTrainingJobCallbacks& callbacks) {
    const auto job_start = std::chrono::steady_clock::now();
    PreparedGraphJob prepared;
    if (auto failure = PrepareGraphJob(request, prepared); !failure.error.empty()) return failure;
    auto& datasets = prepared.datasets;
    auto& disk_datasets = prepared.disk_datasets;
    auto& source_paths = prepared.source_paths;
    auto& config = prepared.config;
    if (request.epochs_override > 0) config.epochs = request.epochs_override;
    if (request.batch_size_override > 0) config.batch_size = request.batch_size_override;
    if (!request.checkpoint_dir_override.empty()) config.checkpoint_dir = request.checkpoint_dir_override;
    const int epochs = std::max(1, config.epochs);
    const int batch_size = std::max(1, config.batch_size);
    const int tokens_per_sample = config.sequence_batch.enabled ? config.sequence_batch.max_sequence_length : 0;

    const auto role = [&datasets](const DatasetSourceRef& source) -> std::shared_ptr<ArrowDataset> {
        if (!source.IsSupplied()) return nullptr;
        const auto it = datasets.find(source.dataset_name);
        return it == datasets.end() ? nullptr : it->second;
    };
    const auto disk = [&disk_datasets](const std::string& name) -> std::shared_ptr<ParquetBackedDataset> {
        const auto it = disk_datasets.find(name);
        return it == disk_datasets.end() ? nullptr : it->second;
    };
    const auto roles = config.dataset_roles;
    const bool train_on_disk = disk(config.dataset_name) != nullptr;
    const bool role_on_disk = (roles.dev.IsSupplied() && disk(roles.dev.dataset_name)) ||
                              (roles.test.IsSupplied() && disk(roles.test.dataset_name));
    const auto train_it = datasets.find(config.dataset_name);

    std::unique_ptr<TrainingExecutor> executor;
    if (config.sequence_batch.enabled && (train_on_disk || role_on_disk)) {
        if (!train_on_disk) {
            return Fail(TrainingFailureKind::DataError,
                        "a supplied role is kept on disk, so the training set must be too: raise the data budget "
                        "or keep all of them in memory");
        }
        // Token windows on disk stream a row group at a time (ParquetSequenceBatcher).
        const auto source_of = [&](const DatasetSourceRef& source) {
            SequenceWindowSource window_source;
            if (!source.IsSupplied()) return window_source;
            if (auto memory = role(source)) window_source.table = memory->GetArrowTable();
            else window_source.file = disk(source.dataset_name);
            return window_source;
        };
        SequenceWindowSource train_source;
        train_source.file = disk(config.dataset_name);
        auto sequence = BuildSequenceBatcherFromWindowSources(train_source, config, batch_size,
                                                              source_of(roles.dev), source_of(roles.test));
        if (!sequence.success()) return Fail(TrainingFailureKind::DataError, "sequence batcher: " + sequence.error_message);
        ApplySequenceBatcherBuildResultToTrainingConfig(sequence, config);
        executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(sequence.batcher),
                                                      std::move(sequence.id_to_label));
    } else if (config.sequence_batch.enabled) {
        auto dev = role(config.dataset_roles.dev);
        auto test = role(config.dataset_roles.test);
        auto sequence = BuildSequenceBatcherFromArrowDataset(train_it->second, config, batch_size, dev, test);
        if (!sequence.success()) return Fail(TrainingFailureKind::DataError, "sequence batcher: " + sequence.error_message);
        ApplySequenceBatcherBuildResultToTrainingConfig(sequence, config);
        executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(sequence.batcher),
                                                      std::move(sequence.id_to_label));
    } else if (role_on_disk || (train_on_disk && (roles.dev.IsSupplied() || roles.test.IsSupplied()))) {
        // Supplied roles with any of them on disk: one batcher per role, each
        // from memory or from its file (the Engine's TabularLoader path).
        ResolvedTabularDatasets resolved;
        const auto pick = [&](const std::string& name, std::shared_ptr<ArrowDataset>& memory,
                              std::shared_ptr<ParquetBackedDataset>& file) {
            if (name.empty()) return;
            if (const auto it = datasets.find(name); it != datasets.end()) memory = it->second;
            file = disk(name);
        };
        pick(config.dataset_name, resolved.train_arrow, resolved.train_parquet);
        if (roles.dev.IsSupplied()) pick(roles.dev.dataset_name, resolved.dev_arrow, resolved.dev_parquet);
        if (roles.test.IsSupplied()) pick(roles.test.dataset_name, resolved.test_arrow, resolved.test_parquet);
        auto built = BuildResolvedTabularTrainingBatchers(config, resolved, batch_size);
        if (!built.ok()) return Fail(TrainingFailureKind::DataError, "training batchers: " + built.error_message);
        executor = std::make_unique<TrainingExecutor>(std::move(config),
                                                      TakeResolvedExternalBatchers(std::move(built.batchers)));
    } else if (train_on_disk) {
        const std::string label = config.dataset_roles.train.label_column;
        auto file = disk(config.dataset_name);
        executor = std::make_unique<TrainingExecutor>(std::move(config), std::move(file), label);
    } else {
        const std::string label = config.dataset_roles.train.label_column;
        executor = std::make_unique<TrainingExecutor>(std::move(config), train_it->second, label);
    }

    // Resume checkpoints (TOFIX118 P4e): the graph and data files identify the
    // run, so a checkpoint cannot be resumed into a different job.
    if (!request.resume_checkpoint_root.empty() || !request.resume_from.empty()) {
        TrainingResumeIdentity identity;
        identity.run_id = "graph-job";
        identity.graph_json = request.graph_json;
        nlohmann::json files = nlohmann::json::object();
        for (const auto& [name, path] : source_paths) {
            std::error_code size_error;
            files[name] = {{"path", path}, {"bytes", std::filesystem::file_size(path, size_error)}};
        }
        identity.dataset_manifest_json = files.dump(2);
        std::string hash_error;
        for (const auto& [text, digest] :
             {std::pair{&identity.graph_json, &identity.graph_fingerprint},
              std::pair{&identity.dataset_manifest_json, &identity.dataset_fingerprint}}) {
            Sha256Hasher hasher;
            if (!hasher.Update(*text, hash_error) || !hasher.Finish(*digest, hash_error)) digest->clear();
        }
        if (!request.resume_checkpoint_root.empty()) {
            executor->EnableResumeCheckpoints(request.resume_checkpoint_root, identity, 2,
                                              request.resume_checkpoint_every_steps);
            if (callbacks.on_checkpoint) {
                executor->SetResumeCheckpointCallback(
                    [&callbacks](const std::filesystem::path& checkpoint, int epoch, int next_batch) {
                        callbacks.on_checkpoint(checkpoint.string(), epoch, next_batch);
                    });
            }
        }
        if (!request.resume_from.empty()) {
            std::filesystem::path from = request.resume_from;
            if (request.resume_from == "latest") {
                const auto latest = FindLatestTrainingResumeCheckpoint(request.resume_checkpoint_root);
                if (!latest) {
                    return Fail(TrainingFailureKind::DataError,
                                "no resume checkpoint under " + request.resume_checkpoint_root);
                }
                from = *latest;
            }
            executor->SetResumeFrom(from);
        }
    }

    // Cancellation: a watcher asks the executor to stop cooperatively.
    GraphTrainingJobResult result;
    result.timing.tokens_per_sample = tokens_per_sample;
    std::atomic<long long> batches{0};
    const BatchCallback on_batch = [&](int epoch, int batch, int total, float loss, float accuracy) {
        ++batches;
        if (callbacks.on_batch) callbacks.on_batch(epoch, batch, total, loss, accuracy);
        // A pause asked for now holds the next batch: the executor waits at
        // the top of each batch. The watcher's 200 ms poll alone let batches
        // run on after the request.
        if (callbacks.should_pause && !executor->IsPaused() && callbacks.should_pause()) executor->Pause();
    };
    const auto seconds_since = [](std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    std::atomic<bool> finished{false};
    std::thread watcher;
    if (callbacks.should_cancel || callbacks.should_pause || callbacks.should_stop_with_checkpoint) {
        // Cancel and pause reach the executor from here; it honours both
        // between batches (Stop, and Pause/Resume via WaitWhilePaused). The
        // batch callback can pause too, so the watcher follows the
        // executor's state rather than its own.
        watcher = std::thread([&] {
            while (!finished.load()) {
                if (callbacks.should_cancel && callbacks.should_cancel()) {
                    result.cancelled = true;
                    executor->Stop();
                    return;
                }
                if (callbacks.should_stop_with_checkpoint && callbacks.should_stop_with_checkpoint()) {
                    result.cancelled = true;
                    executor->StopWithCheckpoint();
                    return;
                }
                const bool pause = callbacks.should_pause && callbacks.should_pause();
                if (pause && !executor->IsPaused()) {
                    executor->Pause();
                } else if (!pause && executor->IsPaused()) {
                    executor->Resume();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }
    result.timing.prepare_seconds = seconds_since(job_start);
    if (callbacks.on_start) callbacks.on_start(epochs, batch_size);
    const auto train_start = std::chrono::steady_clock::now();
    try {
        executor->Train(epochs, batch_size, on_batch, callbacks.on_epoch);
    } catch (const std::exception& e) {
        finished.store(true);
        if (watcher.joinable()) watcher.join();
        auto failed = Fail(ClassifyTrainingFailure(e.what()), std::string("training failed: ") + e.what());
        failed.timing = result.timing;
        failed.timing.train_seconds = seconds_since(train_start);
        failed.timing.batches_trained = batches.load();
        failed.timing.samples_trained = batches.load() * batch_size;
        return failed;
    }
    finished.store(true);
    if (watcher.joinable()) watcher.join();
    result.timing.train_seconds = seconds_since(train_start);
    result.timing.batches_trained = batches.load();
    result.timing.samples_trained = batches.load() * batch_size;

    result.metrics = executor->GetMetrics();
    if (result.metrics.terminal_status == "failed") {
        result.error = result.metrics.terminal_reason.empty() ? "training failed" : result.metrics.terminal_reason;
        result.failure = ClassifyTrainingFailure(result.error);
        return result;
    }
    if (result.cancelled) result.failure = TrainingFailureKind::Cancelled;
    result.ok = !result.cancelled;
    // A cancelled run keeps what it learned so far (partial weights).
    result.model = executor->ReleaseModel();
    return result;
}


bool PlanGraphJobDatasets(const std::string& graph_json, GraphJobDatasetPlan& plan, std::string& error) {
    plan = {};
    GraphDocument graph;
    if (!ParseGraphDocument(graph_json, graph, error)) return false;
    if (FindPreparationNode(graph, error)) return false;
    // Extension nodes run code the remote node does not have (TOFIX125,
    // decision D3): refused before anything is sent.
    for (const auto& node : graph.nodes) {
        if (node.type != gui::NodeType::PluginCustom) continue;
        error = "This graph uses extension node " + node.extension_type_id + " ('" + node.name +
                "'). Remote nodes do not run extension nodes yet; train it on this computer.";
        return false;
    }

    // Data Split's third input (dataset.v2) is the supplied Test role.
    std::set<int> test_pins;
    for (const auto& node : graph.nodes) {
        if (node.type == gui::NodeType::DataSplit && node.inputs.size() >= 3) test_pins.insert(node.inputs[2].id);
    }
    std::set<int> test_inputs;
    for (const auto& link : graph.links) {
        if (test_pins.count(link.to_pin) > 0) test_inputs.insert(link.from_node);
    }
    std::set<std::string> seen;
    for (const auto& node : graph.nodes) {
        if (node.type != gui::NodeType::DataInput) continue;
        const std::string name = Parameter(node, "dataset_name");
        if (name.empty()) {
            error = "Data Input '" + node.name + "' has no dataset_name";
            return false;
        }
        if (!seen.insert(name).second) continue;
        if (test_inputs.count(node.id) > 0) {
            plan.kept_private.push_back(name);
            continue;
        }
        const std::string cloud_file_id = Parameter(node, "cloud_file_id");
        if (cloud_file_id.empty()) {
            plan.ship.push_back(name);
            continue;
        }
        CloudJobDataset cloud;
        cloud.dataset_name = name;
        cloud.dataset_id = Parameter(node, "cloud_dataset_id");
        cloud.file_id = cloud_file_id;
        cloud.file_name = Parameter(node, "cloud_file_name");
        if (cloud.dataset_id.empty()) {
            error = "Data Input '" + node.name + "' names a CyxCloud file but not its dataset (cloud_dataset_id)";
            return false;
        }
        if (!CloudDatasetFileFormatKnown(cloud.file_name)) {
            error = "Data Input '" + node.name + "': CyxCloud file '" + cloud.file_name +
                    "' is not Arrow IPC or Parquet; remote nodes train from those two formats only";
            return false;
        }
        const std::string size = Parameter(node, "cloud_size_bytes");
        if (!size.empty()) cloud.size_bytes = std::strtoull(size.c_str(), nullptr, 10);
        plan.cloud.push_back(std::move(cloud));
    }
    if (plan.ship.empty() && plan.cloud.empty()) {
        error = "the graph has no training Data Input";
        return false;
    }
    return true;
}

bool BindGraphJobDatasets(const std::string& graph_json, const std::map<std::string, std::string>& files,
                          const std::vector<std::string>& drop, std::string& bound_json, std::string& error) {
    nlohmann::json graph;
    try {
        graph = nlohmann::json::parse(graph_json);
    } catch (const std::exception& e) {
        error = std::string("the graph is not valid JSON: ") + e.what();
        return false;
    }
    if (!graph.contains("nodes") || !graph["nodes"].is_array() || !graph.contains("links") ||
        !graph["links"].is_array()) {
        error = "the graph has no nodes and links arrays";
        return false;
    }
    const std::set<std::string> dropped(drop.begin(), drop.end());
    std::set<int> removed_nodes;
    nlohmann::json nodes = nlohmann::json::array();
    for (auto node : graph["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataInput)) {
            auto& params = node["parameters"];
            const std::string name = params.is_object() ? params.value("dataset_name", std::string()) : std::string();
            if (dropped.count(name) > 0) {
                removed_nodes.insert(node.value("id", 0));
                continue;
            }
            const auto file = files.find(name);
            if (file == files.end()) {
                error = "no local file for dataset '" + name + "'";
                return false;
            }
            params["file_path"] = file->second;
            const std::string extension = Lower(std::filesystem::path(file->second).extension().string());
            params["file_type"] = extension == ".parquet" ? "parquet" : "arrow";
        }
        nodes.push_back(std::move(node));
    }
    nlohmann::json links = nlohmann::json::array();
    for (const auto& link : graph["links"]) {
        if (removed_nodes.count(link.value("from_node", 0)) > 0 || removed_nodes.count(link.value("to_node", 0)) > 0) {
            continue;
        }
        links.push_back(link);
    }
    graph["nodes"] = std::move(nodes);
    graph["links"] = std::move(links);
    bound_json = graph.dump();
    return true;
}

}  // namespace cyxwiz
