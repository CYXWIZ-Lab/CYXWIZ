// Engine vs Server Node training parity (TOFIX118 P2 slice 4): the same graph,
// data and seed trained through the Engine's Train path (graph loader,
// compiler, launcher preparation, MainWindow's dispatch, TrainingManager) and
// through the node's RunGraphTrainingJob give the same per-epoch losses.
#include "../src/core/async_task_manager.h"
#include "../src/core/data_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/training_manager.h"
#include "../src/gui/engine_graph_training_dispatch.h"
#include "../src/gui/graph_training_launcher.h"
#include "../src/gui/loaders/data_loader.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <parquet/arrow/reader.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

// No file loaders in this test: the dataset is registered as an Arrow table,
// and sequence graphs never reach the loader dispatch.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool WaitFor(const std::function<bool()>& predicate, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    cyxwiz::AsyncTaskManager::Instance().ProcessCompletedCallbacks();
    return predicate();
}

std::shared_ptr<arrow::Table> ReadParquet(const fs::path& path) {
    auto input = arrow::io::ReadableFile::Open(path.string());
    if (!input.ok()) return nullptr;
    auto reader = parquet::arrow::OpenFile(*input, arrow::default_memory_pool());
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::Table> table;
    if (!(*reader)->ReadTable(&table).ok()) return nullptr;
    return table;
}

std::string Describe(const std::vector<float>& values) {
    std::string text;
    for (const float v : values) text += (text.empty() ? "" : ", ") + std::to_string(v);
    return "[" + text + "]";
}

bool Same(const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
    if (a.size() != b.size() || a.empty()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(std::fabs(a[i] - b[i]) <= tolerance * std::max(1.0f, std::fabs(a[i])))) return false;
    }
    return true;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_engine_node_parity";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    constexpr int kEpochs = 3;
    const std::string dataset = "tiny_causal_lm_tokens";
    auto graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& node : graph["nodes"]) {
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataLoader)) {
            node["parameters"]["epochs"] = std::to_string(kEpochs);
            // Parity is defined for a seeded run (unset = -1 keeps the
            // process's random state, as the Berean graphs never do).
            node["parameters"]["model_seed"] = "52";
            node["parameters"]["checkpoint_dir"] = (work / "engine_checkpoints").string();
        }
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataInput)) {
            node["parameters"]["file_path"] = parquet.string();
        }
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataSplit)) {
            // 8 windows: 80/10/10 leaves no validation window; 60/20/20 leaves one.
            node["parameters"]["train_ratio"] = "0.6";
            node["parameters"]["val_ratio"] = "0.2";
            node["parameters"]["test_ratio"] = "0.2";
        }
    }
    const std::string graph_json = graph.dump();

    std::cout << "Engine: Train through the launcher and TrainingManager\n";
    std::vector<float> engine_loss;
    std::vector<float> engine_val_loss;
    {
        auto& registry = cyxwiz::DataRegistry::Instance();
        auto table = ReadParquet(parquet);
        Check(table != nullptr && registry.RegisterArrowTable(table, dataset) != nullptr,
              "dataset applied in the Engine registry");

        // What the Engine's loaders answer for an applied Arrow table (the
        // Tabular loader owns it); the loaders themselves are not linked here.
        cyxwiz::GraphCompilerDatasetHooks hooks;
        hooks.is_dataset_registered = [&registry](const std::string& name) {
            return registry.GetArrowDataset(name) != nullptr;
        };
        cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));

        cyxwiz::GraphDocument document;
        std::string error;
        Check(cyxwiz::ParseGraphDocument(graph_json, document, error), "graph loaded " + error);
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(document.nodes, document.links);
        Check(config.is_valid, "graph compiles in the Engine " + config.error_message);

        const auto preflight = gui::PreflightGraphMaterialization(document.nodes, document.links, config, registry);
        std::atomic<bool> finished{false};
        auto& tm = cyxwiz::TrainingManager::Instance();
        const auto launch = gui::StartGraphTrainingFromCompiledConfig(
            document.nodes, document.links, std::move(config), registry, std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
            [&finished](bool training) {
                if (!training) finished = true;
            },
            gui::MakeEngineGraphTrainingDispatch(registry, tm), {},
            preflight.estimate_available ? std::make_optional(preflight.evidence) : std::nullopt, work);
        Check(launch.started, "Engine launch started " + launch.error_message);
        Check(WaitFor([&] { return finished.load() && !tm.IsTrainingActive(); }, std::chrono::minutes(3)),
              "Engine run finished");
        tm.WaitForTrainingStop();
        const auto metrics = tm.GetCurrentMetrics();
        engine_loss = metrics.loss_history;
        engine_val_loss = metrics.val_loss_history;
        std::cout << "    train loss " << Describe(engine_loss) << "\n    val loss   " << Describe(engine_val_loss)
                  << "\n";
    }

    std::cout << "Node: RunGraphTrainingJob\n";
    std::vector<float> node_loss;
    std::vector<float> node_val_loss;
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph_json;
        request.checkpoint_dir_override = (work / "node_checkpoints").string();
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_epoch = [&](int, float train_loss, float, float val_loss, float, float) {
            node_loss.push_back(train_loss);
            if (val_loss >= 0.0f) node_val_loss.push_back(val_loss);
        };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "node run finished " + result.error);
        std::cout << "    train loss " << Describe(node_loss) << "\n    val loss   " << Describe(node_val_loss) << "\n";
    }

    std::cout << "Node again (same process)\n";
    std::vector<float> node_loss_2;
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph_json;
        request.checkpoint_dir_override = (work / "node_checkpoints_2").string();
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_epoch = [&](int, float train_loss, float, float, float, float) {
            node_loss_2.push_back(train_loss);
        };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "second node run finished " + result.error);
        std::cout << "    train loss " << Describe(node_loss_2) << "\n";
    }
    Check(Same(node_loss, node_loss_2, 1e-5f), "a seeded run repeats itself");

    std::cout << "parity\n";
    Check(engine_loss.size() == static_cast<size_t>(kEpochs) && node_loss.size() == engine_loss.size(),
          "both ran " + std::to_string(kEpochs) + " epochs");
    Check(Same(engine_loss, node_loss, 1e-5f), "same training loss per epoch");
    Check(engine_val_loss.size() == static_cast<size_t>(kEpochs) && engine_val_loss.front() > 0.0f,
          "validation ran on a real partition");
    Check(Same(engine_val_loss, node_val_loss, 1e-5f), "same validation loss per epoch");

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Engine/node parity test passed\n";
    return EXIT_SUCCESS;
}

