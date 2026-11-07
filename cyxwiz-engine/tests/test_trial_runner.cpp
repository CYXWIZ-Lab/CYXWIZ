// A search trial trains from Train's recipe (TOFIX178 trial runner): the
// graph prepared by Train's launcher, the recipe its dispatch would start,
// then RunTrial with node properties changed. A trial with no changes gives
// the same validation loss as Train for the same seed; a changed learning
// rate gives another; a changed epoch count runs that many; a property that
// is not a training target is refused; cancel stops it; TrainingManager (the
// model in use) is never touched.
#include "../src/core/async_task_manager.h"
#include "../src/core/data_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_document.h"
#include "../src/core/training_manager.h"
#include "../src/core/trial_runner.h"
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
#include <optional>
#include <thread>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

// The dataset is registered as an Arrow table and the graph is a sequence
// graph, so the loader dispatch is never reached.
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

bool Close(float a, float b) { return std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(a)); }

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_trial_runner";
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
            node["parameters"]["model_seed"] = "52";
            node["parameters"]["checkpoint_dir"] = (work / "train_checkpoints").string();
        }
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataInput)) {
            node["parameters"]["file_path"] = parquet.string();
        }
        if (node.value("type", -1) == static_cast<int>(gui::NodeType::DataSplit)) {
            node["parameters"]["train_ratio"] = "0.6";
            node["parameters"]["val_ratio"] = "0.2";
            node["parameters"]["test_ratio"] = "0.2";
        }
    }

    auto& registry = cyxwiz::DataRegistry::Instance();
    auto table = ReadParquet(parquet);
    Check(table != nullptr && registry.RegisterArrowTable(table, dataset) != nullptr, "dataset applied");
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [&registry](const std::string& name) {
        return registry.GetArrowDataset(name) != nullptr;
    };
    cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));

    cyxwiz::GraphDocument document;
    std::string error;
    Check(cyxwiz::ParseGraphDocument(graph.dump(), document, error), "graph loaded " + error);
    const auto launch_with = [&](gui::GraphTrainingDispatch dispatch, std::atomic<bool>& done) {
        cyxwiz::GraphCompiler compiler;
        cyxwiz::TrainingConfiguration config = compiler.Compile(document.nodes, document.links);
        const auto preflight = gui::PreflightGraphMaterialization(document.nodes, document.links, config, registry);
        return gui::StartGraphTrainingFromCompiledConfig(
            document.nodes, document.links, std::move(config), registry, std::weak_ptr<cyxwiz::TrainingPlotPanel>{},
            [&done](bool training) {
                if (!training) done = true;
            },
            std::move(dispatch), {}, preflight.estimate_available ? std::make_optional(preflight.evidence) : std::nullopt,
            work);
    };

    std::cout << "Train (the reference)\n";
    auto& tm = cyxwiz::TrainingManager::Instance();
    float train_val_loss = 0.0f;
    {
        std::atomic<bool> done{false};
        const auto launch = launch_with(gui::MakeEngineGraphTrainingDispatch(registry, tm), done);
        Check(launch.started, "Train started " + launch.error_message);
        Check(WaitFor([&] { return done.load() && !tm.IsTrainingActive(); }, std::chrono::minutes(3)), "Train finished");
        tm.WaitForTrainingStop();
        const auto metrics = tm.GetCurrentMetrics();
        Check(metrics.val_loss_history.size() == static_cast<size_t>(kEpochs), "Train validated every epoch");
        if (!metrics.val_loss_history.empty()) train_val_loss = metrics.val_loss_history.back();
    }
    const auto model_in_use = tm.GetActiveModel();

    std::cout << "The trial setup: what Train's launcher hands its dispatch\n";
    cyxwiz::TrialSetup setup;
    setup.nodes = document.nodes;
    setup.links = document.links;
    bool captured = false;
    {
        std::atomic<bool> done{false};
        const auto launch = launch_with(
            [&](cyxwiz::TrainingConfiguration config, const std::string& dataset_name, const std::string& label,
                int, int, std::weak_ptr<cyxwiz::TrainingPlotPanel>, std::function<void(bool)>) {
                setup.recipe = gui::GraphTrainingRecipe(registry, config, dataset_name, label);
                setup.launched = std::move(config);
                captured = true;
                return true;
            },
            done);
        Check(launch.started && WaitFor([&] { return captured; }, std::chrono::minutes(1)),
              "the launcher handed on the prepared graph " + launch.error_message);
    }
    Check(static_cast<bool>(setup.recipe), "the recipe builds runs " + setup.recipe.error);

    int optimizer = setup.launched.optimizer_node_id;
    std::string lr_key = "learning_rate";
    int loader = -1, data_input = -1;
    for (const auto& node : setup.nodes) {
        if (node.id == optimizer && !node.parameters.count(lr_key) && node.parameters.count("lr")) lr_key = "lr";
        if (node.type == gui::NodeType::DataLoader) loader = node.id;
        if (node.type == gui::NodeType::DataInput) data_input = node.id;
    }

    std::cout << "A trial with no changes is Train\n";
    const auto same = cyxwiz::RunTrial(setup, {});
    Check(same.ok, "the trial ran " + same.error);
    Check(same.epochs == kEpochs, "for the graph's " + std::to_string(kEpochs) + " epochs");
    Check(same.score.has_validation && Close(same.score.val_loss, train_val_loss),
          "the same validation loss as Train: " + std::to_string(same.score.val_loss) + " vs " +
              std::to_string(train_val_loss));

    std::cout << "Changes\n";
    std::vector<int> epochs_seen;
    const auto faster = cyxwiz::RunTrial(setup, {{optimizer, lr_key, "0.05"}},
                                         {nullptr, [&](const cyxwiz::TrialProgress& p) { epochs_seen.push_back(p.epoch); }});
    Check(faster.ok && !Close(faster.score.val_loss, same.score.val_loss),
          "another learning rate trains to another loss " + faster.error);
    std::string seen;
    for (const int e : epochs_seen) seen += std::to_string(e) + " ";
    Check(epochs_seen == std::vector<int>({1, 2, 3}), "each epoch is reported: " + seen);
    const auto shorter = cyxwiz::RunTrial(setup, {{loader, "epochs", "1"}});
    Check(shorter.ok && shorter.epochs == 1, "the DataLoader's epochs set the trial's length " + shorter.error);

    std::cout << "Refusals\n";
    const auto file = cyxwiz::RunTrial(setup, {{data_input, "file_path", "elsewhere.parquet"}});
    Check(!file.ok && file.error.find("not a layer") != std::string::npos, "a data property is refused: " + file.error);
    const auto missing = cyxwiz::RunTrial(setup, {{optimizer, "no_such_property", "1"}});
    Check(!missing.ok && missing.error.find("no property") != std::string::npos, "an unknown property is refused");
    const auto cancelled = cyxwiz::RunTrial(setup, {}, {[] { return true; }, nullptr});
    Check(!cancelled.ok && cancelled.cancelled, "a cancel stops the trial");

    std::cout << "TrainingManager is untouched\n";
    Check(!tm.IsTrainingActive() && tm.GetActiveModel() == model_in_use, "the model in use is still Train's");
    Check(!fs::exists(fs::temp_directory_path() / "cyxwiz_trials") ||
              fs::is_empty(fs::temp_directory_path() / "cyxwiz_trials"),
          "the trials' scratch checkpoints are gone");

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Trial runner test passed\n";
    return EXIT_SUCCESS;
}
