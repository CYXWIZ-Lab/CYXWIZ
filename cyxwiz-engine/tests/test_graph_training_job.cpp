// Headless graph training (TOFIX118 P2): trains the tracked causal-LM example
// graph through RunGraphTrainingJob, linked against cyxwiz-training-core only
// (what a Server Node links), and checks the fail-closed refusals.
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_training_job.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;

namespace {

using cyxwiz::test::LoadTokenWindowGraph;
using cyxwiz::test::ReadText;
using cyxwiz::test::WriteTokenWindows;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

nlohmann::json& NodeNamed(nlohmann::json& graph, int id) {
    for (auto& node : graph["nodes"]) {
        if (node.value("id", 0) == id) return node;
    }
    throw std::runtime_error("node not found");
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_graph_training_job_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    const fs::path validation = work / "validation.parquet";
    if (!WriteTokenWindows(root, parquet, 0, 8) || !WriteTokenWindows(root, validation, 8, 2)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }

    // The executor's route preflight refuses unqualified devices; qualify the
    // inventory as a verified host would have.
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    // A host catalog the job must restore afterwards.
    bool host_catalog_called = false;
    cyxwiz::GraphDatasetCatalog host;
    host.source_path = [&host_catalog_called](const std::string&) -> std::optional<std::string> {
        host_catalog_called = true;
        return std::string("host");
    };
    cyxwiz::SetGraphDatasetCatalog(host);

    std::cout << "trains the causal-LM example from Parquet\n";
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 2;
        request.checkpoint_dir_override = (work / "checkpoints").string();
        int batches = 0;
        int epochs = 0;
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_batch = [&batches](int, int, int, float, float) { ++batches; };
        callbacks.on_epoch = [&epochs](int, float, float, float, float, float) { ++epochs; };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        Check(!result.cancelled, "not cancelled");
        Check(result.model != nullptr, "trained model returned");
        Check(epochs == 2, "two epochs reported (got " + std::to_string(epochs) + ")");
        Check(batches > 0, "batches reported");
        Check(result.metrics.terminal_status != "failed", "metrics not failed");
    }

    const auto host_now = cyxwiz::GetGraphDatasetCatalog();
    Check(static_cast<bool>(host_now.source_path) && !host_now.arrow_dataset,
          "host catalog restored after the job");
    if (host_now.source_path) host_now.source_path("x");
    Check(host_catalog_called, "restored catalog is the host's");

    std::cout << "cancels cooperatively\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 50;
        request.checkpoint_dir_override = (work / "checkpoints_cancel").string();
        std::atomic<int> batches{0};
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_batch = [&batches](int, int, int, float, float) { ++batches; };
        callbacks.should_cancel = [&batches] { return batches.load() > 0; };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.cancelled && !result.ok, "reported as cancelled");
    }

    std::cout << "trains with a supplied validation input\n";
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        nlohmann::json input = NodeNamed(graph, 1);
        input["id"] = 20;
        input["name"] = "Validation tokens";
        input["parameters"]["dataset_name"] = "tiny_causal_lm_validation";
        graph["nodes"].push_back(input);
        graph["links"].push_back({{"id", 40}, {"from_node", 20}, {"from_pin", 60}, {"from_pin_index", 0},
                                  {"to_node", 3}, {"to_pin", 6}, {"to_pin_index", 1}, {"link_type", 0}});
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.dataset_files["tiny_causal_lm_validation"] = validation.string();
        request.epochs_override = 1;
        request.checkpoint_dir_override = (work / "checkpoints_roles").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(result.ok, "job with a validation role succeeded" +
                             (result.error.empty() ? "" : " (" + result.error + ")"));
    }

    std::cout << "refuses what it cannot run\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] =
            (root / "examples/cyxgraph/text/causal_lm_tiny_token_sequences.csv").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "not supported"), "CSV input refused: " + result.error);
    }
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = (work / "missing.parquet").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "file not found"), "missing file refused: " + result.error);
    }
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        nlohmann::json scaler = NodeNamed(graph, 3);
        scaler["id"] = 99;
        scaler["type"] = static_cast<int>(gui::NodeType::StandardScaler);
        scaler["name"] = "Scale tokens";
        scaler["category"] = static_cast<int>(gui::NodeCategory::Preprocessing);
        scaler["parameters"] = nlohmann::json::object();
        graph["nodes"].push_back(scaler);
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "data preparation step"),
              "preprocessing node refused: " + result.error);
    }
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        nlohmann::json scaler = NodeNamed(graph, 3);
        scaler["id"] = 98;
        scaler["type"] = static_cast<int>(gui::NodeType::StandardScaler);
        scaler["name"] = "Scale tokens (uncategorized)";
        scaler.erase("category");
        scaler["parameters"] = nlohmann::json::object();
        graph["nodes"].push_back(scaler);
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "data preparation step"),
              "operator node without a category refused: " + result.error);
    }
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = "{ not json";
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "not valid JSON"), "malformed graph refused");
    }

    fs::remove_all(work, ec);
    cyxwiz::SetGraphDatasetCatalog({});
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "graph training job test passed\n";
    return EXIT_SUCCESS;
}
