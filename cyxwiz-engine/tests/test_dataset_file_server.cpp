// Remote jobs ship whole dataset files (TOFIX118 P2): the Engine's
// DatasetFileServer plans a job from its graph, serves the training inputs in
// ordered SHA-256-checked chunks (resumable), keeps a supplied Test input on
// the Engine, and the node side binds the received file into the graph and
// trains it with RunGraphTrainingJob.
#include "../src/core/arrow_dataset.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/sha256_digest.h"
#include "../src/network/dataset_file_server.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef STATUS_ERROR
#undef STATUS_ERROR
#endif

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

const std::string kTrain = "tiny_causal_lm_tokens";
const std::string kTest = "tiny_causal_lm_heldout";

// The token-window graph with a supplied Test input on Data Split's third pin.
nlohmann::json GraphWithTestInput(const fs::path& root) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    nlohmann::json test_input;
    for (const auto& node : graph["nodes"]) {
        if (node.value("id", 0) == 1) test_input = node;
    }
    test_input["id"] = 21;
    test_input["name"] = "Held-out tokens";
    test_input["parameters"]["dataset_name"] = kTest;
    graph["nodes"].push_back(test_input);
    graph["links"].push_back({{"id", 41}, {"from_node", 21}, {"from_pin", 0}, {"from_pin_index", 0},
                              {"to_node", 3}, {"to_pin", 0}, {"to_pin_index", 2}, {"link_type", 0}});
    return graph;
}

struct Received {
    std::vector<cyxwiz::protocol::DatasetFileChunk> chunks;
    std::string bytes;
};

Received Request(network::DatasetFileServer& server, const std::string& job, const std::string& dataset,
                 int64_t offset = 0, size_t chunk_bytes = 256) {
    cyxwiz::protocol::DatasetFileRequest request;
    request.set_job_id(job);
    request.set_dataset_name(dataset);
    request.set_offset(offset);
    request.set_request_id(7);
    Received received;
    server.HandleRequest(
        request,
        [&received](const cyxwiz::protocol::DatasetFileChunk& chunk) {
            received.chunks.push_back(chunk);
            received.bytes += chunk.data();
            return true;
        },
        chunk_bytes);
    return received;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_dataset_file_server_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    const fs::path heldout = work / "heldout.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8) ||
        !cyxwiz::test::WriteTokenWindows(root, heldout, 8, 2)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    // The Engine registry, as the catalog the graph compiler reads.
    std::map<std::string, std::shared_ptr<cyxwiz::ArrowDataset>> loaded;
    loaded[kTrain] = cyxwiz::ArrowDataset::FromParquet(parquet.string(), kTrain);
    loaded[kTest] = cyxwiz::ArrowDataset::FromParquet(heldout.string(), kTest);
    cyxwiz::GraphDatasetCatalog catalog;
    catalog.arrow_dataset = [&loaded](const std::string& name) -> std::shared_ptr<cyxwiz::ArrowDataset> {
        const auto it = loaded.find(name);
        return it == loaded.end() ? nullptr : it->second;
    };
    cyxwiz::SetGraphDatasetCatalog(catalog);

    const std::string graph = GraphWithTestInput(root).dump();

    std::cout << "plans the job from its graph\n";
    {
        cyxwiz::GraphJobDatasetPlan plan;
        std::string error;
        Check(cyxwiz::PlanGraphJobDatasets(graph, plan, error), "plan ok " + error);
        Check(plan.ship == std::vector<std::string>{kTrain}, "ships only the training input");
        Check(plan.kept_private == std::vector<std::string>{kTest}, "Test input stays on the Engine");

        nlohmann::json with_scaler = GraphWithTestInput(root);
        with_scaler["nodes"].push_back({{"id", 90}, {"type", static_cast<int>(gui::NodeType::StandardScaler)},
                                        {"name", "Scale"}, {"parameters", nlohmann::json::object()}});
        Check(!cyxwiz::PlanGraphJobDatasets(with_scaler.dump(), plan, error) &&
                  Contains(error, "data preparation step"),
              "graph with a preparation step refused: " + error);
    }

    network::DatasetFileServer server;
    const fs::path cache = work / "engine_cache" / "job_1";

    std::cout << "refuses a job whose dataset is not loaded\n";
    {
        nlohmann::json other = GraphWithTestInput(root);
        for (auto& node : other["nodes"]) {
            if (node.value("id", 0) == 1) node["parameters"]["dataset_name"] = "not_loaded";
        }
        std::string error;
        Check(!server.RegisterJob("job_x", other.dump(), cache, error) && Contains(error, "not loaded"),
              "missing dataset refused: " + error);
    }

    std::string error;
    Check(server.RegisterJob("job_1", graph, cache, error), "job registered " + error);

    std::cout << "sends the training input in ordered, hashed chunks\n";
    std::string shipped_sha;
    fs::path received_file = work / "node_cache" / (kTrain + ".arrow");
    {
        const Received received = Request(server, "job_1", kTrain);
        Check(received.chunks.size() > 1, "several chunks (" + std::to_string(received.chunks.size()) + ")");
        bool ordered = true;
        int64_t expected_offset = 0;
        for (size_t i = 0; i < received.chunks.size(); ++i) {
            const auto& chunk = received.chunks[i];
            ordered = ordered && chunk.status() == cyxwiz::protocol::STATUS_SUCCESS &&
                      chunk.offset() == expected_offset && chunk.request_id() == 7 &&
                      chunk.last() == (i + 1 == received.chunks.size()) && chunk.format() == "arrow_ipc";
            expected_offset += static_cast<int64_t>(chunk.data().size());
        }
        Check(ordered, "chunks in order, last flag on the final one");
        const auto& first = received.chunks.front();
        Check(first.total_size() == static_cast<int64_t>(received.bytes.size()), "total size matches");
        cyxwiz::Sha256Hasher hasher;
        std::string digest;
        Check(hasher.Update(received.bytes, error) && hasher.Finish(digest, error) && digest == first.sha256(),
              "SHA-256 of the received bytes matches");
        shipped_sha = digest;

        fs::create_directories(received_file.parent_path());
        std::ofstream(received_file, std::ios::binary) << received.bytes;
        auto reopened = cyxwiz::ArrowDataset::FromFile(received_file.string(), kTrain);
        Check(reopened && reopened->GetNumRows() == loaded[kTrain]->GetNumRows(), "received file has the rows");
        const auto metadata = reopened ? reopened->GetArrowTable()->schema()->metadata() : nullptr;
        Check(metadata && metadata->Contains("cyxwiz.token_windows.vocabulary"),
              "token-window schema metadata survives");
    }

    std::cout << "resumes from an offset\n";
    {
        const Received full = Request(server, "job_1", kTrain);
        const Received tail = Request(server, "job_1", kTrain, 700);
        Check(!tail.chunks.empty() && tail.chunks.front().offset() == 700 && full.bytes.substr(700) == tail.bytes,
              "tail from offset 700 equals the file's tail");
        Check(tail.chunks.front().sha256() == shipped_sha, "resume carries the same whole-file hash");
    }

    std::cout << "refuses what must not be sent\n";
    {
        const Received test = Request(server, "job_1", kTest);
        Check(test.chunks.size() == 1 && test.chunks[0].status() != cyxwiz::protocol::STATUS_SUCCESS &&
                  Contains(test.chunks[0].error().message(), "stays on the Engine"),
              "Test input refused");
        const Received other = Request(server, "job_1", "some_other_dataset");
        Check(other.chunks.size() == 1 && Contains(other.chunks[0].error().message(), "not an input"),
              "dataset outside the graph refused");
        const Received unknown = Request(server, "job_unknown", kTrain);
        Check(unknown.chunks.size() == 1 && Contains(unknown.chunks[0].error().message(), "no datasets registered"),
              "unknown job refused");
    }

    std::cout << "node side: binds the received file and trains\n";
    {
        cyxwiz::SetGraphDatasetCatalog({});  // the node has no Engine registry
        std::string bound;
        Check(cyxwiz::BindGraphJobDatasets(graph, {{kTrain, received_file.string()}}, {kTest}, bound, error),
              "bound " + error);
        const auto bound_json = nlohmann::json::parse(bound);
        bool test_node_gone = true, test_link_gone = true, path_set = false;
        for (const auto& node : bound_json["nodes"]) {
            if (node.value("id", 0) == 21) test_node_gone = false;
            if (node.value("id", 0) == 1) {
                path_set = node["parameters"].value("file_path", std::string()) == received_file.string() &&
                           node["parameters"].value("file_type", std::string()) == "arrow";
            }
        }
        for (const auto& link : bound_json["links"]) {
            if (link.value("from_node", 0) == 21) test_link_gone = false;
        }
        Check(test_node_gone && test_link_gone, "Test input and its link removed");
        Check(path_set, "training input reads the received file");
        std::string unbound_error;
        Check(!cyxwiz::BindGraphJobDatasets(graph, {}, {kTest}, bound, unbound_error) &&
                  Contains(unbound_error, "no local file"),
              "an input without a file is refused");

        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = bound_json.dump();
        request.epochs_override = 1;
        request.checkpoint_dir_override = (work / "checkpoints").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(result.ok, "trained from the shipped file" + (result.error.empty() ? "" : " (" + result.error + ")"));
    }

    std::cout << "unregister deletes the export\n";
    {
        Check(fs::exists(cache / (kTrain + ".arrow")), "export exists while registered");
        server.UnregisterJob("job_1");
        Check(!fs::exists(cache / (kTrain + ".arrow")) && !server.HasJob("job_1"), "export deleted");
    }

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "dataset file server test passed\n";
    return EXIT_SUCCESS;
}
