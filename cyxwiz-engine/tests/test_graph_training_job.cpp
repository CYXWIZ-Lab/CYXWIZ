// Headless graph training (TOFIX118 P2): trains the tracked causal-LM example
// graph through RunGraphTrainingJob, linked against cyxwiz-training-core only
// (what a Server Node links), and checks the fail-closed refusals.
#include "../src/core/arrow_dataset.h"
#include "../src/core/data_budget.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_job_memory_probe.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/graph_node_factory.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/arrow/writer.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <thread>
#include <algorithm>
#include <optional>
#include <mutex>
#include <chrono>
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

// Three separable classes over four features, written in row groups of 100.
bool WriteTabularBlobs(const fs::path& path, int rows, int seed) {
    arrow::DoubleBuilder features[4];
    arrow::Int64Builder labels;
    uint32_t state = static_cast<uint32_t>(seed);
    const auto noise = [&state] {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>(state >> 8) / 16777216.0 - 0.5) * 0.6;
    };
    for (int row = 0; row < rows; ++row) {
        const int label = row % 3;
        for (int column = 0; column < 4; ++column) {
            if (!features[column].Append((column == label ? 2.0 : 0.0) + noise()).ok()) return false;
        }
        if (!labels.Append(label).ok()) return false;
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(5);
    for (int column = 0; column < 4; ++column) {
        if (!features[column].Finish(&arrays[column]).ok()) return false;
    }
    if (!labels.Finish(&arrays[4]).ok()) return false;
    auto table = arrow::Table::Make(arrow::schema({arrow::field("x0", arrow::float64()),
                                                   arrow::field("x1", arrow::float64()),
                                                   arrow::field("x2", arrow::float64()),
                                                   arrow::field("x3", arrow::float64()),
                                                   arrow::field("class", arrow::int64())}),
                                    arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 100).ok() &&
           (*out)->Close().ok();
}

// Data Input -> Split -> Loader -> Dense 16 -> ReLU -> Dense 3 -> cross
// entropy -> Adam -> Output; with a validation input on the Split's second pin.
nlohmann::json TabularGraph(bool validation_input) {
    using gui::NodeType;
    nlohmann::json nodes = nlohmann::json::array();
    const auto add = [&nodes](int id, NodeType type, const char* name, nlohmann::json parameters) {
        nodes.push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name}, {"parameters", parameters}});
    };
    add(1, NodeType::DataInput, "Blobs",
        {{"dataset_name", "tabular_blobs"}, {"file_category", "tabular"}, {"file_type", "parquet"},
         {"label_column", "class"}});
    add(2, NodeType::DataSplit, "Split",
        {{"seed", "42"}, {"stratified", "false"}, {"train_ratio", validation_input ? "0.9" : "0.8"},
         {"val_ratio", validation_input ? "0.1" : "0.1"}, {"test_ratio", validation_input ? "0" : "0.1"}});
    add(3, NodeType::DataLoader, "Loader",
        {{"batch_size", "32"}, {"epochs", "2"}, {"shuffle", "true"}, {"seed", "11"}, {"model_seed", "7"}});
    add(4, NodeType::Dense, "Dense 16", {{"units", "16"}});
    add(5, NodeType::ReLU, "ReLU", nlohmann::json::object());
    add(6, NodeType::Dense, "Classes", {{"units", "3"}});
    add(7, NodeType::CrossEntropyLoss, "Cross entropy", nlohmann::json::object());
    add(8, NodeType::Adam, "Adam", {{"learning_rate", "0.01"}});
    add(9, NodeType::Output, "Output", {{"num_classes", "3"}});
    nlohmann::json links = nlohmann::json::array();
    int link_id = 1;
    const auto link = [&](int from, int from_index, int to, int to_index) {
        links.push_back({{"id", link_id}, {"from_node", from}, {"from_pin", 100 + link_id}, {"from_pin_index", from_index},
                         {"to_node", to}, {"to_pin", 200 + link_id}, {"to_pin_index", to_index}, {"link_type", 0}});
        ++link_id;
    };
    link(1, 0, 2, 0);
    link(2, 0, 3, 0);
    link(3, 0, 4, 0);
    link(4, 0, 5, 0);
    link(5, 0, 6, 0);
    link(6, 0, 7, 0);
    link(3, 1, 7, 1);
    link(7, 0, 8, 0);
    link(8, 0, 9, 0);
    if (validation_input) {
        add(10, NodeType::DataInput, "Validation blobs",
            {{"dataset_name", "tabular_blobs_validation"}, {"file_category", "tabular"}, {"file_type", "parquet"},
             {"label_column", "class"}});
        link(10, 0, 2, 1);
    }
    return {{"version", "1.0"}, {"data_boundary_version", 2}, {"nodes", nodes}, {"links", links}};
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
        bool fake_validation = false;
        callbacks.on_epoch = [&epochs, &fake_validation](int, float, float, float val_loss, float, float) {
            ++epochs;
            // 80/10/10 of 8 windows leaves no validation window: no metric, not 0.
            fake_validation = fake_validation || val_loss >= 0.0f;
        };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        Check(result.failure == cyxwiz::TrainingFailureKind::None, "a completed job has no failure category");
        Check(!result.cancelled, "not cancelled");
        Check(result.model != nullptr, "trained model returned");
        Check(epochs == 2, "two epochs reported (got " + std::to_string(epochs) + ")");
        Check(batches > 0, "batches reported");
        Check(!fake_validation, "an empty validation partition reports no validation loss");
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
        Check(result.failure == cyxwiz::TrainingFailureKind::Cancelled, "cancel is its own category");
        Check(result.model != nullptr, "a cancelled run returns its partially trained model");
    }

    std::cout << "pauses between batches and resumes\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 3;
        request.checkpoint_dir_override = (work / "checkpoints_pause").string();
        using Clock = std::chrono::steady_clock;
        std::mutex mutex;
        std::vector<Clock::time_point> batch_times;
        std::optional<Clock::time_point> pause_started;
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_batch = [&](int, int, int, float, float) {
            std::lock_guard<std::mutex> lock(mutex);
            batch_times.push_back(Clock::now());
            if (!pause_started) pause_started = Clock::now();  // pause after the first batch
        };
        // Held for 1.5 s after the first batch, then released.
        callbacks.should_pause = [&] {
            std::lock_guard<std::mutex> lock(mutex);
            return pause_started && Clock::now() - *pause_started < std::chrono::milliseconds(1500);
        };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "a paused job completes after resuming" + (result.error.empty() ? "" : " (" + result.error + ")"));
        double longest_gap = 0.0;
        for (size_t i = 1; i < batch_times.size(); ++i) {
            longest_gap = std::max(longest_gap,
                                   std::chrono::duration<double>(batch_times[i] - batch_times[i - 1]).count());
        }
        Check(longest_gap >= 1.0, "no batch ran while paused (longest gap " + std::to_string(longest_gap) + " s)");
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

    std::cout << "keeps inputs over the data budget on disk (TOFIX154)\n";
    {
        // The same graph with a validation input and no test share, unshuffled
        // and seeded: in memory, then with a 1-byte budget so both files stay
        // on disk and stream a row group at a time. The losses match.
        nlohmann::json graph = LoadTokenWindowGraph(root);
        nlohmann::json input = NodeNamed(graph, 1);
        input["id"] = 20;
        input["name"] = "Validation tokens";
        input["parameters"]["dataset_name"] = "tiny_causal_lm_validation";
        graph["nodes"].push_back(input);
        graph["links"].push_back({{"id", 40}, {"from_node", 20}, {"from_pin", 60}, {"from_pin_index", 0},
                                  {"to_node", 3}, {"to_pin", 6}, {"to_pin_index", 1}, {"link_type", 0}});
        for (auto& node : graph["nodes"]) {
            auto& params = node["parameters"];
            if (params.contains("batch_size")) {
                params["model_seed"] = "7";
                params["seed"] = "11";
                params["shuffle"] = "false";
                params["batch_size"] = "2";
            }
        }
        for (auto& node : graph["nodes"]) {
            if (!node["parameters"].contains("test_ratio")) continue;
            node["parameters"]["train_ratio"] = "0.9";
            node["parameters"]["val_ratio"] = "0.1";
            node["parameters"]["test_ratio"] = "0";
        }
        const auto run = [&](const std::string& name, std::vector<float>& losses, std::vector<float>& val_losses) {
            cyxwiz::GraphTrainingJobRequest request;
            request.graph_json = graph.dump();
            request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
            request.dataset_files["tiny_causal_lm_validation"] = validation.string();
            request.epochs_override = 2;
            request.checkpoint_dir_override = (work / name).string();
            cyxwiz::GraphTrainingJobCallbacks callbacks;
            callbacks.on_batch = [&losses](int, int, int, float loss, float) { losses.push_back(loss); };
            callbacks.on_epoch = [&val_losses](int, float, float, float val_loss, float, float) {
                val_losses.push_back(val_loss);
            };
            return cyxwiz::RunGraphTrainingJob(request, callbacks);
        };
        std::vector<float> memory_losses, memory_val, disk_losses, disk_val;
        const auto memory = run("checkpoints_memory", memory_losses, memory_val);
        Check(memory.ok, "in memory: trained" + (memory.error.empty() ? "" : " (" + memory.error + ")"));
        cyxwiz::SetDataBudgetBytes(1);
        const auto on_disk = run("checkpoints_disk", disk_losses, disk_val);
        Check(on_disk.ok, "on disk: trained" + (on_disk.error.empty() ? "" : " (" + on_disk.error + ")"));
        bool same = !memory_losses.empty() && memory_losses.size() == disk_losses.size() &&
                    memory_val.size() == disk_val.size();
        for (size_t i = 0; same && i < memory_losses.size(); ++i)
            same = std::abs(memory_losses[i] - disk_losses[i]) <= 1e-5f * std::max(1.0f, std::abs(memory_losses[i]));
        for (size_t i = 0; same && i < memory_val.size(); ++i)
            same = std::abs(memory_val[i] - disk_val[i]) <= 1e-5f * std::max(1.0f, std::abs(memory_val[i]));
        Check(same, "on disk: the same losses as in memory (" + std::to_string(memory_losses.size()) + " vs " +
                        std::to_string(disk_losses.size()) + " batches)");

        // A test share left to cut from Train is refused on disk, with the reason.
        nlohmann::json share = graph;
        for (auto& node : share["nodes"])
            if (node["parameters"].contains("test_ratio")) node["parameters"]["test_ratio"] = "0.1";
        cyxwiz::GraphTrainingJobRequest share_request;
        share_request.graph_json = share.dump();
        share_request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        share_request.dataset_files["tiny_causal_lm_validation"] = validation.string();
        share_request.epochs_override = 1;
        share_request.checkpoint_dir_override = (work / "checkpoints_disk_share").string();
        const auto refused_share = cyxwiz::RunGraphTrainingJob(share_request);
        Check(!refused_share.ok && Contains(refused_share.error, "test windows"),
              "on disk: a test share without an input is refused (" + refused_share.error + ")");

        // A ratio split cannot cut a windows file on disk: refused with the reason.
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 1;
        request.checkpoint_dir_override = (work / "checkpoints_disk_split").string();
        const auto split = cyxwiz::RunGraphTrainingJob(request);
        Check(!split.ok && Contains(split.error, "ratio splitting"),
              "on disk: a ratio split is refused with the reason (" + split.error + ")");
        cyxwiz::SetDataBudgetBytes(0);
    }

    std::cout << "trains a table kept on disk (TOFIX154)\n";
    {
        const fs::path blobs = work / "blobs.parquet";
        const fs::path blobs_validation = work / "blobs_validation.parquet";
        Check(WriteTabularBlobs(blobs, 600, 1) && WriteTabularBlobs(blobs_validation, 90, 2), "tabular fixtures written");
        const auto run = [&](bool validation_input, const std::string& name) {
            cyxwiz::GraphTrainingJobRequest request;
            request.graph_json = TabularGraph(validation_input).dump();
            request.dataset_files["tabular_blobs"] = blobs.string();
            if (validation_input) request.dataset_files["tabular_blobs_validation"] = blobs_validation.string();
            request.checkpoint_dir_override = (work / name).string();
            return cyxwiz::RunGraphTrainingJob(request);
        };
        const auto memory = run(false, "checkpoints_tabular_memory");
        Check(memory.ok, "tabular in memory: trained" + (memory.error.empty() ? "" : " (" + memory.error + ")"));
        cyxwiz::SetDataBudgetBytes(1);
        const auto on_disk = run(false, "checkpoints_tabular_disk");
        Check(on_disk.ok && on_disk.model != nullptr,
              "tabular on disk: trained" + (on_disk.error.empty() ? "" : " (" + on_disk.error + ")"));
        const auto with_validation = run(true, "checkpoints_tabular_disk_validation");
        Check(with_validation.ok, "tabular on disk with a validation input on disk: trained" +
                                      (with_validation.error.empty() ? "" : " (" + with_validation.error + ")"));
        cyxwiz::SetDataBudgetBytes(0);
    }

    std::cout << "resumes exactly from an epoch checkpoint\n";
    {
        // Seeded, shuffled and with dropout, so randomness and data order
        // both have to replay for the losses to match.
        nlohmann::json graph = LoadTokenWindowGraph(root);
        for (auto& node : graph["nodes"]) {
            auto& params = node["parameters"];
            if (params.contains("dropout")) params["dropout"] = "0.1";
            if (params.contains("batch_size")) {
                params["model_seed"] = "7";
                params["seed"] = "11";
                params["shuffle"] = "true";
                params["batch_size"] = "2";
            }
        }
        struct Step {
            int epoch;
            int batch;
            float loss;
        };
        const auto run = [&](const std::string& root_name, const std::string& resume_from, int stop_at_epoch,
                             std::vector<Step>& steps, int stop_at_batch = 1, int every_steps = 0,
                             bool checkpoint_on_stop = false) {
            cyxwiz::GraphTrainingJobRequest request;
            request.graph_json = graph.dump();
            request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
            request.epochs_override = 3;
            request.checkpoint_dir_override = (work / (root_name + "_best")).string();
            request.resume_checkpoint_root = (work / root_name).string();
            request.resume_from = resume_from;
            request.resume_checkpoint_every_steps = every_steps;
            std::atomic<bool> stop{false};
            cyxwiz::GraphTrainingJobCallbacks callbacks;
            callbacks.on_batch = [&](int epoch, int batch, int, float loss, float) {
                if (stop_at_epoch > 0 && (epoch > stop_at_epoch || (epoch == stop_at_epoch && batch >= stop_at_batch))) {
                    // Hold this batch until the runner's watcher (200 ms poll)
                    // has stopped the executor, so the run cannot finish the
                    // epoch and checkpoint it first.
                    stop = true;
                    std::this_thread::sleep_for(std::chrono::milliseconds(600));
                    return;
                }
                steps.push_back({epoch, batch, loss});
            };
            if (checkpoint_on_stop) {
                callbacks.should_stop_with_checkpoint = [&] { return stop.load(); };
            } else {
                callbacks.should_cancel = [&] { return stop.load(); };
            }
            return cyxwiz::RunGraphTrainingJob(request, callbacks);
        };

        std::vector<Step> uninterrupted;
        const auto full = run("resume_full", "", 0, uninterrupted);
        Check(full.ok, "uninterrupted seeded run" + (full.error.empty() ? "" : " (" + full.error + ")"));
        Check(!full.metrics.last_resume_checkpoint.empty(), "checkpoints are written at epoch ends");

        std::vector<Step> first_part;
        const auto interrupted = run("resume_split", "", 2, first_part);
        Check(interrupted.cancelled, "the second run stops during epoch 2");
        std::vector<Step> resumed_steps;
        const auto resumed = run("resume_split", "latest", 0, resumed_steps);
        Check(resumed.ok, "resumed from the newest checkpoint" + (resumed.error.empty() ? "" : " (" + resumed.error + ")"));
        Check(resumed.metrics.resumed_after_epoch == 1, "resume continues after epoch 1");
        Check(resumed.metrics.resume_exact, "the resume is declared exact (seeded run and data order)");

        std::vector<Step> expected;
        for (const auto& step : uninterrupted) {
            if (step.epoch >= 2) expected.push_back(step);
        }
        bool identical = !expected.empty() && expected.size() == resumed_steps.size();
        for (size_t i = 0; identical && i < expected.size(); ++i) {
            identical = expected[i].epoch == resumed_steps[i].epoch && expected[i].batch == resumed_steps[i].batch &&
                        expected[i].loss == resumed_steps[i].loss;
        }
        Check(identical, "epochs 2-3 after resuming match the uninterrupted run bit for bit (" +
                             std::to_string(resumed_steps.size()) + " batches)");
        Check(resumed.metrics.loss_history.size() == 3, "the loss history spans all three epochs");

        std::cout << "resumes exactly from inside an epoch\n";
        {
            std::vector<Step> every_step;
            const auto baseline = run("resume_steps_full", "", 0, every_step, 1, 1);
            Check(baseline.ok, "uninterrupted run with a checkpoint every step");
            std::vector<Step> before;
            const auto stopped = run("resume_steps_split", "", 2, before, 2, 1);
            Check(stopped.cancelled, "the run stops at epoch 2, batch 2");
            std::vector<Step> after;
            const auto continued = run("resume_steps_split", "latest", 0, after, 1, 1);
            Check(continued.ok, "resumed inside epoch 2" + (continued.error.empty() ? "" : " (" + continued.error + ")"));
            Check(Contains(continued.metrics.last_resume_checkpoint, "resume-") && continued.metrics.resumed_after_epoch == 1,
                  "the newest checkpoint was taken inside epoch 2");
            std::vector<Step> expected_rest;
            for (const auto& step : every_step) {
                if (step.epoch > 2 || (step.epoch == 2 && step.batch >= 2)) expected_rest.push_back(step);
            }
            bool same = !expected_rest.empty() && expected_rest.size() == after.size();
            for (size_t i = 0; same && i < after.size(); ++i) {
                same = expected_rest[i].epoch == after[i].epoch && expected_rest[i].batch == after[i].batch &&
                       expected_rest[i].loss == after[i].loss;
            }
            Check(same, "the rest of epoch 2 and epoch 3 match the uninterrupted run bit for bit (" +
                            std::to_string(after.size()) + " batches)");
            Check(continued.metrics.loss_history == baseline.metrics.loss_history,
                  "epoch losses, including the resumed epoch's total, are identical");
        }

        std::cout << "a stop with a checkpoint keeps the batches already trained\n";
        {
            // A node's reservation ends mid-epoch (TOFIX118 gap 6): no
            // periodic checkpoints, the stop itself writes one after batch 1.
            std::vector<Step> before;
            const auto stopped = run("resume_stop_split", "", 2, before, 1, 0, true);
            Check(stopped.cancelled, "the run stops at epoch 2, batch 1");
            std::vector<Step> after;
            const auto continued = run("resume_stop_split", "latest", 0, after);
            Check(continued.ok, "resumed from the stop's checkpoint" +
                                    (continued.error.empty() ? "" : " (" + continued.error + ")"));
            Check(Contains(continued.metrics.last_resume_checkpoint, "resume-") && !after.empty() &&
                      after.front().epoch == 2 && after.front().batch == 2,
                  "the resume continues at epoch 2, batch 2");
            std::vector<Step> expected_rest;
            for (const auto& step : uninterrupted) {
                if (step.epoch > 2 || (step.epoch == 2 && step.batch >= 2)) expected_rest.push_back(step);
            }
            bool same = !expected_rest.empty() && expected_rest.size() == after.size();
            for (size_t i = 0; same && i < after.size(); ++i) {
                same = expected_rest[i].epoch == after[i].epoch && expected_rest[i].batch == after[i].batch &&
                       expected_rest[i].loss == after[i].loss;
            }
            Check(same, "the rest of epoch 2 and epoch 3 match the uninterrupted run bit for bit (" +
                            std::to_string(after.size()) + " batches)");
        }

        // A checkpoint does not resume into a different graph.
        nlohmann::json other = graph;
        for (auto& node : other["nodes"]) {
            if (node["parameters"].contains("dropout")) node["parameters"]["dropout"] = "0.2";
        }
        cyxwiz::GraphTrainingJobRequest wrong;
        wrong.graph_json = other.dump();
        wrong.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        wrong.epochs_override = 3;
        wrong.checkpoint_dir_override = (work / "resume_wrong_best").string();
        wrong.resume_checkpoint_root = (work / "resume_split").string();
        wrong.resume_from = "latest";
        const auto refused = cyxwiz::RunGraphTrainingJob(wrong);
        Check(!refused.ok && Contains(refused.error, "different graph"),
              "a different graph refuses the checkpoint: " + refused.error);
    }

    std::cout << "measures one step's device memory\n";
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        const auto probe = cyxwiz::ProbeGraphJobMemory(request);
        Check(probe.ok, "probe measured the job" + (probe.error.empty() ? "" : " (" + probe.error + ")"));
        Check(probe.training_bytes > 0, "one step allocates device memory (" + std::to_string(probe.training_bytes) +
                                            " bytes)");
        Check(!probe.backend.empty(), "probe names the route that measured it");

        request.dataset_files["tiny_causal_lm_tokens"] = (work / "missing.parquet").string();
        const auto missing = cyxwiz::ProbeGraphJobMemory(request);
        Check(!missing.ok && missing.failure == cyxwiz::TrainingFailureKind::DataError,
              "a job that cannot run reports why instead of a size");
    }

    std::cout << "failure categories from executor reasons\n";
    {
        using cyxwiz::ClassifyTrainingFailure;
        using cyxwiz::TrainingFailureKind;
        Check(ClassifyTrainingFailure("ArrayFire Exception (Device out of memory:101)") == TrainingFailureKind::OutOfMemory,
              "an ArrayFire allocation failure is out of memory");
        Check(ClassifyTrainingFailure("device_preflight_failed: This route has no retained qualification evidence") ==
                  TrainingFailureKind::DeviceError,
              "a failed device preflight is a device error");
        Check(ClassifyTrainingFailure("CW-T-0501 Training failed: vector subscript") == TrainingFailureKind::Internal,
              "anything else is internal");
        Check(std::string(cyxwiz::TrainingFailureCode(TrainingFailureKind::OutOfMemory)) == "OUT_OF_MEMORY" &&
                  std::string(cyxwiz::TrainingFailureLabel("OUT_OF_MEMORY")) == "Out of memory" &&
                  std::string(cyxwiz::TrainingFailureLabel("TRAINING_FAILED")) == "Training failed",
              "wire codes and labels (older nodes' TRAINING_FAILED still reads)");
    }

    std::cout << "refuses what it cannot run\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] =
            (root / "examples/cyxgraph/text/causal_lm_tiny_token_sequences.csv").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "not supported"), "CSV input refused: " + result.error);
        Check(result.failure == cyxwiz::TrainingFailureKind::Refused, "an unsupported input format is a refusal");
    }
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = (work / "missing.parquet").string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "file not found"), "missing file refused: " + result.error);
        Check(result.failure == cyxwiz::TrainingFailureKind::DataError, "a missing file is a data error");
    }
    {
        // An Arrow IPC file with a schema and no rows: what an Engine shipped
        // when an Apply had registered nothing (E3 2026-10-08).
        const fs::path empty_file = work / "empty.arrow";
        const auto rows = cyxwiz::ArrowDataset::FromParquet(parquet.string(), "tokens");
        auto empty_table = arrow::Table::MakeEmpty(rows->GetArrowTable()->schema()).ValueOrDie();
        Check(cyxwiz::ArrowDataset(empty_table, "empty").ExportFeather(empty_file.string()),
              "schema-only Arrow IPC file written");
        const auto reopened = cyxwiz::ArrowDataset::FromFile(empty_file.string(), "empty");
        Check(reopened && reopened->GetNumRows() == 0, "a schema-only Arrow IPC file reads as 0 rows");
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = LoadTokenWindowGraph(root).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = empty_file.string();
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "empty (0 rows)"), "empty dataset refused: " + result.error);
        Check(result.failure == cyxwiz::TrainingFailureKind::DataError, "an empty dataset is a data error");
    }
    {
        nlohmann::json graph = LoadTokenWindowGraph(root);
        nlohmann::json scaler = NodeNamed(graph, 3);
        scaler["id"] = 99;
        scaler["type"] = static_cast<int>(gui::NodeType::StandardScaler);
        scaler["name"] = "Scale tokens";
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
        // Categories come from the Node Browser's registry (TOFIX140): image
        // transforms compile into the trainer and inspection nodes do not
        // change the data, so neither is a preparation step; the refusal
        // check runs before the (missing) file is opened.
        Check(gui::GetNodeCategoryForType(gui::NodeType::ThresholdFilter) == gui::NodeCategory::DNN,
              "Threshold Filter is a DNN output filter");
        Check(gui::GetNodeCategoryForType(gui::NodeType::Resize) == gui::NodeCategory::Preprocessing,
              "Resize has the registry's category");
        for (const auto type : {gui::NodeType::Resize, gui::NodeType::HorizontalFlip,
                                gui::NodeType::AugmentationPreset, gui::NodeType::DataValidator}) {
            nlohmann::json graph = LoadTokenWindowGraph(root);
            nlohmann::json extra = NodeNamed(graph, 3);
            extra["id"] = 98;
            extra["type"] = static_cast<int>(type);
            extra["name"] = "Extra step";
            extra["parameters"] = nlohmann::json::object();
            graph["nodes"].push_back(extra);
            cyxwiz::GraphTrainingJobRequest request;
            request.graph_json = graph.dump();
            request.dataset_files["tiny_causal_lm_tokens"] = (work / "missing.parquet").string();
            const auto result = cyxwiz::RunGraphTrainingJob(request);
            Check(!Contains(result.error, "data preparation step"),
                  "node type " + std::to_string(static_cast<int>(type)) + " is not a preparation step: " +
                      result.error);
        }
    }
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = "{ not json";
        const auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok && Contains(result.error, "not valid JSON"), "malformed graph refused");
        Check(result.failure == cyxwiz::TrainingFailureKind::Refused, "a malformed graph is a refusal");
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
