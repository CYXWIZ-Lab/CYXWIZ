// Headless graph training (TOFIX118 P2): trains the tracked causal-LM example
// graph through RunGraphTrainingJob, linked against cyxwiz-training-core only
// (what a Server Node links), and checks the fail-closed refusals.
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_training_job.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <cyxwiz/tokenizer.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

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

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

std::string ReadText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// The example's token rows (t0..t7, trailing padding dropped) as token-window
// Parquet in the shape the Engine writes: document_id + token_ids list, window
// bounds, and the frozen vocabulary in the schema metadata.
bool WriteTokenWindows(const fs::path& root, const fs::path& parquet_path, size_t first_row, size_t row_count) {
    std::vector<std::string> words;
    {
        std::ifstream vocab(root / "examples/cyxgraph/text/causal_lm_tiny_vocab.txt");
        std::string word;
        while (std::getline(vocab, word)) {
            if (!word.empty() && word.back() == '\r') word.pop_back();
            if (!word.empty()) words.push_back(word);
        }
    }
    cyxwiz::Vocabulary vocabulary;
    vocabulary.SetVocabulary(words);
    std::ostringstream artifact;
    if (words.empty() || !vocabulary.SaveToStream(artifact)) return false;

    std::ifstream in(root / "examples/cyxgraph/text/causal_lm_tiny_token_sequences.csv");
    std::string line;
    if (!std::getline(in, line)) return false;
    arrow::StringBuilder documents;
    auto values = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder tokens(arrow::default_memory_pool(), values);
    arrow::Int64Builder starts, ends, targets;
    size_t row_index = 0;
    size_t written = 0;
    int64_t offset = 0;
    while (std::getline(in, line) && written < row_count) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (row_index++ < first_row) continue;
        std::stringstream row(line);
        std::string cell;
        std::vector<int64_t> ids;
        for (int i = 0; i < 8 && std::getline(row, cell, ','); ++i) ids.push_back(std::stoll(cell));
        while (!ids.empty() && ids.back() == 0) ids.pop_back();
        for (auto& id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= words.size()) return false;
            id = vocabulary.WordToIndex(words[static_cast<size_t>(id)]);
        }
        const auto length = static_cast<int64_t>(ids.size());
        if (!documents.Append("doc" + std::to_string(row_index)).ok() || !tokens.Append().ok() ||
            !values->AppendValues(ids).ok() || !starts.Append(offset).ok() || !ends.Append(offset + length).ok() ||
            !targets.Append(length - 1).ok()) {
            return false;
        }
        offset += length;
        ++written;
    }
    std::shared_ptr<arrow::Array> document_array, token_array, start_array, end_array, target_array;
    if (!documents.Finish(&document_array).ok() || !tokens.Finish(&token_array).ok() ||
        !starts.Finish(&start_array).ok() || !ends.Finish(&end_array).ok() || !targets.Finish(&target_array).ok()) {
        return false;
    }
    const auto metadata = arrow::key_value_metadata(
        {"cyxwiz.token_windows.version", "cyxwiz.token_windows.context", "cyxwiz.token_windows.vocabulary",
         "cyxwiz.token_windows.tokenizer_type", "cyxwiz.token_windows.lowercase"},
        {"1", "8", artifact.str(), "0", "false"});
    const auto table = arrow::Table::Make(
        arrow::schema({arrow::field("document_id", arrow::utf8()),
                       arrow::field("token_ids", arrow::list(arrow::int64()), false),
                       arrow::field("__token_start", arrow::int64(), false),
                       arrow::field("__token_end", arrow::int64(), false),
                       arrow::field("__valid_targets", arrow::int64(), false)},
                      metadata),
        {document_array, token_array, start_array, end_array, target_array});
    auto out = arrow::io::FileOutputStream::Open(parquet_path.string());
    if (!out.ok()) return false;
    auto properties = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    return parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 1024,
                                      parquet::default_writer_properties(), properties)
        .ok();
}

// The tracked causal-LM example with its Data Input on the token-window
// contract (token_ids, loader-built next-token targets).
nlohmann::json LoadTokenWindowGraph(const fs::path& root) {
    nlohmann::json graph = nlohmann::json::parse(
        ReadText(root / "examples/cyxgraph/text/causal_lm_decoder_generation_experiment.cyxgraph"));
    graph["data_boundary_version"] = 2;
    for (auto& node : graph["nodes"]) {
        if (node.value("type", 0) != static_cast<int>(gui::NodeType::DataInput)) continue;
        auto& params = node["parameters"];
        for (const char* key : {"feature_columns", "label_column", "columns", "sequence_target_columns",
                                "sequence_create_causal_lm_targets", "sequence_length", "sequence_vocab_size"}) {
            params.erase(key);
        }
        params["file_type"] = "parquet";
        params["token_column"] = "token_ids";
        params["sentence_id_column"] = "document_id";
        params["create_causal_lm_targets"] = "true";
        params["model_task"] = "causal_lm";
        params["target_ignore_index"] = "-100";
        params["shape"] = "[8]";
    }
    for (auto& node : graph["nodes"]) {
        if (node.value("type", 0) != static_cast<int>(gui::NodeType::DataLoader)) continue;
        node["parameters"]["create_causal_lm_targets"] = "true";
        node["parameters"]["max_sequence_length"] = "8";
        node["parameters"]["sentence_id_column"] = "document_id";
    }
    // The example leaves its training tail unwired (no loss labels, no loss ->
    // optimizer) and its Sample Rows output unconsumed; wire it as the Engine
    // saves a trainable causal-LM graph (Loader labels -> Loss, Loss -> Adam).
    auto& nodes = graph["nodes"];
    for (auto& node : nodes) {
        // Vocabulary logits per position: TimeDistributed, as in the Engine's
        // causal-LM templates (a plain Dense takes one vector per sample).
        if (node.value("id", 0) == 9) {
            node["type"] = static_cast<int>(gui::NodeType::TimeDistributed);
            node["parameters"] = {{"units", "64"}, {"use_bias", "true"}};
        }
    }
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                               [](const nlohmann::json& node) { return node.value("id", 0) == 2; }),
                nodes.end());
    auto& links = graph["links"];
    links.erase(std::remove_if(links.begin(), links.end(),
                               [](const nlohmann::json& link) {
                                   return link.value("to_node", 0) == 2 ||
                                          (link.value("from_node", 0) == 4 && link.value("to_node", 0) == 10);
                               }),
                links.end());
    links.push_back({{"id", 30}, {"from_node", 4}, {"from_pin", 0}, {"from_pin_index", 1},
                     {"to_node", 11}, {"to_pin", 0}, {"to_pin_index", 1}, {"link_type", 0}});
    links.push_back({{"id", 31}, {"from_node", 11}, {"from_pin", 0}, {"from_pin_index", 0},
                     {"to_node", 12}, {"to_pin", 0}, {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
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
