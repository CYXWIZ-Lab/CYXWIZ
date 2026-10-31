// The shipped examples compile: the text classifiers (TOFIX152 finding) and
// the image ResNet (TOFIX178 1.8). The graphs are loaded the way the canvas
// loads a starter: each node made by CreateGraphNode, so its pins are the
// registry's, then linked by pin index, and compiled by GraphCompiler.
// Before, every test built its own pins, so a stale pin type (Text
// Tokenizer's Tensor output feeding Data Split) shipped and no example with a
// tokenizer compiled.

#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_node_factory.h"
#include "../src/gui/node_type_import_registry.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

fs::path FindRepoRoot() {
    for (auto dir = fs::current_path(); !dir.empty(); dir = dir.parent_path()) {
        if (fs::exists(dir / "examples" / "cyxgraph") && fs::exists(dir / "cyxwiz-engine" / "CMakeLists.txt"))
            return dir;
        if (dir.parent_path() == dir) break;
    }
    return fs::current_path();
}

// "$name" -> the template parameter's default.
std::string Substitute(std::string value, const std::map<std::string, std::string>& defaults) {
    for (const auto& [name, default_value] : defaults) {
        const std::string token = "$" + name;
        for (size_t at = value.find(token); at != std::string::npos; at = value.find(token, at + default_value.size()))
            value.replace(at, token.size(), default_value);
    }
    return value;
}

bool LoadTemplate(const fs::path& path, std::vector<gui::MLNode>& nodes, std::vector<gui::NodeLink>& links) {
    std::ifstream in(path);
    if (!in) return false;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json document = json::parse(text);
    if (!document.contains("template")) {
        // A saved graph: read by the editor's own document loader.
        cyxwiz::GraphDocument graph;
        std::string error;
        const bool parsed = cyxwiz::ParseGraphDocument(text, graph, error);
        Check(parsed, path.filename().string() + ": " + error);
        nodes = std::move(graph.nodes);
        links = std::move(graph.links);
        return parsed;
    }
    std::map<std::string, std::string> defaults;
    for (const auto& parameter : document.value("parameters", json::array()))
        defaults[parameter.at("name").get<std::string>()] = parameter.value("default_value", std::string());
    const json& graph = document.at("template");
    int next_node_id = 1, next_pin_id = 1, next_link_id = 1;
    std::map<std::string, size_t> index_of;
    for (const auto& node_json : graph.at("nodes")) {
        const std::string type_name = node_json.at("type").get<std::string>();
        const auto type = gui::ResolveNodeTypeImportName(type_name);
        Check(type.has_value(), path.filename().string() + ": unknown node type " + type_name);
        if (!type) return false;
        gui::MLNode node = gui::CreateGraphNode(*type, node_json.value("name", type_name), next_node_id, next_pin_id);
        const json params = node_json.value("params", json::object());
        for (const auto& [key, value] : params.items())
            node.parameters[key] = Substitute(value.is_string() ? value.get<std::string>() : value.dump(), defaults);
        index_of[node_json.at("id").get<std::string>()] = nodes.size();
        nodes.push_back(std::move(node));
    }
    for (const auto& link_json : graph.at("links")) {
        const auto& from = nodes.at(index_of.at(link_json.at("from").get<std::string>()));
        const auto& to = nodes.at(index_of.at(link_json.at("to").get<std::string>()));
        const size_t from_pin = link_json.value("from_pin", 0), to_pin = link_json.value("to_pin", 0);
        Check(from_pin < from.outputs.size() && to_pin < to.inputs.size(),
              path.filename().string() + ": link " + from.name + " -> " + to.name + " names a missing pin");
        if (from_pin >= from.outputs.size() || to_pin >= to.inputs.size()) return false;
        links.push_back({next_link_id++, from.id, from.outputs[from_pin].id, to.id, to.inputs[to_pin].id});
    }
    return true;
}

}  // namespace

int main() {
    // What the Engine's data loaders tell the compiler about a Data Input's
    // category (gui/loaders/data_loader.cpp installs these in the Engine).
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        return category == "text" || category == "image" || category == "audio";
    };
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (category == "text") return cyxwiz::PreprocessingDomain::Text;
        if (category == "image") return cyxwiz::PreprocessingDomain::Image;
        if (category == "audio") return cyxwiz::PreprocessingDomain::Audio;
        if (category == "timeseries") return cyxwiz::PreprocessingDomain::TimeSeries;
        return cyxwiz::PreprocessingDomain::Tabular;
    };
    cyxwiz::SetGraphCompilerDatasetHooks(std::move(hooks));

    const fs::path root = FindRepoRoot() / "examples" / "cyxgraph";
    const std::vector<fs::path> graphs = {
        root / "Sentiment analysis" / "sentiment_analysis_gru_classifier.cyxgraph",
        root / "Sentiment analysis" / "sentiment_analysis_lstm_classifier.cyxgraph",
        root / "Sentiment analysis" / "sentiment_analysis_lstm_word_vectors.cyxgraph",
        root / "Sentiment analysis" / "sentiment_analysis_tfidf_mlp_classifier.cyxgraph",
        root / "Sentiment analysis" / "sentiment_analysis_tfidf_mlp_classifier_unigram_bigram_candidate.cyxgraph",
        root / "mental_health_sentiment_classifier.cyxgraph",
        root / "mental_health_sentiment_classifier_v2.cyxgraph",
        root / "call_center_sentiment.cyxgraph",
        root / "code_lang_feedback" / "codefeedback_lang_classifier_char.cyxgraph",
        root / "text" / "test_01_sentiment_basic.cyxgraph",
        root / "text" / "test_02_sentiment_lstm.cyxgraph",
        root / "text" / "test_03_sentiment_lstm_mini.cyxgraph",
        root / "text" / "test_04_sentiment_gru_mini.cyxgraph",
        root / "text" / "test_05_sentiment_transformer_mini.cyxgraph",
        root / "cats_dogs_resnet_scratch.cyxgraph",
    };
    for (const auto& path : graphs) {
        std::vector<gui::MLNode> nodes;
        std::vector<gui::NodeLink> links;
        const std::string name = path.filename().string();
        try {
            if (!LoadTemplate(path, nodes, links)) {
                Check(false, name + ": not loaded");
                continue;
            }
        } catch (const std::exception& e) {
            Check(false, name + ": not loaded: " + e.what());
            continue;
        }
        // The data is not loaded here; everything else must hold.
        cyxwiz::TrainingConfiguration config;
        try {
            config = cyxwiz::GraphCompiler{}.Compile(nodes, links, /*allow_unloaded_data=*/true);
        } catch (const std::exception& e) {
            Check(false, name + ": compile threw: " + e.what());
            continue;
        }
        for (const auto& issue : config.issues) {
            Check(issue.level != cyxwiz::IssueLevel::Error,
                  name + ": [" + issue.node_name + "] " + issue.message + " " + issue.error_code);
        }
        if (config.is_valid) std::cout << name << ": compiles (" << config.layers.size() << " layers)\n";
    }
    if (failures > 0) return 1;
    std::cout << "example graphs compile\n";
    return 0;
}
