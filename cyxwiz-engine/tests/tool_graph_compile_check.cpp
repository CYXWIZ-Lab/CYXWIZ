// Compiles saved .cyxgraph files with the Engine's GraphCompiler, without the
// GUI, and prints the issues plus the resolved training recipe. Used to check
// graphs before a (long) run is launched on another machine.
//
// Usage: cyxwiz-graph-compile-check [--reference known_good.cyxgraph] <graph.cyxgraph> [more ...]
// Exit code: 0 when every graph compiles valid (or, with --reference, has no
// issue beyond the reference's), 1 otherwise.
//
// No datasets are loaded, so checks that need them (named dataset roles,
// generation-preview vocabulary metadata) report errors even for graphs that
// train fine in the Engine. --reference compiles a graph known to train and
// reports only the issues it does not have (node names are normalized so a
// renamed node does not count as a new issue).
#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Graph-only check: no file loaders are registered, so datasets are not read
// (the compiler validates structure, parameters and the training recipe).
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace {

using json = nlohmann::json;

std::string ParameterText(const json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    return value.dump();
}

bool LoadGraph(const std::filesystem::path& path, std::vector<gui::MLNode>& nodes,
               std::vector<gui::NodeLink>& links, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open file";
        return false;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    json root;
    try {
        root = json::parse(buffer.str());
    } catch (const std::exception& e) {
        error = std::string("invalid JSON: ") + e.what();
        return false;
    }
    if (!root.contains("nodes") || !root["nodes"].is_array() || !root.contains("links") || !root["links"].is_array()) {
        error = "graph needs nodes and links arrays";
        return false;
    }
    for (const auto& node_json : root["nodes"]) {
        gui::MLNode node;
        node.id = node_json.value("id", 0);
        node.type = static_cast<gui::NodeType>(node_json.value("type", 0));
        node.name = node_json.value("name", std::string("node"));
        node.category = static_cast<gui::NodeCategory>(node_json.value("category", static_cast<int>(node.category)));
        if (node_json.contains("parameters") && node_json["parameters"].is_object()) {
            for (auto it = node_json["parameters"].begin(); it != node_json["parameters"].end(); ++it) {
                node.parameters[it.key()] = ParameterText(it.value());
            }
        }
        nodes.push_back(std::move(node));
    }
    for (const auto& link_json : root["links"]) {
        gui::NodeLink link;
        link.id = link_json.value("id", 0);
        link.from_node = link_json.value("from_node", 0);
        link.to_node = link_json.value("to_node", 0);
        link.from_pin = link_json.value("from_pin", 0);
        link.to_pin = link_json.value("to_pin", 0);
        links.push_back(std::move(link));
    }
    return true;
}

struct Compiled {
    cyxwiz::TrainingConfiguration config;
    std::vector<std::string> issue_keys;  // level + message, node names normalized
};

std::string ReplaceAll(std::string text, const std::string& from, const std::string& to) {
    if (from.empty()) return text;
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
    return text;
}

const char* LevelName(cyxwiz::IssueLevel level);

bool CompileGraph(const std::filesystem::path& path, Compiled& out, std::string& error) {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    if (!LoadGraph(path, nodes, links, error)) return false;
    cyxwiz::GraphCompiler compiler;
    out.config = compiler.Compile(nodes, links, true);
    for (const auto& issue : out.config.issues) {
        std::string message = issue.message;
        for (const auto& node : nodes) message = ReplaceAll(message, node.name, "<node>");
        out.issue_keys.push_back(std::string(LevelName(issue.level)) + " " + message);
    }
    return true;
}

const char* LevelName(cyxwiz::IssueLevel level) {
    switch (level) {
    case cyxwiz::IssueLevel::Error: return "ERROR";
    case cyxwiz::IssueLevel::Warning: return "WARN";
    default: return "INFO";
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> graphs;
    std::filesystem::path reference_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reference" && i + 1 < argc) {
            reference_path = argv[++i];
        } else {
            graphs.emplace_back(arg);
        }
    }
    if (graphs.empty()) {
        std::cerr << "usage: cyxwiz-graph-compile-check [--reference known_good.cyxgraph] <graph.cyxgraph> [more ...]\n";
        return 2;
    }
    std::multiset<std::string> reference_keys;
    if (!reference_path.empty()) {
        Compiled reference;
        std::string error;
        if (!CompileGraph(reference_path, reference, error)) {
            std::cerr << "reference " << reference_path.string() << ": " << error << "\n";
            return 2;
        }
        reference_keys.insert(reference.issue_keys.begin(), reference.issue_keys.end());
        std::cout << "reference " << reference_path.filename().string() << ": " << reference_keys.size()
                  << " issues treated as expected\n";
    }
    bool all_ok = true;
    for (const auto& path : graphs) {
        std::cout << "== " << path.filename().string() << "\n";
        Compiled compiled;
        std::string error;
        if (!CompileGraph(path, compiled, error)) {
            std::cout << "   load failed: " << error << "\n";
            all_ok = false;
            continue;
        }
        const auto& config = compiled.config;
        auto expected = reference_keys;
        size_t new_errors = 0, reported = 0;
        for (size_t k = 0; k < config.issues.size(); ++k) {
            const auto& issue = config.issues[k];
            const auto match = expected.find(compiled.issue_keys[k]);
            if (!reference_path.empty() && match != expected.end()) {
                expected.erase(match);
                continue;
            }
            ++reported;
            if (issue.level == cyxwiz::IssueLevel::Error) ++new_errors;
            std::cout << "   [" << LevelName(issue.level) << "] "
                      << (issue.node_name.empty() ? std::string("graph") : issue.node_name) << ": " << issue.message
                      << "\n";
        }
        if (!reference_path.empty()) {
            std::cout << "   " << reported << " issue(s) beyond the reference\n";
        }
        std::cout << "   valid=" << (config.is_valid ? "true" : "false") << " layers=" << config.layers.size()
                  << " batch=" << config.batch_size << " accum=" << config.grad_accum_steps
                  << " epochs=" << config.epochs << " optimizer_type=" << static_cast<int>(config.optimizer_type)
                  << " lr=" << config.learning_rate << " schedule=" << config.lr_schedule
                  << " warmup=" << config.warmup_ratio << " min_lr=" << config.min_lr_ratio
                  << " clip=" << config.grad_clip_norm << " weight_decay=" << config.weight_decay
                  << " decay_exclude=" << config.weight_decay_exclude
                  << " checkpoint_dir=" << config.checkpoint_dir << "\n";
        const bool ok = reference_path.empty() ? config.is_valid : new_errors == 0;
        if (!ok) all_ok = false;
    }
    return all_ok ? 0 : 1;
}
