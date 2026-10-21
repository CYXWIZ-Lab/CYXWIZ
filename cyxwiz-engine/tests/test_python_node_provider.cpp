// The Python node provider (TOFIX125 P4 step 4.4): bundles in the roots
// register their nodes without Python; an unapproved bundle's nodes are
// refused at compile with the reason (D6); an approval, an edit, a removal
// are picked up by Refresh() without a restart; the project's bundle wins
// over a user bundle with the same id. A loss node registers with a loss
// factory (P7).
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"
#include "../src/extension/python/python_node_provider.h"
#include "causal_lm_token_window_fixture.h"

#include <cyxwiz/losses/loss_base.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

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

const fs::path kFixture = fs::path(CYXWIZ_SOURCE_ROOT) / "cyxwiz-engine" / "tests" / "fixtures" /
                          "python_bundles" / "example-acts.cyxnode";
const std::string kSnake = "com.example.acts:Snake";

void CopyBundle(const fs::path& to) {
    fs::create_directories(to);
    fs::copy_file(kFixture / "node.json", to / "node.json", fs::copy_options::overwrite_existing);
    fs::copy_file(kFixture / "node.py", to / "node.py", fs::copy_options::overwrite_existing);
}

// The snake node's compile issues in the token-window graph.
std::string CompileErrorOnSnake(const fs::path& root) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& link : graph["links"]) {
        if (link["from_node"] == 8 && link["to_node"] == 9) {
            link["to_node"] = 50;
            link["to_pin_index"] = 0;
        }
    }
    graph["nodes"].push_back(
        {{"id", 50}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", "Snake"}, {"description", ""},
         {"parameters", nlohmann::json::object()}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", kSnake}, {"version", "1.0.0"}, {"content_hash", ""},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", 50}, {"from_pin", 0}, {"from_pin_index", 0},
                              {"to_node", 9}, {"to_pin", 0}, {"to_pin_index", 0}, {"link_type", 0}});
    cyxwiz::GraphDocument document;
    std::string error;
    if (!cyxwiz::ParseGraphDocument(graph.dump(), document, error)) return "graph not parsed: " + error;
    const auto config = cyxwiz::GraphCompiler{}.Compile(document.nodes, document.links, true);
    std::string issues;
    for (const auto& issue : config.issues) {
        if (issue.node_id == 50 && issue.level == cyxwiz::IssueLevel::Error) issues += issue.message + "\n";
    }
    return issues;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_test_python_node_provider";
    std::error_code ec;
    fs::remove_all(work, ec);
    const fs::path project_nodes = cyxwiz::PythonNodeProvider::ProjectNodesFolder(work / "project");
    const fs::path user_nodes = work / "user_nodes";
    CopyBundle(project_nodes / "example-acts.cyxnode");
    fs::create_directories(user_nodes);

    cyxwiz::PythonNodeApprovals approvals(work / "node_approvals.json");
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    const std::vector<fs::path> roots = {project_nodes, user_nodes};

    std::cout << "a bundle registers its nodes, not approved yet\n";
    {
        cyxwiz::PythonNodeProvider provider(approvals);
        Check(provider.Refresh(roots), "the first refresh scans");
        Check(registry.Has(kSnake) && registry.Has("com.example.acts:Softsign"), "both nodes are registered");
        Check(provider.Bundles().size() == 1 && !provider.Bundles()[0].approved && provider.Issues().empty(),
              "one bundle, not approved, no issues");
        const auto d = registry.Find(kSnake);
        Check(d && d->source == cyxwiz::ExtensionSource::PythonBundle &&
                  d->metadata.properties_editor == cyxwiz::NodePropertiesEditor::Custom,
              "listed as a Python bundle node with the extension Properties");
        const auto refused = registry.TensorFactory(kSnake)->InferOutputShape({8, 32}, {});
        Check(!refused.ok && refused.error == cyxwiz::PythonBundleNotApprovedMessage("com.example.acts"),
              "an unapproved node refuses: " + refused.error);
        const std::string issues = CompileErrorOnSnake(root);
        Check(issues.find("not approved yet") != std::string::npos,
              "compiling a graph with it reports the reason on the node: " + issues);
        Check(!provider.Refresh(roots), "nothing changed: no rescan");

        std::cout << "approving the code makes it trainable, picked up by Refresh\n";
        const auto& bundle = provider.Bundles()[0].bundle;
        Check(approvals.Approve({bundle.id, bundle.content_hash, bundle.folder.string(), "2026-10-17", "approved"}),
              "approve");
        Check(provider.Refresh(roots), "an approval triggers a rescan");
        Check(provider.Bundles()[0].approved, "the bundle is approved");
        const auto shape = registry.TensorFactory(kSnake)->InferOutputShape({8, 32}, {{"initial_a", "2.0"}});
        Check(shape.ok && shape.output_shape == std::vector<size_t>({8, 32}),
              "the Python factory answers the shape (no Python needed)");
        Check(CompileErrorOnSnake(root).empty(), "the graph compiles without an error on the node");

        std::cout << "an edited node.py is noticed; the old code is not used\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        {
            std::ofstream code(project_nodes / "example-acts.cyxnode" / "node.py", std::ios::app | std::ios::binary);
            code << "\n# edited\n";
        }
        Check(provider.Refresh(roots), "an edit triggers a rescan");
        Check(!registry.Has(kSnake), "the nodes are removed: the manifest no longer matches the code");
        Check(provider.Issues().size() == 1 &&
                  provider.Issues()[0].error.find("node.py changed") != std::string::npos,
              "the issue says to bundle again: " +
                  (provider.Issues().empty() ? std::string("none") : provider.Issues()[0].error));

        std::cout << "a bundle written again with other code needs approval again\n";
        CopyBundle(project_nodes / "example-acts.cyxnode");
        Check(provider.Refresh(roots) && registry.Has(kSnake) && provider.Bundles()[0].approved,
              "restored: the approved code is back and runs");

        std::cout << "the project's bundle wins over a user bundle with the same id\n";
        CopyBundle(user_nodes / "copy.cyxnode");
        Check(provider.Refresh(roots), "a new user bundle triggers a rescan");
        Check(provider.Bundles().size() == 1 &&
                  provider.Bundles()[0].bundle.folder == project_nodes / "example-acts.cyxnode",
              "the project's bundle is used");
        Check(provider.Issues().size() == 1 &&
                  provider.Issues()[0].error.find("already loaded") != std::string::npos,
              "the user copy is reported");

        std::cout << "removing the bundle folders removes the nodes\n";
        fs::remove_all(project_nodes / "example-acts.cyxnode");
        fs::remove_all(user_nodes / "copy.cyxnode");
        Check(provider.Refresh(roots) && !registry.Has(kSnake) && provider.Bundles().empty(),
              "the nodes are gone");
        CopyBundle(project_nodes / "example-acts.cyxnode");
        provider.Refresh(roots);
        Check(registry.Has(kSnake), "back for the next check");
    }
    Check(!registry.Has(kSnake), "the provider removes its nodes when it ends");

    std::cout << "a loss node registers with a loss factory\n";
    {
        // The fixture with Snake declared as a loss (registering reads no Python).
        const fs::path folder = project_nodes / "example-losses.cyxnode";
        CopyBundle(folder);
        nlohmann::json manifest;
        {
            std::ifstream in(folder / "node.json");
            manifest = nlohmann::json::parse(in);
        }
        manifest["id"] = "com.example.losses";
        nlohmann::json& snake = manifest["nodes"][1];
        snake["tier"] = "loss";
        snake["role"] = "loss";
        snake["target"] = "categorical";
        snake.erase("shape_rule");
        snake["inputs"] = nlohmann::json::array({{{"name", "Predictions"}, {"type", "Tensor"}},
                                                 {{"name", "Targets"}, {"type", "Labels"}}});
        snake["outputs"] = nlohmann::json::array({{{"name", "Loss"}, {"type", "Loss"}}});
        {
            std::ofstream out(folder / "node.json", std::ios::binary);
            out << manifest.dump(2);
        }
        const std::string loss_id = "com.example.losses:Snake";
        cyxwiz::PythonNodeProvider provider(approvals);
        provider.Refresh(roots);
        const auto d = registry.Find(loss_id);
        Check(d && d->kind == cyxwiz::ExtensionNodeKind::Loss, "registered as a loss node");
        Check(registry.LossFactory(loss_id) && !registry.TensorFactory(loss_id),
              "with a loss factory and no tensor factory");
        // Each call runs before its Check, so the message shows the reason.
        std::string error;
        bool created = registry.LossFactory(loss_id)->CreateLoss({}, error) != nullptr;
        Check(!created && error == cyxwiz::PythonBundleNotApprovedMessage("com.example.losses"),
              "not approved: creating it is refused with the reason: " + error);
        cyxwiz::PythonBundle bundle;
        for (const auto& state : provider.Bundles()) {
            if (state.bundle.id == "com.example.losses") bundle = state.bundle;
        }
        Check(approvals.Approve({bundle.id, bundle.content_hash, bundle.folder.string(), "2026-10-21", "approved"}),
              "approve");
        provider.Refresh(roots);
        error.clear();
        created = registry.LossFactory(loss_id)->CreateLoss({}, error) != nullptr;
        Check(!created && error.find("Python is not running yet") != std::string::npos,
              "approved: the Python factory asks for Python (not running in this test): " + error);
        error.clear();
        created = registry.LossFactory(loss_id)->CreateLoss({{"no_such", "1"}}, error) != nullptr;
        Check(!created && error.find("no_such") != std::string::npos,
              "a setting it does not have is refused without Python: " + error);
        fs::remove_all(folder);
    }

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_python_node_provider: all checks passed\n";
    return 0;
}
