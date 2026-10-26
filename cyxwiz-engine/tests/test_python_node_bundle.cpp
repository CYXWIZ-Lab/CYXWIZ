// Python node bundles (TOFIX125 P4 step 4.1): the reader accepts the bundle
// pycyxwiz.nodes.bundle() writes (fixture example-acts.cyxnode, made in P3)
// and refuses broken ones with a reason, without running any Python.
#include "../src/core/python_node_bundle.h"
#include "../src/core/sha256_digest.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
    std::cout << "ok   " << message << '\n';
}

std::string ReadText(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void WriteText(const fs::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary);
    file << text;
}

fs::path FixtureBundle() {
    fs::path here = fs::path(__FILE__).parent_path();
    return here / "fixtures" / "python_bundles" / "example-acts.cyxnode";
}

// A copy of the fixture whose manifest `edit` changes, in its own folder.
fs::path Variant(const fs::path& work, const std::string& name, const std::function<void(json&)>& edit) {
    const fs::path folder = work / (name + ".cyxnode");
    fs::create_directories(folder);
    fs::copy_file(FixtureBundle() / "node.py", folder / "node.py", fs::copy_options::overwrite_existing);
    json manifest = json::parse(ReadText(FixtureBundle() / "node.json"));
    edit(manifest);
    WriteText(folder / "node.json", manifest.dump(2));
    return folder;
}

void ExpectRefused(const fs::path& folder, const std::string& expected) {
    cyxwiz::PythonBundle bundle;
    std::string error;
    const bool ok = cyxwiz::ReadPythonBundle(folder, bundle, error);
    Check(!ok && error.find(expected) != std::string::npos,
          "refused (" + expected + "): " + (ok ? std::string("accepted") : error));
}

}  // namespace

int main() {
    // --- the bundle P3 wrote ---------------------------------------------------------
    cyxwiz::PythonBundle bundle;
    std::string error;
    Check(cyxwiz::ReadPythonBundle(FixtureBundle(), bundle, error), "the fixture bundle reads: " + error);
    Check(bundle.id == "com.example.acts" && bundle.nodes.size() == 2, "bundle id and two nodes");
    std::string digest;
    Check(cyxwiz::Sha256File(FixtureBundle() / "node.py", digest, error) &&
              bundle.content_hash == "sha256:" + digest,
          "content_hash is node.py's SHA-256");
    Check(bundle.requires_pycyxwiz == ">=0.2.0", "requires pycyxwiz");

    const auto& soft = bundle.nodes[0];
    const auto& snake = bundle.nodes[1];
    Check(soft.descriptor.type_name == "Softsign" && soft.tier == "activation" && soft.entry_class == "Softsign",
          "Softsign: name, tier 1, entry class");
    Check(snake.descriptor.type_name == "Snake" && snake.tier == "module" && snake.same_shape,
          "Snake: name, tier 2, same shape");
    const auto& d = snake.descriptor;
    Check(d.provider_id == "com.example.acts" && d.version == "1.0.0" && d.content_hash == bundle.content_hash,
          "Snake: provider, version, hash");
    Check(d.source == cyxwiz::ExtensionSource::PythonBundle && d.kind == cyxwiz::ExtensionNodeKind::TensorLayer &&
              d.role == cyxwiz::ExtensionLayerRole::Activation,
          "Snake: Python bundle source, trainable, activation role");
    Check(fs::path(d.source_path) == FixtureBundle(), "Snake: source path is the bundle folder");
    Check(d.menu_category == "Activation" && d.metadata.name == "Snake" &&
              d.metadata.brief_description == "x + sin^2(a x) / a, learnable a",
          "Snake: category, name, description");
    Check(d.metadata.inputs.size() == 1 && d.metadata.inputs[0].name == "Input" &&
              d.metadata.outputs.size() == 1 && d.metadata.outputs[0].name == "Output",
          "Snake: one Tensor in, one Tensor out");
    Check(d.metadata.parameters.size() == 1, "Snake: one setting");
    const auto& p = d.metadata.parameters[0];
    Check(p.name == "initial_a" && p.type == "float" && p.default_value == "1.0" && p.validation == "0.01-" &&
              p.description == "Starting value of a",
          "Snake: initial_a as a ParameterDefinition");
    Check(d.verification.status == cyxwiz::ExtensionVerification::Status::Passed &&
              d.verification.max_relative_error > 0.0 && d.verification.tolerance == 0.02 &&
              d.verification.checked_on == "2026-10-17",
          "Snake: the bundle's gradient check is carried");

    // --- finding bundles -------------------------------------------------------------
    const fs::path work = fs::temp_directory_path() / "cyxwiz_test_python_node_bundle";
    fs::remove_all(work);
    fs::create_directories(work / "nodes" / "not-a-bundle");
    const auto found = cyxwiz::FindPythonBundles({FixtureBundle().parent_path(), work / "nodes", work / "missing"});
    Check(found.size() == 1 && found[0] == FixtureBundle(), "FindPythonBundles lists *.cyxnode folders only");
    Check(cyxwiz::UserPythonNodesFolder().filename() == "nodes", "the user nodes folder is .../cyxwiz/nodes");

    // --- refusals ----------------------------------------------------------------------
    ExpectRefused(work / "nodes" / "not-a-bundle", "node.json is missing");
    {
        const fs::path folder = Variant(work, "no-code", [](json&) {});
        fs::remove(folder / "node.py");
        ExpectRefused(folder, "node.py is missing");
    }
    {
        const fs::path folder = Variant(work, "changed", [](json&) {});
        WriteText(folder / "node.py", ReadText(folder / "node.py") + "\n# edited\n");
        ExpectRefused(folder, "node.py changed since the bundle was written");
    }
    {
        const fs::path folder = Variant(work, "not-json", [](json&) {});
        WriteText(folder / "node.json", "{ not json");
        ExpectRefused(folder, "node.json is not valid JSON");
    }
    ExpectRefused(Variant(work, "newer", [](json& m) { m["bundle_format"] = 2; }), "newer CyxWiz");
    ExpectRefused(Variant(work, "bad-id", [](json& m) { m["id"] = "Com.Example"; }), "Com.Example");
    ExpectRefused(Variant(work, "no-nodes", [](json& m) { m["nodes"] = json::array(); }), "lists no nodes");
    ExpectRefused(Variant(work, "two-inputs", [](json& m) {
                      m["nodes"][1]["inputs"].push_back({{"name", "Other"}, {"type", "Tensor"}});
                  }),
                  "node Snake: needs exactly one input pin");
    ExpectRefused(Variant(work, "dataset-pin", [](json& m) { m["nodes"][1]["outputs"][0]["type"] = "Dataset"; }),
                  "pins must be Tensor pins");
    ExpectRefused(Variant(work, "entry", [](json& m) { m["nodes"][1]["entry"] = "other.py:Snake"; }),
                  "entry must be node.py:<Class>");
    ExpectRefused(Variant(work, "tier", [](json& m) { m["nodes"][1]["tier"] = "autograd"; }),
                  "tier must be activation, module or loss");
    ExpectRefused(Variant(work, "shape", [](json& m) { m["nodes"][1]["shape_rule"] = "guess"; }),
                  "shape_rule must be");
    ExpectRefused(Variant(work, "param-type", [](json& m) { m["nodes"][1]["parameters"][0]["type"] = "tensor"; }),
                  "setting 'initial_a' has the unknown type 'tensor'");

    // A setting with choices is an enum.
    {
        const fs::path folder = Variant(work, "choices", [](json& m) {
            m["nodes"][1]["parameters"][0] = {{"name", "mode"}, {"type", "string"}, {"default", "fast"},
                                              {"validation", "fast|exact"}, {"description", ""},
                                              {"choices", {"fast", "exact"}}};
        });
        cyxwiz::PythonBundle with_choices;
        Check(cyxwiz::ReadPythonBundle(folder, with_choices, error), "a bundle with a choice setting reads");
        const auto& mode = with_choices.nodes[1].descriptor.metadata.parameters[0];
        Check(mode.type == "enum" && mode.enum_values == std::vector<std::string>{"fast", "exact"},
              "a setting with choices becomes an enum");
    }

    // A loss node (P7): tier and role loss, the built-in losses' pins, a target.
    const auto as_loss = [](json& m) {
        json& n = m["nodes"][1];
        n["tier"] = "loss";
        n["role"] = "loss";
        n["target"] = "continuous";
        n.erase("shape_rule");
        n["inputs"] = json::array({{{"name", "Predictions"}, {"type", "Tensor"}}, {{"name", "Targets"}, {"type", "Labels"}}});
        n["outputs"] = json::array({{{"name", "Loss"}, {"type", "Loss"}}});
    };
    {
        cyxwiz::PythonBundle with_loss;
        const bool read = cyxwiz::ReadPythonBundle(Variant(work, "loss", as_loss), with_loss, error);
        Check(read, "a loss node reads" + (error.empty() ? std::string() : ": " + error));
        const auto& loss_node = with_loss.nodes[1].descriptor;
        Check(loss_node.kind == cyxwiz::ExtensionNodeKind::Loss && with_loss.nodes[1].tier == "loss" &&
                  loss_node.loss_target == cyxwiz::ExtensionLossTarget::Continuous,
              "it is a Loss node with continuous targets");
        Check(loss_node.metadata.inputs.size() == 2 && loss_node.metadata.inputs[1].type == cyxwiz::PinType::Labels &&
                  loss_node.metadata.outputs.size() == 1 &&
                  loss_node.metadata.outputs[0].type == cyxwiz::PinType::Loss,
              "with the built-in losses' pins");
        Check(with_loss.nodes[0].descriptor.kind == cyxwiz::ExtensionNodeKind::TensorLayer,
              "the other node of the bundle stays a layer");
    }
    ExpectRefused(Variant(work, "loss-target", [&](json& m) { as_loss(m); m["nodes"][1]["target"] = "ranked"; }),
                  "target must be categorical or continuous");
    ExpectRefused(Variant(work, "loss-pins", [&](json& m) { as_loss(m); m["nodes"][1]["inputs"][1]["type"] = "Tensor"; }),
                  "a loss has the pins Predictions (Tensor), Targets (Labels) -> Loss (Loss)");
    ExpectRefused(Variant(work, "loss-tier", [&](json& m) { as_loss(m); m["nodes"][1]["tier"] = "module"; }),
                  "a loss has tier and role loss");

    fs::remove_all(work);
    std::cout << "test_python_node_bundle: all checks passed\n";
    return 0;
}
