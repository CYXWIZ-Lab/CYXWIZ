#include "python_node_bundle.h"

#include "sha256_digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

namespace cyxwiz {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string Text(const json& object, const char* key) {
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : std::string();
}

bool ReadFile(const fs::path& path, std::string& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::ostringstream text;
    text << file.rdbuf();
    out = text.str();
    return true;
}

// One Tensor in and one Tensor out (decision D5).
bool ReadPins(const json& node, const char* key, std::vector<PortDefinition>& pins, std::string& error) {
    const auto it = node.find(key);
    if (it == node.end() || !it->is_array() || it->size() != 1 || !(*it)[0].is_object()) {
        error = std::string("needs exactly one ") + (std::string(key) == "inputs" ? "input" : "output") +
                " pin";
        return false;
    }
    const json& pin = (*it)[0];
    if (Text(pin, "type") != "Tensor") {
        error = "its pins must be Tensor pins";
        return false;
    }
    const std::string name = Text(pin, "name");
    if (name.empty()) {
        error = "a pin has no name";
        return false;
    }
    pins.emplace_back(name, PinType::Tensor);
    return true;
}

// A loss node's pins: exactly the built-in losses' (TOFIX125 P7).
bool ReadLossPins(const json& node, NodeMetadata& metadata, std::string& error) {
    const json inputs = json::array({{{"name", "Predictions"}, {"type", "Tensor"}},
                                     {{"name", "Targets"}, {"type", "Labels"}}});
    const json outputs = json::array({{{"name", "Loss"}, {"type", "Loss"}}});
    const auto pins = [](const json& list) {
        json plain = json::array();
        for (const auto& pin : list) plain.push_back({{"name", Text(pin, "name")}, {"type", Text(pin, "type")}});
        return plain;
    };
    if (pins(node.value("inputs", json::array())) != inputs || pins(node.value("outputs", json::array())) != outputs) {
        error = "a loss has the pins Predictions (Tensor), Targets (Labels) -> Loss (Loss)";
        return false;
    }
    SetExtensionLossPorts(metadata);
    return true;
}

bool ReadParameters(const json& node, std::vector<ParameterDefinition>& out, std::string& error) {
    const auto it = node.find("parameters");
    if (it == node.end()) return true;
    if (!it->is_array()) {
        error = "'parameters' must be a list";
        return false;
    }
    for (const json& p : *it) {
        const std::string name = p.is_object() ? Text(p, "name") : std::string();
        if (name.empty()) {
            error = "a setting has no name";
            return false;
        }
        std::string type = Text(p, "type");
        if (type != "float" && type != "int" && type != "bool" && type != "string") {
            error = "setting '" + name + "' has the unknown type '" + type + "'";
            return false;
        }
        std::vector<std::string> choices;
        if (const auto c = p.find("choices"); c != p.end() && c->is_array()) {
            for (const json& choice : *c) {
                if (choice.is_string()) choices.push_back(choice.get<std::string>());
            }
        }
        if (!choices.empty()) type = "enum";
        out.emplace_back(name, type, Text(p, "default"), Text(p, "description"), choices,
                         choices.empty() ? Text(p, "validation") : std::string());
    }
    return true;
}

ExtensionVerification ReadVerification(const json& node) {
    ExtensionVerification v;
    const auto it = node.find("verification");
    if (it == node.end() || !it->is_object()) return v;
    const std::string status = Text(*it, "status");
    v.status = status == "passed"   ? ExtensionVerification::Status::Passed
               : status == "failed" ? ExtensionVerification::Status::Failed
                                    : ExtensionVerification::Status::NotRun;
    v.checked_on = Text(*it, "checked_on");
    v.tolerance = it->value("tolerance", 0.0);
    v.max_relative_error = it->value("max_relative_error", 0.0);
    v.note = !Text(*it, "error").empty() ? Text(*it, "error") : Text(*it, "where");
    return v;
}

}  // namespace

bool ReadPythonBundle(const fs::path& folder, PythonBundle& out, std::string& error) {
    out = PythonBundle{};
    out.folder = folder;
    const fs::path manifest_path = folder / "node.json";
    const fs::path code_path = folder / "node.py";
    std::string manifest_text;
    if (!ReadFile(manifest_path, manifest_text)) {
        error = "node.json is missing; write the bundle with pycyxwiz.nodes.bundle()";
        return false;
    }
    std::error_code ec;
    if (!fs::is_regular_file(code_path, ec)) {
        error = "node.py is missing; write the bundle again with pycyxwiz.nodes.bundle()";
        return false;
    }

    json manifest;
    try {
        manifest = json::parse(manifest_text);
    } catch (const json::exception& e) {
        error = std::string("node.json is not valid JSON: ") + e.what();
        return false;
    }
    if (!manifest.is_object()) {
        error = "node.json must hold an object";
        return false;
    }
    const int format = manifest.value("bundle_format", 0);
    if (format > kPythonBundleFormat) {
        error = "the bundle was written for a newer CyxWiz (bundle format " + std::to_string(format) +
                "); update CyxWiz";
        return false;
    }
    if (format != kPythonBundleFormat) {
        error = "node.json has no supported bundle_format";
        return false;
    }

    out.id = Text(manifest, "id");
    out.content_hash = Text(manifest, "content_hash");
    if (const auto r = manifest.find("requires"); r != manifest.end() && r->is_object()) {
        out.requires_pycyxwiz = Text(*r, "pycyxwiz");
    }

    // The code must be the code the manifest describes (D2: the graph and
    // approvals (D6) refer to this hash).
    std::string digest;
    std::string hash_error;
    if (!Sha256File(code_path, digest, hash_error)) {
        error = "node.py could not be read: " + hash_error;
        return false;
    }
    if (out.content_hash != "sha256:" + digest) {
        error = "node.py changed since the bundle was written; bundle it again with pycyxwiz.nodes.bundle()";
        return false;
    }

    const auto nodes = manifest.find("nodes");
    if (nodes == manifest.end() || !nodes->is_array() || nodes->empty()) {
        error = "node.json lists no nodes";
        return false;
    }
    for (const json& node : *nodes) {
        if (!node.is_object()) {
            error = "a node entry in node.json is not an object";
            return false;
        }
        PythonBundleNode entry;
        ExtensionNodeDescriptor& d = entry.descriptor;
        d.provider_id = out.id;
        d.type_name = Text(node, "name");
        std::string id_error;
        if (!ValidateExtensionTypeId(d.provider_id, d.type_name, id_error)) {
            error = id_error;
            return false;
        }
        const std::string where = "node " + d.type_name + ": ";
        const std::string target = Text(node, "entry");
        if (target.rfind("node.py:", 0) != 0 || target.size() == 8) {
            error = where + "entry must be node.py:<Class>";
            return false;
        }
        entry.entry_class = target.substr(8);
        entry.tier = Text(node, "tier");
        if (entry.tier != "activation" && entry.tier != "module" && entry.tier != "loss") {
            error = where + "tier must be activation, module or loss";
            return false;
        }
        const std::string role = Text(node, "role");
        if (role != "activation" && role != "layer" && role != "loss") {
            error = where + "role must be activation, layer or loss";
            return false;
        }
        const bool loss = role == "loss";
        if (loss != (entry.tier == "loss")) {
            error = where + "a loss has tier and role loss";
            return false;
        }
        if (loss) {
            const std::string loss_target = Text(node, "target");
            if (loss_target != "categorical" && loss_target != "continuous") {
                error = where + "target must be categorical or continuous";
                return false;
            }
            d.loss_target = loss_target == "continuous" ? ExtensionLossTarget::Continuous : ExtensionLossTarget::Categorical;
        } else {
            const std::string shape_rule = Text(node, "shape_rule");
            if (shape_rule != "same_as_input" && shape_rule != "python") {
                error = where + "shape_rule must be same_as_input or python";
                return false;
            }
            entry.same_shape = shape_rule == "same_as_input";
        }

        d.version = Text(node, "version");
        d.content_hash = out.content_hash;
        d.source = ExtensionSource::PythonBundle;
        d.source_path = folder.string();
        d.kind = loss ? ExtensionNodeKind::Loss : ExtensionNodeKind::TensorLayer;
        d.role = role == "activation" ? ExtensionLayerRole::Activation : ExtensionLayerRole::Layer;
        d.menu_category = Text(node, "category");
        const std::string display = Text(node, "display_name");
        d.metadata.name = display.empty() ? d.type_name : display;
        d.metadata.brief_description = Text(node, "description");
        std::string pin_error;
        const bool pins_read = loss ? ReadLossPins(node, d.metadata, pin_error)
                                    : ReadPins(node, "inputs", d.metadata.inputs, pin_error) &&
                                          ReadPins(node, "outputs", d.metadata.outputs, pin_error);
        if (!pins_read || !ReadParameters(node, d.metadata.parameters, pin_error)) {
            error = where + pin_error;
            return false;
        }
        d.verification = ReadVerification(node);
        out.nodes.push_back(std::move(entry));
    }
    return true;
}

std::vector<fs::path> FindPythonBundles(const std::vector<fs::path>& roots) {
    std::vector<fs::path> found;
    for (const auto& root : roots) {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) continue;
        for (const auto& item : fs::directory_iterator(root, ec)) {
            if (item.is_directory(ec) && item.path().extension() == ".cyxnode") found.push_back(item.path());
        }
    }
    std::sort(found.begin(), found.end());
    return found;
}

fs::path UserPythonNodesFolder() {
#ifdef _WIN32
    if (const char* appdata = std::getenv("APPDATA")) return fs::path(appdata) / "cyxwiz" / "nodes";
#else
    if (const char* home = std::getenv("HOME")) return fs::path(home) / ".cyxwiz" / "nodes";
#endif
    return "nodes";
}

}  // namespace cyxwiz
