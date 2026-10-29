#pragma once

// Python node bundles (TOFIX125 P4 step 4.1, Part C7): a folder
// <name>.cyxnode/ holding node.json (written by pycyxwiz.nodes.bundle) and
// node.py (the author's code). Reading a bundle never runs its code and never
// starts Python: the manifest alone describes the nodes, so the palette can
// list them before the interpreter exists.

#include "extension_node_contract.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr int kPythonBundleFormat = 1;

struct PythonBundleNode {
    ExtensionNodeDescriptor descriptor;  // source PythonBundle, kind TensorLayer or Loss
    std::string entry_class;             // class in node.py ("Snake")
    std::string tier;                    // "activation" (f, df), "module" or "loss"
    bool same_shape = true;              // shape_rule same_as_input: no Python needed for shapes
};

struct PythonBundle {
    std::filesystem::path folder;  // the .cyxnode folder
    std::string id;                // provider id of its nodes
    std::string content_hash;      // "sha256:<hex>" of node.py
    std::string requires_pycyxwiz; // ">=0.2.0"
    std::vector<PythonBundleNode> nodes;
};

// Reads <folder>/node.json and checks it against <folder>/node.py. False with
// a reason the user can act on: missing files, not JSON, a newer bundle
// format, an invalid id or node name, a node that is not one Tensor in and
// one Tensor out, or node.py changed since the bundle was written.
bool ReadPythonBundle(const std::filesystem::path& folder, PythonBundle& out, std::string& error);

// The *.cyxnode folders directly inside each root: root by root in the
// order given (a bundle of an earlier root wins an id clash), sorted by
// path within a root. Missing roots are skipped.
std::vector<std::filesystem::path> FindPythonBundles(const std::vector<std::filesystem::path>& roots);

// %APPDATA%\cyxwiz\nodes on Windows, ~/.cyxwiz/nodes elsewhere.
std::filesystem::path UserPythonNodesFolder();

}  // namespace cyxwiz
