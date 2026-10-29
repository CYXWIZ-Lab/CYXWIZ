#pragma once

// Registers Python node bundles in the extension registry (TOFIX125 P4
// step 4.4, Part C7/C8). Listing never runs code: node.json is enough.
//
// - Roots: <project>/nodes, the user nodes folder, nodes/ inside plugin
//   folders (the caller passes them).
// - An approved bundle (D6) gets the Python factory; an unapproved one is
//   listed, saved and loaded like any node, but compiling it reports that
//   its code is not approved yet.
// - Refresh() re-reads when a bundle changed on disk (files added, removed,
//   edited): its nodes are replaced with the new version, without a restart.
// - A bundle that cannot be read is reported in Issues(); its nodes are not
//   registered (a graph using them shows them as not installed).

#include "../../core/graph_model.h"
#include "../../core/python_node_approvals.h"
#include "../../core/python_node_bundle.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace cyxwiz {

struct PythonBundleIssue {
    std::filesystem::path folder;
    std::string error;
};

struct PythonBundleState {
    PythonBundle bundle;
    bool approved = false;
};

class PythonNodeProvider {
public:
    explicit PythonNodeProvider(PythonNodeApprovals& approvals);
    ~PythonNodeProvider();

    // Rescans when the roots or anything in their bundles changed since the
    // last call (cheap: file names, sizes and times). True when it rescanned.
    bool Refresh(const std::vector<std::filesystem::path>& roots);
    // Rescans now (after an approval, for example).
    void Rescan();
    // Removes every node this provider registered.
    void Clear();

    const std::vector<PythonBundleState>& Bundles() const { return bundles_; }

    // True when one of these nodes comes from a bundle of this provider
    // whose code is approved: the graph will need Python.
    bool UsesApprovedNode(const std::vector<gui::MLNode>& nodes) const;
    const std::vector<PythonBundleIssue>& Issues() const { return issues_; }

    // "<project>/nodes"
    static std::filesystem::path ProjectNodesFolder(const std::filesystem::path& project_root);

private:
    std::string Stamp(const std::vector<std::filesystem::path>& roots) const;
    void Register();

    PythonNodeApprovals& approvals_;
    std::vector<std::filesystem::path> roots_;
    std::string stamp_;
    bool scanned_ = false;
    std::set<std::string> registered_providers_;
    std::vector<PythonBundleState> bundles_;
    std::vector<PythonBundleIssue> issues_;
};

// The message a not yet approved bundle's nodes give at compile.
std::string PythonBundleNotApprovedMessage(const std::string& bundle_id);

}  // namespace cyxwiz
