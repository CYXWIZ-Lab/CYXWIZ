#pragma once

// Python node bundles in the Engine's windows (TOFIX125 P4 step 4.5, mockup
// row 6): the Plugin Manager's bundle section, the approval dialog (opened
// from the Plugin Manager, Properties and the Apps menu), and the lookups the
// Info panel, Node Browser and Apps menu use. Drawing only; the words and
// facts come from core/python_bundle_presentation and the extension models.

#include "../core/extension_node_presentation.h"
#include "../core/python_bundle_presentation.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {
class PythonNodeProvider;
}

namespace cyxwiz::python_ui {

// The provider MainWindow owns; nullptr before the first scan.
void SetProvider(PythonNodeProvider* provider);
PythonNodeProvider* Provider();

// Where bundles are looked for (the open project's nodes folder, the user's).
PythonBundleRoots CurrentRoots();

// The loaded bundle with this id, or nullptr.
const PythonBundle* FindBundle(const std::string& bundle_id);
// The approval of a bundle's current code, if any.
std::optional<PythonNodeApproval> ApprovalOf(const PythonBundle& bundle);
// For the Info panel and Properties.
ExtensionPythonState StateOf(const std::string& bundle_id);

// Display names of loaded bundles (id, folder name), for the Node Browser.
std::vector<std::pair<std::string, std::string>> BundleNames();
// Apps menu entries: id, folder name, "2 nodes", "2 nodes · not approved".
struct BundleMenuEntry {
    std::string id;
    std::string name;
    std::string detail;
};
std::vector<BundleMenuEntry> MenuEntries();

// Opens node.py; set by MainWindow (the Script Editor opens it).
void SetOpenCodeHandler(std::function<void(const std::filesystem::path&)> handler);
void OpenCode(const PythonBundle& bundle);

// Asks the user to approve the bundle's code (opens the dialog next frame).
void RequestApproval(const std::string& bundle_id);
// Draws the approval dialog; MainWindow calls it every frame.
void RenderApprovalDialog();

// The Plugin Manager's "Python node bundles" section. `filter` narrows by
// name, id and node names (case-insensitive).
void RenderBundleSection(const std::string& filter);

}  // namespace cyxwiz::python_ui
