#pragma once

// What the Engine shows for Python node bundles (TOFIX125 P4 step 4.5): the
// Plugin Manager's bundle cards and unreadable bundles, the approval dialog,
// and the Script Editor's "Create node from script" dialog. Pure data in and
// out. Approved mockup: canvas row 6,
// https://claude.ai/artifact/EtpzDhx9brMw43wA5Dt9Xy

#include "python_node_approvals.h"
#include "python_node_bundle.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class PythonCheckKind { Passed, Failed, NotChecked };

struct PythonBundleNodeRow {
    std::string name;   // "Softsign 1.0.0"
    std::string check;  // "Gradient passed 4.5e-05"
    PythonCheckKind kind = PythonCheckKind::NotChecked;
};

struct PythonBundleCard {
    std::string key;     // stable: the folder
    std::string title;   // folder name without .cyxnode ("example-acts")
    std::string id;      // "com.example.acts"
    std::string scope;   // "This project", "All your projects", "With a plugin"
    std::string folder;
    std::string adds;    // "Adds: 2 trainable nodes (Activation)"
    std::vector<PythonBundleNodeRow> nodes;
    bool approved = false;
    std::string status;       // "Approved 2026-10-17" / "Not approved yet"
    std::string status_note;
    std::vector<std::pair<std::string, std::string>> details;  // hash, requires, approval
};

// Where a bundle folder sits, for the card's scope label.
struct PythonBundleRoots {
    std::filesystem::path project_nodes;  // empty without a project
    std::filesystem::path user_nodes;
};

PythonBundleCard BuildPythonBundleCard(const PythonBundle& bundle, bool approved,
                                       const PythonNodeApproval* approval, const PythonBundleRoots& roots);

struct PythonBundleIssueCard {
    std::string key;
    std::string title;   // "my-filters"
    std::string status;  // "could not be read"
    std::string folder;
    std::string error;
};

PythonBundleIssueCard BuildPythonBundleIssueCard(const std::filesystem::path& folder, const std::string& error);

// "1 bundle · 1 not read", "No bundles found".
std::string PythonBundleSectionSummary(size_t bundles, size_t issues);
// The empty section: where to put bundles.
std::vector<std::string> PythonBundleEmptyLines(const PythonBundleRoots& roots);

// "Run this Python code?"
struct PythonBundleApprovalView {
    std::string title;
    std::string intro;
    std::vector<std::pair<std::string, std::string>> rows;     // always shown
    std::vector<std::pair<std::string, std::string>> details;  // under Details
    std::string note;
};

PythonBundleApprovalView BuildPythonBundleApproval(const PythonBundle& bundle);

// The record Approve writes. how: "approved" (asked) or "created" (written in the Engine).
PythonNodeApproval MakePythonNodeApproval(const PythonBundle& bundle, const std::string& how,
                                          const std::string& today);

// "2026-10-17", local date.
std::string TodayIsoDate();

// ---- Script Editor: Create node from script ------------------------------------

enum class CreateNodeStage { Inspecting, Found, NoNodes, Failed, Checking, Checked, Writing, Written };

struct CreateNodeFound {
    std::string name;
    std::string version;
    std::string category;
    std::string tier;                 // "activation" / "module"
    std::string pins;                 // "Input -> Output"
    std::vector<std::string> settings; // "initial_a (float, at least 0.01, default 1.0)"
    // After the check:
    bool checked = false;
    PythonCheckKind kind = PythonCheckKind::NotChecked;
    std::string check;  // "Gradient passed 4.5e-05"
};

struct CreateNodeInputs {
    CreateNodeStage stage = CreateNodeStage::Inspecting;
    std::string script_name;    // "my_acts.py"
    std::vector<CreateNodeFound> nodes;
    std::string error;          // Failed: the Python message (and traceback)
    std::string bundle_name;    // as typed
    std::filesystem::path project_nodes;  // empty: no project
    std::string written_folder; // Written
};

struct CreateNodeView {
    std::string title;          // "Create nodes from my_acts.py"
    std::string lead;           // "Found 2 nodes (...)"
    std::string note;
    std::string target;         // where the bundle goes
    std::string name_error;     // bundle name refused
    std::string primary_label;  // "Check gradients" / "Create nodes" / "Close"
    bool primary_enabled = false;
    std::string primary_disabled_reason;
    bool show_name = false;
    bool busy = false;
};

CreateNodeView BuildCreateNodeView(const CreateNodeInputs& in);

// "initial_a (float, at least 0.01, default 1.0)", "mode (one of fast, exact, default fast)".
// validation as pycyxwiz writes it: "0.01-", "-5", "0-1".
std::string CreateNodeSettingText(const std::string& name, const std::string& type, const std::string& default_value,
                                  const std::string& validation, const std::vector<std::string>& choices);

// "my_acts.py" -> "my-acts": lower case, '_' and spaces to '-', others dropped.
std::string SuggestBundleName(const std::string& script_name);
// Empty when acceptable; else why not.
std::string CheckBundleName(const std::string& name);

// One node's check line: "Gradient passed 4.5e-05", "Gradient failed 5.0e-01 in input",
// "Gradient not checked: <reason>".
std::string PythonCheckLine(const std::string& status, double max_relative_error, const std::string& where,
                            const std::string& error, PythonCheckKind& kind);

}  // namespace cyxwiz
