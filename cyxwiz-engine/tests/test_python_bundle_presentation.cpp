// Python node bundles in the GUI (TOFIX125 P4 step 4.5, mockup row 6): the
// Plugin Manager's bundle card and unreadable bundle, the approval dialog
// and the Create node from script dialog, from the P3 fixture bundle.
#include "../src/core/python_bundle_presentation.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::string Find(const std::vector<std::pair<std::string, std::string>>& rows, const std::string& key) {
    for (const auto& [k, v] : rows) {
        if (k == key) return v;
    }
    return "(none)";
}

}  // namespace

int main() {
    const fs::path fixture = fs::path(__FILE__).parent_path() / "fixtures" / "python_bundles" / "example-acts.cyxnode";
    cyxwiz::PythonBundle bundle;
    std::string error;
    if (!cyxwiz::ReadPythonBundle(fixture, bundle, error)) {
        std::cerr << "fixture: " << error << "\n";
        return EXIT_FAILURE;
    }
    const cyxwiz::PythonBundleRoots roots{fixture.parent_path(), fs::path("C:/Users/u/AppData/Roaming/cyxwiz/nodes")};

    std::cout << "bundle card, not approved\n";
    const auto card = cyxwiz::BuildPythonBundleCard(bundle, false, nullptr, roots);
    Check(card.title == "example-acts" && card.id == "com.example.acts", "title is the folder name, then the id");
    Check(card.scope == "This project", "scope: the project's nodes folder");
    Check(card.adds == "Adds: 2 trainable nodes (Activation)", "adds line: " + card.adds);
    Check(card.nodes.size() == 2 && card.nodes[0].name == "Softsign 1.0.0" &&
              card.nodes[0].check == "Gradient passed 4.5e-05" &&
              card.nodes[0].kind == cyxwiz::PythonCheckKind::Passed,
          "node row: " + (card.nodes.empty() ? std::string() : card.nodes[0].name + " / " + card.nodes[0].check));
    Check(card.nodes.size() == 2 && card.nodes[1].check == "Gradient passed 1.3e-03", "Snake's check");
    Check(!card.approved && card.status == "Not approved yet" &&
              card.status_note.find("does not run until you approve it") != std::string::npos,
          "status and note");
    Check(Find(card.details, "Code sha256") ==
              "bcffe763ca1defc79c152f8221da10377c1c2c51d38913d253bb44ecfb247d0d",
          "details: the full code hash");
    Check(Find(card.details, "Requires") == "pycyxwiz 0.2.0 or newer", "details: requires");

    std::cout << "bundle card, approved\n";
    const cyxwiz::PythonNodeApproval approval =
        cyxwiz::MakePythonNodeApproval(bundle, "approved", "2026-10-17");
    Check(approval.bundle_id == "com.example.acts" && approval.content_hash == bundle.content_hash &&
              approval.approved_on == "2026-10-17" && approval.how == "approved",
          "the approval record");
    const auto approved = cyxwiz::BuildPythonBundleCard(bundle, true, &approval, roots);
    Check(approved.approved && approved.status == "Approved 2026-10-17" &&
              approved.status_note == "Its nodes build and train. Editing node.py asks again.",
          "approved status: " + approved.status);
    Check(Find(approved.details, "Approval") == "Approved by you", "details: who approved");
    const auto created = cyxwiz::MakePythonNodeApproval(bundle, "created", "2026-10-17");
    Check(Find(cyxwiz::BuildPythonBundleCard(bundle, true, &created, roots).details, "Approval") ==
              "Created in the Engine",
          "details: created in the Engine");
    Check(cyxwiz::BuildPythonBundleCard(bundle, false, nullptr, {fs::path("D:/other"), fixture.parent_path()}).scope ==
              "All your projects",
          "scope: the user nodes folder");
    Check(cyxwiz::BuildPythonBundleCard(bundle, false, nullptr, {}).scope == "With a plugin", "scope: elsewhere");

    std::cout << "unreadable bundle, summary, empty section\n";
    const auto issue = cyxwiz::BuildPythonBundleIssueCard(
        fs::path("C:/Users/u/AppData/Roaming/cyxwiz/nodes/my-filters.cyxnode"),
        "node.py changed since the bundle was written; bundle it again with pycyxwiz.nodes.bundle()");
    Check(issue.title == "my-filters" && issue.status == "could not be read" && issue.error.find("bundle it again") != std::string::npos,
          "issue card");
    Check(cyxwiz::PythonBundleSectionSummary(1, 1) == "1 bundle \xC2\xB7 1 not read", "summary with an issue");
    Check(cyxwiz::PythonBundleSectionSummary(2, 0) == "2 bundles", "summary");
    Check(cyxwiz::PythonBundleSectionSummary(0, 0) == "No bundles found", "summary, none");
    const auto empty = cyxwiz::PythonBundleEmptyLines(roots);
    Check(empty.size() == 3 && empty[1].find("(this project)") != std::string::npos, "empty section names the folders");

    std::cout << "approval dialog\n";
    const auto dialog = cyxwiz::BuildPythonBundleApproval(bundle);
    Check(dialog.title == "Run this Python code?", "title");
    Check(dialog.intro.find("adds 2 nodes") != std::string::npos &&
              dialog.intro.find("your user's rights") != std::string::npos,
          "intro: what it adds and the risk");
    Check(Find(dialog.rows, "Bundle") == "com.example.acts" && Find(dialog.rows, "Nodes") == "Softsign 1.0.0, Snake 1.0.0",
          "rows: bundle and nodes");
    Check(Find(dialog.rows, "Gradient check") == "Passed when bundled (2026-10-17)", "rows: the bundle's check");
    Check(Find(dialog.details, "Code sha256").size() == 64, "details: the full hash");
    Check(dialog.note.find("If node.py changes") != std::string::npos, "note: per code");

    std::cout << "create node from script\n";
    Check(cyxwiz::SuggestBundleName("My_Acts.py") == "my-acts", "suggested name");
    Check(cyxwiz::SuggestBundleName("___.py") == "my-nodes", "suggested name, nothing usable");
    Check(cyxwiz::CheckBundleName("my-acts").empty() && !cyxwiz::CheckBundleName("My acts").empty() &&
              !cyxwiz::CheckBundleName("").empty(),
          "bundle name rules");
    cyxwiz::CreateNodeInputs in;
    in.script_name = "my_acts.py";
    in.bundle_name = "my-acts";
    in.project_nodes = fs::path("D:/tmp/t125/pyproj/nodes");
    in.stage = cyxwiz::CreateNodeStage::Inspecting;
    auto view = cyxwiz::BuildCreateNodeView(in);
    Check(view.busy && !view.primary_enabled && view.title == "Create nodes from my_acts.py", "inspecting");
    in.stage = cyxwiz::CreateNodeStage::NoNodes;
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.lead.find("mark a class with @nodes.node, @nodes.activation or @nodes.loss") != std::string::npos &&
              view.primary_label == "Close",
          "no nodes found");
    in.stage = cyxwiz::CreateNodeStage::Found;
    in.nodes = {{"Softsign", "1.0.0", "Activation", "activation", "Input -> Output", {}},
                {"Snake", "1.0.0", "Activation", "module", "Input -> Output", {"initial_a (float, at least 0.01, default 1.0)"}}};
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.lead.find("Found 2 nodes") == 0 && view.primary_label == "Check gradients" && view.primary_enabled,
          "found: check first");
    Check(view.target == (fs::path("D:/tmp/t125/pyproj/nodes") / "my-acts.cyxnode").string(), "target folder");
    in.stage = cyxwiz::CreateNodeStage::Checked;
    in.nodes[0].kind = in.nodes[1].kind = cyxwiz::PythonCheckKind::Passed;
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.primary_label == "Create nodes" && view.primary_enabled && view.note == "All pass.", "checked: create");
    in.nodes[1].kind = cyxwiz::PythonCheckKind::Failed;
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.note.find("still created") != std::string::npos && view.primary_enabled, "a failed check still creates");
    in.bundle_name = "Bad Name";
    view = cyxwiz::BuildCreateNodeView(in);
    Check(!view.primary_enabled && view.primary_disabled_reason == cyxwiz::CheckBundleName("Bad Name"),
          "a bad name blocks Create with the reason");
    in.bundle_name = "my-acts";
    in.project_nodes.clear();
    view = cyxwiz::BuildCreateNodeView(in);
    Check(!view.primary_enabled && view.primary_disabled_reason.find("Open a project") == 0, "no project: Create off");
    in.stage = cyxwiz::CreateNodeStage::Written;
    in.written_folder = "D:/tmp/t125/pyproj/nodes/my-acts.cyxnode";
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.lead == "Created D:/tmp/t125/pyproj/nodes/my-acts.cyxnode" &&
              view.note.find("approved for you") != std::string::npos && view.primary_label == "Close",
          "written");
    in.stage = cyxwiz::CreateNodeStage::Failed;
    in.error = "NameError: name 'x' is not defined";
    view = cyxwiz::BuildCreateNodeView(in);
    Check(view.note == in.error && view.lead.find("Python stopped") == 0, "a Python error is shown");

    cyxwiz::PythonCheckKind kind;
    Check(cyxwiz::PythonCheckLine("failed", 0.5, "input", "", kind) == "Gradient failed 5.0e-01 in input" &&
              kind == cyxwiz::PythonCheckKind::Failed,
          "failed check line");
    Check(cyxwiz::PythonCheckLine("not_checked", 0, "", "initial_a must be at least 0.01", kind) ==
              "Gradient not checked: initial_a must be at least 0.01",
          "not checked line");
    Check(cyxwiz::TodayIsoDate().size() == 10, "today's date");
    Check(cyxwiz::CreateNodeSettingText("initial_a", "float", "1.0", "0.01-", {}) ==
              "initial_a (float, at least 0.01, default 1.0)",
          "setting text: minimum");
    Check(cyxwiz::CreateNodeSettingText("p", "float", "0.5", "0-1", {}) == "p (float, 0 to 1, default 0.5)",
          "setting text: range");
    Check(cyxwiz::CreateNodeSettingText("n", "int", "3", "-10", {}) == "n (int, at most 10, default 3)",
          "setting text: maximum");
    Check(cyxwiz::CreateNodeSettingText("lo", "float", "0", "-1-1", {}) == "lo (float, -1 to 1, default 0)",
          "setting text: negative minimum");
    Check(cyxwiz::CreateNodeSettingText("mode", "string", "fast", "fast|exact", {"fast", "exact"}) ==
              "mode (string, one of fast, exact, default fast)",
          "setting text: choices");

    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_python_bundle_presentation: all checks passed\n";
    return 0;
}
