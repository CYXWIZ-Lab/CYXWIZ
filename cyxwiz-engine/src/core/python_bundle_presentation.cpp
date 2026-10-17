#include "python_bundle_presentation.h"

#include "plugin_presentation.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>

namespace cyxwiz {

namespace fs = std::filesystem;

namespace {

std::string Title(const fs::path& folder) {
    std::string name = folder.filename().string();
    if (name.size() > 8 && name.compare(name.size() - 8, 8, ".cyxnode") == 0) name.resize(name.size() - 8);
    return name;
}

std::string Format(double value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1e", value);
    return text;
}

bool Under(const fs::path& folder, const fs::path& root) {
    if (root.empty()) return false;
    return folder.parent_path().lexically_normal() == root.lexically_normal();
}

std::string HashDigits(const std::string& hash) {
    const auto colon = hash.find(':');
    return colon == std::string::npos ? hash : hash.substr(colon + 1);
}

PluginContribution Contribution(const PythonBundle& bundle) {
    PluginContribution c;
    for (const auto& node : bundle.nodes) {
        ++c.trainable_nodes;
        const std::string& category = node.descriptor.menu_category;
        if (!category.empty() &&
            std::find(c.node_categories.begin(), c.node_categories.end(), category) == c.node_categories.end()) {
            c.node_categories.push_back(category);
        }
    }
    return c;
}

std::string RequiresText(const std::string& requirement) {
    if (requirement.rfind(">=", 0) == 0) return "pycyxwiz " + requirement.substr(2) + " or newer";
    return requirement.empty() ? std::string() : "pycyxwiz " + requirement;
}

}  // namespace

std::string PythonCheckLine(const std::string& status, double max_relative_error, const std::string& where,
                            const std::string& error, PythonCheckKind& kind) {
    if (status == "passed") {
        kind = PythonCheckKind::Passed;
        return "Gradient passed " + Format(max_relative_error);
    }
    if (status == "failed") {
        kind = PythonCheckKind::Failed;
        return "Gradient failed " + Format(max_relative_error) + (where.empty() ? "" : " in " + where);
    }
    kind = PythonCheckKind::NotChecked;
    return error.empty() ? std::string("Gradient not checked") : "Gradient not checked: " + error;
}

PythonBundleCard BuildPythonBundleCard(const PythonBundle& bundle, bool approved,
                                       const PythonNodeApproval* approval, const PythonBundleRoots& roots) {
    PythonBundleCard card;
    card.key = bundle.folder.string();
    card.title = Title(bundle.folder);
    card.id = bundle.id;
    card.folder = bundle.folder.string();
    card.scope = Under(bundle.folder, roots.project_nodes) ? "This project"
                 : Under(bundle.folder, roots.user_nodes)  ? "All your projects"
                                                           : "With a plugin";
    card.adds = PluginAddsLine(Contribution(bundle));
    for (const auto& node : bundle.nodes) {
        const auto& d = node.descriptor;
        PythonBundleNodeRow row;
        row.name = d.metadata.name + (d.version.empty() ? "" : " " + d.version);
        const auto& v = d.verification;
        const std::string status = v.status == ExtensionVerification::Status::Passed   ? "passed"
                                   : v.status == ExtensionVerification::Status::Failed ? "failed"
                                                                                       : "not_checked";
        const bool has_where = status != "not_checked";
        row.check = PythonCheckLine(status, v.max_relative_error, has_where ? v.note : std::string(),
                                    has_where ? std::string() : v.note, row.kind);
        card.nodes.push_back(std::move(row));
    }
    card.approved = approved;
    if (approved) {
        card.status = approval && !approval->approved_on.empty() ? "Approved " + approval->approved_on : "Approved";
        card.status_note = "Its nodes build and train. Editing node.py asks again.";
    } else {
        card.status = "Not approved yet";
        card.status_note =
            "Its nodes show in the palette and save in graphs, but its Python code does not run until you "
            "approve it.";
    }
    card.details.emplace_back("Code sha256", HashDigits(bundle.content_hash));
    if (!bundle.requires_pycyxwiz.empty()) card.details.emplace_back("Requires", RequiresText(bundle.requires_pycyxwiz));
    if (approval) {
        card.details.emplace_back("Approval", approval->how == "created" ? "Created in the Engine" : "Approved by you");
    }
    return card;
}

PythonBundleIssueCard BuildPythonBundleIssueCard(const fs::path& folder, const std::string& error) {
    PythonBundleIssueCard card;
    card.key = folder.string();
    card.title = Title(folder);
    card.status = "could not be read";
    card.folder = folder.string();
    card.error = error;
    return card;
}

std::string PythonBundleSectionSummary(size_t bundles, size_t issues) {
    if (bundles == 0 && issues == 0) return "No bundles found";
    std::string text = std::to_string(bundles) + (bundles == 1 ? " bundle" : " bundles");
    if (issues > 0) text += " \xC2\xB7 " + std::to_string(issues) + " not read";
    return text;
}

std::vector<std::string> PythonBundleEmptyLines(const PythonBundleRoots& roots) {
    std::vector<std::string> lines;
    lines.push_back("Write a node with pycyxwiz.nodes and bundle it, or use Create node from script in the "
                    "Script Editor. CyxWiz looks in:");
    if (!roots.project_nodes.empty()) lines.push_back(roots.project_nodes.string() + "  (this project)");
    lines.push_back(roots.user_nodes.string() + "  (all your projects)");
    return lines;
}

PythonBundleApprovalView BuildPythonBundleApproval(const PythonBundle& bundle) {
    PythonBundleApprovalView view;
    view.title = "Run this Python code?";
    const size_t count = bundle.nodes.size();
    view.intro = "The bundle " + Title(bundle.folder) + " adds " + std::to_string(count) +
                 (count == 1 ? " node" : " nodes") +
                 ". Its code runs inside CyxWiz with your user's rights: it can read and change your files. "
                 "Approve only code you wrote or trust.";
    std::string names;
    bool all_passed = !bundle.nodes.empty();
    bool any_failed = false;
    std::string checked_on;
    for (const auto& node : bundle.nodes) {
        const auto& d = node.descriptor;
        names += (names.empty() ? "" : ", ") + d.metadata.name + (d.version.empty() ? "" : " " + d.version);
        all_passed = all_passed && d.verification.status == ExtensionVerification::Status::Passed;
        any_failed = any_failed || d.verification.status == ExtensionVerification::Status::Failed;
        if (checked_on.empty()) checked_on = d.verification.checked_on;
    }
    view.rows.emplace_back("Bundle", bundle.id);
    view.rows.emplace_back("Nodes", names);
    view.rows.emplace_back("Folder", bundle.folder.string());
    view.rows.emplace_back("Gradient check",
                           all_passed  ? "Passed when bundled" + (checked_on.empty() ? "" : " (" + checked_on + ")")
                           : any_failed ? std::string("Failed for at least one node when bundled")
                                        : std::string("Not checked when bundled"));
    view.details.emplace_back("Code sha256", HashDigits(bundle.content_hash));
    if (!bundle.requires_pycyxwiz.empty()) view.details.emplace_back("Requires", RequiresText(bundle.requires_pycyxwiz));
    view.note = "The approval is for this exact code. If node.py changes, CyxWiz asks again.";
    return view;
}

PythonNodeApproval MakePythonNodeApproval(const PythonBundle& bundle, const std::string& how, const std::string& today) {
    return {bundle.id, bundle.content_hash, bundle.folder.string(), today, how};
}

std::string TodayIsoDate() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%Y-%m-%d", &local);
    return text;
}

std::string CreateNodeSettingText(const std::string& name, const std::string& type, const std::string& default_value,
                                  const std::string& validation, const std::vector<std::string>& choices) {
    std::string text = name + " (" + type;
    if (!choices.empty()) {
        text += ", one of ";
        for (size_t i = 0; i < choices.size(); ++i) text += (i ? ", " : "") + choices[i];
    } else if (!validation.empty()) {
        // "<min>-<max>": the separator is the first '-' that is not a sign.
        size_t split = std::string::npos;
        for (size_t i = 1; i < validation.size(); ++i) {
            if (validation[i] == '-' && validation[i - 1] != 'e' && validation[i - 1] != 'E') {
                split = i;
                break;
            }
        }
        if (validation[0] == '-' && split == std::string::npos) {
            text += ", at most " + validation.substr(1);
        } else if (split != std::string::npos) {
            const std::string lo = validation.substr(0, split);
            const std::string hi = validation.substr(split + 1);
            if (!lo.empty() && !hi.empty()) {
                text += ", " + lo + " to " + hi;
            } else if (!lo.empty()) {
                text += ", at least " + lo;
            } else if (!hi.empty()) {
                text += ", at most " + hi;
            }
        }
    }
    if (!default_value.empty()) text += ", default " + default_value;
    return text + ")";
}

std::string SuggestBundleName(const std::string& script_name) {
    std::string stem = fs::path(script_name).stem().string();
    std::string name;
    for (char c : stem) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u)) {
            name += static_cast<char>(std::tolower(u));
        } else if ((c == '_' || c == '-' || c == ' ') && !name.empty() && name.back() != '-') {
            name += '-';
        }
    }
    while (!name.empty() && name.back() == '-') name.pop_back();
    return name.empty() ? std::string("my-nodes") : name;
}

std::string CheckBundleName(const std::string& name) {
    if (name.empty()) return "Enter a bundle name";
    if (!std::isalnum(static_cast<unsigned char>(name[0])) || std::isupper(static_cast<unsigned char>(name[0]))) {
        return "Start with a lower-case letter or a digit";
    }
    for (char c : name) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!(std::islower(u) || std::isdigit(u) || c == '-' || c == '_')) return "Use a-z, 0-9, '-' and '_'";
    }
    return {};
}

CreateNodeView BuildCreateNodeView(const CreateNodeInputs& in) {
    CreateNodeView view;
    view.title = "Create nodes from " + in.script_name;
    const size_t count = in.nodes.size();
    const std::string found = "Found " + std::to_string(count) + (count == 1 ? " node" : " nodes") +
                              " (classes marked with @nodes.activation or @nodes.node).";
    view.target = in.project_nodes.empty()
        ? std::string()
        : (in.project_nodes / ((in.bundle_name.empty() ? std::string("?") : in.bundle_name) + ".cyxnode")).string();
    view.name_error = CheckBundleName(in.bundle_name);

    switch (in.stage) {
        case CreateNodeStage::Inspecting:
            view.lead = "Reading " + in.script_name + "...";
            view.primary_label = "Check gradients";
            view.busy = true;
            view.primary_disabled_reason = "Reading the script";
            break;
        case CreateNodeStage::NoNodes:
            view.lead = "No node in " + in.script_name +
                        ": mark a class with @nodes.node or @nodes.activation (see docs/extending/python-nodes.md).";
            view.primary_label = "Close";
            view.primary_enabled = true;
            break;
        case CreateNodeStage::Failed:
            view.lead = "Python stopped while reading " + in.script_name + ":";
            view.note = in.error;
            view.primary_label = "Close";
            view.primary_enabled = true;
            break;
        case CreateNodeStage::Found:
            view.lead = found;
            view.note = "The gradient check runs each node with its default settings (a few seconds).";
            view.primary_label = "Check gradients";
            view.primary_enabled = true;
            view.show_name = true;
            break;
        case CreateNodeStage::Checking:
            view.lead = found;
            view.note = "Checking gradients...";
            view.primary_label = "Check gradients";
            view.busy = true;
            view.primary_disabled_reason = "Checking";
            view.show_name = true;
            break;
        case CreateNodeStage::Checked: {
            view.lead = found;
            const bool failed = std::any_of(in.nodes.begin(), in.nodes.end(),
                                            [](const CreateNodeFound& n) { return n.kind == PythonCheckKind::Failed; });
            view.note = failed
                ? "A node that fails is still created, marked \"gradient check failed\"; the Engine warns when you "
                  "use it. Fix its backward step and create it again."
                : (count == 1 ? std::string("It passes.") : std::string("All pass."));
            view.primary_label = "Create nodes";
            view.show_name = true;
            if (in.project_nodes.empty()) {
                view.primary_disabled_reason = "Open a project: the bundle goes into its nodes folder";
            } else if (!view.name_error.empty()) {
                view.primary_disabled_reason = view.name_error;
            } else {
                view.primary_enabled = true;
            }
            break;
        }
        case CreateNodeStage::Writing:
            view.lead = found;
            view.note = "Writing the bundle...";
            view.primary_label = "Create nodes";
            view.busy = true;
            view.primary_disabled_reason = "Writing";
            view.show_name = true;
            break;
        case CreateNodeStage::Written:
            view.lead = "Created " + in.written_folder;
            view.note = "The bundle is approved for you (you wrote it). Its nodes appear in the Node Browser within "
                        "2 seconds.";
            view.primary_label = "Close";
            view.primary_enabled = true;
            break;
    }
    return view;
}

}  // namespace cyxwiz
