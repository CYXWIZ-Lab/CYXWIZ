#include "python_bundle_ui.h"

#include "file_browser_launch.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "ui_widgets.h"
#include "../core/project_manager.h"
#include "../extension/python/python_node_provider.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <set>

namespace cyxwiz::python_ui {

namespace fs = std::filesystem;

namespace {

PythonNodeProvider* g_provider = nullptr;
std::function<void(const fs::path&)> g_open_code;
std::string g_pending_approval;   // bundle id waiting for the dialog
bool g_open_approval_popup = false;
bool g_show_approval_details = false;
std::set<std::string> g_open_details;  // card keys with Details open

constexpr const char* kApprovalPopup = "Run this Python code?###PythonBundleApproval";

bool ContainsCI(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(), [](char a, char b) {
               return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
           }) != haystack.end();
}

bool Matches(const PythonBundle& bundle, const std::string& filter) {
    if (filter.empty()) return true;
    if (ContainsCI(bundle.folder.filename().string(), filter) || ContainsCI(bundle.id, filter)) return true;
    return std::any_of(bundle.nodes.begin(), bundle.nodes.end(), [&](const PythonBundleNode& node) {
        return ContainsCI(node.descriptor.metadata.name, filter);
    });
}

std::string FolderName(const PythonBundle& bundle) {
    std::string name = bundle.folder.filename().string();
    if (name.size() > 8 && name.compare(name.size() - 8, 8, ".cyxnode") == 0) name.resize(name.size() - 8);
    return name;
}

void Rescan() {
    if (g_provider) g_provider->Rescan();
}

ImVec4 CheckColour(PythonCheckKind kind, const ui::Tokens& t) {
    switch (kind) {
        case PythonCheckKind::Passed: return ui::StatusStyleFor(ui::Status::Verified, t).colour;
        case PythonCheckKind::Failed: return ui::StatusStyleFor(ui::Status::Failed, t).colour;
        default: return ui::StatusStyleFor(ui::Status::NotVerifiedYet, t).colour;
    }
}

void RenderBundleCard(const PythonBundleState& state) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto approval = ApprovalOf(state.bundle);
    const PythonBundleCard card = BuildPythonBundleCard(state.bundle, state.approved,
                                                        approval ? &*approval : nullptr, CurrentRoots());
    const ImVec4 status_colour = card.approved ? ui::StatusStyleFor(ui::Status::Verified, t).colour : t.warning;
    ImGui::PushID(card.key.c_str());
    ui::BeginCard(card.key.c_str());

    ImGui::TextColored(status_colour, ICON_FA_CIRCLE);
    ImGui::SameLine();
    ImGui::TextUnformatted(card.title.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", card.id.c_str());
    ui::SameLineRight(ImGui::CalcTextSize(card.scope.c_str()).x);
    ImGui::TextColored(t.text_dim, "%s", card.scope.c_str());
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", card.folder.c_str());
    if (!card.adds.empty()) ImGui::TextColored(t.accent_text, "%s", card.adds.c_str());
    ImGui::PopTextWrapPos();

    if (ImGui::BeginTable("##nodes", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("node", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("check", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        for (const auto& row : card.nodes) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.name.c_str());
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(CheckColour(row.kind, t), "%s", row.check.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }

    ImGui::TextColored(status_colour, "%s", card.status.c_str());
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", card.status_note.c_str());
    ImGui::PopTextWrapPos();

    if (!card.approved) {
        if (ui::PrimaryButton("Approve code...", true, nullptr, ui::ButtonSize::Small)) {
            RequestApproval(card.id);
        }
        ImGui::SameLine();
    }
    if (ui::LinkButton("View code")) OpenCode(state.bundle);
    ImGui::SameLine();
    const bool details_open = g_open_details.count(card.key) > 0;
    if (ui::LinkButton(details_open ? "Hide" : "Details")) {
        if (details_open) {
            g_open_details.erase(card.key);
        } else {
            g_open_details.insert(card.key);
        }
    }
    const char* actions = "Actions";
    ui::SameLineRight(ui::ButtonWidth(actions, ui::ButtonSize::Small));
    if (ui::SecondaryButton(actions)) ImGui::OpenPopup("##bundle_actions");
    if (ImGui::BeginPopup("##bundle_actions")) {
        if (ImGui::MenuItem("Revoke approval", nullptr, false, card.approved)) {
            std::string error;
            if (!PythonNodeApprovals::User().Revoke(card.id, &error)) {
                spdlog::warn("Python bundle {}: approval not removed from disk: {}", card.id, error);
            }
            spdlog::info("Python bundle {}: approval revoked", card.id);
            Rescan();
        }
        if (!card.approved && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ui::Tooltip("Its code is not approved");
        }
        if (ImGui::MenuItem("Open folder")) LaunchAssetInFileBrowser(card.folder, true);
        if (ImGui::MenuItem("View code")) OpenCode(state.bundle);
        ImGui::EndPopup();
    }
    if (details_open) {
        std::vector<ui::KeyValue> rows;
        for (const auto& [key, value] : card.details) rows.push_back({key, value});
        ui::KeyValueTable("##details", rows, 1, true);
    }

    ui::EndCard();
    ImGui::PopID();
}

void RenderIssueCard(const PythonBundleIssue& issue) {
    const ui::Tokens& t = ui::CurrentTokens();
    const PythonBundleIssueCard card = BuildPythonBundleIssueCard(issue.folder, issue.error);
    ImGui::PushID(card.key.c_str());
    ui::BeginCard(card.key.c_str());
    const ImVec4 failed = ui::StatusStyleFor(ui::Status::Failed, t).colour;
    ImGui::TextColored(failed, ICON_FA_CIRCLE);
    ImGui::SameLine();
    ImGui::TextUnformatted(card.title.c_str());
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", card.status.c_str());
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", card.folder.c_str());
    ImGui::TextUnformatted(card.error.c_str());
    ImGui::PopTextWrapPos();
    if (ui::LinkButton("Open folder")) LaunchAssetInFileBrowser(card.folder, true);
    ui::EndCard();
    ImGui::PopID();
}

}  // namespace

void SetProvider(PythonNodeProvider* provider) { g_provider = provider; }
PythonNodeProvider* Provider() { return g_provider; }

PythonBundleRoots CurrentRoots() {
    PythonBundleRoots roots;
    auto& projects = ProjectManager::Instance();
    if (projects.HasActiveProject()) {
        roots.project_nodes = PythonNodeProvider::ProjectNodesFolder(projects.GetProjectRoot());
    }
    roots.user_nodes = UserPythonNodesFolder();
    return roots;
}

const PythonBundle* FindBundle(const std::string& bundle_id) {
    if (!g_provider) return nullptr;
    for (const auto& state : g_provider->Bundles()) {
        if (state.bundle.id == bundle_id) return &state.bundle;
    }
    return nullptr;
}

std::optional<PythonNodeApproval> ApprovalOf(const PythonBundle& bundle) {
    for (const auto& a : PythonNodeApprovals::User().All()) {
        if (a.bundle_id == bundle.id && a.content_hash == bundle.content_hash) return a;
    }
    return std::nullopt;
}

ExtensionPythonState StateOf(const std::string& bundle_id) {
    ExtensionPythonState state;
    if (const PythonBundle* bundle = FindBundle(bundle_id)) {
        if (const auto approval = ApprovalOf(*bundle)) {
            state.approved = true;
            state.approved_on = approval->approved_on;
        }
    }
    return state;
}

std::vector<std::pair<std::string, std::string>> BundleNames() {
    std::vector<std::pair<std::string, std::string>> names;
    if (!g_provider) return names;
    for (const auto& state : g_provider->Bundles()) names.emplace_back(state.bundle.id, FolderName(state.bundle));
    return names;
}

std::vector<BundleMenuEntry> MenuEntries() {
    std::vector<BundleMenuEntry> entries;
    if (!g_provider) return entries;
    for (const auto& state : g_provider->Bundles()) {
        const size_t n = state.bundle.nodes.size();
        std::string detail = std::to_string(n) + (n == 1 ? " node" : " nodes");
        if (!state.approved) detail += " \xC2\xB7 not approved";
        entries.push_back({state.bundle.id, FolderName(state.bundle), detail});
    }
    return entries;
}

void SetOpenCodeHandler(std::function<void(const fs::path&)> handler) { g_open_code = std::move(handler); }

void OpenCode(const PythonBundle& bundle) {
    const fs::path code = bundle.folder / "node.py";
    if (g_open_code) {
        g_open_code(code);
    } else {
        LaunchAssetInFileBrowser(code.string(), false);
    }
}

void RequestApproval(const std::string& bundle_id) {
    g_pending_approval = bundle_id;
    g_open_approval_popup = true;
    g_show_approval_details = false;
}

void RenderApprovalDialog() {
    if (g_open_approval_popup) {
        ImGui::OpenPopup(kApprovalPopup);
        g_open_approval_popup = false;
    }
    ui::DialogOptions options;
    options.size = ImVec2(580.0f, 440.0f);
    if (!ui::BeginDialog(kApprovalPopup, options)) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const PythonBundle* bundle = FindBundle(g_pending_approval);
    if (!bundle) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(("The bundle " + g_pending_approval + " is no longer loaded.").c_str());
        ImGui::PopTextWrapPos();
        ui::EndDialog("Close", nullptr);
        return;
    }
    const PythonBundleApprovalView view = BuildPythonBundleApproval(*bundle);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(view.intro.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    std::vector<ui::KeyValue> rows;
    for (const auto& [key, value] : view.rows) rows.push_back({key, value});
    ui::FactsTable("##approval_rows", rows);
    if (ui::LinkButton(g_show_approval_details ? "Hide" : "Details")) g_show_approval_details = !g_show_approval_details;
    if (g_show_approval_details) {
        std::vector<ui::KeyValue> details;
        for (const auto& [key, value] : view.details) details.push_back({key, value});
        ui::FactsTable("##approval_details", details);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", view.note.c_str());
    ImGui::PopTextWrapPos();
    if (ui::LinkButton("View code")) OpenCode(*bundle);

    if (ui::EndDialog("Approve", "Not now") == ui::DialogResult::Primary) {
        std::string error;
        if (!PythonNodeApprovals::User().Approve(MakePythonNodeApproval(*bundle, "approved", TodayIsoDate()), &error)) {
            spdlog::warn("Python bundle {}: approval holds for this session but was not saved: {}", bundle->id, error);
        }
        spdlog::info("Python bundle {} approved (code {})", bundle->id, bundle->content_hash);
        Rescan();
    }
}

void RenderBundleSection(const std::string& filter) {
    const ui::Tokens& t = ui::CurrentTokens();
    const size_t bundles = g_provider ? g_provider->Bundles().size() : 0;
    const size_t issues = g_provider ? g_provider->Issues().size() : 0;
    const std::string summary = PythonBundleSectionSummary(bundles, issues);
    ImGui::TextUnformatted("Python node bundles");
    ui::SameLineRight(ImGui::CalcTextSize(summary.c_str()).x);
    ImGui::TextColored(t.text_dim, "%s", summary.c_str());
    if (bundles == 0 && issues == 0) {
        ImGui::PushTextWrapPos(0.0f);
        for (const auto& line : PythonBundleEmptyLines(CurrentRoots())) ImGui::TextColored(t.text_dim, "%s", line.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    size_t shown = 0;
    if (g_provider) {
        for (const auto& state : g_provider->Bundles()) {
            if (!Matches(state.bundle, filter)) continue;
            RenderBundleCard(state);
            ImGui::Spacing();
            ++shown;
        }
        for (const auto& issue : g_provider->Issues()) {
            if (!filter.empty() && !ContainsCI(issue.folder.filename().string(), filter)) continue;
            RenderIssueCard(issue);
            ImGui::Spacing();
            ++shown;
        }
    }
    if (shown == 0) ImGui::TextColored(t.text_dim, "No bundles match '%s'", filter.c_str());
}

}  // namespace cyxwiz::python_ui
