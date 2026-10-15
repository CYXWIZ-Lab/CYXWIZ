#include "readiness_view.h"

#include "../../core/async_task_manager.h"
#include "../../core/column_role_store.h"
#include "../../core/dataset_catalog.h"
#include "../../core/session_query_service.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

ImVec4 ToneColour(studio::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case studio::Tone::Ok: return t.success;
        case studio::Tone::Warn: return t.warning;
        case studio::Tone::Error: return t.error;
        default: return t.text;
    }
}

const char* ToneIcon(studio::Tone tone) {
    switch (tone) {
        case studio::Tone::Ok: return ICON_FA_CIRCLE_CHECK;
        case studio::Tone::Warn: return ICON_FA_TRIANGLE_EXCLAMATION;
        case studio::Tone::Error: return ICON_FA_CIRCLE_XMARK;
        default: return ICON_FA_CIRCLE;
    }
}

// The verdict as a tinted pill, no outline.
void VerdictChip(const std::string& text, studio::Tone tone, const std::string& tip) {
    const ImVec4 c = tone == studio::Tone::Plain ? ui::CurrentTokens().text_dim : ToneColour(tone);
    const std::string label = std::string(ToneIcon(tone)) + " " + text;
    const ImVec2 ts = ImGui::CalcTextSize(label.c_str());
    const ImVec2 size(ts.x + 18.0f, ImGui::GetFrameHeight());
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ui::ToU32(ui::WithAlpha(c, 0.16f)), size.y * 0.5f);
    dl->AddText(ImVec2(at.x + 9.0f, at.y + (size.y - ts.y) * 0.5f), ui::ToU32(c), label.c_str());
    if (!tip.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip.c_str());
}

void RenderCard(const readiness::Card& card, int index) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushID(index);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.035f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::TextColored(t.text_dim, "%s", card.title.c_str());
    if (!card.header.empty() && !card.rows.empty() &&
        ImGui::BeginTable("##figures", static_cast<int>(card.header.size()), ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
        for (size_t i = 0; i < card.header.size(); ++i)
            ImGui::TableSetupColumn(card.header[i].empty() ? "##split" : card.header[i].c_str(), ImGuiTableColumnFlags_WidthStretch,
                                    i == 0 ? 1.3f : 1.0f);
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (size_t i = 0; i < card.header.size(); ++i) {
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_faint, "%s", card.header[i].c_str());
        }
        for (const auto& row : card.rows) {
            ImGui::TableNextRow();
            for (size_t i = 0; i < row.size() && i < card.header.size(); ++i) {
                ImGui::TableNextColumn();
                if (i == 0) ImGui::TextColored(t.text_dim, "%s", row[i].c_str());
                else ImGui::TextUnformatted(row[i].c_str());
            }
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& line : card.lines) {
        if (line.faint) ImGui::TextColored(t.text_faint, "%s", line.text.c_str());
        else ImGui::TextColored(ToneColour(line.tone), "%s", line.text.c_str());
    }
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::PopID();
}

}  // namespace

ReadinessView::~ReadinessView() {
    if (task_id_) AsyncTaskManager::Instance().Cancel(task_id_);
}

void ReadinessView::SetSplitSets(const std::vector<studio::SplitSet>& sets, const std::string& balance) {
    // Keep showing the same set when the list is rebuilt.
    const std::string chosen = chosen_ < sets_.size() ? sets_[chosen_].name : std::string();
    sets_ = sets;
    balance_ = balance;
    chosen_ = 0;
    for (size_t i = 0; i < sets_.size(); ++i)
        if (sets_[i].name == chosen) chosen_ = i;
}

void ReadinessView::Activate() { measured_.clear(); }

std::string ReadinessView::Signature() const {
    if (chosen_ >= sets_.size()) return "";
    std::string sig = sets_[chosen_].name + "|" + balance_;
    for (const auto& p : sets_[chosen_].parts) {
        const auto entry = DatasetCatalog::Instance().Resolve(p.dataset);
        sig += "|" + p.split + "=" + p.dataset + "@" + std::to_string(entry ? entry->generation : 0);
    }
    return sig;
}

void ReadinessView::Measure() {
    if (chosen_ >= sets_.size()) return;
    const studio::SplitSet& set = sets_[chosen_];
    measured_ = Signature();
    error_.clear();
    if (set.parts.empty()) {
        result_.reset();
        return;
    }
    // Roles come from the train split (the first part): read on the UI thread.
    std::vector<readiness::SplitTable> tables;
    for (const auto& p : set.parts) tables.push_back({p.split, p.dataset, p.label});
    const auto entry = DatasetCatalog::Instance().Resolve(set.parts.front().dataset);
    if (!entry) {
        error_ = "'" + set.parts.front().label + "' is no longer loaded.";
        result_.reset();
        return;
    }
    const std::string key = RoleSourceKey(entry->source_path, entry->Shown());
    std::map<std::string, ColumnRole> contract_roles;
    if (!entry->target_column.empty()) contract_roles[entry->target_column] = ColumnRole::Target;
    const std::map<std::string, ColumnRole> user_roles = ProjectColumnRoles().RolesFor(key);
    ProfileOptions options;
    if (const auto* s = ProjectColumnRoles().Find(key)) options.missing_text = s->missing_text;
    const std::string name = set.name, balance = balance_;

    if (task_id_) AsyncTaskManager::Instance().Cancel(task_id_);
    std::weak_ptr<int> alive = alive_;
    auto facts = std::make_shared<readiness::Facts>();
    task_id_ = AsyncTaskManager::Instance().RunAsync(
        "ML readiness: " + name,
        [=](LambdaTask& task) {
            const QueryRunner run = [](const QueryRequest& r) { return SessionQueryService::Instance().RunNow(r); };
            const auto stop = [&task] { return task.ShouldStop(); };
            ProfileOptions opts = options;
            opts.should_stop = stop;
            task.ReportProgress(0.1f, "Roles of " + tables.front().split);
            const DatasetProfile profile = ProfileTable(tables.front().dataset, run, opts);
            if (!profile.ok()) {
                facts->error = profile.error;
                task.MarkFailed(profile.error);
                return;
            }
            const DatasetContract contract = BuildContract(key, profile.Facts(), contract_roles, user_roles);
            task.ReportProgress(0.5f, "Comparing the splits");
            *facts = readiness::Measure(name, tables, readiness::PickRoles(contract, profile.Facts()), run, stop);
            facts->balance = balance;
            if (!facts->error.empty()) task.MarkFailed(facts->error);
        },
        nullptr,
        [this, alive, facts](bool, const std::string&) {
            if (alive.expired()) return;
            task_id_ = 0;
            if (!facts->error.empty()) {
                error_ = facts->error == "Cancelled." ? std::string() : facts->error;
                if (!error_.empty()) result_.reset();
                return;
            }
            result_ = facts;
            view_ = readiness::Build(*facts);
            spdlog::info("ML readiness '{}': {} in {:.0f} ms", facts->name, view_.verdict, facts->elapsed_ms);
        },
        alive_);
}

void ReadinessView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (sets_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No splits to compare yet.");
        ImGui::TextColored(t.text_faint, "Connect train, validation and test Data Inputs to a Data Split's inputs, or load datasets named "
                                         "<name>_train, <name>_validation and <name>_test. A Data Split that splits one input by ratio "
                                         "does so during training; it is not previewed here.");
        ImGui::PopTextWrapPos();
        return;
    }
    if (chosen_ >= sets_.size()) chosen_ = 0;
    if (!IsRunning() && Signature() != measured_) Measure();

    // Which set, when there are several.
    if (sets_.size() > 1) {
        for (size_t i = 0; i < sets_.size(); ++i) {
            if (i) ImGui::SameLine(0.0f, 4.0f);
            ImGui::PushID(static_cast<int>(i));
            const std::string count = std::to_string(sets_[i].parts.size());
            if (ui::FilterChip("##set", sets_[i].name.c_str(), count.c_str(), i == chosen_) && i != chosen_) {
                chosen_ = i;
                measured_.clear();
            }
            ImGui::PopID();
        }
        ImGui::Spacing();
    }
    const studio::SplitSet& set = sets_[chosen_];

    // Header: the set, its splits, the verdict, actions.
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", set.name.c_str());
    std::string splits;
    for (const auto& p : set.parts) splits += (splits.empty() ? "" : " \xC2\xB7 ") + p.split;
    if (!splits.empty()) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::TextColored(t.text_dim, "%s", splits.c_str());
    }
    if (result_ && !IsRunning()) {
        ImGui::SameLine(0.0f, 12.0f);
        VerdictChip(view_.verdict, view_.verdict_tone, view_.verdict_tip);
    }
    const float again_w = ui::ButtonWidth(ICON_FA_ROTATE " Check again", ui::ButtonSize::Small);
    const float profile_w = ui::ButtonWidth("Set roles in Profile", ui::ButtonSize::Small);
    const float node_w = set.node_id >= 0 ? ImGui::CalcTextSize("Show the Data Split").x + 16.0f : 0.0f;
    const float right = again_w + profile_w + node_w + ImGui::GetStyle().ItemSpacing.x * 2.0f;
    ImGui::SameLine(0.0f, 14.0f);
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - right));
    if (set.node_id >= 0) {
        if (ui::LinkButton("Show the Data Split", static_cast<bool>(on_show_node))) on_show_node(set.node_id);
        ImGui::SameLine();
    }
    if (ui::SecondaryButton("Set roles in Profile", !set.parts.empty() && static_cast<bool>(on_profile), "No split is loaded", ui::ButtonSize::Small))
        on_profile(set.parts.front().dataset);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Roles (Text, ID, Group, Target) are read from the train split; set them in Explore > Profile.");
    ImGui::SameLine();
    if (ui::SecondaryButton(ICON_FA_ROTATE " Check again", !IsRunning() && !set.parts.empty(), "Checking now", ui::ButtonSize::Small))
        measured_.clear();

    if (!set.note.empty()) ImGui::TextColored(t.warning, ICON_FA_TRIANGLE_EXCLAMATION " %s", set.note.c_str());
    if (IsRunning()) ImGui::TextColored(t.info, ICON_FA_SPINNER " Comparing the splits (Task View shows the steps)");
    if (!error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, "%s", error_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (set.parts.empty()) {
        ImGui::TextColored(t.text_dim, "None of this Data Split's inputs is loaded: open each Data Input's dialog and Apply, or run the graph.");
        return;
    }
    if (!result_) return;
    ImGui::Spacing();

    // The cards, as many per row as fit (at least 280 px each).
    const float avail = ImGui::GetContentRegionAvail().x;
    const int columns = std::clamp(static_cast<int>(avail / 280.0f), 1, 3);
    ImGui::BeginChild("##ready_cards", ImVec2(0.0f, 0.0f));
    if (ImGui::BeginTable("##cards", columns, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings,
                          ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
        for (size_t i = 0; i < view_.cards.size(); ++i) {
            ImGui::TableNextColumn();
            RenderCard(view_.cards[i], static_cast<int>(i));
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Compared in %.1f s on the whole tables.", result_->elapsed_ms / 1000.0);
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
}

}  // namespace cyxwiz
