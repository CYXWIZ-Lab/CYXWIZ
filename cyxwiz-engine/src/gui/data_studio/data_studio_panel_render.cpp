// Data Studio's shell (TOFIX148 step 3, approved mockup v8): header, the
// data rail, the job tabs, each job's "In the graph" strip and body, the
// status bar.

#include "data_studio_panel.h"

#include "../node_editor.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../separate_windows.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <string>

namespace cyxwiz {

namespace {

ImVec4 ToneColour(studio::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case studio::Tone::Ok: return t.success;
        case studio::Tone::Warn: return t.warning;
        case studio::Tone::Error: return t.error;
        default: return t.text_dim;
    }
}

const char* KindIcon(studio::RailItem::Kind kind) {
    switch (kind) {
        case studio::RailItem::Kind::DataInput: return ICON_FA_RIGHT_TO_BRACKET;
        case studio::RailItem::Kind::NodeResult: return ICON_FA_TABLE;
        case studio::RailItem::Kind::Published: return ICON_FA_FILE_EXPORT;
        default: return ICON_FA_DATABASE;
    }
}

// Moves to the next line when `width` does not fit after the last item.
void SameLineOrWrap(float width, float right_edge) {
    ImGui::SameLine(0.0f, 4.0f);
    if (ImGui::GetCursorScreenPos().x + width > right_edge) ImGui::NewLine();
}

}  // namespace

void DataStudioPanel::Render() {
    if (query_editor_) query_editor_->RenderWindows();
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(1200, 800), ImGuiCond_FirstUseEver);
    ::gui::NextWindowMayLeave("Data Studio");
    const bool expanded = ImGui::Begin("Data Studio", &visible_);
    ::gui::TabMenu("Data Studio", &visible_);
    if (expanded) {
        const double now = ImGui::GetTime();
        if (view_time_ < 0.0 || now - view_time_ > 0.5) {
            RefreshView();
            view_time_ = now;
        }
        RenderHeader();
        const float status_h = ImGui::GetFrameHeightWithSpacing() + 2.0f;
        const float body_h = std::max(120.0f, ImGui::GetContentRegionAvail().y - status_h);
        // Side by side, a narrow panel left the job ~120 px (tables one
        // letter per line); there the rail goes above the job instead.
        float main_h = body_h;
        if (ImGui::GetContentRegionAvail().x < rail_width_ + kMinJobWidth) {
            RenderRail(0.0f, body_h * 0.35f);
            main_h = std::max(120.0f, body_h - ImGui::GetItemRectSize().y - ImGui::GetStyle().ItemSpacing.y);
        } else {
            RenderRail(body_h);
            ImGui::SameLine(0.0f, 0.0f);
        }
        ImGui::BeginChild("##studio_main", ImVec2(0.0f, main_h), ImGuiChildFlags_AlwaysUseWindowPadding);
        RenderJobTabs();
        RenderNodeStrip(view_.jobs[static_cast<size_t>(job_)]);
        ImGui::Spacing();
        RenderJobBody();
        ImGui::EndChild();
        RenderStatusBar();
    }
    ImGui::End();
}

void DataStudioPanel::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "Data Studio");
    ImGui::SameLine(0.0f, 14.0f);

    const float show_w = ui::ButtonWidth("Show graph", ui::ButtonSize::Small);
    const float publish_w = ui::ButtonWidth("Publish...", ui::ButtonSize::Regular);
    const float right = show_w + publish_w + ImGui::GetStyle().ItemSpacing.x;

    // The graph it works on: a tinted chip, no outline. In a narrow panel the
    // chip shrinks (its text clipped, whole on hover) so the buttons never
    // draw over it.
    const std::string graph = view_.graph_label;
    const float pad = 9.0f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float label_w = ImGui::CalcTextSize("Graph").x;
    const float full_w = pad * 2.0f + label_w + 8.0f + ImGui::CalcTextSize(graph.c_str()).x;
    const float w = std::min(full_w, std::max(pad * 2.0f + label_w, ImGui::GetContentRegionAvail().x - right - 14.0f));
    const float h = ImGui::GetFrameHeight();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), ui::ToU32(t.bg_raised), 6.0f);
    const float ty = pos.y + (h - ImGui::GetTextLineHeight()) * 0.5f;
    dl->PushClipRect(pos, ImVec2(pos.x + w - pad * 0.5f, pos.y + h), true);
    dl->AddText(ImVec2(pos.x + pad, ty), ui::ToU32(t.text_dim), "Graph");
    dl->AddText(ImVec2(pos.x + pad + label_w + 8.0f, ty), ui::ToU32(t.text), graph.c_str());
    dl->PopClipRect();
    ImGui::Dummy(ImVec2(w, h));
    const char* note = "Everything here is a node in this graph; changes show on the canvas.";
    if (ImGui::IsItemHovered()) {
        if (w < full_w) ImGui::SetTooltip("%s\n%s", graph.c_str(), note);
        else ImGui::SetTooltip("%s", note);
    }
    const float note_w = ImGui::CalcTextSize(note).x;
    const float room = ImGui::GetContentRegionAvail().x - w - right - 40.0f;
    if (room > note_w) {
        ImGui::SameLine(0.0f, 14.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, "%s", note);
    }
    ImGui::SameLine(0.0f, 14.0f);
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - right));
    if (ui::SecondaryButton("Show graph", static_cast<bool>(open_node_editor_callback_), "CyxWiz Studio is not available"))
        open_node_editor_callback_();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open CyxWiz Studio on this graph");
    ImGui::SameLine();
    if (ui::PrimaryButton("Publish...", node_editor_ != nullptr && publish_ && !publish_->IsRunning(),
                          !node_editor_ ? "No graph is open" : "The Data Pipeline is running")) {
        job_ = studio::Job::Publish;
        view_time_ = -1.0;  // read what feeds the outputs now
        publish_->OpenPublish();
    }
    ImGui::Spacing();
}

void DataStudioPanel::RenderRail(float height, float max_height) {
    const ui::Tokens& t = ui::CurrentTokens();
    // A tinted area, no outline (owner rule: no lines around areas).
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.03f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 10.0f));
    const bool stacked = height <= 0.0f;
    if (stacked) ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(FLT_MAX, max_height));
    ImGui::BeginChild("##studio_rail", ImVec2(stacked ? 0.0f : rail_width_, stacked ? 0.0f : height),
                      ImGuiChildFlags_AlwaysUseWindowPadding | (stacked ? ImGuiChildFlags_AutoResizeY : 0));
    ImGui::PopStyleVar();
    for (const auto& section : view_.rail) {
        ImGui::TextColored(t.text_dim, "%s", section.title.c_str());
        if (section.items.empty()) {
            ImGui::TextColored(t.text_faint, section.title == "Data Inputs" ? "None in this graph" : "None yet");
        }
        for (const auto& item : section.items) {
            ImGui::PushID(item.node_id >= 0 ? item.node_id : static_cast<int>(std::hash<std::string>{}(item.dataset)));
            const bool chosen = !item.dataset.empty() && item.dataset == active_dataset_;
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            ui::PushListRowColours();
            if (ImGui::Selectable("##item", chosen, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(w, 0.0f))) {
                if (!item.dataset.empty()) SetActiveDataset(item.dataset);
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && item.node_id >= 0) ShowOnCanvas({item.node_id});
            }
            ui::PopListRowColours();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                std::string tip = item.tip;
                tip += std::string(tip.empty() ? "" : "\n") +
                       (item.dataset.empty() ? "No table to show yet." : "Click: show it in Explore.");
                if (item.node_id >= 0) tip += "\nDouble-click: show the node on the canvas.";
                ImGui::SetTooltip("%s", tip.c_str());
            }
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float meta_w = ImGui::CalcTextSize(item.meta.c_str()).x;
            const float icon_w = ImGui::CalcTextSize(ICON_FA_TABLE).x + 6.0f;
            dl->AddText(ImVec2(pos.x + 2.0f, pos.y), ui::ToU32(t.text_faint), KindIcon(item.kind));
            const ImVec4 clip(pos.x, pos.y, pos.x + w - meta_w - 10.0f, pos.y + ImGui::GetTextLineHeight());
            dl->AddText(nullptr, 0.0f, ImVec2(pos.x + icon_w, pos.y), ui::ToU32(item.dataset.empty() ? t.text_dim : t.text),
                        item.label.c_str(), nullptr, 0.0f, &clip);
            dl->AddText(ImVec2(pos.x + w - meta_w, pos.y), ui::ToU32(ToneColour(item.tone)), item.meta.c_str());
            ImGui::PopID();
        }
        ImGui::Spacing();
        ImGui::Spacing();
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Data enters the graph through a Data Input (Bring in).");
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void DataStudioPanel::RenderJobTabs() {
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (size_t j = 0; j < studio::kJobCount; ++j) {
        const auto& job = view_.jobs[j];
        const std::string caption = std::to_string(j + 1) + " \xC2\xB7 " + job.state;
        if (j > 0) SameLineOrWrap(ui::JobTabWidth(job.name.c_str(), caption.c_str()), right_edge);
        if (ui::JobTab(("##job" + std::to_string(j)).c_str(), job.name.c_str(), caption.c_str(), job.job == job_) && job_ != job.job) {
            job_ = job.job;
            if (job_ == studio::Job::Ready && readiness_) readiness_->Activate();  // roles may have changed in Profile
            if (job_ == studio::Job::Results && results_) results_->Activate();   // runs may have finished since
            if (job_ == studio::Job::Validate || job_ == studio::Job::Publish) view_time_ = -1.0;  // read what feeds the checks / outputs now
        }
    }
}

void DataStudioPanel::RenderNodeStrip(const studio::JobView& job) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->ChannelsSplit(2);
    dl->ChannelsSetCurrent(1);
    ImGui::SetCursorScreenPos(ImVec2(start.x + 10.0f, start.y + 6.0f));
    ImGui::BeginGroup();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_faint, "In the graph:");
    const char* const empty_note = "no node does this yet";
    // Results keeps the latest test as a node (TOFIX157, approved mockup).
    const bool keep = job.job == studio::Job::Results && results_;
    const char* const keep_label = "Keep as node";
    const std::string keep_why = keep ? results_->KeepBlocker(node_editor_ != nullptr) : std::string();
    const float keep_w = keep ? ui::ButtonWidth(keep_label, ui::ButtonSize::Small) + ImGui::GetStyle().ItemSpacing.x : 0.0f;
    const float link_w = ImGui::CalcTextSize("Show on canvas").x + 12.0f + keep_w;
    // The link sits at the top right while the first line still fits beside
    // it; in a narrow panel it goes under the chips instead of over them.
    float first_w = ImGui::CalcTextSize(empty_note).x;
    if (!job.chips.empty()) {
        const auto& chip = job.chips.front();
        first_w = ImGui::CalcTextSize((chip.name + (chip.note.empty() ? "" : "  " + chip.note)).c_str()).x + 18.0f;
    }
    const bool link_beside = 10.0f + ImGui::CalcTextSize("In the graph:").x + 4.0f + first_w + 16.0f + link_w <= width;
    const float right_edge = start.x + width - (link_beside ? link_w + 16.0f : 10.0f);
    if (job.chips.empty()) {
        SameLineOrWrap(ImGui::CalcTextSize(empty_note).x, right_edge);
        ImGui::AlignTextToFramePadding();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + (right_edge - ImGui::GetCursorScreenPos().x));
        ImGui::TextColored(t.text_dim, "%s", empty_note);
        ImGui::PopTextWrapPos();
    }
    for (size_t i = 0; i < job.chips.size(); ++i) {
        const auto& chip = job.chips[i];
        const std::string text = chip.name + (chip.note.empty() ? "" : "  " + chip.note);
        const float cw = ImGui::CalcTextSize(text.c_str()).x + 18.0f;
        SameLineOrWrap(cw, right_edge);
        ImGui::PushID(static_cast<int>(i));
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float h = ImGui::GetFrameHeight() - 2.0f;
        if (ImGui::InvisibleButton("##chip", ImVec2(cw, h))) ShowOnCanvas(chip.ids);
        const bool hovered = ImGui::IsItemHovered();
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            ImGui::SetTooltip("Show on the canvas");
        }
        dl->AddRectFilled(p, ImVec2(p.x + cw, p.y + h), ui::ToU32(ui::Mix(t.bg_window, t.info, hovered ? 0.30f : 0.18f)), h * 0.5f);
        const float ty = p.y + (h - ImGui::GetTextLineHeight()) * 0.5f;
        dl->AddText(ImVec2(p.x + 9.0f, ty), ui::ToU32(t.info), chip.name.c_str());
        if (!chip.note.empty())
            dl->AddText(ImVec2(p.x + 9.0f + ImGui::CalcTextSize((chip.name + "  ").c_str()).x, ty), ui::ToU32(t.text_dim),
                        chip.note.c_str());
        ImGui::PopID();
    }
    ImGui::EndGroup();
    float bottom = ImGui::GetItemRectMax().y;
    ImGui::SetCursorScreenPos(link_beside ? ImVec2(start.x + width - link_w - 6.0f, start.y + 6.0f)
                                          : ImVec2(start.x + 10.0f, bottom + 4.0f));
    if (keep) {
        if (ui::SecondaryButton(keep_label, keep_why.empty(), keep_why.c_str(), ui::ButtonSize::Small)) results_->RequestKeep();
        if (keep_why.empty() && ImGui::IsItemHovered())
            ImGui::SetTooltip("Saves the latest test's rows in the project and adds a Test Evaluation node reading them");
        ImGui::SameLine();
    }
    ImGui::AlignTextToFramePadding();
    if (ui::LinkButton("Show on canvas", !job.node_ids.empty())) ShowOnCanvas(job.node_ids);
    if (!link_beside) bottom = ImGui::GetItemRectMax().y;
    const float strip_h = bottom - start.y + 6.0f;
    dl->ChannelsSetCurrent(0);
    dl->AddRectFilled(start, ImVec2(start.x + width, start.y + strip_h), ui::ToU32(ui::Mix(t.bg_window, t.text, 0.035f)), 8.0f);
    dl->ChannelsMerge();
    ImGui::SetCursorScreenPos(ImVec2(start.x, start.y + strip_h));
    ImGui::Dummy(ImVec2(width, 0.0f));
}

void DataStudioPanel::RenderJobBody() {
    switch (job_) {
        case studio::Job::BringIn: RenderBringIn(); break;
        case studio::Job::Explore: RenderExplore(); break;
        case studio::Job::Prepare: if (prepare_) prepare_->Render(); break;
        case studio::Job::Validate: if (validate_) validate_->Render(); break;
        case studio::Job::Ready: if (readiness_) readiness_->Render(); break;
        case studio::Job::Publish: if (publish_) publish_->Render(); break;
        case studio::Job::Results: if (results_) results_->Render(); break;
    }
}

void DataStudioPanel::RenderBringIn() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ui::PrimaryButton(ICON_FA_PLUS " Add Data Input...", node_editor_ != nullptr, "No graph is open", ui::ButtonSize::Small)) {
        // The node is added when CyxWiz Studio draws, so bring it forward first.
        if (open_node_editor_callback_) open_node_editor_callback_();
        node_editor_->AddDataInputAndConfigure();
    }
    // Beside the button when it fits, else under it, wrapped.
    const char* hint =
        "Adds a Data Input node to this graph and opens its dialog: the file, its format, a preview, Apply.";
    ImGui::SameLine(0.0f, 12.0f);
    if (ImGui::CalcTextSize(hint).x > ImGui::GetContentRegionAvail().x) ImGui::NewLine();
    ImGui::AlignTextToFramePadding();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s", hint);
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    // Too narrow for the five columns (name and file squeezed to a letter
    // each): one block per Data Input with the same facts and actions.
    constexpr float kTableMinWidth = 620.0f;
    if (view_.sources.empty()) {
        ImGui::TextColored(t.text_dim, "No Data Input in this graph yet.");
    } else if (ImGui::GetContentRegionAvail().x < kTableMinWidth) {
        ImFont* mono = gui::GetCodeFont();
        for (const auto& row : view_.sources) {
            ImGui::PushID(row.node_id);
            ImGui::TextColored(t.text_bright, "%s", row.name.c_str());
            if (mono) ImGui::PushFont(mono);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", row.file.empty() ? "(no file set)" : row.file.c_str());
            ImGui::PopTextWrapPos();
            if (mono) ImGui::PopFont();
            ImGui::TextUnformatted(row.size.empty() ? "-" : row.size.c_str());
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::TextColored(ToneColour(row.tone), ICON_FA_CIRCLE " %s", row.state.c_str());
            if (!row.detail.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", row.detail.c_str());
            if (ui::LinkButton("Open dialog", node_editor_ != nullptr)) {
                if (open_node_editor_callback_) open_node_editor_callback_();
                node_editor_->OpenNodeConfiguration(row.node_id);
            }
            if (!row.dataset.empty()) {
                ImGui::SameLine(0.0f, 12.0f);
                if (ui::LinkButton("Explore")) {
                    SetActiveDataset(row.dataset);
                    job_ = studio::Job::Explore;
                    explore_tab_ = ExploreTab::Profile;
                }
            }
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::PopID();
        }
    } else if (ImGui::BeginTable("##bring_in", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("Data Input node", ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn("file", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        // Size and state are short: fixed from their longest text so nothing clips.
        const char* size_header = "rows \xC3\x97" " columns";
        ImGui::TableSetupColumn(size_header, ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize(size_header).x + 12.0f);
        ImGui::TableSetupColumn("state", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize(ICON_FA_CIRCLE " not loaded").x + 12.0f);
        const float actions_w = ui::ButtonWidth("Open dialog", ui::ButtonSize::Small) + ImGui::CalcTextSize("Explore").x + 24.0f;
        ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed, actions_w);
        ImGui::TableHeadersRow();
        ImFont* mono = gui::GetCodeFont();
        for (const auto& row : view_.sources) {
            ImGui::PushID(row.node_id);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_bright, "%s", row.name.c_str());
            ImGui::TableNextColumn();
            if (mono) ImGui::PushFont(mono);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", row.file.empty() ? "(no file set)" : row.file.c_str());
            ImGui::PopTextWrapPos();
            if (mono) ImGui::PopFont();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row.size.empty() ? "-" : row.size.c_str());
            ImGui::TableNextColumn();
            ImGui::TextColored(ToneColour(row.tone), ICON_FA_CIRCLE " %s", row.state.c_str());
            if (!row.detail.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", row.detail.c_str());
            ImGui::TableNextColumn();
            if (ui::LinkButton("Open dialog", node_editor_ != nullptr)) {
                if (open_node_editor_callback_) open_node_editor_callback_();
                node_editor_->OpenNodeConfiguration(row.node_id);
            }
            if (!row.dataset.empty()) {
                ImGui::SameLine();
                if (ui::LinkButton("Explore")) {
                    SetActiveDataset(row.dataset);
                    job_ = studio::Job::Explore;
                    explore_tab_ = ExploreTab::Profile;
                }
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Each row is a Data Input node; its dialog is where the file, format and loading are set. "
                                     "Its profile is Explore > Profile. Data enters the graph only through a Data Input.");
    ImGui::PopTextWrapPos();
}

void DataStudioPanel::RenderExplore() {
    static const std::pair<const char*, ExploreTab> kTabs[] = {
        {"Query", ExploreTab::Query}, {"Profile", ExploreTab::Profile}, {"Missing", ExploreTab::Missing},
        {"Outliers", ExploreTab::Outliers}, {"Correlation", ExploreTab::Correlation}, {"Visualize", ExploreTab::Visualize}};
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    bool first = true;
    for (const auto& [label, tab] : kTabs) {
        // Wraps in a narrow dock.
        if (!first) SameLineOrWrap(ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 4.0f, right_edge);
        first = false;
        if (ui::FilterChip(label, label, "", explore_tab_ == tab)) explore_tab_ = tab;
    }
    ImGui::Spacing();
    switch (explore_tab_) {
        case ExploreTab::Query: if (query_editor_) query_editor_->Render(); break;
        case ExploreTab::Profile: if (profile_) profile_->Render(); break;
        case ExploreTab::Missing: if (missing_) missing_->Render(MakeExploreContext()); break;
        case ExploreTab::Outliers: if (outliers_) outliers_->Render(MakeExploreContext()); break;
        case ExploreTab::Correlation: if (correlation_) correlation_->Render(MakeExploreContext()); break;
        case ExploreTab::Visualize: if (visualizer_) visualizer_->Render(); break;
    }
}

void DataStudioPanel::RenderStatusBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    if (query_editor_ && query_editor_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Query running");
    else if (profile_ && profile_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Profiling");
    else if (missing_ && missing_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Finding missing values");
    else if (outliers_ && outliers_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Finding outliers");
    else if (correlation_ && correlation_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Computing correlations");
    else if (readiness_ && readiness_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Comparing splits");
    else if (validate_ && validate_->IsRunning()) ImGui::TextColored(t.info, ICON_FA_CIRCLE " Data Pipeline running");
    else ImGui::TextColored(t.success, ICON_FA_CIRCLE " Ready");
    ImGui::SameLine(0.0f, 14.0f);
    if (view_.last_run.empty()) ImGui::TextColored(t.text_dim, "No run in this session");
    else ImGui::TextColored(view_.last_run_tone == studio::Tone::Ok ? t.text_dim : ToneColour(view_.last_run_tone), "Last run: %s",
                            view_.last_run.c_str());
    // Right-aligned, and only where the last-run line leaves room for it.
    const char* how = "Runs through the Data Pipeline executor";
    const float how_x = ImGui::GetWindowContentRegionMax().x - ImGui::CalcTextSize(how).x;
    ImGui::SameLine();
    if (ImGui::GetCursorPosX() + 20.0f <= how_x) {
        ImGui::SetCursorPosX(how_x);
        ImGui::TextColored(t.text_faint, "%s", how);
    } else {
        ImGui::NewLine();
    }
}

}  // namespace cyxwiz
