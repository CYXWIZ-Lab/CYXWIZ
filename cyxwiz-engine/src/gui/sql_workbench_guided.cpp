// SQL Workbench, Guided mode (TOFIX146 step 5, mockup v5 board 3): build a
// step from plain operations. The operations (gui/guided_editor, core/
// guided_sql) write the step's SQL, which the rest of the Workbench runs,
// counts and saves.
#include "sql_workbench.h"

#include "editor_fonts.h"
#include "icons.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace guided = cyxwiz::guided;
using cyxwiz::sqlstep::FormatCount;
using namespace sqlwb;

std::string Param(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}
}  // namespace

// ---- State ----------------------------------------------------------------------

void SqlWorkbenchDialog::LoadGuided(const MLNode& node) {
    std::string error;
    if (!guided_.Load(Param(node, guided::kGuidedOpsParameter), &error)) {
        AddMessage("The guided operations could not be read (" + error + "); the step opens as SQL.", true);
    }
    mode_ = guided_.Active() ? 0 : 1;
    pane_.Editor().SetReadOnly(guided_.Active());
}

void SqlWorkbenchDialog::OnGuidedChanged() {
    // The generated SQL is the draft the rest of the Workbench runs.
    pane_.Editor().SetReadOnly(false);
    pane_.Editor().SetText(guided_.Generated().sql);
    pane_.Editor().SetReadOnly(true);
    guided_changed_at_ = ImGui::GetTime();
}

int SqlWorkbenchDialog::GuidedPartIndex(int op) const {
    const auto& names = guided_.Generated().part_names;
    if (op < 0 || op >= static_cast<int>(names.size())) return -1;
    const std::string& name = names[static_cast<size_t>(op)];
    if (name.empty()) return -1;
    for (size_t i = 0; i < pane_.Outline().parts.size(); ++i) {
        if (pane_.Outline().parts[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

int64_t SqlWorkbenchDialog::GuidedInputRows() const {
    if (inputs_.empty() || !inputs_.front().table) return -1;
    const int64_t rows = inputs_.front().table->num_rows();
    return sample_mode_ == 0 ? rows : std::min<int64_t>(1000, rows);
}

void SqlWorkbenchDialog::StartGuidedPreview() {
    if (run_ || !guided_.Generated().ok) return;
    for (const auto& input : inputs_) {
        if (!input.table) return;
    }
    guided_changed_at_ = -1.0;
    if (pane_.Outline().structured && pane_.Outline().parts.size() > 1) {
        preview_part_after_counts_ = true;
        StartRun(RunKind::Counts, 0, true);
    } else {
        StartRun(RunKind::Step, 0, true);
    }
}

// ---- Layout ---------------------------------------------------------------------

void SqlWorkbenchDialog::RenderGuided(float height) {
    // The SQL is written again when the inputs' columns change (they load
    // after the window opens): a calculated column replaces or adds by them.
    std::vector<GuidedEditor::Input> inputs;
    for (const auto& input : inputs_) inputs.push_back({input.alias, input.table});
    const bool unchanged = pane_.Text() == saved_query_;
    if (guided_.SetInputs(std::move(inputs)) && unchanged && guided_.Generated().sql != saved_query_ && !guided_.Edited()) {
        // Opening a saved step is not an edit when the SQL comes out the same.
        AddMessage("The operations write a different query for the inputs as they are now; Save to step keeps it.", false);
    }
    guided_.busy = run_ != nullptr;
    if (!guided_.Active()) {
        RenderSqlStepInGuidedMode(height);
        return;
    }
    // Preview half a second after the last change.
    if (guided_changed_at_ >= 0.0 && ImGui::GetTime() - guided_changed_at_ > 0.5) StartGuidedPreview();

    const ui::Tokens& t = ui::CurrentTokens();
    guided_.RenderLibrary(height);
    ImGui::SameLine();
    const bool counted = !part_rows_.empty() && part_rows_draft_ == pane_.Text();
    guided_.RenderCards(height, {counted ? &part_rows_ : nullptr, GuidedInputRows(), context_.checks.size()});
    ImGui::SameLine();
    ImGui::BeginGroup();
    const float editor_h = std::floor(height * 0.5f);
    if (BeginPanel("##guided_editor", ImVec2(ImGui::GetContentRegionAvail().x, editor_h), t.bg_panel)) {
        guided_.RenderEditor();
    }
    EndPanel();
    RenderGuidedPreview(height - editor_h - ImGui::GetStyle().ItemSpacing.y);
    ImGui::EndGroup();
}

void SqlWorkbenchDialog::RenderGuidedToolbar() {
    const bool pipeline = node_editor_ && node_editor_->IsDataPipelineRunning();
    const bool running = run_ != nullptr || pipeline;
    const char* busy = run_ ? "A query is running" : pipeline ? "A pipeline run is in progress" : nullptr;
    const bool ready = guided_.Active() && guided_.Generated().ok;
    if (!guided_.Active()) {
        // A step written in SQL: Guided mode only explains it until a guided
        // step is started.
        Dim("Guided mode explains this SQL step; start a guided step below to build it from operations.");
        const float reset_w = ui::ButtonWidth("Reset to saved", ui::ButtonSize::Regular);
        AlignRight(reset_w);
        if (ui::SecondaryButton("Reset to saved", pane_.Text() != saved_query_, "Nothing changed", ui::ButtonSize::Regular)) Reset();
        return;
    }
    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running && ready, running ? busy : "Fix the operations marked in red")) {
        StartRun(RunKind::Step);
    }
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("Run the operations on all rows (Ctrl+Enter)");
    ImGui::SameLine();
    if (ui::SecondaryButton("Preview on sample", !running && ready, running ? busy : "Fix the operations marked in red",
                            ui::ButtonSize::Regular)) {
        StartGuidedPreview();
    }
    ImGui::SameLine();
    static const char* const kSamples[] = {"All rows", "First 1,000", "Random 1,000"};
    ImGui::SetNextItemWidth(150.0f);
    if (ImGui::Combo("##sample", &sample_mode_, kSamples, 3)) guided_changed_at_ = ImGui::GetTime();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rows the live preview uses; Run step always uses all rows");
    ImGui::SameLine();
    if (ui::GhostButton("Undo", guided_.CanUndo(), "Nothing to undo", false, ui::ButtonSize::Regular)) guided_.Undo();
    ImGui::SameLine();
    if (ui::GhostButton("Redo", guided_.CanRedo(), "Nothing to redo", false, ui::ButtonSize::Regular)) guided_.Redo();
    ImGui::SameLine();
    guided_.RenderTemplateButton();

    const bool edited = pane_.Text() != saved_query_ || guided_.Edited();
    const float reset_w = ui::ButtonWidth("Reset to saved", ui::ButtonSize::Regular);
    const float save_w = ui::ButtonWidth("Save to step", ui::ButtonSize::Regular);
    AlignRight(reset_w + save_w + ImGui::GetStyle().ItemSpacing.x);
    if (ui::SecondaryButton("Reset to saved", edited, "Nothing changed", ui::ButtonSize::Regular)) Reset();
    ImGui::SameLine();
    if (ui::SecondaryButton("Save to step", edited && guided_.Generated().ok,
                            !edited ? "Nothing to save" : "Fix the operations marked in red", ui::ButtonSize::Regular)) {
        Apply();
    }
}

void SqlWorkbenchDialog::RenderGuidedPreview(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##guided_preview", ImVec2(ImGui::GetContentRegionAvail().x, height), t.bg_panel)) {
        static const char* const kViews[] = {"After this operation", "Summary", "SQL"};
        ui::SegmentedControl("##guided_view", kViews, 3, &guided_view_);
        if (run_) {
            ImGui::SameLine();
            Coloured(ICON_FA_SPINNER " Updating the preview...", t.running);
        } else if (sample_mode_ != 0) {
            ImGui::SameLine();
            Dim(sample_mode_ == 1 ? "on the first 1,000 rows" : "on 1,000 random rows");
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        if (ImGui::BeginChild("##guided_preview_body", ImGui::GetContentRegionAvail())) {
            if (guided_view_ == 0) {
                RenderResults();
            } else if (guided_view_ == 1) {
                const bool counted = !part_rows_.empty() && part_rows_draft_ == pane_.Text();
                if (!counted) {
                    Dim("The rows after each operation appear after the preview runs.");
                } else if (ImGui::BeginTable("##guided_summary", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30.0f);
                    ImGui::TableSetupColumn("operation", ImGuiTableColumnFlags_WidthStretch, 3.0f);
                    ImGui::TableSetupColumn("rows after", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                    ImGui::TableSetupColumn("change", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                    ImGui::TableHeadersRow();
                    int64_t previous = GuidedInputRows();
                    const auto& ops = guided_.Step().ops;
                    for (size_t i = 0; i < ops.size(); ++i) {
                        const int64_t rows = guided_.RowsAfter(i, part_rows_);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        Dim(std::to_string(i + 1));
                        ImGui::TableNextColumn();
                        ImGui::PushTextWrapPos(0.0f);
                        ImGui::TextUnformatted(guided::Summarise(ops[i]).c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::TableNextColumn();
                        if (rows < 0) {
                            Dim(ops[i].on ? "-" : "off");
                            continue;
                        }
                        ImGui::TextUnformatted(FormatCount(rows).c_str());
                        ImGui::TableNextColumn();
                        if (previous >= 0 && previous != rows) {
                            const int64_t change = rows - previous;
                            Coloured(change > 0 ? "+" + FormatCount(change) : "-" + FormatCount(-change), change > 0 ? t.success : t.caution);
                        }
                        previous = rows;
                    }
                    ImGui::EndTable();
                }
            } else {
                DimWrapped("The SQL these operations make (read-only). Edit as SQL in SQL mode to go further.");
                ImFont* mono = cyxwiz::gui::GetCodeFont();
                if (mono) ImGui::PushFont(mono);
                Coloured(guided_.Generated().sql, t.text);
                if (mono) ImGui::PopFont();
            }
        }
        ImGui::EndChild();
    }
    EndPanel();
}

// A step written in SQL, seen in Guided mode: what it does, part by part.
void SqlWorkbenchDialog::RenderSqlStepInGuidedMode(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##guided_sql_step", ImVec2(ImGui::GetContentRegionAvail().x, height), t.bg_panel)) {
        Coloured("This step is written in SQL", t.text_bright);
        DimWrapped("Here is what each part does in plain words. To build the step from operations instead, start a guided "
                   "step: it replaces this query when you save (Reset to saved brings the query back until then).");
        if (ui::SecondaryButton("Start a guided step instead")) confirm_start_guided_ = true;
        if (confirm_start_guided_) {
            ImGui::SameLine();
            Coloured("Replace the query with guided operations?", t.warning);
            ImGui::SameLine();
            if (ui::LinkButton("Yes, start")) {
                confirm_start_guided_ = false;
                guided_.Start();
            }
            ImGui::SameLine();
            if (ui::LinkButton("Cancel")) confirm_start_guided_ = false;
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_sm));
        for (size_t i = 0; i < pane_.Outline().parts.size(); ++i) {
            const auto& part = pane_.Outline().parts[i];
            Coloured(part.name, t.accent_text);
            ImGui::SameLine(200.0f);
            std::string description = part.description;
            if (description.empty()) {
                for (const auto& name : part.reads) description += (description.empty() ? "reads " : ", ") + name;
            }
            ColouredWrapped(description.empty() ? std::string("-") : description, part.description.empty() ? t.text_dim : t.text);
        }
    }
    EndPanel();
}

}  // namespace gui
