// Studio Debugger > Overview > Runs and a training run's snapshots (TOFIX178
// 6.3d, approved mockup board Snapshots): Local Debug runs and training runs
// that kept layer snapshots in one list, newest first; picking a training
// run shows its snapshots in Model > Layers and its snapshot table in
// Overview > Compare. The wording comes from training_snapshots_presentation.

#include "studio_debugger_panel.h"
#include "studio_debugger_colors.h"
#include "studio_debugger_widgets.h"

#include "../../core/layer_view_presentation.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <algorithm>

namespace cyxwiz {

namespace ts = training_snapshots;

namespace {

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

std::string StudioDebuggerPanel::DataItemsNotStarted(const DataItemsStart& start, const std::string& what) {
    if (!start.loading.empty())
        return "loading " + start.loading + " first (its data is not loaded in this session); press Compute when it "
               "is loaded";
    return what + " need the graph's data and a clean compile";
}

const StudioDebuggerSampleFacts& StudioDebuggerPanel::ShownSampleFacts() const {
    return shown_training_run_.empty() ? session_.sample_facts : shown_training_facts_;
}

const std::optional<TrainingRunRecord>& StudioDebuggerPanel::ShownTrainingRecord() const {
    static const std::optional<TrainingRunRecord> none;
    return shown_training_run_.empty() ? none : shown_training_record_;
}

void StudioDebuggerPanel::ShowTrainingRun(const std::string& run_id) {
    shown_training_run_ = run_id;
    // Model > Layers reads this run from now on (a snapshot opened from
    // Compare before Layers is shown is kept).
    layer_capture_.reset();
    ++layer_capture_id_;
    layer_capture_run_id_ = run_id;
    layer_capture_label_.clear();
    layer_capture_labels_.clear();
    layer_capture_listed_ = false;
    selected_layer_ = 0;
    shown_training_record_.reset();
    shown_training_captures_.clear();
    shown_training_read_ = false;
    shown_training_facts_ = {};
    struct Loaded {
        std::optional<TrainingRunRecord> record;
        std::vector<LayerCaptureInfo> captures;
    };
    auto loaded = std::make_shared<Loaded>();
    AsyncTaskManager::Instance().RunAsync(
        "Studio Debugger: read a training run",
        [loaded, run_id](LambdaTask&) {
            loaded->record = LoadTrainingRunRecord(run_id);
            loaded->captures = ListLayerCaptureInfo(run_id);
        },
        nullptr,
        [this, loaded, run_id](bool, const std::string&) {
            if (shown_training_run_ != run_id) return;  // another run is shown now
            shown_training_record_ = loaded->record;
            shown_training_captures_ = loaded->captures;
            shown_training_read_ = true;
            if (const auto& r = loaded->record) {
                shown_training_facts_.item_names = r->item_names;
                shown_training_facts_.class_names = r->class_names;
                shown_training_facts_.output_rule = r->output_rule;
                if (!r->label_values.empty())
                    shown_training_facts_.labels = layer_view::ItemLabels(r->label_values, r->label_shape);
            }
        });
}

void StudioDebuggerPanel::RenderRunHistory() {
    if (!history_loaded_) RequestRunHistoryRefresh();

    std::vector<ts::RunRow> rows = training_run_rows_;
    for (const auto& run : session_.run_history) rows.push_back(ts::DebugRunRow(run));
    std::stable_sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.when > b.when; });

    ImGui::BeginChild("StudioDebuggerRunHistory", ImVec2(0.0f, 0.0f), false);
    Wrapped(DebuggerMuted(), "Saved runs, newest first. A training run is listed when it kept layer snapshots.");
    ImGui::Spacing();
    if (rows.empty()) {
        Wrapped(DebuggerMuted(), "No saved runs yet: run Local Debug, or train with Layer capture on.");
        ImGui::EndChild();
        return;
    }
    const ui::Tokens& t = ui::CurrentTokens();
    for (const auto& row : rows) {
        ImGui::PushID(row.run_id.c_str());
        const bool shown = row.training ? row.run_id == shown_training_run_
                                        : shown_training_run_.empty() && row.run_id == session_.run_id;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, shown ? t.bg_raised : DebuggerPanelBg());
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 8.0f));
        ImGui::BeginChild("run", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);

        // Kind badge, title and time on one line (wrapping when narrow).
        const ImVec2 badge = ImGui::CalcTextSize(row.kind.c_str());
        const ImVec2 pad(8.0f, 2.0f);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(badge.x + pad.x * 2, badge.y + pad.y * 2));
        ImVec4 fill = row.training ? t.accent : t.bg_window;
        if (row.training) fill.w = 0.35f;
        ImGui::GetWindowDrawList()->AddRectFilled(at, ImVec2(at.x + badge.x + pad.x * 2, at.y + badge.y + pad.y * 2),
                                                  ui::ToU32(fill), 9.0f);
        ImGui::GetWindowDrawList()->AddText(ImVec2(at.x + pad.x, at.y + pad.y),
                                            ui::ToU32(row.training ? t.text_bright : t.text_dim), row.kind.c_str());
        const bool current = !row.training && row.run_id == current_run_id_;
        const std::string title = row.title + (current ? "  (latest)" : "");
        SameLineIfFits(ImGui::CalcTextSize(title.c_str()).x, 10.0f);
        ImGui::TextColored(DebuggerBright(), "%s", title.c_str());
        if (!row.when.empty()) {
            SameLineIfFits(ImGui::CalcTextSize(row.when.c_str()).x, 10.0f);
            ImGui::TextColored(DebuggerMuted(), "%s", row.when.c_str());
        }
        Wrapped(DebuggerText(), row.detail);

        // A Local Debug run's other facts, under Details.
        bool details_clicked = false;
        if (!row.details.empty()) {
            const bool open = open_run_details_.count(row.run_id) > 0;
            if (ui::LinkButton(open ? "Hide" : "Details")) {
                details_clicked = true;
                if (open)
                    open_run_details_.erase(row.run_id);
                else
                    open_run_details_.insert(row.run_id);
            }
            if (open && ImGui::BeginTable("##details", 4, ImGuiTableFlags_SizingStretchProp)) {
                for (size_t i = 0; i < row.details.size(); ++i) {
                    if (i % 2 == 0) ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(DebuggerMuted(), "%s", row.details[i].first.c_str());
                    ImGui::TableNextColumn();
                    Wrapped(DebuggerText(), row.details[i].second);
                }
                ImGui::EndTable();
            }
        }
        const bool clicked = !details_clicked && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
                             ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsAnyItemActive() &&
                             !ImGui::IsAnyItemHovered();
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (clicked && !shown) {
            if (row.training) {
                ShowTrainingRun(row.run_id);
            } else {
                shown_training_run_.clear();
                LoadStoredRun(row.run_id);
            }
        }
        ImGui::Spacing();
        ImGui::PopID();
    }
    Wrapped(DebuggerFaint(), "Click a run to show it; Compare and Model \xE2\x80\xBA Layers follow the shown run.");
    ImGui::EndChild();
}

void StudioDebuggerPanel::RenderSnapshotTable() {
    const auto table = ts::SnapshotTableFor(shown_training_run_, shown_training_record_, shown_training_captures_);
    ImGui::BeginChild("StudioDebuggerSnapshots", ImVec2(0.0f, 0.0f), false);
    CardHeading(table.heading.c_str());
    ImGui::Spacing();
    if (table.rows.empty()) {
        Wrapped(DebuggerMuted(), shown_training_read_ ? "This run has no snapshots left in the run store."
                                                      : "Reading the run's snapshots...");
        ImGui::EndChild();
        return;
    }
    if (ImGui::BeginTable("##snapshots", 6, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX)) {
        ImGui::TableSetupColumn("Epoch");
        ImGui::TableSetupColumn("Training loss");
        ImGui::TableSetupColumn("Validation loss");
        ImGui::TableSetupColumn(table.metric_header.c_str());
        ImGui::TableSetupColumn("Size");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();
        for (const auto& row : table.rows) {
            ImGui::PushID(row.label.c_str());
            ImGui::TableNextRow();
            // Text cells at the link's height, so every row lines up.
            const auto cell = [](const std::string& text) {
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(text.c_str());
            };
            cell(std::to_string(row.epoch));
            cell(row.train_loss);
            cell(row.val_loss);
            cell(row.val_metric);
            cell(row.size);
            ImGui::TableNextColumn();
            if (ui::LinkButton("Open in Model \xE2\x80\xBA Layers")) {
                ShowLayerSnapshot(row.label);
                active_section_ = StudioDebuggerSection::Model;
                section_view_[static_cast<int>(StudioDebuggerSection::Model)] = 0;
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::Spacing();
    Wrapped(DebuggerMuted(), table.note);
    ImGui::EndChild();
}

}  // namespace cyxwiz
