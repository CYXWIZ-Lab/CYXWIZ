#include "publish_view.h"

#include "../../core/project_manager.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cfloat>

namespace cyxwiz {

namespace {

namespace fs = std::filesystem;

ImVec4 ToneColour(studio::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case studio::Tone::Ok: return t.success;
        case studio::Tone::Warn: return t.warning;
        case studio::Tone::Error: return t.error;
        default: return t.text_dim;
    }
}

void Dim(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
}

// Where a published file goes, relative to the project when it is inside it.
std::string ProjectRelative(const std::string& path) {
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    if (root.empty() || path.empty()) return path;
    std::error_code ec;
    const fs::path rel = fs::relative(fs::path(path), fs::path(root), ec);
    // generic_string(): native() is a wide string only on Windows.
    if (ec || rel.empty() || rel.generic_string().rfind("..", 0) == 0) return path;
    return rel.generic_string();
}

}  // namespace

void PublishView::SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes,
                           const std::string& active_dataset) {
    editor_ = editor;
    nodes_ = nodes;
    const auto& pm = ProjectManager::Instance();
    rows_ = studio::OutputRows(nodes_, pm.GetProjectRoot(), pm.GetExportsPath());

    targets_.clear();
    int active = -1, last_step = -1;
    for (const auto& n : nodes_) {
        studio::Job job;
        if (!studio::JobOf(n.type, job) || n.type == ::gui::NodeType::Subgraph) continue;
        if (n.type != ::gui::NodeType::DataInput && job != studio::Job::Prepare) continue;
        targets_.push_back({n.id, n.name});
        if (job == studio::Job::Prepare) last_step = n.id;
        if (!active_dataset.empty() && (n.output_dataset == active_dataset || n.dataset_name == active_dataset)) active = n.id;
    }
    default_target_ = active >= 0 ? active : last_step >= 0 ? last_step : targets_.empty() ? -1 : targets_.front().id;

    // A Parquet export Publish added runs once it is on the canvas.
    if (!pending_name_.empty() && editor_) {
        for (const auto& r : rows_) {
            if (r.name != pending_name_ || std::find(known_outputs_.begin(), known_outputs_.end(), r.id) != known_outputs_.end())
                continue;
            pending_name_.clear();
            selected_ = r.id;
            if (editor_->ExecuteDataPipelineThrough(r.id, false)) {
                message_ = "Publishing \"" + r.name + "\": the run writes the file and its lineage.";
                message_error_ = false;
            } else {
                message_ = "\"" + r.name + "\" was added, but a pipeline run is in progress; Publish again when it ends.";
                message_error_ = true;
            }
            break;
        }
    }
    if (std::none_of(rows_.begin(), rows_.end(), [&](const studio::OutputRow& r) { return r.id == selected_; }))
        selected_ = rows_.empty() ? -1 : rows_.front().id;
    for (const auto& r : rows_)
        if (r.id == selected_) LoadLineage(r.writes);
    if (rows_.empty()) LoadLineage({});
}

bool PublishView::IsRunning() const { return editor_ && editor_->IsDataPipelineRunning(); }

void PublishView::LoadLineage(const std::string& path) {
    std::error_code ec;
    const std::string file = path.empty() ? std::string() : LineagePathFor(path);
    const auto time = file.empty() ? fs::file_time_type{} : fs::last_write_time(file, ec);
    if (ec) {
        // No lineage file (yet).
        lineage_path_ = path;
        lineage_ok_ = false;
        lineage_error_.clear();
        return;
    }
    if (path == lineage_path_ && time == lineage_time_ && (lineage_ok_ || !lineage_error_.empty())) return;
    lineage_path_ = path;
    lineage_time_ = time;
    lineage_error_.clear();
    lineage_ok_ = !path.empty() && ReadLineage(path, lineage_, lineage_error_);
}

void PublishView::OpenPublish() {
    open_dialog_ = true;
    dialog_error_.clear();
    dialog_output_ = selected_;
    dialog_target_ = default_target_;
    dialog_name_.clear();
    dialog_path_.clear();
    for (const auto& r : rows_)
        if (r.id == dialog_output_) {
            dialog_name_ = r.name;
            dialog_path_ = ProjectRelative(r.writes);
        }
    if (dialog_output_ < 0) {
        for (const auto& t : targets_)
            if (t.id == dialog_target_) {
                dialog_name_ = "Export " + t.name;
                dialog_path_ = studio::SuggestedExportPath(t.name);
            }
    }
}

void PublishView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (rows_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "No output in this graph yet. Publish... adds a Parquet export after a step and runs it: "
                                       "it writes the file and its lineage.");
        ImGui::PopTextWrapPos();
    } else {
        RenderTable();
        ImGui::Spacing();
        RenderLineage();
    }
    ImGui::Spacing();
    const bool can = editor_ != nullptr;
    if (ui::PrimaryButton("Publish...", can && !IsRunning(), !can ? "No graph is open" : "The Data Pipeline is running",
                          ui::ButtonSize::Small))
        OpenPublish();
    ImGui::SameLine();
    if (ui::SecondaryButton("Add Data Output", can && default_target_ >= 0, "The graph has no step to output yet",
                            ui::ButtonSize::Small))
        AddDataOutput();
    if (ImGui::IsItemHovered() && default_target_ >= 0)
        ImGui::SetTooltip("Adds a Data Output node reading the step chosen in the rail (or the last step); its format and file "
                          "are set in its dialog");
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Publishing goes through the graph's Export / Data Output nodes. Each time one writes its "
                                     "file, the run writes <file>.lineage.json beside it: the recipe and its revision, the "
                                     "input files' SHA-256, the result's shape and SHA-256, and the checks.");
    ImGui::PopTextWrapPos();
    RenderDialog();
}

void PublishView::RenderTable() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < 760.0f;
    if (!ImGui::BeginTable("##outputs", narrow ? 3 : 4,
                           ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                               ImGuiTableFlags_NoSavedSettings))
        return;
    float last_w = ImGui::CalcTextSize("last written").x;
    for (const auto& r : rows_) last_w = std::max(last_w, ImGui::CalcTextSize((ICON_FA_CIRCLE " " + r.last).c_str()).x);
    const float show_w = ImGui::CalcTextSize("Show").x + 8.0f;
    ImGui::TableSetupColumn("output (node)", ImGuiTableColumnFlags_WidthStretch, 1.4f);
    if (!narrow) ImGui::TableSetupColumn("reads", ImGuiTableColumnFlags_WidthStretch, 0.9f);
    ImGui::TableSetupColumn("writes", ImGuiTableColumnFlags_WidthStretch, 2.4f);
    ImGui::TableSetupColumn("last written", ImGuiTableColumnFlags_WidthFixed, std::max(last_w, show_w) + 8.0f);
    ImGui::TableHeadersRow();
    ImFont* mono = gui::GetCodeFont();
    for (const auto& r : rows_) {
        ImGui::PushID(r.id);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ui::PushListRowColours();
        if (ImGui::Selectable("##row", r.id == selected_, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                              ImVec2(0.0f, 0.0f)))
            selected_ = r.id;
        ui::PopListRowColours();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s node\nClick: show its lineage", r.kind.c_str());
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_bright, "%s", r.name.c_str());
        if (narrow) ImGui::TextColored(t.text_dim, "reads %s", r.reads.empty() ? "nothing linked" : r.reads.c_str());
        else {
            ImGui::TableNextColumn();
            ImGui::TextColored(r.reads.empty() ? t.text_faint : t.text_dim, "%s", r.reads.empty() ? "nothing linked" : r.reads.c_str());
        }
        ImGui::TableNextColumn();
        if (mono) ImGui::PushFont(mono);
        const std::string shown = r.writes.empty() ? std::string("(named after its table when it runs)") : ProjectRelative(r.writes);
        ImGui::TextColored(r.written ? t.text : t.text_dim, "%s", shown.c_str());
        if (mono) ImGui::PopFont();
        if (!r.writes.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r.writes.c_str());
        if (!r.detail.empty()) ImGui::TextColored(t.error, "%s", r.detail.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        ImGui::TextColored(ToneColour(r.tone), ICON_FA_CIRCLE " %s", r.last.c_str());
        if (ui::LinkButton("Show") && show_on_canvas) show_on_canvas({r.id});
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void PublishView::RenderLineage() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.035f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("##lineage", ImVec2(std::min(760.0f, ImGui::GetContentRegionAvail().x), 0.0f),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();
    ImGui::TextColored(t.text_dim, "LINEAGE BESIDE THE FILE");
    ImGui::PushTextWrapPos(0.0f);
    if (lineage_path_.empty()) {
        ImGui::TextColored(t.text_faint, "This output names its file after its table when it runs; its lineage shows after that.");
    } else if (!lineage_ok_) {
        ImGui::TextColored(t.text_faint, "%s",
                           lineage_error_.empty() ? "No lineage file yet: the next time this output writes its file, the run "
                                                    "writes the lineage beside it."
                                                  : ("The lineage file could not be read: " + lineage_error_).c_str());
    } else {
        const LineageSummary s = SummarizeLineage(lineage_);
        if (ImGui::BeginTable("##lineage_rows", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) {
            ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Written").x + 24.0f);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            const auto row = [&](const char* what, const std::string& value, const ImVec4& colour) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(t.text_dim, "%s", what);
                ImGui::TableNextColumn();
                ImGui::TextColored(colour, "%s", value.c_str());
            };
            row("Made by", s.made_by, t.text);
            row("Inputs", s.inputs, t.text);
            row("Result", s.result, t.text);
            row("Checks", s.checks, lineage_.checks.empty() ? t.text_dim : s.checks_ok ? t.success : t.error);
            row("Written", s.written, t.text_dim);
            ImGui::EndTable();
        }
        if (ImGui::TreeNodeEx("Steps", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            for (const auto& step : lineage_.steps)
                ImGui::TextColored(t.text_dim, "%s  \xC2\xB7 %s%s", step.name.c_str(), step.type.c_str(),
                                   step.recipe.empty() ? "" : ("  \xC2\xB7 recipe " + step.recipe).c_str());
            ImGui::TreePop();
        }
        ImGui::TextColored(t.text_faint, "%s", LineagePathFor(lineage_path_).c_str());
    }
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void PublishView::RenderDialog() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (open_dialog_) {
        ImGui::OpenPopup("Publish dataset");
        open_dialog_ = false;
    }
    // A fixed width, and a height that follows the fields (Reads appears for a
    // new export), so the buttons never scroll out of view.
    const float width = std::min(560.0f, ImGui::GetMainViewport()->WorkSize.x - 32.0f);
    ImGui::SetNextWindowSizeConstraints(ImVec2(width, 0.0f), ImVec2(width, FLT_MAX));
    if (!ImGui::BeginPopupModal("Publish dataset", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize))
        return;

    // Which output it goes through, or a new Parquet export after a step.
    Dim("Output");
    ImGui::SameLine(110.0f);
    std::string current = "New Parquet export";
    for (const auto& r : rows_)
        if (r.id == dialog_output_) current = r.name;
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##output", current.c_str())) {
        for (const auto& r : rows_) {
            ImGui::PushID(r.id);
            if (ImGui::Selectable(r.name.c_str(), r.id == dialog_output_)) {
                dialog_output_ = r.id;
                dialog_name_ = r.name;
                dialog_path_ = ProjectRelative(r.writes);
            }
            ImGui::PopID();
        }
        if (ImGui::Selectable("New Parquet export", dialog_output_ < 0)) {
            dialog_output_ = -1;
            for (const auto& target : targets_)
                if (target.id == dialog_target_) {
                    dialog_name_ = "Export " + target.name;
                    dialog_path_ = studio::SuggestedExportPath(target.name);
                }
        }
        ImGui::EndCombo();
    }
    if (dialog_output_ < 0) {
        Dim("Reads");
        ImGui::SameLine(110.0f);
        std::string reads;
        for (const auto& target : targets_)
            if (target.id == dialog_target_) reads = target.name;
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##reads", reads.empty() ? "choose a step" : reads.c_str())) {
            for (const auto& target : targets_) {
                ImGui::PushID(target.id);
                if (ImGui::Selectable(target.name.c_str(), target.id == dialog_target_)) {
                    dialog_target_ = target.id;
                    dialog_name_ = "Export " + target.name;
                    dialog_path_ = studio::SuggestedExportPath(target.name);
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
    }
    Dim("Name");
    ImGui::SameLine(110.0f);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##name", "the node's name", &dialog_name_);
    Dim("Location");
    ImGui::SameLine(110.0f);
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##path", "datasets/published/name.parquet (relative to the project)", &dialog_path_);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim, "%s",
                       dialog_output_ < 0 ? "Adds an Export Parquet node reading the step (a recipe step: the recipe's result) and runs "
                                            "it. It writes the file and <file>.lineage.json beside it: recipe revision, input "
                                            "digests, result digest, checks."
                                          : "Sets this node's name and file and runs it. It writes the file and <file>.lineage.json "
                                            "beside it: recipe revision, input digests, result digest, checks.");
    if (!dialog_error_.empty()) ImGui::TextColored(t.error, "%s", dialog_error_.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    const float buttons_w = ui::ButtonWidth("Cancel", ui::ButtonSize::Regular) + ui::ButtonWidth("Publish", ui::ButtonSize::Regular) +
                            ImGui::GetStyle().ItemSpacing.x;
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - buttons_w));
    if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::SameLine();
    if (ui::PrimaryButton("Publish", !IsRunning(), "The Data Pipeline is running")) {
        Publish();
        if (dialog_error_.empty()) ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void PublishView::Publish() {
    dialog_error_.clear();
    if (!editor_) return;
    std::string path = dialog_path_;
    while (!path.empty() && path.back() == ' ') path.pop_back();
    if (path.empty()) {
        dialog_error_ = "Enter where the file goes.";
        return;
    }
    std::string name = dialog_name_.empty() ? "Export " + fs::path(path).stem().string() : dialog_name_;
    const std::map<std::string, std::string> parameters{{"file_path", path}, {"path_base", "project"}};
    if (dialog_output_ >= 0) {
        if (!editor_->SetStepParameters(dialog_output_, parameters, {"path"}, name)) {
            dialog_error_ = "The output is no longer in the graph.";
            return;
        }
        selected_ = dialog_output_;
        if (!editor_->ExecuteDataPipelineThrough(dialog_output_, false)) {
            message_ = "Saved the output's name and file, but a pipeline run is in progress; Publish again when it ends.";
            message_error_ = true;
            return;
        }
        message_ = "Publishing \"" + name + "\" (Ctrl+Z on the canvas undoes the change): the run writes the file and its lineage.";
        message_error_ = false;
        return;
    }
    if (dialog_target_ < 0) {
        dialog_error_ = "Choose the step it reads.";
        return;
    }
    // Runs once the node is on the canvas (SetGraph): known outputs tell the new one apart.
    known_outputs_.clear();
    for (const auto& r : rows_) known_outputs_.push_back(r.id);
    pending_name_ = name;
    editor_->RequestBranchAfter(dialog_target_, ::gui::NodeType::ExportParquet, parameters, name);
    message_ = "Added \"" + name + "\" (Ctrl+Z on the canvas undoes it); it runs as soon as it is in the graph.";
    message_error_ = false;
}

void PublishView::AddDataOutput() {
    if (!editor_ || default_target_ < 0) return;
    std::string reads;
    for (const auto& target : targets_)
        if (target.id == default_target_) reads = target.name;
    editor_->RequestBranchAfter(default_target_, ::gui::NodeType::DataOutput);
    message_ = "Added a Data Output reading \"" + reads + "\" (Ctrl+Z on the canvas undoes it); set its format and file in its "
               "dialog, then Publish through it.";
    message_error_ = false;
}

}  // namespace cyxwiz
