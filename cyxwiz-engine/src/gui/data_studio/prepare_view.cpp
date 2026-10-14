#include "prepare_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/guided_sql.h"
#include "../../core/sql_step_presentation.h"
#include "../../core/sql_transform.h"
#include "../../data/arrow_data_table.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>

namespace cyxwiz {

namespace {

constexpr int64_t kGridRows = 1000;

ImVec4 ToneColour(studio::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case studio::Tone::Ok: return t.success;
        case studio::Tone::Warn: return t.warning;
        case studio::Tone::Error: return t.error;
        default: return t.text_dim;
    }
}

std::string Param(const ::gui::MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

// A tinted pill naming what a step is ("SQL Query node"), no outline.
void KindChip(const std::string& text) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
    const ImVec2 size(ts.x + 16.0f, ImGui::GetFrameHeight() - 4.0f);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImGui::GetWindowDrawList()->AddRectFilled(at, ImVec2(at.x + size.x, at.y + size.y), ui::ToU32(ui::Mix(t.bg_window, t.info, 0.18f)),
                                              size.y * 0.5f);
    ImGui::GetWindowDrawList()->AddText(ImVec2(at.x + 8.0f, at.y + (size.y - ts.y) * 0.5f), ui::ToU32(t.info), text.c_str());
}

}  // namespace

// Rows per part of the step's query, counted on its inputs off the UI thread
// (the Workbench's Parts count, SqlTransform on the inputs in memory).
struct PrepareView::PartCounts {
    std::string key;
    std::thread worker;
    std::atomic<bool> done{false};
    SqlTransform transform;
    std::vector<std::pair<std::string, int64_t>> counts;  // in the query's order
    std::string error;
    ~PartCounts() {
        if (worker.joinable()) {
            transform.Interrupt();
            worker.join();
        }
    }
};

PrepareView::PrepareView() {
    grid_.options.data_bars = false;
    grid_.options.rows_per_page = 200;
}

PrepareView::~PrepareView() = default;

const studio::GraphNode* PrepareView::NodeById(int id) const {
    for (const auto& n : nodes_)
        if (n.id == id) return &n;
    return nullptr;
}

int PrepareView::RecipeOf(int step_id) const {
    if (!editor_) return -1;
    for (int r : recipes_)
        for (const auto& s : editor_->GetSubgraphSummary(r).steps)
            if (s.id == step_id) return r;
    return -1;
}

void PrepareView::SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes, const std::vector<int>& prepare_nodes) {
    editor_ = editor;
    nodes_ = nodes;
    recipes_.clear();
    loose_.clear();
    std::set<int> in_recipes;
    for (const auto& n : nodes_)
        if (n.recipe && editor_) {
            recipes_.push_back(n.id);
            for (const auto& s : editor_->GetSubgraphSummary(n.id).steps) in_recipes.insert(s.id);
        }
    for (int id : prepare_nodes)
        if (!in_recipes.count(id) && std::find(recipes_.begin(), recipes_.end(), id) == recipes_.end()) loose_.push_back(id);
    if (recipe_ >= 0 && std::find(recipes_.begin(), recipes_.end(), recipe_) == recipes_.end()) recipe_ = -1;
    if (recipe_ < 0 && !recipes_.empty() && loose_.empty()) recipe_ = recipes_.front();

    std::vector<studio::GraphNode> steps;
    const auto add = [&](int id) {
        if (const auto* n = NodeById(id)) steps.push_back(*n);
    };
    if (recipe_ >= 0)
        for (const auto& s : editor_->GetSubgraphSummary(recipe_).steps) add(s.id);
    else
        for (int id : loose_) add(id);
    steps_ = studio::StepRows(steps);
    if (loaded_ >= 0 && editor_) {
        reads_.clear();
        for (const auto& input : editor_->GetDataStepContext(loaded_).inputs) reads_ += (reads_.empty() ? "" : ", ") + input.name;
    }
    const bool still_there = std::any_of(steps_.begin(), steps_.end(), [&](const studio::StepRow& r) { return r.id == selected_; });
    if (!still_there) {
        selected_ = -1;
        for (const auto& r : steps_)
            if (r.sql) {
                selected_ = r.id;
                break;
            }
        if (selected_ < 0 && !steps_.empty()) selected_ = steps_.front().id;
    }
}

void PrepareView::Select(int step_id) {
    const int recipe = RecipeOf(step_id);
    if (recipe >= 0) recipe_ = recipe;
    selected_ = step_id;
}

void PrepareView::LoadStep(int step_id) {
    loaded_ = step_id;
    message_.clear();
    const ::gui::MLNode* node = editor_ ? editor_->FindStep(step_id) : nullptr;
    saved_query_ = node ? Param(*node, "query") : std::string();
    guided_ = node && !Param(*node, guided::kGuidedOpsParameter).empty();
    pane_.Editor().SetText(saved_query_);
    pane_.Editor().SetReadOnly(guided_);
    pane_.Refresh();
    // What it reads, and completion: the columns of its inputs when they are in memory.
    reads_.clear();
    std::vector<std::string> columns;
    if (editor_) {
        const auto& facts = editor_->GetNodeRunFacts();
        for (const auto& input : editor_->GetDataStepContext(step_id).inputs) {
            reads_ += (reads_.empty() ? "" : ", ") + input.name;
            const auto it = facts.find(input.id);
            if (it == facts.end() || it->second.output_dataset.empty()) continue;
            if (auto ds = DataRegistry::Instance().GetArrowDataset(it->second.output_dataset))
                if (auto table = ds->GetArrowTable())
                    for (const auto& field : table->schema()->fields()) columns.push_back(field->name());
        }
    }
    pane_.SetCompletionColumns(std::move(columns));
}

void PrepareView::SaveDraft() {
    if (!editor_ || loaded_ < 0 || guided_) return;
    const std::string text = pane_.Editor().GetText();
    if (!editor_->SetStepParameters(loaded_, {{"query", text}})) {
        message_ = "The step is no longer in the graph.";
        message_error_ = true;
        return;
    }
    saved_query_ = text;
    message_ = "Saved to the step (Ctrl+Z on the canvas undoes it).";
    message_error_ = false;
}

void PrepareView::RunStep(int step_id) {
    if (!editor_) return;
    if (step_id == loaded_ && pane_.Text() != saved_query_) SaveDraft();
    if (!editor_->ExecuteDataPipelineThrough(step_id, true)) {
        message_ = "A pipeline run is in progress; run again when it ends.";
        message_error_ = true;
        return;
    }
    message_.clear();  // the result line below says it is running, then what it made
}

void PrepareView::OpenInDialog(int step_id) {
    if (!editor_) return;
    if (raise_canvas) raise_canvas();
    if (editor_->GetNodeForConfiguration(step_id)) {
        editor_->OpenNodeConfiguration(step_id);
        return;
    }
    // Inside a collapsed recipe: the canvas expands it, shows the step, then opens it.
    const int recipe = RecipeOf(step_id);
    if (recipe >= 0) editor_->RequestSubgraphAction(recipe, ::gui::NodeEditor::SubgraphRequest::OpenStep, step_id);
}

void PrepareView::AddStepAfter(int step_id, bool guided) {
    if (!editor_) return;
    // A step is added on the canvas: a collapsed recipe opens first.
    if (!editor_->GetNodeForConfiguration(step_id)) {
        const int recipe = RecipeOf(step_id);
        if (recipe >= 0) editor_->RequestSubgraphAction(recipe, ::gui::NodeEditor::SubgraphRequest::Expand);
    }
    if (guided) editor_->RequestGuidedStepAfter(step_id);
    else editor_->RequestSqlStepAfter(step_id, "SQL step", "SELECT *\nFROM input");
    if (raise_canvas) raise_canvas();
    message_ = guided ? "Added a guided step after this one; edit its operations in the Workbench (Open in Workbench)."
                      : "Added an SQL step after this one; pick it in the list to write its query.";
    message_error_ = false;
}

void PrepareView::StartPartCounts(const studio::GraphNode& node) {
    const ::gui::MLNode* step = editor_ ? editor_->FindStep(node.id) : nullptr;
    if (!step) return;
    const std::string query = Param(*step, "query");
    const auto outline = sqlstep::OutlineSqlQuery(query);
    const std::string counts_sql = sqlstep::PartCountsQuery(query, outline);
    auto job = std::make_shared<PartCounts>();
    job->key = node.output_dataset + "@" + node.finished_at;
    parts_ = job;
    if (counts_sql.empty() || outline.parts.size() < 2) {
        job->done = true;
        return;
    }
    const auto names = sqlstep::InputNames(step->parameters);
    const auto& facts = editor_->GetNodeRunFacts();
    std::vector<SqlTransformInput> inputs;
    for (const auto& input : editor_->GetDataStepContext(node.id).inputs) {
        const auto it = facts.find(input.id);
        std::shared_ptr<arrow::Table> table;
        if (it != facts.end() && !it->second.output_dataset.empty())
            if (auto ds = DataRegistry::Instance().GetArrowDataset(it->second.output_dataset)) table = ds->GetArrowTable();
        if (!table) {
            job->error = "its input \"" + input.name + "\" is not in memory";
            job->done = true;
            return;
        }
        inputs.push_back({input.pin < static_cast<int>(names.size()) ? names[static_cast<size_t>(input.pin)] : std::string("input"), table});
    }
    std::vector<std::string> order;
    for (const auto& part : outline.parts) order.push_back(part.name);
    PartCounts* raw = job.get();
    job->worker = std::thread([raw, counts_sql, inputs, order] {
        const SqlTransformResult r = raw->transform.Run(counts_sql, inputs);
        if (!r.ok || !r.table) {
            raw->error = r.cancelled ? std::string() : r.error;
        } else {
            std::map<std::string, int64_t> by_part;
            const auto parts = r.table->GetColumnByName("part");
            const auto rows = r.table->GetColumnByName("row_count");
            for (int64_t i = 0; parts && rows && i < r.table->num_rows(); ++i) {
                auto p = parts->GetScalar(i);
                auto n = rows->GetScalar(i);
                if (!p.ok() || !n.ok() || !(*p)->is_valid || !(*n)->is_valid) continue;
                const std::string name = std::static_pointer_cast<arrow::StringScalar>(*p)->ToString();
                by_part[name] = std::stoll((*n)->ToString());
            }
            for (const auto& name : order)
                if (const auto it = by_part.find(name); it != by_part.end()) raw->counts.emplace_back(name, it->second);
        }
        raw->done = true;
    });
}

void PrepareView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!editor_) {
        ImGui::TextColored(t.text_dim, "No graph is open.");
        return;
    }
    if (recipes_.empty() && loose_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "This graph has no preparation steps yet.");
        ImGui::TextColored(t.text_faint, "Explore > Query's \"Turn into SQL step\" adds one after a step of the graph; on the canvas, "
                                         "an SQL Query node after a Data Input, grouped as a Preparation Recipe, is a recipe.");
        ImGui::PopTextWrapPos();
        return;
    }
    const float height = ImGui::GetContentRegionAvail().y;
    RenderSteps(height);
    ImGui::SameLine(0.0f, 14.0f);
    ImGui::BeginChild("##prepare_step", ImVec2(0.0f, height));
    const studio::GraphNode* node = NodeById(selected_);
    if (!node) {
        ImGui::TextColored(t.text_dim, "Pick a step on the left.");
    } else {
        ImGui::PushID(node->id);
        if (node->type == ::gui::NodeType::SQLQuery) RenderSqlStep(*node, height);
        else RenderOtherStep(*node);
        ImGui::PopID();
    }
    ImGui::EndChild();
}

void PrepareView::RenderSteps(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.03f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
    ImGui::BeginChild("##prepare_steps", ImVec2(list_width_, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    // Which recipe (or the steps outside one), when there is a choice.
    if (recipes_.size() + (loose_.empty() ? 0 : 1) > 1) {
        for (size_t i = 0; i <= recipes_.size(); ++i) {
            const bool loose = i == recipes_.size();
            if (loose && loose_.empty()) continue;
            const int id = loose ? -1 : recipes_[i];
            const studio::GraphNode* r = loose ? nullptr : NodeById(id);
            const std::string label = loose ? "Steps outside a recipe" : r ? r->name : "recipe";
            ImGui::PushID(id);
            if (ImGui::Selectable(label.c_str(), recipe_ == id)) recipe_ = id;
            ImGui::PopID();
        }
        ImGui::Spacing();
    }
    ImGui::TextColored(t.text_dim, recipe_ >= 0 ? "RECIPE STEPS (NODES)" : "STEPS (NODES)");
    for (const auto& row : steps_) {
        ImGui::PushID(row.id);
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        const float line = ImGui::GetTextLineHeight();
        if (ImGui::Selectable("##step", row.id == selected_, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(w, line * 2.0f + 4.0f))) {
            selected_ = row.id;
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && show_on_canvas) show_on_canvas({row.id});
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            const std::string tip = row.kind + (row.tip.empty() ? "" : "\n" + row.tip) + "\nDouble-click: show it on the canvas.";
            ImGui::SetTooltip("%s", tip.c_str());
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const char* icon = row.sql ? ICON_FA_CODE : row.kind == "Row Count Check" ? ICON_FA_CIRCLE_CHECK : ICON_FA_GEAR;
        dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(row.sql ? t.accent_text : t.text_faint), icon);
        const ImVec4 clip(at.x, at.y, at.x + w - 4.0f, at.y + line * 2.0f + 4.0f);
        dl->AddText(nullptr, 0.0f, ImVec2(at.x + 26.0f, at.y + 2.0f), ui::ToU32(t.text_bright), row.name.c_str(), nullptr, 0.0f, &clip);
        const std::string meta = row.kind + " \xC2\xB7 " + row.meta;
        dl->AddText(nullptr, 0.0f, ImVec2(at.x + 26.0f, at.y + line + 3.0f),
                    ui::ToU32(row.tone == studio::Tone::Plain ? t.text_dim : ToneColour(row.tone)), meta.c_str(), nullptr, 0.0f, &clip);
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, recipe_ >= 0 ? "The recipe is the Preparation Recipe node on the canvas; its steps run in this order."
                                                  : "These nodes prepare data outside a Preparation Recipe.");
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void PrepareView::RenderSqlStep(const studio::GraphNode& node, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (loaded_ != node.id) LoadStep(node.id);
    pane_.Refresh();
    // The query changed elsewhere (the Workbench, Properties) and nothing is typed here: show it.
    if (const ::gui::MLNode* step = editor_->FindStep(node.id)) {
        const std::string now = Param(*step, "query");
        if (now != saved_query_ && pane_.Text() == saved_query_) LoadStep(node.id);
    }
    const bool dirty = pane_.Text() != saved_query_;
    const bool running = editor_->IsDataPipelineRunning();

    // Header: the step, what it is, what it reads.
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", node.name.c_str());
    ImGui::SameLine(0.0f, 10.0f);
    KindChip(guided_ ? "SQL Query node \xC2\xB7 guided" : "SQL Query node");
    if (!reads_.empty()) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "reads %s", reads_.c_str());
    }
    if (dirty) {
        ImGui::SameLine(0.0f, 10.0f);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.warning, "not saved");
    }
    if (guided_) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "A guided step: its SQL is made from its operations, edited in the Workbench's Guided mode "
                                       "(Open in Workbench). Shown here read-only.");
        ImGui::PopTextWrapPos();
    }

    pane_.Render(std::clamp(height * 0.38f, 140.0f, 360.0f));
    // Ctrl+Enter runs the step, as in the Workbench.
    if (pane_.Editor().IsFocused() && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false) && !running) RunStep(node.id);

    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running, "The Data Pipeline is running", ui::ButtonSize::Small)) RunStep(node.id);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Saves the query to the step, then runs the Data Pipeline through it and its checks (Ctrl+Enter).");
    ImGui::SameLine();
    if (ui::SecondaryButton("Save to step", dirty && !guided_, "No changes to save")) SaveDraft();
    if (dirty) {
        ImGui::SameLine();
        if (ui::LinkButton("Reset")) LoadStep(node.id);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Open in Workbench")) OpenInDialog(node.id);
    ImGui::SameLine();
    if (ui::SecondaryButton("Add a step after " ICON_FA_CARET_DOWN)) ImGui::OpenPopup("##add_after");
    if (ImGui::BeginPopup("##add_after")) {
        if (ImGui::MenuItem("SQL step")) AddStepAfter(node.id, false);
        if (ImGui::MenuItem("Guided step")) AddStepAfter(node.id, true);
        ImGui::EndPopup();
    }
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    RenderResult(node);
}

void PrepareView::RenderOtherStep(const studio::GraphNode& node) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ::gui::MLNode* step = editor_->FindStep(node.id);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", node.name.c_str());
    for (const auto& row : steps_)
        if (row.id == node.id) {
            ImGui::SameLine(0.0f, 10.0f);
            KindChip(row.kind == "step node" ? "step node" : row.kind + " node");
            if (!row.tip.empty()) {
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(ToneColour(row.tone), "%s", row.tip.c_str());
                ImGui::PopTextWrapPos();
            }
        }
    const bool running = editor_->IsDataPipelineRunning();
    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running, "The Data Pipeline is running", ui::ButtonSize::Small)) RunStep(node.id);
    ImGui::SameLine();
    if (ui::SecondaryButton("Open dialog")) OpenInDialog(node.id);
    ImGui::SameLine();
    if (ui::LinkButton("Show on canvas") && show_on_canvas) show_on_canvas({node.id});
    if (!message_.empty() && loaded_ == node.id) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (loaded_ != node.id) {
        loaded_ = node.id;
        message_.clear();
    }
    // Its settings, as saved in the graph.
    if (step && !step->parameters.empty() &&
        ImGui::BeginTable("##settings", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn("setting", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        for (const auto& [key, value] : step->parameters) {
            if (value.empty()) continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", key.c_str());
            ImGui::TableNextColumn();
            const std::string shown = value.size() > 160 ? value.substr(0, 157) + "..." : value;
            ImGui::TextUnformatted(shown.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::TextColored(t.text_faint, "Settings are changed in the node's dialog.");
    RenderResult(node);
}

void PrepareView::RenderResult(const studio::GraphNode& node) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    if (editor_->IsDataPipelineRunning()) {
        ImGui::TextColored(t.info, ICON_FA_SPINNER " Running (Task View shows the steps)");
        return;
    }
    if (!node.error.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, ICON_FA_CIRCLE_XMARK " %s", node.error.c_str());
        ImGui::PopTextWrapPos();
        return;
    }
    if (!node.ran || node.output_dataset.empty()) {
        ImGui::TextColored(t.text_dim, "Not run in this session: Run step shows its rows here.");
        return;
    }
    const std::string key = node.output_dataset + "@" + node.finished_at;
    if (key != grid_key_) {
        grid_key_ = key;
        grid_.SetTable(nullptr);
        if (auto ds = DataRegistry::Instance().GetArrowDataset(node.output_dataset))
            if (auto table = ds->GetArrowTable()) grid_.SetTable(DataTableFromArrow(*table, node.name, kGridRows));
        if (node.type == ::gui::NodeType::SQLQuery) StartPartCounts(node);
        else parts_.reset();
    }
    ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " %s rows", studio::Thousands(node.rows).c_str());
    ImGui::SameLine(0.0f, 8.0f);
    ImGui::TextColored(t.text_dim, "%lld columns \xC2\xB7 at %s%s", static_cast<long long>(node.columns), node.finished_at.c_str(),
                       node.rows > kGridRows ? " \xC2\xB7 the first 1,000 rows below" : "");
    // Rows each part of the query keeps.
    if (parts_ && parts_->key == key) {
        if (!parts_->done) {
            ImGui::TextColored(t.text_faint, "Counting the rows per part...");
        } else if (!parts_->error.empty()) {
            ImGui::TextColored(t.text_faint, "Rows per part not counted: %s", parts_->error.c_str());
        } else if (!parts_->counts.empty()) {
            std::string line = "rows per part: ";
            for (size_t i = 0; i < parts_->counts.size(); ++i)
                line += (i ? " " ICON_FA_ANGLE_RIGHT " " : "") + parts_->counts[i].first + " " + studio::Thousands(parts_->counts[i].second);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", line.c_str());
            ImGui::PopTextWrapPos();
        }
    }
    if (grid_.HasData()) grid_.Render("##prepare_grid", ImVec2(0.0f, std::max(120.0f, ImGui::GetContentRegionAvail().y)));
}

}  // namespace cyxwiz
