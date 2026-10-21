#include "prepare_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/guided_sql.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/sql_step_presentation.h"
#include "../../core/sql_transform.h"
#include "../../data/arrow_data_table.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../sql_workbench_ui.h"
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
    std::string query;  // the query counted
    std::thread worker;
    std::atomic<bool> done{false};
    SqlTransform transform;
    std::vector<std::pair<std::string, int64_t>> counts;  // in the query's order
    std::map<std::string, int64_t> by_part;               // the same, by part name (guided cards)
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
    guided_.on_changed = [this] { OnGuidedChanged(); };
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
        const auto context = editor_->GetDataStepContext(loaded_);
        for (const auto& input : context.inputs) reads_ += (reads_.empty() ? "" : ", ") + input.name;
        checks_after_ = context.checks.size();
        // A run puts the inputs in memory: the operations see their columns.
        guided_.SetInputs(StepInputs(loaded_));
    }
    // A step just added (Add a step, Data > Transform) is picked once it is on the canvas.
    if (pick_after_ != -2 && editor_ && editor_->LastAddedStep() != pick_after_ && NodeById(editor_->LastAddedStep())) {
        Select(editor_->LastAddedStep());
        if (pick_compare_) transform_.CompareOn(selected_);
        pick_after_ = -2;
    }
    // A step replaced (a method switch) stays picked as its new node.
    if (selected_ >= 0 && !NodeById(selected_) && editor_ && editor_->ReplacedBy(selected_) >= 0) selected_ = editor_->ReplacedBy(selected_);
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
    pane_.Editor().SetReadOnly(false);
    pane_.Editor().SetText(saved_query_);
    std::string error;
    if (!guided_.Load(node ? Param(*node, guided::kGuidedOpsParameter) : std::string(), &error)) {
        message_ = "The guided operations could not be read (" + error + "); the step opens as SQL.";
        message_error_ = true;
    }
    mode_ = guided_.Active() ? 0 : 1;
    confirm_start_guided_ = confirm_edit_as_sql_ = false;
    pane_.Editor().SetReadOnly(guided_.Active());
    pane_.Refresh();
    // What it reads, and completion: the columns of its inputs when they are in memory.
    reads_.clear();
    checks_after_ = 0;
    std::vector<std::string> columns;
    if (editor_) {
        const auto context = editor_->GetDataStepContext(step_id);
        for (const auto& input : context.inputs) reads_ += (reads_.empty() ? "" : ", ") + input.name;
        checks_after_ = context.checks.size();
        auto inputs = StepInputs(step_id);
        for (const auto& input : inputs)
            if (input.table)
                for (const auto& field : input.table->schema()->fields()) columns.push_back(field->name());
        // A guided step: its operations write the query for the inputs as they are.
        if (guided_.SetInputs(std::move(inputs)) && guided_.Generated().sql != saved_query_ && message_.empty()) {
            message_ = "The operations write a different query for the inputs as they are now; Save to step keeps it.";
            message_error_ = false;
        }
    }
    pane_.SetCompletionColumns(std::move(columns));
}

std::vector<::gui::GuidedEditor::Input> PrepareView::StepInputs(int step_id) const {
    std::vector<::gui::GuidedEditor::Input> inputs;
    const ::gui::MLNode* step = editor_ ? editor_->FindStep(step_id) : nullptr;
    if (!step) return inputs;
    const auto names = sqlstep::InputNames(step->parameters);
    const auto& facts = editor_->GetNodeRunFacts();
    for (const auto& input : editor_->GetDataStepContext(step_id).inputs) {
        ::gui::GuidedEditor::Input entry;
        entry.alias = input.pin < static_cast<int>(names.size()) ? names[static_cast<size_t>(input.pin)] : std::string("input");
        const auto it = facts.find(input.id);
        if (it != facts.end() && !it->second.output_dataset.empty()) {
            const auto source = SqlInputForDataset(entry.alias, it->second.output_dataset);
            if (source.table) {
                entry.table = source.table;
                entry.rows = source.table->num_rows();
            } else if (!source.parquet_path.empty()) {
                // A table kept on disk: its columns here, its rows read by DuckDB.
                entry.parquet_path = source.parquet_path;
                entry.table = arrow::Table::MakeEmpty(source.schema).ValueOr(nullptr);
                const auto ref = DataRegistry::Instance().GetTableRef(it->second.output_dataset);
                entry.rows = ref ? ref->rows : 0;
            }
        }
        inputs.push_back(std::move(entry));
    }
    return inputs;
}

void PrepareView::OnGuidedChanged() {
    pane_.Editor().SetReadOnly(false);
    pane_.Editor().SetText(guided_.Generated().sql);
    pane_.Editor().SetReadOnly(true);
    pane_.Refresh();
}

void PrepareView::SaveDraft() {
    if (!editor_ || loaded_ < 0) return;
    if (guided_.Active() && !guided_.Generated().ok) {
        message_ = "Fix the operations marked in red first.";
        message_error_ = true;
        return;
    }
    // Its "Add checks" become Check nodes after the step (TOFIX148 D4).
    const auto checks = guided_.TakeChecks();
    const std::string text = pane_.Editor().GetText();
    std::map<std::string, std::string> values{{"query", text}};
    std::vector<std::string> erase;
    // A guided step keeps its operations next to the query they wrote; an
    // SQL step keeps none (Edit as SQL).
    if (guided_.Active()) values[guided::kGuidedOpsParameter] = guided_.Json();
    else erase.push_back(guided::kGuidedOpsParameter);
    if (!editor_->SetStepParameters(loaded_, values, erase)) {
        message_ = "The step is no longer in the graph.";
        message_error_ = true;
        return;
    }
    saved_query_ = text;
    guided_.MarkSaved();
    message_ = guided_.Active() ? "Saved the operations and their query to the step (Ctrl+Z on the canvas undoes it)."
                                : "Saved to the step (Ctrl+Z on the canvas undoes it).";
    message_error_ = false;
    if (!checks.empty()) {
        editor_->RequestChecksAfter(loaded_, checks);
        message_ = "Saved; its " + std::to_string(checks.size()) + " check(s) became Check nodes after the step (Ctrl+Z on the canvas undoes it).";
    }
}

void PrepareView::RunStep(int step_id) {
    if (!editor_) return;
    if (step_id == loaded_ && (pane_.Text() != saved_query_ || guided_.Edited())) SaveDraft();
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

void PrepareView::PickWhenAdded(bool compare) {
    pick_after_ = editor_ ? editor_->LastAddedStep() : -2;
    pick_compare_ = compare;
}

void PrepareView::AddAfter(int after, ::gui::NodeType type, std::map<std::string, std::string> parameters, bool compare) {
    if (!editor_) return;
    PickWhenAdded(compare);
    editor_->RequestStepAfter(after, type, {}, std::move(parameters));
    const auto* meta = NodeMetadataRegistry::Instance().GetMetadata(type);
    const auto* from = NodeById(after);
    message_ = "Added a " + (meta ? meta->name : std::string("step")) + " step after " +
               (from ? from->name : std::string("the data")) + " (Ctrl+Z on the canvas undoes it).";
    message_error_ = false;
}

void PrepareView::AddStepAfter(int step_id, ::gui::NodeType type, bool guided) {
    if (!editor_) return;
    PickWhenAdded(false);
    // A step is added on the canvas: a collapsed recipe opens first.
    if (!editor_->GetNodeForConfiguration(step_id)) {
        const int recipe = RecipeOf(step_id);
        if (recipe >= 0) editor_->RequestSubgraphAction(recipe, ::gui::NodeEditor::SubgraphRequest::Expand);
    }
    if (type != ::gui::NodeType::SQLQuery) {
        editor_->RequestStepAfter(step_id, type);
        const auto* meta = NodeMetadataRegistry::Instance().GetMetadata(type);
        message_ = "Added a " + (meta ? meta->name : std::string("step")) + " node after this one (Ctrl+Z on the canvas undoes it).";
    } else if (guided) {
        editor_->RequestGuidedStepAfter(step_id);
        message_ = "Added a guided step after this one; add its operations.";
    } else {
        editor_->RequestSqlStepAfter(step_id, "SQL step", "SELECT *\nFROM input");
        message_ = "Added an SQL step after this one; write its query.";
    }
    // It joins the step list here; the canvas shows it when you go there.
    message_error_ = false;
}

void PrepareView::RenderAddStepMenu(int step_id) {
    using ::gui::NodeType;
    if (ImGui::MenuItem("SQL step")) AddStepAfter(step_id, NodeType::SQLQuery);
    if (ImGui::MenuItem("Guided step")) AddStepAfter(step_id, NodeType::SQLQuery, true);
    ImGui::Separator();
    const auto& registry = NodeMetadataRegistry::Instance();
    const auto item = [&](const NodeMetadata& meta) {
        if (ImGui::MenuItem((meta.icon + "  " + meta.name).c_str())) AddStepAfter(step_id, meta.type);
        if (ImGui::IsItemHovered() && !meta.brief_description.empty()) ImGui::SetTooltip("%s", meta.brief_description.c_str());
    };
    // The step nodes used most, then every other preparation node.
    static const NodeType kCommon[] = {NodeType::FilterRows, NodeType::JoinTables, NodeType::GroupByAggregate, NodeType::TextCleanNode};
    for (const NodeType type : kCommon)
        if (const auto* meta = registry.GetMetadata(type)) item(*meta);
    // Scaling and shape (TOFIX129 D2): a Scale step's view switches between the scalers.
    if (ImGui::BeginMenu("Scale and transform")) {
        if (ImGui::MenuItem("Scale (Z-score, Min-Max, Robust, Max-Abs, Quantile)")) AddStepAfter(step_id, NodeType::StandardScaler);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds a Z-score step; its Method switch picks another scaler");
        if (ImGui::MenuItem("Log transform")) AddStepAfter(step_id, NodeType::LogTransform);
        if (ImGui::MenuItem("Power transform (Box-Cox, Yeo-Johnson)")) AddStepAfter(step_id, NodeType::PowerTransform);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("More nodes")) {
        std::vector<const NodeMetadata*> more;
        for (const auto* meta : registry.GetAllMetadata()) {
            studio::Job job;
            if (!meta || meta->status != NodeImplementationStatus::Implemented || !studio::JobOf(meta->type, job) ||
                job != studio::Job::Prepare || meta->type == NodeType::SQLQuery || meta->type == NodeType::Subgraph ||
                transform_step::KindOf(meta->type) != transform_step::Kind::None ||
                std::find(std::begin(kCommon), std::end(kCommon), meta->type) != std::end(kCommon))
                continue;
            more.push_back(meta);
        }
        std::sort(more.begin(), more.end(), [](const NodeMetadata* a, const NodeMetadata* b) { return a->name < b->name; });
        for (const auto* meta : more) item(*meta);
        ImGui::EndMenu();
    }
}

void PrepareView::StartPartCounts(const studio::GraphNode& node) {
    const ::gui::MLNode* step = editor_ ? editor_->FindStep(node.id) : nullptr;
    if (!step) return;
    const std::string query = Param(*step, "query");
    const auto outline = sqlstep::OutlineSqlQuery(query);
    const std::string counts_sql = sqlstep::PartCountsQuery(query, outline);
    auto job = std::make_shared<PartCounts>();
    job->key = node.output_dataset + "@" + node.finished_at;
    job->query = query;
    parts_ = job;
    if (counts_sql.empty() || outline.parts.size() < 2) {
        job->done = true;
        return;
    }
    std::vector<SqlTransformInput> inputs;
    for (const auto& input : StepInputs(node.id)) {
        if (!input.table) {
            job->error = "its input (" + input.alias + " in the query) is not in memory";
            job->done = true;
            return;
        }
        SqlTransformInput source;
        source.alias = input.alias;
        if (input.parquet_path.empty()) {
            source.table = input.table;
        } else {
            source.parquet_path = input.parquet_path;
            source.schema = input.table->schema();
        }
        inputs.push_back(std::move(source));
    }
    std::vector<std::string> order;
    for (const auto& part : outline.parts) order.push_back(part.name);
    PartCounts* raw = job.get();
    job->worker = std::thread([raw, counts_sql, inputs, order] {
        const SqlTransformResult r = raw->transform.Run(counts_sql, inputs);
        if (!r.ok || !r.table) {
            raw->error = r.cancelled ? std::string() : r.error;
        } else {
            auto& by_part = raw->by_part;
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
        RenderEmpty();
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

// No steps yet: Add a step after a Data Input (found while drawing the D2
// mockup: before, Prepare said so and offered nothing).
void PrepareView::RenderEmpty() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TextColored(t.text_dim, "This graph has no preparation steps yet.");
    std::vector<const studio::GraphNode*> inputs;
    for (const auto& n : nodes_)
        if (n.type == ::gui::NodeType::DataInput) inputs.push_back(&n);
    if (inputs.empty()) {
        ImGui::TextColored(t.text_faint, "Bring in data first: a Data Input node (Bring in).");
    } else {
        const std::string label = inputs.size() == 1 ? "Add a step after " + inputs.front()->name + " " ICON_FA_CARET_DOWN
                                                     : std::string("Add a step after " ICON_FA_CARET_DOWN);
        if (ui::SecondaryButton(label.c_str())) ImGui::OpenPopup("##add_first_step");
        if (ImGui::BeginPopup("##add_first_step")) {
            if (inputs.size() == 1) {
                RenderAddStepMenu(inputs.front()->id);
            } else {
                for (const auto* input : inputs) {
                    ImGui::PushID(input->id);
                    if (ImGui::BeginMenu(input->name.c_str())) {
                        RenderAddStepMenu(input->id);
                        ImGui::EndMenu();
                    }
                    ImGui::PopID();
                }
            }
            ImGui::EndPopup();
        }
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "or Explore > Query's \"Turn into SQL step\". On the canvas, an SQL Query node after a "
                                     "Data Input, grouped as a Preparation Recipe, is a recipe.");
    if (!message_.empty()) ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
    ImGui::PopTextWrapPos();
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
            ui::PushListRowColours();
            if (ImGui::Selectable(label.c_str(), recipe_ == id)) recipe_ = id;
            ui::PopListRowColours();
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
        ui::PushListRowColours();
        if (ImGui::Selectable("##step", row.id == selected_, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(w, line * 2.0f + 4.0f))) {
            selected_ = row.id;
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && show_on_canvas) show_on_canvas({row.id});
        }
        ui::PopListRowColours();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            const std::string tip = row.kind + (row.tip.empty() ? "" : "\n" + row.tip) + "\nDouble-click: show it on the canvas.";
            ImGui::SetTooltip("%s", tip.c_str());
        }
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const char* icon = row.sql ? ICON_FA_CODE : row.kind == "Check" ? ICON_FA_CIRCLE_CHECK : ICON_FA_GEAR;
        dl->AddText(ImVec2(at.x + 4.0f, at.y + 2.0f), ui::ToU32(row.sql ? t.accent_text : t.text_faint), icon);
        const ImVec4 clip(at.x, at.y, at.x + w - 4.0f, at.y + line * 2.0f + 4.0f);
        dl->AddText(nullptr, 0.0f, ImVec2(at.x + 26.0f, at.y + 2.0f), ui::ToU32(t.text_bright), row.name.c_str(), nullptr, 0.0f, &clip);
        const std::string meta = row.kind + " \xC2\xB7 " + row.meta;
        dl->AddText(nullptr, 0.0f, ImVec2(at.x + 26.0f, at.y + line + 3.0f),
                    ui::ToU32(row.tone == studio::Tone::Plain ? t.text_dim : ToneColour(row.tone)), meta.c_str(), nullptr, 0.0f, &clip);
        ImGui::PopID();
    }
    // Add a step after the one selected (the mockup's "Add a step").
    if (selected_ >= 0) {
        ImGui::Spacing();
        if (ui::SecondaryButton(ICON_FA_PLUS " Add a step " ICON_FA_CARET_DOWN)) ImGui::OpenPopup("##add_step");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds an SQL step, a guided step or a step node after the selected step");
        if (ImGui::BeginPopup("##add_step")) {
            RenderAddStepMenu(selected_);
            ImGui::EndPopup();
        }
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
        if (now != saved_query_ && pane_.Text() == saved_query_ && !guided_.Edited()) LoadStep(node.id);
    }
    const bool dirty = pane_.Text() != saved_query_ || guided_.Edited();
    const bool running = editor_->IsDataPipelineRunning();
    const bool ready = !guided_.Active() || guided_.Generated().ok;
    guided_.busy = running;

    // Header: the step, what it is, what it reads.
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", node.name.c_str());
    ImGui::SameLine(0.0f, 10.0f);
    KindChip(guided_.Active() ? "SQL Query node \xC2\xB7 guided" : "SQL Query node");
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
    // How the step is edited: from operations (Guided) or as its query (SQL).
    static const char* const kModes[] = {"Guided", "SQL"};
    ui::SegmentedControl("##prepare_mode", kModes, 2, &mode_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Guided: build the step from operations. SQL: the query itself.");
    if (mode_ == 0) {
        RenderGuided(std::clamp(height * 0.46f, 240.0f, 480.0f));
    } else {
        if (guided_.Active()) {
            // SQL mode on a guided step: the query is read-only.
            ImGui::SameLine(0.0f, 12.0f);
            ImGui::AlignTextToFramePadding();
            if (!confirm_edit_as_sql_) {
                ImGui::TextColored(t.info, ICON_FA_CIRCLE_INFO " Written from the operations (read-only).");
                ImGui::SameLine();
                if (ui::LinkButton("Edit as SQL")) confirm_edit_as_sql_ = true;
            } else {
                ImGui::TextColored(t.warning, "The operations go when you save.");
                ImGui::SameLine();
                if (ui::LinkButton("Edit as SQL")) {
                    confirm_edit_as_sql_ = false;
                    guided_.Stop();
                    pane_.Editor().SetReadOnly(false);
                    message_ = "The query is yours to edit; Save to step keeps it as an SQL step (Reset undoes this).";
                    message_error_ = false;
                }
                ImGui::SameLine();
                if (ui::LinkButton("Cancel")) confirm_edit_as_sql_ = false;
            }
        }
        pane_.Render(std::clamp(height * 0.38f, 140.0f, 360.0f));
        // Ctrl+Enter runs the step, as in the Workbench.
        if (pane_.Editor().IsFocused() && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false) && !running && ready)
            RunStep(node.id);
    }

    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running && ready, running ? "The Data Pipeline is running" : "Fix the operations marked in red",
                          ui::ButtonSize::Small))
        RunStep(node.id);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Saves the step, then runs the Data Pipeline through it and its checks (Ctrl+Enter in the query).");
    ImGui::SameLine();
    if (ui::SecondaryButton("Save to step", dirty && ready, !dirty ? "No changes to save" : "Fix the operations marked in red")) SaveDraft();
    if (dirty) {
        ImGui::SameLine();
        if (ui::LinkButton("Reset")) LoadStep(node.id);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Open in Workbench")) OpenInDialog(node.id);
    ImGui::SameLine();
    if (ui::SecondaryButton("Add a step after " ICON_FA_CARET_DOWN)) ImGui::OpenPopup("##add_after");
    if (ImGui::BeginPopup("##add_after")) {
        RenderAddStepMenu(node.id);
        ImGui::EndPopup();
    }
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    RenderResult(node);
}

// Guided mode: the operations as cards and the picked one's fields (the
// Workbench's guided editor); the library is a menu, the screen being narrower.
void PrepareView::RenderGuided(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!guided_.Active()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "This step is written in SQL. A guided step builds it from operations instead; it replaces "
                                       "the query when you save (Reset brings the query back until then).");
        ImGui::PopTextWrapPos();
        if (ui::SecondaryButton("Start a guided step instead")) confirm_start_guided_ = true;
        if (confirm_start_guided_) {
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.warning, "Replace the query with operations?");
            ImGui::SameLine();
            if (ui::LinkButton("Yes, start")) {
                confirm_start_guided_ = false;
                guided_.Start();
            }
            ImGui::SameLine();
            if (ui::LinkButton("Cancel")) confirm_start_guided_ = false;
        }
        ImGui::Spacing();
        return;
    }
    // The tools beside Guided | SQL, or on the next line when they do not fit.
    const float tools = ui::ButtonWidth(ICON_FA_PLUS " Add an operation " ICON_FA_CHEVRON_DOWN, ui::ButtonSize::Small) +
                        ui::ButtonWidth("Start from a template " ICON_FA_CHEVRON_DOWN, ui::ButtonSize::Small) +
                        ui::ButtonWidth("Undo", ui::ButtonSize::Small) + ui::ButtonWidth("Redo", ui::ButtonSize::Small) +
                        ImGui::GetStyle().ItemSpacing.x * 3.0f;
    ImGui::SameLine(0.0f, 12.0f);
    if (ImGui::GetContentRegionAvail().x < tools) ImGui::NewLine();
    guided_.RenderAddMenu(ui::ButtonSize::Small);
    ImGui::SameLine();
    guided_.RenderTemplateButton(ui::ButtonSize::Small);
    ImGui::SameLine();
    if (ui::GhostButton("Undo", guided_.CanUndo(), "Nothing to undo")) guided_.Undo();
    ImGui::SameLine();
    if (ui::GhostButton("Redo", guided_.CanRedo(), "Nothing to redo")) guided_.Redo();

    // Rows after each operation: the last run's count per part, while the
    // operations still write that query.
    ::gui::GuidedEditor::CardFacts facts;
    if (parts_ && parts_->done && parts_->error.empty() && parts_->query == pane_.Text()) facts.rows = &parts_->by_part;
    if (!guided_.Inputs().empty() && guided_.Inputs().front().table) facts.input_rows = guided_.Inputs().front().rows;
    facts.checks_after = checks_after_;

    const float width = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.y;
    if (width >= 760.0f) {
        guided_.RenderCards(ImVec2(std::clamp(width * 0.4f, 300.0f, 420.0f), height), facts);
        ImGui::SameLine();
        if (::gui::sqlwb::BeginPanel("##prepare_guided_fields", ImVec2(0.0f, height), t.bg_panel)) guided_.RenderEditor();
        ::gui::sqlwb::EndPanel();
    } else {
        // Narrow (Properties docked beside): the fields below the cards.
        const float cards_h = std::floor(height * 0.45f);
        guided_.RenderCards(ImVec2(0.0f, cards_h), facts);
        if (::gui::sqlwb::BeginPanel("##prepare_guided_fields", ImVec2(0.0f, height - cards_h - spacing), t.bg_panel))
            guided_.RenderEditor();
        ::gui::sqlwb::EndPanel();
    }
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
        guided_.Load({});  // no operations: this step is not an SQL step
    }
    // A scale, power or log step: its own view (settings, before / after).
    if (transform_step::KindOf(node.type) != transform_step::Kind::None) {
        const auto context = editor_->GetDataStepContext(node.id);
        const studio::GraphNode* input = context.inputs.empty() ? nullptr : NodeById(context.inputs.front().id);
        transform_.Render(editor_, node, input);
        RenderResult(node);
        return;
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
        if (auto table = DataRegistry::Instance().ReadTablePreview(node.output_dataset, kGridRows))
            grid_.SetTable(DataTableFromArrow(*table, node.name, kGridRows));
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
