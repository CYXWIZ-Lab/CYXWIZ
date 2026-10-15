#include "validate_view.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <map>

namespace cyxwiz {

namespace {

using studio::CheckKind;

ImVec4 ToneColour(studio::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case studio::Tone::Ok: return t.success;
        case studio::Tone::Warn: return t.warning;
        case studio::Tone::Error: return t.error;
        default: return t.text_dim;
    }
}

// The next item on this line when `width` still fits, else on the next line
// (Data Studio docked beside the canvas is narrow; nothing may clip).
void Next(float width) {
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

void Dim(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
}

bool TextField(const char* id, std::string& value, const char* hint, float width) {
    ImGui::SetNextItemWidth(std::min(width, ImGui::GetContentRegionAvail().x));
    return ImGui::InputTextWithHint(id, hint, &value);
}

bool Choice(const char* id, int& value, const char* const* options, int count, float width) {
    bool changed = false;
    ImGui::SetNextItemWidth(std::min(width, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo(id, options[std::clamp(value, 0, count - 1)])) {
        for (int i = 0; i < count; ++i)
            if (ImGui::Selectable(options[i], i == value)) {
                value = i;
                changed = true;
            }
        ImGui::EndCombo();
    }
    return changed;
}

std::vector<std::string> SplitList(const std::string& list) {
    std::vector<std::string> out;
    std::string item;
    const auto flush = [&] {
        const size_t a = item.find_first_not_of(' '), b = item.find_last_not_of(' ');
        if (a != std::string::npos) out.push_back(item.substr(a, b - a + 1));
        item.clear();
    };
    for (char ch : list) {
        if (ch == ',') flush();
        else item += ch;
    }
    flush();
    return out;
}

}  // namespace

void ValidateView::SetGraph(::gui::NodeEditor* editor, const std::vector<studio::GraphNode>& nodes, const std::vector<int>& recipes,
                            const std::string& active_dataset, const std::string& pipeline_error) {
    editor_ = editor;
    pipeline_error_ = pipeline_error;
    nodes_ = nodes;
    recipes_ = recipes;
    rows_ = studio::CheckRows(nodes_);

    // The steps a check can follow: Data Inputs and Prepare's steps.
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
    // A new check follows the rail's dataset, else the step most checks read, else the last step.
    std::map<int, int> reads;
    for (const auto& r : rows_)
        if (r.reads_id >= 0) ++reads[r.reads_id];
    int most = -1;
    for (const auto& [id, count] : reads)
        if (most < 0 || count > reads[most]) most = id;
    default_target_ = active >= 0 ? active : most >= 0 ? most : last_step >= 0 ? last_step : targets_.empty() ? -1 : targets_.front().id;

    if (form_open_) {
        const bool target_there = std::any_of(targets_.begin(), targets_.end(), [&](const Target& t) { return t.id == target_; });
        const bool edit_there = std::any_of(rows_.begin(), rows_.end(), [&](const studio::CheckRow& r) { return r.id == editing_; });
        if (editing_ >= 0 && !edit_there) form_open_ = false;  // the check was deleted on the canvas
        if (editing_ < 0 && !target_there) target_ = default_target_;
        LoadColumns();
    }
}

bool ValidateView::IsRunning() const { return editor_ && editor_->IsDataPipelineRunning(); }

int ValidateView::RecipeOf(int step_id) const {
    if (!editor_) return -1;
    for (int r : recipes_)
        for (const auto& s : editor_->GetSubgraphSummary(r).steps)
            if (s.id == step_id) return r;
    return -1;
}

void ValidateView::LoadColumns() {
    int id = target_;
    if (editing_ >= 0)
        for (const auto& r : rows_)
            if (r.id == editing_) id = r.reads_id;
    const studio::GraphNode* node = nullptr;
    for (const auto& n : nodes_)
        if (n.id == id) node = &n;
    const std::string dataset = !node ? std::string() : !node->output_dataset.empty() ? node->output_dataset : node->dataset_name;
    const std::string key = std::to_string(id) + "@" + dataset + "@" + (node ? node->finished_at : std::string());
    if (key == columns_key_) return;
    columns_key_ = key;
    columns_.clear();
    if (dataset.empty()) return;
    if (auto ds = DataRegistry::Instance().GetArrowDataset(dataset))
        if (auto table = ds->GetArrowTable())
            for (const auto& field : table->schema()->fields()) columns_.push_back(field->name());
}

void ValidateView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    size_t checks = 0, ran = 0, passed = 0;
    for (const auto& r : rows_) {
        if (r.validator) continue;
        ++checks;
        if (r.result != "not run") ++ran;
        if (r.tone == studio::Tone::Ok) ++passed;
    }
    ImGui::AlignTextToFramePadding();
    if (rows_.empty()) ImGui::TextColored(t.text_dim, "No checks in this graph yet.");
    else if (!ran) ImGui::TextColored(t.text_bright, "%zu check%s, not run in this session", checks, checks == 1 ? "" : "s");
    else ImGui::TextColored(passed == ran ? t.success : t.error, "%zu of %zu checks pass", passed, ran);
    // Run checks: right-aligned, on the next line when the panel is narrow.
    const float run_w = ui::ButtonWidth(ICON_FA_PLAY " Run checks", ui::ButtonSize::Small);
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < run_w) ImGui::NewLine();
    else ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - run_w);
    const char* run_reason = !editor_ ? "No graph is open" : rows_.empty() ? "Add a check first" : "The Data Pipeline is running";
    if (!pipeline_error_.empty()) run_reason = "The graph cannot run (see the line below)";
    if (ui::PrimaryButton(ICON_FA_PLAY " Run checks", editor_ && !rows_.empty() && !IsRunning() && pipeline_error_.empty(), run_reason,
                          ui::ButtonSize::Small))
        RunChecks();
    if (ImGui::IsItemHovered() && !rows_.empty() && pipeline_error_.empty())
        ImGui::SetTooltip("Runs the Data Pipeline through every check (and the steps before them)");
    if (!pipeline_error_.empty()) {
        // What the checks read comes from the graph as it runs; say why it cannot.
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.error, ICON_FA_TRIANGLE_EXCLAMATION " The graph cannot run: %s", pipeline_error_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();

    if (!rows_.empty()) RenderTable();
    ImGui::Spacing();
    RenderAddRow();
    if (form_open_) {
        ImGui::Spacing();
        RenderForm();
    }
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "Each check is a Row Count Check node after the step it reads: it passes the rows on when its rule "
                                     "holds and stops the run when it does not, so nothing after it sees wrong data. A guided step's "
                                     "\"Add checks\" makes these nodes too.");
    ImGui::PopTextWrapPos();
}

void ValidateView::RenderTable() {
    const ui::Tokens& t = ui::CurrentTokens();
    // Narrow (Data Studio docked beside the canvas): what it reads goes under
    // the rule and the row's links under its last run, so no column is
    // squeezed into breaking words.
    const bool narrow = ImGui::GetContentRegionAvail().x < 760.0f;
    const int columns = narrow ? 3 : 5;
    if (!ImGui::BeginTable("##checks", columns, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_PadOuterX |
                                                    ImGuiTableFlags_NoSavedSettings))
        return;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float links_w = ImGui::CalcTextSize("Editing").x + ImGui::CalcTextSize("Show").x + spacing + 8.0f;
    float result_w = ImGui::CalcTextSize("last run").x;
    for (const auto& r : rows_) result_w = std::max(result_w, ImGui::CalcTextSize((ICON_FA_CIRCLE " " + r.result).c_str()).x);
    ImGui::TableSetupColumn("check (node)", ImGuiTableColumnFlags_WidthStretch, 1.3f);
    if (!narrow) ImGui::TableSetupColumn("reads", ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("rule", ImGuiTableColumnFlags_WidthStretch, 2.4f);
    ImGui::TableSetupColumn("last run", ImGuiTableColumnFlags_WidthFixed, (narrow ? std::max(result_w, links_w) : result_w) + 8.0f);
    if (!narrow) ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed, links_w);
    ImGui::TableHeadersRow();
    const studio::CheckRow* edit = nullptr;
    for (const auto& r : rows_) {
        ImGui::PushID(r.id);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_bright, "%s", r.name.c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s node\nDouble-click: show it on the canvas", r.validator ? "Data Validator" : "Row Count Check");
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && show_on_canvas) show_on_canvas({r.id});
        }
        ImGui::TableNextColumn();
        if (!narrow) {
            if (r.reads.empty()) ImGui::TextColored(t.text_faint, "nothing linked");
            else ImGui::TextColored(t.text_dim, "%s", r.reads.c_str());
            ImGui::TableNextColumn();
        }
        ImGui::TextUnformatted(r.rule.c_str());
        if (narrow) ImGui::TextColored(t.text_dim, "%s", r.reads.empty() ? "reads nothing linked" : ("reads " + r.reads).c_str());
        if (r.tone == studio::Tone::Error && !r.detail.empty()) ImGui::TextColored(t.error, "%s", r.detail.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TableNextColumn();
        ImGui::TextColored(ToneColour(r.tone), ICON_FA_CIRCLE " %s", r.result.c_str());
        if (!r.detail.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r.detail.c_str());
        if (!narrow) ImGui::TableNextColumn();
        if (ui::LinkButton(editing_ == r.id && form_open_ ? "Editing" : "Edit", !(editing_ == r.id && form_open_))) edit = &r;
        ImGui::SameLine();
        if (ui::LinkButton("Show") && show_on_canvas) show_on_canvas({r.id});
        ImGui::PopID();
    }
    ImGui::EndTable();
    if (edit) EditCheck(*edit);
}

void ValidateView::RenderAddRow() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool can = editor_ && !targets_.empty();
    const char* reason = !editor_ ? "No graph is open" : "The graph has no step to check yet (Bring in adds a Data Input)";
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Add a check");
    for (int k = 0; k < studio::kCheckKindCount; ++k) {
        const auto kind = static_cast<CheckKind>(k);
        const char* label = studio::CheckKindName(kind);
        Next(ui::ButtonWidth(label, ui::ButtonSize::Small));
        if (ui::SecondaryButton(label, can, reason, ui::ButtonSize::Small)) OpenForm(kind);
    }
    Next(ui::ButtonWidth("Data Validator...", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Data Validator...", can, reason, ui::ButtonSize::Small)) OpenValidatorForm();
    if (can && ImGui::IsItemHovered())
        ImGui::SetTooltip("A Data Validator node: required, not-empty and unique columns; it reports the issues as a table "
                          "instead of stopping the run");
}

void ValidateView::OpenForm(CheckKind kind) {
    form_open_ = true;
    validator_form_ = false;
    editing_ = -1;
    target_ = default_target_;
    form_ = {};
    form_.kind = kind;
    form_error_.clear();
    message_.clear();
    columns_key_.clear();
    LoadColumns();
}

void ValidateView::OpenValidatorForm() {
    form_open_ = true;
    validator_form_ = true;
    editing_ = -1;
    target_ = default_target_;
    vform_ = {};
    form_error_.clear();
    message_.clear();
    columns_key_.clear();
    LoadColumns();
}

void ValidateView::EditCheck(const studio::CheckRow& row) {
    const ::gui::MLNode* node = editor_ ? editor_->FindStep(row.id) : nullptr;
    if (!node) return;
    form_open_ = true;
    editing_ = row.id;
    validator_form_ = row.validator;
    if (row.validator) {
        vform_ = studio::ValidatorFormFrom(node->parameters);
        form_error_.clear();
        message_.clear();
        columns_key_.clear();
        LoadColumns();
        return;
    }
    form_ = studio::CheckFormFrom(node->parameters);
    if (form_.name.empty()) form_.name = node->name;
    // A name the rule gave follows the rule when it changes.
    if (form_.name == studio::CheckDefaultName(node->parameters)) form_.name.clear();
    form_error_.clear();
    message_.clear();
    columns_key_.clear();
    LoadColumns();
}

void ValidateView::ColumnField(const char* id, std::string& value, float width) {
    ImGui::SetNextItemWidth(std::min(width, ImGui::GetContentRegionAvail().x));
    if (columns_.empty()) {
        ImGui::InputTextWithHint(id, "column name", &value);
        return;
    }
    if (ImGui::BeginCombo(id, value.empty() ? "choose a column" : value.c_str())) {
        for (const auto& c : columns_)
            if (ImGui::Selectable(c.c_str(), c == value)) value = c;
        ImGui::EndCombo();
    }
}

void ValidateView::ColumnSetField(const char* id, std::string& list, const char* hint) {
    if (columns_.empty()) {
        ImGui::SameLine();
        TextField(id, list, hint, 320.0f);
        return;
    }
    // The columns as checkboxes (as the guided editor's column sets).
    ImGui::PushID(id);
    auto chosen = SplitList(list);
    for (size_t i = 0; i < columns_.size(); ++i) {
        const std::string& c = columns_[i];
        bool on = std::find(chosen.begin(), chosen.end(), c) != chosen.end();
        Next(ImGui::CalcTextSize(c.c_str()).x + ImGui::GetFrameHeight() + 16.0f);
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Checkbox(c.c_str(), &on)) {
            if (on) chosen.push_back(c);
            else chosen.erase(std::remove(chosen.begin(), chosen.end(), c), chosen.end());
            list.clear();
            for (const auto& s : chosen) list += (list.empty() ? "" : ", ") + s;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
}

void ValidateView::RenderForm() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.text, 0.035f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("##check_form", ImVec2(0.0f, 0.0f), ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PopStyleVar();
    if (validator_form_) {
        RenderValidatorForm();
        ImGui::EndChild();
        ImGui::PopStyleColor();
        return;
    }

    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", editing_ >= 0 ? "Edit check" : "New check");
    ImGui::SameLine(0.0f, 12.0f);
    static const char* kKinds[] = {"Row count", "Allowed values", "Unique", "Not empty", "Range"};
    int kind = static_cast<int>(form_.kind);
    float kinds_w = 0.0f;
    for (const char* k : kKinds) kinds_w += ImGui::CalcTextSize(k).x + 24.0f;
    if (ImGui::GetContentRegionAvail().x < kinds_w) ImGui::NewLine();
    if (ui::SegmentedControl("##kind", kKinds, studio::kCheckKindCount, &kind)) {
        form_.kind = static_cast<CheckKind>(kind);
        form_error_.clear();
    }

    // What it checks, and its name.
    if (editing_ >= 0) {
        std::string reads;
        for (const auto& r : rows_)
            if (r.id == editing_) reads = r.reads;
        Dim("Reads");
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(reads.empty() ? "nothing linked" : reads.c_str());
    } else {
        Dim("After");
        ImGui::SameLine();
        std::string current;
        for (const auto& target : targets_)
            if (target.id == target_) current = target.name;
        ImGui::SetNextItemWidth(std::min(260.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::BeginCombo("##after", current.empty() ? "choose a step" : current.c_str())) {
            for (const auto& target : targets_) {
                ImGui::PushID(target.id);
                if (ImGui::Selectable(target.name.c_str(), target.id == target_)) {
                    target_ = target.id;
                    LoadColumns();
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The step whose rows it checks; the check goes right after it");
    }
    Next(ImGui::CalcTextSize("Name").x + 280.0f);
    Dim("Name");
    ImGui::SameLine();
    {
        // The hint is the name it gets when left empty.
        std::map<std::string, std::string> p;
        std::string ignored;
        const std::string hint = studio::CheckParameters(form_, p, ignored) ? studio::CheckDefaultName(p) : std::string("shown when it fails");
        TextField("##name", form_.name, hint.c_str(), 260.0f);
    }

    // The rule.
    switch (form_.kind) {
        case CheckKind::RowCount: {
            static const char* kModes[] = {"exactly", "at least", "at most", "between"};
            Dim("Rows");
            ImGui::SameLine();
            Choice("##mode", form_.count_mode, kModes, 4, 110.0f);
            Next(110.0f);
            TextField("##count", form_.count, "15349", 110.0f);
            if (form_.count_mode == 3) {
                Next(ImGui::CalcTextSize("and").x + 120.0f);
                Dim("and");
                ImGui::SameLine();
                TextField("##count_max", form_.count_max, "20000", 110.0f);
            }
            static const char* kWhere[] = {"of all rows", "where a column equals a value", "where a true/false column is true"};
            Next(250.0f);
            Choice("##where", form_.where, kWhere, 3, 250.0f);
            if (form_.where != 0) {
                Next(220.0f);
                ColumnField("##where_column", form_.where_column, 220.0f);
                if (form_.where == 1) {
                    Next(ImGui::CalcTextSize("=").x + 170.0f);
                    Dim("=");
                    ImGui::SameLine();
                    TextField("##where_value", form_.where_value, "value", 160.0f);
                }
            }
            break;
        }
        case CheckKind::AllowedValues:
            Dim("Column");
            ImGui::SameLine();
            ColumnField("##column", form_.column, 220.0f);
            Next(ImGui::CalcTextSize("is one of").x + 330.0f);
            Dim("is one of");
            ImGui::SameLine();
            TextField("##allowed", form_.allowed, "book, chapter, verse", 320.0f);
            break;
        case CheckKind::Unique:
            // Together the columns must not repeat.
            Dim("No two rows share");
            ColumnSetField("##columns", form_.columns, "id, or book, chapter");
            break;
        case CheckKind::NotEmpty:
            Dim("Column");
            ImGui::SameLine();
            ColumnField("##column", form_.column, 220.0f);
            Next(ImGui::CalcTextSize("is never empty (missing or empty text)").x);
            Dim("is never empty (missing or empty text)");
            break;
        case CheckKind::Range:
            Dim("Column");
            ImGui::SameLine();
            ColumnField("##column", form_.column, 220.0f);
            Next(ImGui::CalcTextSize("from").x + 120.0f);
            Dim("from");
            ImGui::SameLine();
            TextField("##min", form_.min_value, "no limit", 110.0f);
            Next(ImGui::CalcTextSize("to").x + 120.0f);
            Dim("to");
            ImGui::SameLine();
            TextField("##max", form_.max_value, "no limit", 110.0f);
            break;
    }
    if (columns_.empty())
        ImGui::TextColored(t.text_faint, "Run the step it follows to pick from its columns; until then type the names.");

    // The rule as it will read, or what is missing.
    std::map<std::string, std::string> parameters;
    std::string error;
    const bool complete = studio::CheckParameters(form_, parameters, error);
    ImGui::PushTextWrapPos(0.0f);
    if (!form_error_.empty()) ImGui::TextColored(t.error, "%s", form_error_.c_str());
    else if (complete) ImGui::TextColored(t.text_dim, "Rule: %s", studio::CheckRule(parameters).c_str());
    ImGui::PopTextWrapPos();

    const char* submit = editing_ >= 0 ? "Save check" : "Add check";
    if (ui::PrimaryButton(submit, complete && (editing_ >= 0 || target_ >= 0), complete ? "Choose the step it follows" : error.c_str(),
                          ui::ButtonSize::Small))
        SubmitForm();
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Small)) {
        form_open_ = false;
        editing_ = -1;
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void ValidateView::SubmitForm() {
    if (!editor_) return;
    std::map<std::string, std::string> parameters;
    std::string error;
    if (!studio::CheckParameters(form_, parameters, error)) {
        form_error_ = error;
        return;
    }
    // The name labels the node and the failure; by default the rule says it.
    std::string name = parameters["check_name"];
    if (name.empty()) {
        name = studio::CheckDefaultName(parameters);
        parameters["check_name"] = name;
    }
    if (editing_ >= 0) {
        if (!editor_->SetStepParameters(editing_, parameters, {}, name)) {
            message_ = "The check is no longer in the graph.";
            message_error_ = true;
            return;
        }
        message_ = "Saved \"" + name + "\" (Ctrl+Z on the canvas undoes it). Run checks to see it hold.";
    } else {
        std::string after;
        for (const auto& target : targets_)
            if (target.id == target_) after = target.name;
        // Added on the canvas: a collapsed recipe opens first.
        if (!editor_->GetNodeForConfiguration(target_))
            if (const int recipe = RecipeOf(target_); recipe >= 0)
                editor_->RequestSubgraphAction(recipe, ::gui::NodeEditor::SubgraphRequest::Expand);
        editor_->RequestStepAfter(target_, ::gui::NodeType::RowCountCheck, name, parameters);
        message_ = "Added \"" + name + "\" after \"" + after + "\" (Ctrl+Z on the canvas undoes it). Run checks to see it hold.";
    }
    message_error_ = false;
    form_open_ = false;
    editing_ = -1;
}

int ValidateView::ValidatorSource(int target, std::string* what) const {
    int from = target;
    if (what)
        for (const auto& t : targets_)
            if (t.id == target) *what = "\"" + t.name + "\"";
    // A recipe's steps all lead to its output, so a validator of a step
    // inside one reads the recipe's result: the recipe node, or its output
    // step while it is expanded on the canvas.
    if (const int recipe = RecipeOf(target); recipe >= 0 && editor_) {
        const auto summary = editor_->GetSubgraphSummary(recipe);
        from = summary.expanded && !summary.steps.empty() ? summary.steps.back().id : recipe;
        if (what)
            for (const auto& n : nodes_)
                if (n.id == recipe) *what = "the recipe \"" + n.name + "\" (its result)";
    }
    return from;
}

void ValidateView::RenderValidatorForm() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_bright, "%s", editing_ >= 0 ? "Edit Data Validator" : "New Data Validator");
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "reports the rows that break a rule as an issues table; the run goes on");
    if (editing_ >= 0) {
        std::string reads;
        for (const auto& r : rows_)
            if (r.id == editing_) reads = r.reads;
        Dim("Reads");
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(reads.empty() ? "nothing linked" : reads.c_str());
    } else {
        Dim("Reads");
        ImGui::SameLine();
        std::string current;
        for (const auto& target : targets_)
            if (target.id == target_) current = target.name;
        ImGui::SetNextItemWidth(std::min(260.0f, ImGui::GetContentRegionAvail().x));
        if (ImGui::BeginCombo("##reads", current.empty() ? "choose a step" : current.c_str())) {
            for (const auto& target : targets_) {
                ImGui::PushID(target.id);
                if (ImGui::Selectable(target.name.c_str(), target.id == target_)) {
                    target_ = target.id;
                    LoadColumns();
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        std::string what;
        ValidatorSource(target_, &what);
        if (what.rfind("the recipe", 0) == 0) {
            Next(ImGui::CalcTextSize(what.c_str()).x + 60.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_faint, "it reads %s", what.c_str());
        }
    }
    // Each rule's columns wrap under the rule, not under its label.
    const float label_w = ImGui::CalcTextSize("No repeated value").x + 16.0f;
    const auto rule = [&](const char* label, const char* id, std::string& list, const char* hint) {
        Dim(label);
        ImGui::SameLine(label_w);
        ImGui::Indent(label_w);
        ImGui::Dummy(ImVec2(0.0f, ImGui::GetFrameHeight()));
        ColumnSetField(id, list, hint);
        ImGui::Unindent(label_w);
    };
    rule("Required columns", "##required", vform_.required, "text, source_id");
    rule("No missing value", "##not_null", vform_.not_null, "text");
    rule("No repeated value", "##unique", vform_.unique, "paragraph_id");
    if (columns_.empty())
        ImGui::TextColored(t.text_faint, "Run the step it reads to pick from its columns; until then type the names.");

    std::map<std::string, std::string> parameters;
    std::string error;
    const bool complete = studio::ValidatorParameters(vform_, parameters, error);
    ImGui::PushTextWrapPos(0.0f);
    if (!form_error_.empty()) ImGui::TextColored(t.error, "%s", form_error_.c_str());
    else if (complete) ImGui::TextColored(t.text_dim, "Rules: %s", studio::ValidatorRule(parameters).c_str());
    ImGui::PopTextWrapPos();
    const char* submit = editing_ >= 0 ? "Save validator" : "Add validator";
    if (ui::PrimaryButton(submit, complete && (editing_ >= 0 || target_ >= 0), complete ? "Choose the step it reads" : error.c_str(),
                          ui::ButtonSize::Small))
        SubmitValidator();
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", true, nullptr, ui::ButtonSize::Small)) {
        form_open_ = false;
        editing_ = -1;
    }
}

void ValidateView::SubmitValidator() {
    if (!editor_) return;
    std::map<std::string, std::string> parameters;
    std::string error;
    if (!studio::ValidatorParameters(vform_, parameters, error)) {
        form_error_ = error;
        return;
    }
    if (editing_ >= 0) {
        if (!editor_->SetStepParameters(editing_, parameters)) {
            message_ = "The validator is no longer in the graph.";
            message_error_ = true;
            return;
        }
        message_ = "Saved the Data Validator's rules (Ctrl+Z on the canvas undoes it). Run checks to see its issues.";
    } else {
        std::string what;
        const int from = ValidatorSource(target_, &what);
        editor_->RequestBranchAfter(from, ::gui::NodeType::DataValidator, parameters);
        message_ = "Added a Data Validator reading " + what + " (Ctrl+Z on the canvas undoes it). Run checks to see its issues.";
    }
    message_error_ = false;
    form_open_ = false;
    editing_ = -1;
}

void ValidateView::RunChecks() {
    if (!editor_) return;
    std::vector<int> ids;
    for (const auto& r : rows_) ids.push_back(r.id);
    if (!editor_->ExecuteDataPipelineThrough(ids, false)) {
        message_ = "A pipeline run is in progress, or the graph could not be read; see the notice.";
        message_error_ = true;
        return;
    }
    message_.clear();
}

}  // namespace cyxwiz
