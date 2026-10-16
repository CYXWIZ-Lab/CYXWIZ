// The guided step editor (TOFIX146 step 5, mockup v5 board 3), shared by the
// SQL Workbench's Guided mode and Data Studio > Prepare (TOFIX148 D3).
#include "guided_editor.h"

#include "../core/sql_step_presentation.h"
#include "icons.h"
#include "sql_workbench_ui.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cctype>
#include <cstring>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace guided = cyxwiz::guided;
using json = nlohmann::json;
using cyxwiz::sqlstep::FormatCount;
using namespace sqlwb;

constexpr float kLabelWidth = 170.0f;

// ---- Field widgets: each edits one value of an operation's params ----------

std::string& StrRef(json& p, const char* key) {
    if (!p.contains(key) || !p[key].is_string()) p[key] = "";
    return p[key].get_ref<std::string&>();
}

// The next item on this line when `width` still fits, else on the next
// line (a narrow host such as Prepare must not clip a field).
void Next(float width) {
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    Dim(text);
    ImGui::SameLine(kLabelWidth);
}

bool Combo(const char* id, std::string& value, const std::vector<std::string>& options, const char* empty = "choose...") {
    bool changed = false;
    ImGui::SetNextItemWidth(std::min(320.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::BeginCombo(id, value.empty() ? empty : value.c_str())) {
        for (const auto& option : options) {
            if (ImGui::Selectable(option.c_str(), option == value)) {
                value = option;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

// A column: a list when the columns are known, else typed.
bool ColumnField(const char* id, std::string& value, const std::vector<std::string>& columns, bool known) {
    if (known && !columns.empty()) return Combo(id, value, columns, "choose a column");
    ImGui::SetNextItemWidth(std::min(320.0f, ImGui::GetContentRegionAvail().x));
    return ImGui::InputTextWithHint(id, "column name", &value);
}

bool TextField(const char* id, std::string& value, const char* hint = "", float width = 320.0f) {
    ImGui::SetNextItemWidth(std::min(width, ImGui::GetContentRegionAvail().x));
    return ImGui::InputTextWithHint(id, hint, &value);
}

bool Choice(const char* id, std::string& value, const std::vector<const char*>& options) {
    int selected = 0;
    for (size_t i = 0; i < options.size(); ++i) {
        if (value == options[i]) selected = static_cast<int>(i);
    }
    if (options.size() <= 4) {
        if (ui::SegmentedControl(id, options.data(), static_cast<int>(options.size()), &selected)) {
            value = options[static_cast<size_t>(selected)];
            return true;
        }
        return false;
    }
    std::vector<std::string> list(options.begin(), options.end());
    return Combo(id, value, list);
}

bool FlagField(const char* label, json& p, const char* key) {
    bool value = p.contains(key) && p[key].is_boolean() ? p[key].get<bool>() : false;
    if (ImGui::Checkbox(label, &value)) {
        p[key] = value;
        return true;
    }
    return false;
}

// A set of columns as toggles.
bool ColumnSet(const char* id, json& p, const char* key, const std::vector<std::string>& columns) {
    if (!p.contains(key) || !p[key].is_array()) p[key] = json::array();
    std::vector<std::string> chosen;
    for (const auto& v : p[key]) {
        if (v.is_string()) chosen.push_back(v.get<std::string>());
    }
    bool changed = false;
    ImGui::PushID(id);
    float x = 0.0f;
    const float avail = ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < columns.size(); ++i) {
        const bool on = std::find(chosen.begin(), chosen.end(), columns[i]) != chosen.end();
        const float w = ImGui::CalcTextSize(columns[i].c_str()).x + ImGui::GetFrameHeight() + 16.0f;
        if (i > 0 && x + w < avail) ImGui::SameLine();
        else if (i > 0) x = 0.0f;
        x += w + ImGui::GetStyle().ItemSpacing.x;
        bool value = on;
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Checkbox(columns[i].c_str(), &value)) {
            if (value) chosen.push_back(columns[i]);
            else chosen.erase(std::remove(chosen.begin(), chosen.end(), columns[i]), chosen.end());
            changed = true;
        }
        ImGui::PopID();
    }
    if (columns.empty()) Dim("(the columns are known after the inputs load)");
    ImGui::PopID();
    if (changed) p[key] = chosen;
    return changed;
}

const std::vector<const char*> kConditionOps = {
    "equals", "not equals", "greater than", "at least", "less than", "at most", "contains", "does not contain",
    "starts with", "ends with", "matches pattern", "does not match pattern", "is empty", "is not empty",
    "is one of", "is not one of"};

bool ConditionRow(json& c, const std::vector<std::string>& columns, bool known) {
    bool changed = false;
    std::string& column = StrRef(c, "column");
    std::string& op = StrRef(c, "op");
    if (op.empty()) op = "equals";
    ImGui::SetNextItemWidth(200.0f);
    changed |= ColumnField("##column", column, columns, known);
    Next(170.0f);
    ImGui::SetNextItemWidth(170.0f);
    if (ImGui::BeginCombo("##op", op.c_str())) {
        for (const char* o : kConditionOps) {
            if (ImGui::Selectable(o, op == o)) {
                op = o;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    if (op != "is empty" && op != "is not empty") {
        Next(220.0f);
        changed |= TextField("##value", StrRef(c, "value"), op.find("one of") != std::string::npos ? "a, b, c" : "value", 220.0f);
    }
    return changed;
}

// A list editor: each item drawn by `row`, with a remove button and an
// "+ add" link. Returns true when anything changed.
template <typename Row>
bool ListEditor(const char* id, json& p, const char* key, const json& fresh, const char* add_label, Row row) {
    if (!p.contains(key) || !p[key].is_array()) p[key] = json::array();
    json& list = p[key];
    bool changed = false;
    int remove = -1;
    ImGui::PushID(id);
    for (size_t i = 0; i < list.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        changed |= row(list[i]);
        Next(ImGui::GetFrameHeight() + 8.0f);
        if (ui::GhostButton(ICON_FA_XMARK)) remove = static_cast<int>(i);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove");
        ImGui::PopID();
    }
    if (remove >= 0) {
        list.erase(list.begin() + remove);
        changed = true;
    }
    if (ui::LinkButton(add_label)) {
        list.push_back(fresh);
        changed = true;
    }
    ImGui::PopID();
    return changed;
}
}  // namespace

// ---- State ----------------------------------------------------------------------

bool GuidedEditor::Load(const std::string& guided_ops, std::string* error) {
    active_ = false;
    step_ = guided::GuidedStep{};
    bool ok = true;
    if (!guided_ops.empty()) {
        if (auto step = guided::FromJson(guided_ops, error)) {
            step_ = std::move(*step);
            active_ = true;
        } else {
            ok = false;
        }
    }
    saved_json_ = active_ ? Json() : std::string();
    picked_ = step_.ops.empty() ? -1 : 0;
    Regenerate();
    inputs_key_.clear();  // the next SetInputs writes the SQL for the inputs as they are
    return ok;
}

void GuidedEditor::Start() {
    active_ = true;
    step_ = guided::GuidedStep{};
    picked_ = -1;
    Changed({});
}

bool GuidedEditor::Edited() const {
    return active_ ? Json() != saved_json_ : !saved_json_.empty();
}

bool GuidedEditor::SetInputs(std::vector<Input> inputs) {
    std::string key;
    for (const auto& input : inputs) {
        key += input.alias + ":" +
               (input.table ? std::to_string(input.table->num_columns()) + "@" + std::to_string(reinterpret_cast<uintptr_t>(input.table.get()))
                            : std::string("-")) +
               ";";
    }
    inputs_ = std::move(inputs);
    if (key == inputs_key_) return false;
    inputs_key_ = key;
    if (!active_) return false;
    Changed({});
    return true;
}

guided::Context GuidedEditor::MakeContext() const {
    guided::Context context;
    for (const auto& input : inputs_) {
        context.inputs.push_back(input.alias);
        if (!input.table) continue;
        auto& columns = context.columns[input.alias];
        for (const auto& field : input.table->schema()->fields()) {
            columns.push_back(field->name());
            if (arrow::is_integer(field->type()->id())) context.whole_number_columns.insert(field->name());
        }
    }
    // Inputs not loaded yet: what the step last saved.
    if (context.columns.empty() && !step_.saved_context.inputs.empty()) return step_.saved_context;
    if (context.inputs.empty()) context.inputs.push_back("input");
    return context;
}

void GuidedEditor::Regenerate() {
    const auto context = MakeContext();
    step_.saved_context = context;
    generated_ = guided::GenerateSql(step_, context);
    columns_ = guided::TrackColumns(step_, context);
}

void GuidedEditor::Changed(const std::string& before_json) {
    if (!before_json.empty() && before_json != Json()) {
        undo_.push_back(before_json);
        if (undo_.size() > 100) undo_.erase(undo_.begin());
        redo_.clear();
    }
    Regenerate();
    if (on_changed) on_changed();
}

std::vector<guided::CheckNode> GuidedEditor::TakeChecks() {
    if (!active_) return {};
    auto split = guided::SplitOffChecks(step_);
    if (split.checks.empty()) return {};
    const std::string before = Json();
    split.step.saved_context = step_.saved_context;
    step_ = std::move(split.step);
    picked_ = std::min(picked_, static_cast<int>(step_.ops.size()) - 1);
    Changed(before);
    return std::move(split.checks);
}

int64_t GuidedEditor::RowsAfter(size_t op, const std::map<std::string, int64_t>& rows) const {
    if (op >= generated_.part_names.size() || generated_.part_names[op].empty()) return -1;
    const auto found = rows.find(generated_.part_names[op]);
    return found == rows.end() ? -1 : found->second;
}

void GuidedEditor::Undo() {
    if (undo_.empty()) return;
    redo_.push_back(Json());
    if (auto step = guided::FromJson(undo_.back())) step_ = std::move(*step);
    undo_.pop_back();
    Changed({});
}

void GuidedEditor::Redo() {
    if (redo_.empty()) return;
    undo_.push_back(Json());
    if (auto step = guided::FromJson(redo_.back())) step_ = std::move(*step);
    redo_.pop_back();
    Changed({});
}

// ---- Layout ---------------------------------------------------------------------

void GuidedEditor::RenderTemplateButton(ui::ButtonSize size) {
    if (ui::SecondaryButton("Start from a template " ICON_FA_CHEVRON_DOWN, true, nullptr, size)) {
        ImGui::OpenPopup("##guided_templates");
    }
    if (ImGui::BeginPopup("##guided_templates")) {
        std::string text_column;
        const auto& in = columns_.empty() ? guided::Columns{} : columns_.front();
        for (const auto& name : in.names) {
            if (name == "text") text_column = name;
        }
        if (text_column.empty() && !in.names.empty()) text_column = in.names.back();
        for (const auto& tpl : guided::Templates()) {
            if (ImGui::MenuItem(tpl.name)) {
                const std::string before = Json();
                const auto made = guided::MakeTemplate(tpl.id, text_column);
                step_.ops.insert(step_.ops.end(), made.ops.begin(), made.ops.end());
                picked_ = static_cast<int>(step_.ops.size()) - static_cast<int>(made.ops.size());
                Changed(before);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tpl.description);
        }
        ImGui::EndPopup();
    }
}

void GuidedEditor::AddOp(const guided::OpInfo& op) {
    const std::string before = Json();
    const int at = picked_ < 0 ? static_cast<int>(step_.ops.size()) : picked_ + 1;
    const auto& in = at < static_cast<int>(columns_.size()) ? columns_[static_cast<size_t>(at)] : guided::Columns{};
    std::string column;
    for (const auto& name : in.names) {
        if (name == "text") column = name;
    }
    if (column.empty() && !in.names.empty()) column = in.names.front();
    guided::Operation added{op.id, true, guided::DefaultParams(op.id, column)};
    if (std::string(op.id) == "choose_columns") {
        for (const auto& name : in.names) added.params["columns"].push_back({{"name", name}, {"keep", true}});
    }
    step_.ops.insert(step_.ops.begin() + at, std::move(added));
    picked_ = at;
    Changed(before);
}

bool GuidedEditor::Matches(const guided::OpInfo& op) const {
    std::string search = search_;
    for (char& c : search) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (search.empty()) return true;
    std::string haystack = std::string(op.name) + " " + op.hint + " " + op.category;
    for (char& c : haystack) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return haystack.find(search) != std::string::npos;
}

void GuidedEditor::RenderAddMenu(ui::ButtonSize size) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (ui::SecondaryButton(ICON_FA_PLUS " Add an operation " ICON_FA_CHEVRON_DOWN, true, nullptr, size)) {
        search_[0] = '\0';
        ImGui::OpenPopup("##guided_add");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds it after the selected operation");
    if (ImGui::BeginPopup("##guided_add")) {
        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(300.0f);
        ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS " Find an operation", search_, sizeof(search_));
        const char* category = "";
        for (const auto& op : guided::Library()) {
            if (!Matches(op)) continue;
            if (std::strcmp(category, op.category) != 0) {
                category = op.category;
                std::string upper = category;
                for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                Coloured(upper, t.text_faint);
            }
            ImGui::PushID(op.id);
            if (ImGui::MenuItem(op.name, op.hint)) AddOp(op);
            ImGui::PopID();
        }
        ImGui::EndPopup();
    }
}

void GuidedEditor::RenderLibrary(const ImVec2& size) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##guided_library", size, t.bg_panel)) {
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##search", ICON_FA_MAGNIFYING_GLASS " Find an operation", search_, sizeof(search_));
        DimWrapped("Click to add it after the selected operation.");
        const char* category = "";
        ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        for (const auto& op : guided::Library()) {
            if (!Matches(op)) continue;
            if (std::strcmp(category, op.category) != 0) {
                category = op.category;
                ImGui::Dummy(ImVec2(0.0f, t.space_xs));
                std::string upper = category;
                for (char& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                Coloured(upper, t.text_faint);
            }
            ImGui::PushID(op.id);
            const ImVec2 start = ImGui::GetCursorPos();
            if (ImGui::Selectable("##op", false, ImGuiSelectableFlags_AllowOverlap,
                                  ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 2.0f))) {
                AddOp(op);
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Add \"%s\"", op.name);
            const ImVec2 after = ImGui::GetCursorPos();
            ImGui::SetCursorPos(start);
            ImGui::TextUnformatted(op.name);
            Dim(op.hint);
            ImGui::SetCursorPos(after);
            ImGui::PopID();
        }
        ImGui::PopStyleColor(2);
    }
    EndPanel();
}

void GuidedEditor::RenderCards(const ImVec2& size, const CardFacts& facts) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##guided_cards", size, t.bg_window)) {
        // Where the operations start.
        std::string start = "Starts from ";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            const auto& input = inputs_[i];
            start += (i ? " and " : "") + input.alias +
                     (input.table ? " (" + FormatCount(input.rows) + " rows, " +
                                        std::to_string(input.table->num_columns()) + " columns" +
                                        (input.parquet_path.empty() ? ")" : ", on disk)")
                                  : " (not loaded)");
        }
        DimWrapped(start);
        if (step_.ops.empty()) {
            ImGui::Dummy(ImVec2(0.0f, t.space_md));
            DimWrapped("No operations yet: add one, or start from a template.");
        }
        int move_from = -1, move_to = -1, remove = -1;
        bool toggled = false;
        const std::string before = Json();
        for (size_t i = 0; i < step_.ops.size(); ++i) {
            auto& op = step_.ops[i];
            const auto* info = guided::FindOp(op.type);
            ImGui::PushID(static_cast<int>(i));
            const bool picked = picked_ == static_cast<int>(i);
            const ImVec4 fill = picked ? t.selection : t.bg_panel;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, fill);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
            if (ImGui::BeginChild("##card", ImVec2(0.0f, 0.0f),
                                  ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding)) {
                if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) picked_ = static_cast<int>(i);
                // The grip drags the operation to another place in the list.
                Coloured(ICON_FA_GRIP_VERTICAL, t.text_faint);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                    ImGui::SetTooltip("Drag to move this operation");
                }
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                    const int index = static_cast<int>(i);
                    ImGui::SetDragDropPayload("GUIDED_OP", &index, sizeof(index));
                    Coloured(std::to_string(i + 1) + ". " + (info ? info->name : op.type), t.text_bright);
                    ImGui::EndDragDropSource();
                }
                ImGui::SameLine();
                Coloured(std::to_string(i + 1), t.text_dim);
                ImGui::SameLine();
                Coloured(info ? info->name : op.type, op.on ? t.text_bright : t.text_dim);
                // Rows after it and the change.
                std::string rows;
                const int64_t after = facts.rows ? RowsAfter(i, *facts.rows) : -1;
                if (after >= 0) {
                    int64_t before_rows = -1;
                    for (size_t k = i; k > 0 && before_rows < 0; --k) before_rows = RowsAfter(k - 1, *facts.rows);
                    if (before_rows < 0) before_rows = facts.input_rows;
                    rows = FormatCount(after) + " rows";
                    if (before_rows >= 0 && before_rows != after) {
                        const int64_t change = after - before_rows;
                        rows += change > 0 ? "  +" + FormatCount(change) : "  -" + FormatCount(-change);
                    }
                }
                const float toggle_w = ImGui::GetFrameHeight() + ImGui::CalcTextSize("On").x + 10.0f;
                AlignRight(toggle_w);
                bool on = op.on;
                if (ImGui::Checkbox("On", &on)) {
                    op.on = on;
                    toggled = true;
                }
                ColouredWrapped(guided::Summarise(op), op.on ? t.text : t.text_dim);
                if (!rows.empty()) Coloured(rows, t.text_dim);
                const std::string& error = i < generated_.errors.size() ? generated_.errors[i] : std::string();
                if (!error.empty() && op.on) ColouredWrapped(ICON_FA_TRIANGLE_EXCLAMATION " " + error, t.caution);
                if (picked) {
                    if (ui::GhostButton(ICON_FA_CHEVRON_UP, i > 0, "Already first")) {
                        move_from = static_cast<int>(i);
                        move_to = static_cast<int>(i) - 1;
                    }
                    ImGui::SameLine();
                    if (ui::GhostButton(ICON_FA_CHEVRON_DOWN, i + 1 < step_.ops.size(), "Already last")) {
                        move_from = static_cast<int>(i);
                        move_to = static_cast<int>(i) + 1;
                    }
                    ImGui::SameLine();
                    if (ui::DangerButton("Remove")) remove = static_cast<int>(i);
                }
            }
            ImGui::EndChild();
            // The whole card takes a dropped operation: it moves to this place.
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GUIDED_OP")) {
                    move_from = *static_cast<const int*>(payload->Data);
                    move_to = static_cast<int>(i);
                }
                ImGui::EndDragDropTarget();
            }
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
            ImGui::PopID();
        }
        if (move_from >= 0 && move_from != move_to &&
            guided::MoveOp(step_, static_cast<size_t>(move_from), static_cast<size_t>(move_to))) {
            picked_ = move_to;
            Changed(before);
        } else if (remove >= 0) {
            step_.ops.erase(step_.ops.begin() + remove);
            picked_ = std::min(remove, static_cast<int>(step_.ops.size()) - 1);
            Changed(before);
        } else if (toggled) {
            Changed(before);
        }
        // Checks that come from the graph (Check nodes after the step).
        if (facts.checks_after > 0) {
            ImGui::Dummy(ImVec2(0.0f, t.space_sm));
            Dim("Checks after this step in the recipe: " + std::to_string(facts.checks_after));
        }
    }
    EndPanel();
}

void GuidedEditor::RenderEditor() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (picked_ < 0 || picked_ >= static_cast<int>(step_.ops.size())) {
        Dim("Select an operation to change it.");
        return;
    }
    auto& op = step_.ops[static_cast<size_t>(picked_)];
    const auto* info = guided::FindOp(op.type);
    Coloured("Editing: " + std::string(info ? info->name : op.type), t.text_bright);
    ImGui::SameLine();
    Dim(info ? info->hint : "");
    const auto& in = picked_ < static_cast<int>(columns_.size())
                         ? columns_[static_cast<size_t>(picked_)]
                         : guided::Columns{};
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    if (ImGui::BeginChild("##fields", ImGui::GetContentRegionAvail())) {
        const std::string before = Json();
        if (RenderFields(op, in)) Changed(before);
    }
    ImGui::EndChild();
}

bool GuidedEditor::RenderFields(guided::Operation& op, const guided::Columns& in) {
    json& p = op.params;
    const auto& cols = in.names;
    const bool known = in.known;
    bool changed = false;
    const std::string& type = op.type;
    const auto column_row = [&](const char* label, const char* key) {
        Label(label);
        return ColumnField((std::string("##") + key).c_str(), StrRef(p, key), cols, known);
    };
    const auto text_row = [&](const char* label, const char* key, const char* hint = "") {
        Label(label);
        return TextField((std::string("##") + key).c_str(), StrRef(p, key), hint);
    };
    std::vector<std::string> other_inputs;
    for (const auto& input : inputs_) {
        if (!inputs_.empty() && &input != &inputs_.front()) other_inputs.push_back(input.alias);
    }
    const auto other_columns = [&](const std::string& alias) {
        for (const auto& input : inputs_) {
            if (input.alias != alias || !input.table) continue;
            std::vector<std::string> names;
            for (const auto& f : input.table->schema()->fields()) names.push_back(f->name());
            return names;
        }
        return std::vector<std::string>{};
    };

    if (type == "keep_rows") {
        Label("Keep rows where");
        changed |= Choice("##match", StrRef(p, "match"), {"all", "any"});
        Next(ImGui::CalcTextSize("of these are true:").x);
        Dim("of these are true:");
        changed |= ListEditor("conditions", p, "conditions", json{{"column", ""}, {"op", "equals"}, {"value", ""}}, "+ condition",
                              [&](json& c) { return ConditionRow(c, cols, known); });
    } else if (type == "remove_duplicates") {
        DimWrapped("Rows count as duplicates when the chosen columns match (none chosen: every column). The first row is kept.");
        changed |= ColumnSet("columns", p, "columns", cols);
    } else if (type == "sort_rows" || type == "number_rows") {
        const char* key = type == "sort_rows" ? "keys" : "order_by";
        if (type == "number_rows") {
            changed |= text_row("Name", "name", "row_index");
            Label("Within each");
            ImGui::NewLine();
            changed |= ColumnSet("partition", p, "partition_by", cols);
            Label("In the order of");
            ImGui::NewLine();
        }
        changed |= ListEditor("keys", p, key, json{{"column", ""}, {"descending", false}}, "+ column", [&](json& k) {
            bool c = false;
            ImGui::SetNextItemWidth(240.0f);
            c |= ColumnField("##column", StrRef(k, "column"), cols, known);
            Next(ImGui::GetFrameHeight() + ImGui::CalcTextSize("high to low").x + 8.0f);
            bool desc = k.value("descending", false);
            if (ImGui::Checkbox("high to low", &desc)) {
                k["descending"] = desc;
                c = true;
            }
            return c;
        });
    } else if (type == "keep_first") {
        Label("Keep");
        changed |= Choice("##mode", StrRef(p, "mode"), {"first", "sample", "every"});
        changed |= text_row(StrRef(p, "mode") == "every" ? "Every Nth row, N =" : "Number of rows", "count", "1000");
    } else if (type == "choose_columns") {
        if (!p.contains("columns") || !p["columns"].is_array()) p["columns"] = json::array();
        json& list = p["columns"];
        // Columns the input has that the list does not.
        std::vector<std::string> missing;
        for (const auto& name : cols) {
            bool present = false;
            for (const auto& c : list) present = present || c.value("name", std::string()) == name;
            if (!present) missing.push_back(name);
        }
        if (!missing.empty()) {
            if (ui::LinkButton(("Add the " + std::to_string(missing.size()) + " input column(s) not listed").c_str())) {
                for (const auto& name : missing) list.push_back({{"name", name}, {"keep", false}});
                changed = true;
            }
        }
        DimWrapped("Tick to keep, type to rename, arrows to reorder.");
        int move_from = -1, move_to = -1;
        for (size_t i = 0; i < list.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            bool keep = list[i].value("keep", true);
            if (ImGui::Checkbox("##keep", &keep)) {
                list[i]["keep"] = keep;
                changed = true;
            }
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            Coloured(list[i].value("name", std::string("?")), keep ? ui::CurrentTokens().text : ui::CurrentTokens().text_faint);
            ImGui::SameLine(260.0f);
            changed |= TextField("##rename", StrRef(list[i], "rename"), "rename to", 200.0f);
            ImGui::SameLine();
            if (ui::GhostButton(ICON_FA_CHEVRON_UP, i > 0)) {
                move_from = static_cast<int>(i);
                move_to = static_cast<int>(i) - 1;
            }
            ImGui::SameLine();
            if (ui::GhostButton(ICON_FA_CHEVRON_DOWN, i + 1 < list.size())) {
                move_from = static_cast<int>(i);
                move_to = static_cast<int>(i) + 1;
            }
            ImGui::PopID();
        }
        if (move_from >= 0) {
            std::swap(list[static_cast<size_t>(move_from)], list[static_cast<size_t>(move_to)]);
            changed = true;
        }
    } else if (type == "add_column") {
        changed |= text_row("New column", "name", "name");
        Label("Value");
        changed |= Choice("##formula", StrRef(p, "formula"),
                          {"length", "word count", "upper case", "lower case", "combine", "arithmetic", "constant", "condition"});
        const std::string formula = StrRef(p, "formula");
        if (formula == "constant") {
            changed |= text_row("Text", "value");
        } else if (formula == "condition") {
            if (!p.contains("condition") || !p["condition"].is_object()) p["condition"] = json{{"column", ""}, {"op", "equals"}, {"value", ""}};
            Label("When");
            changed |= ConditionRow(p["condition"], cols, known);
            changed |= text_row("Then", "then");
            changed |= text_row("Otherwise", "else");
        } else {
            changed |= column_row("Of column", "column");
            if (formula == "combine") {
                changed |= column_row("And column", "column2");
                changed |= text_row("Separator", "separator", "space");
            } else if (formula == "arithmetic") {
                Label("Operation");
                changed |= Choice("##operator", StrRef(p, "operator"), {"+", "-", "*", "/"});
                changed |= column_row("With column", "column2");
                changed |= text_row("Or a number", "value", "1000");
            }
        }
    } else if (type == "change_type") {
        changed |= column_row("Column", "column");
        Label("Make it");
        changed |= Choice("##to", StrRef(p, "to"), {"text", "whole number", "decimal", "true/false"});
        Label("When a value does not fit");
        changed |= Choice("##on_error", StrRef(p, "on_error"), {"empty", "stop"});
    } else if (type == "split_merge") {
        Label("Mode");
        changed |= Choice("##mode", StrRef(p, "mode"), {"split", "merge"});
        if (StrRef(p, "mode") == "merge") {
            Label("Columns");
            ImGui::NewLine();
            changed |= ColumnSet("columns", p, "columns", cols);
            changed |= text_row("Separator", "separator", "space");
            changed |= text_row("New column", "name");
        } else {
            changed |= column_row("Column", "column");
            changed |= text_row("Separator", "separator", ",");
            std::string names;
            for (const auto& n : p.value("names", json::array())) names += (names.empty() ? "" : ", ") + n.get<std::string>();
            Label("New columns");
            if (TextField("##names", names, "part_1, part_2")) {
                json list = json::array();
                std::string item;
                for (const char c : names + ",") {
                    if (c == ',') {
                        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
                        while (!item.empty() && item.back() == ' ') item.pop_back();
                        if (!item.empty()) list.push_back(item);
                        item.clear();
                    } else {
                        item += c;
                    }
                }
                p["names"] = list;
                changed = true;
            }
        }
    } else if (type == "clean_text") {
        changed |= column_row("Column", "column");
        changed |= FlagField("Remove markup (_italic_, =bold=)", p, "remove_markup");
        changed |= FlagField("Remove footnote marks ([1], [a])", p, "remove_footnote_marks");
        changed |= FlagField("Collapse spaces and line breaks", p, "collapse_spaces");
        changed |= FlagField("Trim the ends", p, "trim");
        Label("Case");
        changed |= Choice("##case", StrRef(p, "case"), {"unchanged", "lower", "upper"});
    } else if (type == "find_replace") {
        changed |= column_row("Column", "column");
        changed |= text_row("Find", "find");
        changed |= text_row("Replace with", "replace", "(nothing)");
        changed |= FlagField("The text to find is a pattern (regular expression)", p, "pattern");
        changed |= FlagField("Match case", p, "match_case");
    } else if (type == "keep_between") {
        changed |= column_row("Column", "column");
        changed |= text_row("Start marker", "start", "(the start of the text)");
        changed |= text_row("End marker", "end", "(the end of the text)");
        changed |= FlagField("Keep the markers", p, "keep_markers");
        Label("When the markers are missing");
        changed |= Choice("##if_missing", StrRef(p, "if_missing"), {"keep", "empty"});
    } else if (type == "split_into_rows") {
        changed |= column_row("Column", "column");
        Label("One row per");
        changed |= Choice("##by", StrRef(p, "by"), {"paragraphs", "lines", "sentences", "separator"});
        if (StrRef(p, "by") == "separator") changed |= text_row("Separator", "separator");
        changed |= text_row("Number them in", "index_name", "piece_index");
        changed |= FlagField("Drop empty pieces", p, "drop_empty");
    } else if (type == "extract_pattern") {
        changed |= column_row("From column", "column");
        Label("Preset");
        if (ui::LinkButton("a number")) {
            p["pattern"] = "[0-9]+";
            changed = true;
        }
        Next(ImGui::CalcTextSize("a Bible reference").x);
        if (ui::LinkButton("a Bible reference")) {
            p["pattern"] = "(?:[1-3] )?[A-Z][a-z]+ [0-9]+:[0-9]+";
            changed = true;
        }
        Next(ImGui::CalcTextSize("a year").x);
        if (ui::LinkButton("a year")) {
            p["pattern"] = "\\b(1[5-9]|20)[0-9]{2}\\b";
            changed = true;
        }
        changed |= text_row("Pattern", "pattern", "regular expression");
        changed |= text_row("Group", "group", "0 = the whole match");
        changed |= text_row("New column", "name");
    } else if (type == "look_up") {
        Label("Look up from");
        changed |= Combo("##table", StrRef(p, "table"), other_inputs, other_inputs.empty() ? "add a second input pin to the step" : "choose a table");
        Label("Match this key");
        ImGui::NewLine();
        changed |= ListEditor("key", p, "key", json{{"kind", "column"}, {"value", ""}}, "+ key part", [&](json& k) {
            bool c = Choice("##kind", StrRef(k, "kind"), {"column", "text"});
            Next(220.0f);
            if (StrRef(k, "kind") == "text") {
                c |= TextField("##value", StrRef(k, "value"), "text, e.g. :1", 180.0f);
            } else {
                ImGui::SetNextItemWidth(220.0f);
                c |= ColumnField("##value", StrRef(k, "value"), cols, known);
            }
            return c;
        });
        const auto theirs = other_columns(StrRef(p, "table"));
        Label("to");
        changed |= ColumnField("##other_key", StrRef(p, "other_key"), theirs, !theirs.empty());
        Label("Bring in");
        ImGui::NewLine();
        changed |= ListEditor("bring", p, "bring", json{{"column", ""}, {"as", ""}}, "+ column", [&](json& b) {
            ImGui::SetNextItemWidth(220.0f);
            bool c = ColumnField("##column", StrRef(b, "column"), theirs, !theirs.empty());
            Next(ImGui::CalcTextSize("as").x + 190.0f);
            Dim("as");
            ImGui::SameLine();
            c |= TextField("##as", StrRef(b, "as"), "same name", 180.0f);
            return c;
        });
        Label("Rows with no match");
        changed |= Choice("##no_match", StrRef(p, "no_match"), {"keep", "remove", "stop"});
        DimWrapped("keep: their new columns stay empty; remove: they are dropped; stop: the step fails and names the first key.");
    } else if (type == "append_rows") {
        Label("Append the rows of");
        changed |= Combo("##table", StrRef(p, "table"), other_inputs, other_inputs.empty() ? "add a second input pin to the step" : "choose a table");
        DimWrapped("Columns are matched by name; a column only one table has is empty in the other's rows.");
    } else if (type == "group_summarise") {
        Label("One row per");
        ImGui::NewLine();
        changed |= ColumnSet("by", p, "by", cols);
        Label("With");
        ImGui::NewLine();
        changed |= ListEditor("measures", p, "measures", json{{"function", "count"}, {"column", ""}, {"name", ""}}, "+ summary", [&](json& m) {
            ImGui::SetNextItemWidth(150.0f);
            std::string& fn = StrRef(m, "function");
            bool c = Combo("##fn", fn, {"count", "count distinct", "sum", "average", "min", "max", "first", "join text"});
            if (fn != "count") {
                Next(200.0f);
                ImGui::SetNextItemWidth(200.0f);
                c |= ColumnField("##column", StrRef(m, "column"), cols, known);
            }
            Next(160.0f);
            c |= TextField("##name", StrRef(m, "name"), "name", 160.0f);
            if (fn == "join text") {
                Next(90.0f);
                c |= TextField("##separator", StrRef(m, "separator"), "separator", 90.0f);
            }
            return c;
        });
    } else if (type == "pivot_unpivot") {
        Label("Mode");
        changed |= Choice("##mode", StrRef(p, "mode"), {"pivot", "unpivot"});
        if (StrRef(p, "mode") == "unpivot") {
            Label("Columns to turn into rows");
            ImGui::NewLine();
            changed |= ColumnSet("columns", p, "columns", cols);
            changed |= text_row("Name column", "name_column", "name");
            changed |= text_row("Value column", "value_column", "value");
        } else {
            changed |= column_row("A column per value of", "names_from");
            changed |= text_row("Those values", "values", "a, b, c");
            Next(ImGui::CalcTextSize("Find values").x);
            const bool can_find = !busy && !inputs_.empty() && inputs_.front().table && !StrRef(p, "names_from").empty();
            if (ui::LinkButton("Find values", can_find)) {
                // Distinct values from the main input (up to 50).
                const auto column = inputs_.front().table->GetColumnByName(StrRef(p, "names_from"));
                std::vector<std::string> seen;
                for (int64_t r = 0; column && r < column->length() && seen.size() < 50; ++r) {
                    auto s = column->GetScalar(r);
                    if (!s.ok() || !(*s)->is_valid) continue;
                    const std::string v = (*s)->ToString();
                    if (std::find(seen.begin(), seen.end(), v) == seen.end()) seen.push_back(v);
                }
                std::string joined;
                for (const auto& v : seen) joined += (joined.empty() ? "" : ", ") + v;
                p["values"] = joined;
                changed = true;
            }
            Label("Fill with");
            changed |= Choice("##function", StrRef(p, "function"), {"sum", "count", "average", "min", "max", "first"});
            if (StrRef(p, "function") != "count") changed |= column_row("Of column", "values_from");
            Label("One row per");
            ImGui::NewLine();
            changed |= ColumnSet("group", p, "group_by", cols);
        }
    } else if (type == "assign_split") {
        Label("Rows with the same");
        ImGui::NewLine();
        changed |= ColumnSet("key", p, "key", cols);
        changed |= text_row("Train %", "train", "90");
        changed |= text_row("Validation %", "validation", "5");
        changed |= text_row("Test %", "test", "5");
        changed |= text_row("Salt", "salt", "change it to get a different split");
        changed |= text_row("Column", "name", "split");
        DimWrapped("A stable hash of the key: the same rows land in the same split on every run and every machine.");
    } else if (type == "add_checks") {
        changed |= ListEditor("checks", p, "checks", json{{"kind", "row_count"}, {"min", "1"}}, "+ check", [&](json& c) {
            ImGui::SetNextItemWidth(150.0f);
            std::string& kind = StrRef(c, "kind");
            bool ch = Combo("##kind", kind, {"row_count", "allowed_values", "unique", "not_empty"});
            Next(ImGui::CalcTextSize("exactly").x + 90.0f);
            if (kind == "row_count") {
                Dim("exactly");
                ImGui::SameLine();
                ch |= TextField("##exact", StrRef(c, "exact"), "-", 80.0f);
                Next(ImGui::CalcTextSize("at least").x + 90.0f);
                Dim("at least");
                ImGui::SameLine();
                ch |= TextField("##min", StrRef(c, "min"), "-", 80.0f);
                Next(ImGui::CalcTextSize("at most").x + 90.0f);
                Dim("at most");
                ImGui::SameLine();
                ch |= TextField("##max", StrRef(c, "max"), "-", 80.0f);
            } else {
                ImGui::SetNextItemWidth(200.0f);
                ch |= ColumnField("##column", StrRef(c, "column"), cols, known);
                if (kind == "allowed_values") {
                    Next(220.0f);
                    ch |= TextField("##values", StrRef(c, "values"), "train, validation, test", 220.0f);
                }
            }
            return ch;
        });
        DimWrapped("While you edit, a failing check stops the preview with its message. Saving the step turns these checks into Check nodes after it (this operation goes last).");
    }
    return changed;
}

}  // namespace gui
