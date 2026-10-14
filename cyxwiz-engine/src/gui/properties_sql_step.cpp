// SQL step Properties (TOFIX146, mockup v5 board 1): what the step does,
// its last run, the query with highlighting, the query part by part, its
// inputs, a preview of its output, the checks after it, and Details with
// every fact the generic sections show.
#include "properties_sql_step.h"

#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/guided_sql.h"
#include "../core/node_metadata_registry.h"
#include "../core/sql_step_presentation.h"
#include "../data/arrow_data_table.h"
#include "editor_fonts.h"
#include "icons.h"
#include "properties_contract.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <optional>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqlstep = cyxwiz::sqlstep;

constexpr int kPreviewRows = 5;

std::string Param(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

// "DataInput" -> "Data Input".
std::string SpacedType(const std::string& type) {
    std::string out;
    for (size_t i = 0; i < type.size(); ++i) {
        if (i > 0 && std::isupper(static_cast<unsigned char>(type[i])) &&
            std::islower(static_cast<unsigned char>(type[i - 1]))) {
            out += ' ';
        }
        out += type[i];
    }
    return out;
}

std::shared_ptr<arrow::Table> RegisteredTable(const std::string& dataset) {
    if (dataset.empty()) return nullptr;
    auto handle = cyxwiz::DataRegistry::Instance().GetArrowDataset(dataset);
    return handle ? handle->GetArrowTable() : nullptr;
}

const cyxwiz::PipelineNodeRunFacts* FactsFor(const NodeEditor* editor, int node_id) {
    if (!editor) return nullptr;
    const auto& facts = editor->GetNodeRunFacts();
    const auto it = facts.find(node_id);
    return it == facts.end() ? nullptr : &it->second;
}

// The count a Row Count Check found, from its outcome "passed: <n> ...".
std::string CheckCount(const std::string& outcome) {
    const std::string prefix = "passed: ";
    if (outcome.rfind(prefix, 0) != 0) return "";
    size_t end = prefix.size();
    while (end < outcome.size() && std::isdigit(static_cast<unsigned char>(outcome[end]))) ++end;
    if (end == prefix.size()) return "";
    try {
        return sqlstep::FormatCount(std::stoll(outcome.substr(prefix.size(), end - prefix.size())));
    } catch (...) {
        return "";
    }
}

void SectionTitle(const char* title, const char* hint = nullptr) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    if (hint) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
        ImGui::TextUnformatted(hint);
        ImGui::PopStyleColor();
    }
}

void DimText(const std::string& text, bool wrap = true) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    if (wrap) ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    if (wrap) ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void WrappedText(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

// A tinted card one step from the window (no outline).
bool BeginCard(const char* id) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, t.rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.card_padding);
    const bool open = ImGui::BeginChild(id, ImVec2(0.0f, 0.0f),
                                        ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding);
    return open;
}

void EndCard() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
}

// Puts the next item at the right edge of this line, or on the next line
// when it would overlap what is already there.
void AlignRight(float width) {
    ImGui::SameLine();
    const float x = ImGui::GetContentRegionMax().x - width;
    if (x < ImGui::GetCursorPosX()) ImGui::NewLine();
    else ImGui::SetCursorPosX(x);
}

// Value over label, for the last-run figures.
void Figure(const std::string& value, const std::string& label, const ImVec4& colour) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(value.c_str());
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(label.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}
}  // namespace

void Properties::RenderSqlStep(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string query = Param(node, "query");
    // Follow the selection: a new step loads its query and starts fresh.
    if (!sql_view_ || sql_view_->node_id != node.id) {
        sql_view_ = std::make_unique<SqlStepView>();
        SqlStepView& view = *sql_view_;
        view.node_id = node.id;
        view.editor.SetLanguage(cyxwiz::editor::Language::Sql);
        view.editor.SetShowMinimap(false);
        view.editor.SetScrollPastEnd(false);
        view.editor.SetTabSize(4);
        view.editor.SetText(query);
        view.loaded_query = query;
    }
    SqlStepView& view = *sql_view_;
    if (query != view.loaded_query) {
        // Changed elsewhere (Details, the Workbench, undo): show it.
        view.editor.SetText(query);
        view.loaded_query = query;
    }

    // Neighbours: refresh every 2 s and when a run ends.
    const bool running = node_editor_ && node_editor_->IsDataPipelineRunning();
    const double now = ImGui::GetTime();
    if (node_editor_ && (view.context_time < 0.0 || now - view.context_time > 2.0 || (view.was_running && !running))) {
        view.context = node_editor_->GetDataStepContext(node.id);
        view.context_time = now;
    }
    view.was_running = running;

    const auto* facts = FactsFor(node_editor_, node.id);
    const sqlstep::SqlOutline outline = sqlstep::OutlineSqlQuery(query);

    // Checks after this step and their state.
    int checks_passed = 0;
    int checks_failed = 0;
    for (const auto& check : view.context.checks) {
        const auto* check_facts = FactsFor(node_editor_, check.id);
        if (check_facts && !check_facts->error.empty()) ++checks_failed;
        else if (check_facts && !check_facts->outcome.empty()) ++checks_passed;
    }

    // ---- Header: name, where it sits, last-run state.
    {
        const char* state_text = "Not run yet";
        ImVec4 state_colour = t.pending;
        if (running) {
            state_text = "Running...";
            state_colour = t.running;
        } else if (facts && !facts->error.empty()) {
            state_text = "Last run failed";
            state_colour = t.caution;
        } else if (checks_failed > 0) {
            state_text = "A check failed";
            state_colour = t.caution;
        } else if (facts && facts->rows >= 0) {
            state_text = "Last run passed";
            state_colour = t.success;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
        ImGui::TextUnformatted(node.name.c_str());
        ImGui::PopStyleColor();
        AlignRight(ui::StatusPillWidth(state_text));
        ui::StatusPill("##sql_state", state_text, state_colour);

        std::string where = "SQL step";
        if (!view.context.recipe_name.empty()) {
            where += " in recipe " + view.context.recipe_name + " (" + std::to_string(view.context.recipe_steps) +
                     (view.context.recipe_steps == 1 ? " step)" : " steps)");
        }
        DimText(where);
        if (!outline.header.empty()) {
            WrappedText(outline.header.front(), t.text);
            if (outline.header.size() > 1) {
                if (ui::LinkButton(view.about_open ? "Hide the query's notes" : "The query's notes")) {
                    view.about_open = !view.about_open;
                }
                if (view.about_open) {
                    std::string notes;
                    for (size_t i = 1; i < outline.header.size(); ++i) notes += outline.header[i] + "\n";
                    DimText(notes);
                }
            }
        }
        if (facts && !facts->error.empty()) WrappedText(facts->error, t.caution);
    }

    // ---- Last run figures.
    if (BeginCard("##sql_last_run")) {
        int64_t rows_in = -1;
        for (const auto& input : view.context.inputs) {
            const auto* input_facts = FactsFor(node_editor_, input.id);
            if (input_facts && input_facts->rows >= 0) rows_in = std::max<int64_t>(rows_in, 0) + input_facts->rows;
        }
        const bool ran = facts && facts->rows >= 0;
        if (ImGui::BeginTable("##figures", 3, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            Figure(rows_in >= 0 ? sqlstep::FormatCount(rows_in) : "-", "rows in", t.text_bright);
            ImGui::TableNextColumn();
            Figure(ran ? sqlstep::FormatCount(facts->rows) : "-", "rows out", t.text_bright);
            ImGui::TableNextColumn();
            Figure(ran ? sqlstep::FormatCount(facts->columns) : "-", "columns", t.text_bright);
            ImGui::TableNextColumn();
            char seconds[32] = "-";
            if (ran) std::snprintf(seconds, sizeof(seconds), "%.1f s", facts->seconds);
            const std::string when = ran && facts->finished_at.size() >= 5
                                         ? "last run at " + facts->finished_at.substr(0, 5)
                                         : std::string("not run in this session");
            Figure(seconds, when, t.text_bright);
            ImGui::TableNextColumn();
            const int checks = static_cast<int>(view.context.checks.size());
            Figure(checks == 0 ? std::string("-")
                               : std::to_string(checks_passed) + " of " + std::to_string(checks),
                   checks == 0 ? "no checks after it" : "checks passed",
                   checks_failed > 0 ? t.caution : t.text_bright);
            ImGui::EndTable();
        }
    }
    EndCard();

    // ---- Guided | SQL, Run step, Preview output.
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    static const char* const kModes[] = {"Guided", "SQL"};
    ui::SegmentedControl("##sql_mode", kModes, 2, &view.mode);
    {
        // One DuckDB run at a time: the SQL Workbench's query counts too.
        const bool query_running = active_dialog_ && active_dialog_->IsOpen() && active_dialog_->IsBusy();
        const char* busy = running ? "A pipeline run is in progress"
                           : query_running ? "The SQL Workbench is running a query" : nullptr;
        const bool can_run = node_editor_ && !running && !query_running;
        const float run_w = ui::ButtonWidth(ICON_FA_PLAY " Run step", ui::ButtonSize::Regular);
        const float preview_w = ui::ButtonWidth("Preview output", ui::ButtonSize::Regular);
        const float both = run_w + preview_w + ImGui::GetStyle().ItemSpacing.x;
        AlignRight(both);
        if (ui::SecondaryButton("Preview output", can_run, busy, ui::ButtonSize::Regular)) {
            node_editor_->ExecuteDataPipelineThrough(node.id, false);
        }
        if (ImGui::IsItemHovered() && can_run) ImGui::SetTooltip("Run this step and the steps before it; skip its checks");
        ImGui::SameLine();
        if (ui::PrimaryButton(ICON_FA_PLAY " Run step", can_run, busy)) {
            node_editor_->ExecuteDataPipelineThrough(node.id, true);
        }
        if (ImGui::IsItemHovered() && can_run) ImGui::SetTooltip("Run this step, the steps before it and its checks");
    }

    // ---- The query (SQL) or how it reads (Guided).
    if (view.mode == 1) {
        SectionTitle("Query");
        ImGui::SameLine();
        DimText(std::to_string(outline.line_count) + " lines  DuckDB  read-only (no file access)", false);
        AlignRight(ImGui::CalcTextSize("Open in SQL Workbench").x);
        if (ui::LinkButton("Open in SQL Workbench")) ConfigureNode(&node);

        ImFont* mono = cyxwiz::gui::GetEditorMonoFont(1.0f);
        if (mono) ImGui::PushFont(mono);
        const float line_h = cyxwiz::CodeEditor::LineHeightFor(ImGui::GetFontSize());
        const int shown = std::clamp(outline.line_count + 1, 6, 18);
        if (view.editor.Render("##sql_query", ImVec2(-FLT_MIN, line_h * static_cast<float>(shown) + 8.0f))) {
            view.loaded_query = view.editor.GetText();
            node.parameters["query"] = view.loaded_query;
            NotifyEdited();
        }
        if (mono) ImGui::PopFont();

        // Input names the query reads, and their columns to insert.
        const auto names = sqlstep::InputNames(node.parameters);
        std::string joined;
        for (const auto& name : names) joined += (joined.empty() ? "" : ", ") + name;
        DimText("Input name in the query: " + (joined.empty() ? std::string("input") : joined), false);
        std::string columns_key;
        for (const auto& input : view.context.inputs) {
            const auto* input_facts = FactsFor(node_editor_, input.id);
            if (input_facts) columns_key += input_facts->output_dataset + "@" + input_facts->finished_at + ";";
        }
        if (columns_key != view.input_columns_key) {
            view.input_columns_key = columns_key;
            view.input_columns.clear();
            for (const auto& input : view.context.inputs) {
                const auto* input_facts = FactsFor(node_editor_, input.id);
                if (const auto table = input_facts ? RegisteredTable(input_facts->output_dataset) : nullptr) {
                    for (const auto& field : table->schema()->fields()) view.input_columns.push_back(field->name());
                }
            }
        }
        const float insert_w = ui::ButtonWidth("Insert column", ui::ButtonSize::Small);
        AlignRight(insert_w);
        if (ui::SecondaryButton("Insert column", !view.input_columns.empty(),
                                "Run the step once to list its input's columns")) {
            ImGui::OpenPopup("##sql_insert_column");
        }
        if (ImGui::BeginPopup("##sql_insert_column")) {
            for (const auto& column : view.input_columns) {
                if (ImGui::MenuItem(column.c_str())) {
                    view.editor.Doc().Type(column);
                    view.loaded_query = view.editor.GetText();
                    node.parameters["query"] = view.loaded_query;
                    NotifyEdited();
                }
            }
            ImGui::EndPopup();
        }
    } else {
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        const std::string guided_json = Param(node, cyxwiz::guided::kGuidedOpsParameter);
        auto guided_step = guided_json.empty() ? std::nullopt : cyxwiz::guided::FromJson(guided_json);
        if (guided_step) {
            // A guided step: its operations, one line each, switchable here.
            const auto generated = cyxwiz::guided::GenerateSql(*guided_step, guided_step->saved_context);
            bool toggled = false;
            for (size_t i = 0; i < guided_step->ops.size(); ++i) {
                auto& op = guided_step->ops[i];
                ImGui::PushID(static_cast<int>(i));
                bool on = op.on;
                if (ImGui::Checkbox("##on", &on)) {
                    op.on = on;
                    toggled = true;
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(on ? "Switch this operation off" : "Switch this operation on");
                ImGui::SameLine();
                WrappedText(std::to_string(i + 1) + ". " + cyxwiz::guided::Summarise(op), op.on ? t.text : t.text_dim);
                if (i < generated.errors.size() && !generated.errors[i].empty() && op.on) {
                    WrappedText(ICON_FA_TRIANGLE_EXCLAMATION " " + generated.errors[i], t.caution);
                }
                ImGui::PopID();
            }
            if (guided_step->ops.empty()) DimText("No operations yet.");
            if (toggled) {
                // Write the query again from the operations, as the editor would.
                const auto again = cyxwiz::guided::GenerateSql(*guided_step, guided_step->saved_context);
                node.parameters[cyxwiz::guided::kGuidedOpsParameter] = cyxwiz::guided::ToJson(*guided_step);
                node.parameters["query"] = again.sql;
                NotifyEdited();
            }
            if (ui::SecondaryButton("Edit in guided editor")) ConfigureNode(&node);
            ImGui::SameLine();
            if (ui::SecondaryButton("Add a guided step after") && node_editor_) {
                node_editor_->RequestGuidedStepAfter(node.id);
            }
        } else if (BeginCard("##sql_guided_note")) {
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_bright);
            ImGui::TextUnformatted("This step was written in SQL");
            ImGui::PopStyleColor();
            DimText("Guided mode shows what each part does in plain words (below). To change the result without "
                    "SQL, add a guided step after this one; to rebuild this step from operations, open the SQL "
                    "Workbench and choose Guided.");
            if (ui::SecondaryButton("Add a guided step after") && node_editor_) {
                node_editor_->RequestGuidedStepAfter(node.id);
            }
            ImGui::SameLine();
            if (ui::SecondaryButton("Open SQL Workbench")) ConfigureNode(&node);
        }
        if (!guided_step) EndCard();
    }

    // ---- Part by part.
    SectionTitle("What it does, part by part", view.mode == 1 ? "click a part to mark it in the query" : nullptr);
    if (!outline.structured) {
        DimText("The query could not be split into parts (an unclosed bracket, string or comment). It runs as one "
                "part.");
    }
    for (int i = 0; i < static_cast<int>(outline.parts.size()); ++i) {
        const auto& part = outline.parts[static_cast<size_t>(i)];
        ImGui::PushID(i);
        const bool picked = view.picked_part == i;
        const std::string lines = part.first_line == part.last_line
                                      ? "line " + std::to_string(part.first_line)
                                      : "lines " + std::to_string(part.first_line) + "-" + std::to_string(part.last_line);
        const ImVec2 start = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
        const float lines_w = ImGui::CalcTextSize(lines.c_str()).x;
        const float name_w = std::max(90.0f, ImGui::CalcTextSize(part.name.c_str()).x + t.space_lg);
        // A narrow panel puts the description under the name instead of
        // squeezing it into a sliver beside it.
        const bool stacked = width - name_w - lines_w - t.space_lg * 2.0f < 180.0f;
        const float text_w = stacked ? width : std::max(40.0f, width - name_w - lines_w - t.space_lg * 2.0f);
        // Its comment, else where it starts from.
        std::string description = part.description;
        if (description.empty()) {
            for (const auto& name : part.reads) description += (description.empty() ? "reads " : ", ") + name;
        }
        if (description.empty()) description = "values written in the query";
        const ImU32 description_colour = ui::ToU32(part.description.empty() ? t.text_dim : t.text);
        const float text_h = ImGui::CalcTextSize(description.c_str(), nullptr, false, text_w).y;
        const float row_h = stacked ? ImGui::GetTextLineHeight() + text_h : std::max(text_h, ImGui::GetTextLineHeight());
        if (ImGui::Selectable("##part", picked, ImGuiSelectableFlags_AllowOverlap, ImVec2(width, row_h))) {
            view.picked_part = picked ? -1 : i;
            if (view.picked_part >= 0 && view.mode == 1) {
                const int count = std::max(1, view.editor.Doc().LineCount());
                const int first = std::clamp(part.first_line - 1, 0, count - 1);
                const int last = std::clamp(part.last_line - 1, first, count - 1);
                const cyxwiz::editor::Pos end{last, static_cast<int>(view.editor.Doc().Line(last).size())};
                view.editor.SetMarks({{{first, 0}, end, false}});
                // Select the part with the caret at its end, so scrolling
                // brings the whole part into view.
                cyxwiz::editor::Selection selection;
                selection.anchor = {first, 0};
                selection.head = end;
                view.editor.Doc().SetSelections({selection});
                view.editor.ScrollToCursor();
            } else {
                view.editor.SetMarks({});
            }
        }
        ImGui::PopStyleColor(3);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddText(start, ui::ToU32(t.accent_text), part.name.c_str());
        const ImVec2 text_at = stacked ? ImVec2(start.x, start.y + ImGui::GetTextLineHeight()) : ImVec2(start.x + name_w, start.y);
        dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), text_at, description_colour, description.c_str(), nullptr, text_w);
        dl->AddText(ImVec2(start.x + width - lines_w, start.y), ui::ToU32(t.text_dim), lines.c_str());
        ImGui::PopID();
    }

    // ---- Inputs.
    SectionTitle(view.context.inputs.size() == 1 ? "Input" : "Inputs");
    {
        const auto names = sqlstep::InputNames(node.parameters);
        if (view.context.inputs.empty()) {
            DimText(view.context.error.empty() ? "Nothing is connected to this step's input yet."
                                               : "The graph could not be read: " + view.context.error);
        }
        for (const auto& input : view.context.inputs) {
            ImGui::PushID(input.id);
            const auto* input_facts = FactsFor(node_editor_, input.id);
            const std::string alias = input.pin < static_cast<int>(names.size()) ? names[static_cast<size_t>(input.pin)]
                                                                                 : std::string("input");
            ImGui::PushStyleColor(ImGuiCol_Text, t.accent_text);
            ImGui::TextUnformatted(alias.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine();
            std::string from = "from " + SpacedType(input.type) + " \"" + input.name + "\"";
            if (input_facts && input_facts->rows >= 0) from += "  " + sqlstep::FormatCount(input_facts->rows) + " rows";
            DimText(from, false);
            const bool open = input_facts && RegisteredTable(input_facts->output_dataset) != nullptr;
            const float link_w = ImGui::CalcTextSize("Open in Table Viewer").x;
            AlignRight(link_w);
            if (ui::LinkButton("Open in Table Viewer", open && open_table_)) {
                open_table_(input_facts->output_dataset, input.name);
            }
            if (!open && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Run the step to load its input");
            }
            ImGui::PopID();
        }
    }

    // ---- Output preview.
    {
        const bool ran = facts && facts->rows >= 0;
        const std::string hint = ran ? "first rows of " + sqlstep::FormatCount(facts->rows) : "run the step to see it";
        SectionTitle("Output preview", hint.c_str());
        const auto table = ran ? RegisteredTable(facts->output_dataset) : nullptr;
        if (table) {
            const float link_w = ImGui::CalcTextSize("Open in Table Viewer").x;
            AlignRight(link_w);
            if (ui::LinkButton("Open in Table Viewer", static_cast<bool>(open_table_))) {
                open_table_(facts->output_dataset, node.name);
            }
            const std::string key = facts->output_dataset + "@" + facts->finished_at;
            if (key != view.preview_key) {
                view.preview_key = key;
                view.preview.SetTable(cyxwiz::DataTableFromArrow(*table, node.name, kPreviewRows));
                view.preview.options.line_numbers = false;
                view.preview.options.data_bars = false;
            }
            // The Table Viewer's grid, as tall as its rows.
            const float rows_h = ImGui::GetTextLineHeightWithSpacing() *
                                 static_cast<float>(view.preview.table->GetRowCount() + 1) +
                                 ImGui::GetStyle().ScrollbarSize + 4.0f;
            view.preview.Render("##sql_preview_grid", ImVec2(0.0f, rows_h));
        } else if (ran) {
            DimText("The output is not in memory any more; run the step again to see it.");
        }
    }

    // ---- Checks after this step.
    SectionTitle("Checks after this step");
    if (view.context.checks.empty()) {
        DimText("None. Add a Row Count Check after this step to stop the run when a count is wrong.");
    }
    for (const auto& check : view.context.checks) {
        ImGui::PushID(check.id);
        const auto* check_facts = FactsFor(node_editor_, check.id);
        ui::Status status = ui::Status::NotVerifiedYet;
        if (check_facts && !check_facts->error.empty()) status = ui::Status::Failed;
        else if (check_facts && !check_facts->outcome.empty()) status = ui::Status::Verified;
        const auto style = ui::StatusStyleFor(status);
        ImGui::PushStyleColor(ImGuiCol_Text, style.colour);
        ImGui::TextUnformatted(style.icon);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        const auto name_it = check.parameters.find("check_name");
        const std::string name = name_it != check.parameters.end() && !name_it->second.empty() ? name_it->second
                                                                                               : check.name;
        ImGui::TextUnformatted(name.c_str());
        ImGui::SameLine();
        DimText(sqlstep::DescribeRowCountRule(check.parameters), false);
        const std::string got = check_facts ? CheckCount(check_facts->outcome) : std::string();
        if (!got.empty()) {
            const float got_w = ImGui::CalcTextSize(got.c_str()).x;
            AlignRight(got_w);
            ImGui::PushStyleColor(ImGuiCol_Text, style.colour);
            ImGui::TextUnformatted(got.c_str());
            ImGui::PopStyleColor();
        }
        if (check_facts && !check_facts->error.empty()) WrappedText(check_facts->error, t.caution);
        ImGui::PopID();
    }
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));
    DimText(std::string(ui::StatusStyleFor(ui::Status::Verified).icon) + " passed   " +
                ui::StatusStyleFor(ui::Status::Failed).icon + " failed   " +
                ui::StatusStyleFor(ui::Status::NotVerifiedYet).icon + " not run yet",
            false);

    // ---- Details: every fact and setting of the generic sections.
    ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    if (ui::LinkButton(view.details_open ? "Hide details" : "Details")) view.details_open = !view.details_open;
    if (view.details_open) RenderSqlStepDetails(node, metadata);
}

void Properties::RenderSqlStepDetails(MLNode& node, const cyxwiz::NodeMetadata* metadata) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto* facts = FactsFor(node_editor_, node.id);
    std::vector<std::pair<std::string, std::string>> rows;
    rows.emplace_back("Node ID", std::to_string(node.id));
    if (metadata) {
        rows.emplace_back("Type", metadata->name + " (" + cyxwiz::GetCategoryDisplayName(metadata->category) + ")");
    }
    const std::string contract = Param(node, "sql_contract_version");
    rows.emplace_back("SQL contract", contract.empty() ? std::string("1 (default): flat result columns")
                                                       : contract + ": flat result columns");
    const auto names = sqlstep::InputNames(node.parameters);
    std::string joined;
    for (const auto& name : names) joined += (joined.empty() ? "" : ", ") + name;
    rows.emplace_back("Input names", joined);
    if (facts) {
        rows.emplace_back("Output dataset", facts->output_dataset);
        if (!facts->outcome.empty()) rows.emplace_back("Outcome", facts->outcome);
    }
    rows.emplace_back("Executor", "Data Studio pipeline (DuckDB, read-only)");
    std::string raw;
    for (const auto& [key, value] : node.parameters) raw += (raw.empty() ? "" : ", ") + key;
    rows.emplace_back("Parameters", raw);
    // Two pairs per line when the panel is wide; one pair per line, with the
    // key column sized to the longest key, when it is docked narrow.
    float key_w = 0.0f;
    for (const auto& row : rows) key_w = std::max(key_w, ImGui::CalcTextSize(row.first.c_str()).x);
    const bool two_pairs = ImGui::GetContentRegionAvail().x >= 520.0f;
    if (ImGui::BeginTable("##sql_details", two_pairs ? 4 : 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("k1", ImGuiTableColumnFlags_WidthFixed, key_w);
        ImGui::TableSetupColumn("v1", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        if (two_pairs) {
            ImGui::TableSetupColumn("k2", ImGuiTableColumnFlags_WidthFixed, key_w);
            ImGui::TableSetupColumn("v2", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        }
        for (const auto& [key, value] : rows) {
            if (value.empty()) continue;
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
            ImGui::TextUnformatted(key.c_str());
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(value.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }

    // The generic sections, unchanged: header, settings, AS COMPILED,
    // fusion, Advanced and executor.
    ImGui::Dummy(ImVec2(0.0f, t.space_sm));
    RenderGenericSections(node);
}

}  // namespace gui
