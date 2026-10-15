// SQL Workbench window (TOFIX146 steps 4a and 4b, mockup v5 board 2).
#include "sql_workbench.h"

#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/file_dialogs.h"
#include "../core/node_executors/row_count_check_operator.h"
#include "../core/sql_transform.h"
#include "editor_fonts.h"
#include "sql_workbench_ui.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <arrow/compute/api.h>
#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iterator>
#include <random>
#include <thread>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqlstep = cyxwiz::sqlstep;
using namespace sqlwb;

constexpr int64_t kResultRows = 500;    // rows shown in the grid
constexpr size_t kHistoryLimit = 50;

struct Snippet {
    const char* name;
    const char* hint;
    const char* text;
};

// Common preparation patterns in DuckDB SQL; inserted at the cursor.
const Snippet kSnippets[] = {
    {"Keep rows that match", "rows whose text matches a pattern",
     "SELECT * FROM input\nWHERE regexp_matches(text, 'pattern')"},
    {"Remove rows that match", "drop rows whose text matches a pattern",
     "SELECT * FROM input\nWHERE NOT regexp_matches(text, 'pattern')"},
    {"Split text into rows", "one row per paragraph, numbered (no lateral join: fast)",
     "SELECT * EXCLUDE (pieces), unnest(pieces) AS piece, unnest(range(1, len(pieces) + 1)) AS piece_index\n"
     "FROM (SELECT *, regexp_split_to_array(text, '\\n[ \\t]*\\n+') AS pieces FROM input)"},
    {"Clean text", "collapse whitespace and trim",
     "trim(regexp_replace(text, '\\s+', ' ', 'g')) AS text"},
    {"Number rows", "a running index in a chosen order",
     "row_number() OVER (ORDER BY id) AS row_index"},
    {"Count by a column", "rows per value, most first",
     "SELECT author, count(*) AS row_count\nFROM input\nGROUP BY author\nORDER BY row_count DESC"},
    {"Remove duplicates", "keep the first row for each text",
     "SELECT DISTINCT ON (text) *\nFROM input\nORDER BY text"},
    {"Add a part", "a new named part after the one before it (inside WITH)",
     "), new_part AS (\n SELECT *\n FROM previous_part\n"},
};

std::string Param(const MLNode& node, const char* key) {
    const auto it = node.parameters.find(key);
    return it == node.parameters.end() ? std::string() : it->second;
}

std::string ClockNow() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char text[16];
    std::strftime(text, sizeof(text), "%H:%M:%S", &local);
    return text;
}

std::string Seconds(double seconds) {
    char text[32];
    std::snprintf(text, sizeof(text), seconds < 10.0 ? "%.2f s" : "%.1f s", seconds);
    return text;
}

std::string Number(double value) {
    if (std::abs(value) < 1e15 && value == std::floor(value)) {
        return sqlstep::FormatCount(static_cast<int64_t>(value));
    }
    char text[48];
    std::snprintf(text, sizeof(text), "%.4g", value);
    return text;
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

}  // namespace

// One query on a worker thread; the UI polls `done`.
struct SqlWorkbenchDialog::Run {
    RunKind kind = RunKind::Step;
    std::string label;
    std::string draft;  // the editor text the run was started from
    cyxwiz::SqlTransform transform;
    std::atomic<bool> done{false};
    cyxwiz::SqlTransformResult result;
    cyxwiz::DatasetProfile profile;                   // of the result, computed on the worker
    cyxwiz::sqlplan::Plan plan;                       // Explain / Profile, read on the worker
    double seconds = 0.0;
    std::thread thread;
};

SqlWorkbenchDialog::SqlWorkbenchDialog(MLNode* node)
    : NodeConfigDialog("SQL Workbench", node), pane_("sql_workbench_" + std::to_string(node ? node->id : -1)) {
    node_id_ = node ? node->id : -1;
    step_name_ = node ? node->name : std::string();
    title_ = "SQL Workbench - " + step_name_ + "###sql_workbench_" + std::to_string(node_id_);
    saved_query_ = node ? Param(*node, "query") : std::string();
    pane_.Editor().SetText(saved_query_);
    guided_.on_changed = [this] { OnGuidedChanged(); };
    if (node) LoadGuided(*node);
}

SqlWorkbenchDialog::~SqlWorkbenchDialog() {
    if (run_) {
        run_->transform.Interrupt();
        if (run_->thread.joinable()) run_->thread.join();
    }
}

MLNode* SqlWorkbenchDialog::CurrentNode() {
    // Look the step up every frame by id: the graph may have changed since
    // the window opened (TOFIX146 defect 6, raw node pointers).
    return node_editor_ ? node_editor_->GetNodeForConfiguration(node_id_) : nullptr;
}

void SqlWorkbenchDialog::Apply() {
    MLNode* node = CurrentNode();
    if (!node) return;
    saved_query_ = pane_.Editor().GetText();
    node->parameters["query"] = saved_query_;
    guided_.MarkSaved();
    if (guided_.Active()) {
        // A guided step keeps its operations next to the query they wrote.
        node->parameters[cyxwiz::guided::kGuidedOpsParameter] = guided_.Json();
        AddMessage("Saved the operations and their query to the step.", false);
    } else {
        // Written in SQL now: no operations stay behind.
        node->parameters.erase(cyxwiz::guided::kGuidedOpsParameter);
        AddMessage("Saved the query to the step.", false);
    }
}

void SqlWorkbenchDialog::Reset() {
    MLNode* node = CurrentNode();
    if (node) saved_query_ = Param(*node, "query");
    pane_.Editor().SetReadOnly(false);
    pane_.Editor().SetText(saved_query_);
    if (node) LoadGuided(*node);
    AddMessage("Back to the step's saved query.", false);
}

void SqlWorkbenchDialog::AddMessage(std::string text, bool error) {
    messages_.push_back({ClockNow(), std::move(text), error});
    if (messages_.size() > 200) messages_.erase(messages_.begin());
}

void SqlWorkbenchDialog::RefreshInputs() {
    if (!node_editor_) return;
    context_ = node_editor_->GetDataStepContext(node_id_);
    MLNode* node = CurrentNode();
    const auto names = node ? sqlstep::InputNames(node->parameters) : std::vector<std::string>{};
    const auto& facts = node_editor_->GetNodeRunFacts();
    inputs_.clear();
    for (const auto& input : context_.inputs) {
        InputTable table;
        table.alias = input.pin < static_cast<int>(names.size()) ? names[static_cast<size_t>(input.pin)]
                                                                 : std::string("input");
        table.source = SpacedType(input.type) + " \"" + input.name + "\"";
        table.node_id = input.id;
        const auto it = facts.find(input.id);
        if (it != facts.end() && !it->second.output_dataset.empty()) {
            if (auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(it->second.output_dataset)) {
                table.table = dataset->GetArrowTable();
            }
        }
        inputs_.push_back(std::move(table));
    }
    std::vector<std::string> columns;
    for (const auto& input : inputs_) {
        if (!input.table) continue;
        for (const auto& field : input.table->schema()->fields()) columns.push_back(field->name());
    }
    pane_.SetCompletionColumns(std::move(columns));
}

void SqlWorkbenchDialog::StartRun(RunKind kind, size_t part_index, bool on_sample) {
    if (run_) return;
    if (node_editor_ && node_editor_->IsDataPipelineRunning()) {
        AddMessage("A pipeline run is in progress; run again when it ends.", true);
        bottom_tab_ = kMessages;
        return;
    }
    std::vector<cyxwiz::SqlTransformInput> inputs;
    for (const auto& input : inputs_) {
        if (!input.table) {
            AddMessage("Input '" + input.alias + "' is not in memory. Use \"Load inputs\" first.", true);
            bottom_tab_ = kMessages;
            return;
        }
        inputs.push_back({input.alias, input.table});
    }
    // The Guided preview's sample: the main input only (look-up tables stay whole).
    if (on_sample && sample_mode_ != 0 && !inputs.empty() && inputs.front().table->num_rows() > 1000) {
        auto& main = inputs.front().table;
        if (sample_mode_ == 1) {
            main = main->Slice(0, 1000);
        } else {
            std::vector<int64_t> rows(static_cast<size_t>(main->num_rows()));
            for (size_t r = 0; r < rows.size(); ++r) rows[r] = static_cast<int64_t>(r);
            std::mt19937_64 random(42);
            std::shuffle(rows.begin(), rows.end(), random);
            rows.resize(1000);
            std::sort(rows.begin(), rows.end());
            arrow::Int64Builder builder;
            std::shared_ptr<arrow::Array> indices;
            if (builder.AppendValues(rows).ok() && builder.Finish(&indices).ok()) {
                auto taken = arrow::compute::Take(main, indices);
                if (taken.ok()) main = taken->table();
            }
        }
    }
    if (inputs.empty()) {
        AddMessage("Nothing is connected to this step's input.", true);
        bottom_tab_ = kMessages;
        return;
    }
    const std::string draft = pane_.Editor().GetText();
    const auto outline = sqlstep::OutlineSqlQuery(draft);
    const auto part_name = [&](size_t i) { return i < outline.parts.size() ? outline.parts[i].name : std::string(); };
    std::string query = draft;
    std::string label = "result";
    cyxwiz::SqlRunMode mode = cyxwiz::SqlRunMode::Run;
    switch (kind) {
        case RunKind::Step:
            break;
        case RunKind::Selection:
            query = pane_.Editor().Doc().SelectedText();
            label = "selection";
            if (query.find_first_not_of(" \t\r\n") == std::string::npos) {
                AddMessage("Select the SQL to run first (one SELECT statement).", true);
                bottom_tab_ = kMessages;
                return;
            }
            break;
        case RunKind::ToPart:
            query = sqlstep::QueryThroughPart(draft, outline, part_index);
            label = part_index + 1 < outline.parts.size() ? "part " + part_name(part_index) : "result";
            break;
        case RunKind::Counts:
            query = sqlstep::PartCountsQuery(draft, outline);
            label = "rows after each part";
            break;
        case RunKind::Removed:
            query = sqlstep::RemovedRowsQuery(draft, outline, part_index);
            label = "rows " + part_name(part_index) + " removed";
            break;
        case RunKind::Check:
            mode = cyxwiz::SqlRunMode::Check;
            label = "check";
            break;
        case RunKind::Explain:
            mode = cyxwiz::SqlRunMode::Explain;
            label = "plan";
            break;
        case RunKind::Profile:
            mode = cyxwiz::SqlRunMode::Profile;
            label = "timed plan";
            break;
    }
    if (query.empty()) {
        AddMessage("The query has no parts to use here (no WITH, or it could not be split).", true);
        bottom_tab_ = kMessages;
        return;
    }
    auto run = std::make_unique<Run>();
    run->kind = kind;
    run->label = label;
    run->draft = draft;
    Run* raw = run.get();
    const bool profile = mode == cyxwiz::SqlRunMode::Run && kind != RunKind::Counts;
    raw->thread = std::thread([raw, query, mode, profile, inputs = std::move(inputs)]() {
        const auto start = std::chrono::steady_clock::now();
        raw->result = raw->transform.Run(query, inputs, mode);
        // The time shown is the query's; profiling the result comes after.
        raw->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (profile && raw->result.ok && raw->result.table) raw->profile = cyxwiz::ProfileArrow(raw->result.table);
        if (raw->result.ok && (mode == cyxwiz::SqlRunMode::Explain || mode == cyxwiz::SqlRunMode::Profile)) {
            raw->plan = cyxwiz::sqlplan::ParsePlanJson(raw->result.plan);
            cyxwiz::sqlplan::ExplainPlan(raw->plan, query, sqlstep::OutlineSqlQuery(query));
        }
        raw->done = true;
    });
    run_ = std::move(run);
    running_kind_ = kind;
    running_since_ = ImGui::GetTime();
    AddMessage("Running " + label + "...", false);
}

void SqlWorkbenchDialog::StopRun() {
    if (run_) run_->transform.Interrupt();
}

void SqlWorkbenchDialog::MarkError(const std::string& error) {
    const int line = sqlstep::ErrorLine(error);
    syntax_error_ = true;
    if (line <= 0 || line > pane_.Editor().Doc().LineCount()) {
        syntax_state_ = "Error (no line given)";
        pane_.Editor().SetSquiggles({});
        return;
    }
    syntax_state_ = "Error on line " + std::to_string(line);
    const int row = line - 1;
    pane_.Editor().SetSquiggles({{{row, 0}, {row, static_cast<int>(pane_.Editor().Doc().Line(row).size())}, true}});
}

void SqlWorkbenchDialog::PollRun() {
    if (!run_ || !run_->done) return;
    run_->thread.join();
    std::unique_ptr<Run> run = std::move(run_);
    const auto& result = run->result;
    // Line numbers in errors refer to the editor text for these kinds (a
    // selection or a part query has other lines).
    const bool lines_match = run->kind == RunKind::Step || run->kind == RunKind::Check ||
                             run->kind == RunKind::Explain || run->kind == RunKind::Profile;
    HistoryEntry entry{ClockNow(), run->label, "", !result.ok, run->draft};
    if (!result.ok) {
        entry.outcome = result.cancelled ? "stopped" : result.error;
        if (result.cancelled) {
            AddMessage("Stopped.", false);
        } else {
            AddMessage(result.error, true);
            if (lines_match) MarkError(result.error);
            bottom_tab_ = kMessages;
        }
    } else {
        if (lines_match) {
            syntax_error_ = false;
            syntax_state_ = "Syntax OK";
            pane_.Editor().SetSquiggles({});
        }
        switch (run->kind) {
            case RunKind::Counts: {
                part_rows_.clear();
                const auto part_column = result.table->GetColumnByName("part");
                const auto rows_column = result.table->GetColumnByName("row_count");
                for (int64_t r = 0; r < result.table->num_rows(); ++r) {
                    auto part = part_column->GetScalar(r);
                    auto rows = rows_column->GetScalar(r);
                    if (!part.ok() || !rows.ok()) continue;
                    part_rows_[(*part)->ToString()] = std::static_pointer_cast<arrow::Int64Scalar>(*rows)->value;
                }
                part_rows_draft_ = run->draft;
                part_rows_seconds_ = run->seconds;
                if (side_tab_ == 0 && mode_ == 1) side_tab_ = 1;
                entry.outcome = std::to_string(part_rows_.size()) + " parts, " + Seconds(run->seconds);
                AddMessage("Counted the rows after each part (" + Seconds(run->seconds) + ").", false);
                break;
            }
            case RunKind::Check:
                entry.outcome = "syntax OK";
                AddMessage("Syntax OK: the query binds against its inputs and its result columns are flat.", false);
                break;
            case RunKind::Explain:
            case RunKind::Profile:
                plan_ = std::move(run->plan);
                plan_json_ = result.plan;
                plan_text_ = result.plan_text;
                plan_draft_ = run->draft;
                plan_when_ = ClockNow();
                plan_picked_ = plan_.slowest >= 0 ? plan_.slowest : (plan_.ops.empty() ? -1 : 0);
                bottom_tab_ = kPlan;
                if (!editor_share_set_by_user_) editor_share_ = std::min(editor_share_, 0.3f);
                if (!plan_.error.empty()) {
                    entry.outcome = plan_.error;
                    AddMessage("The plan could not be read: " + plan_.error, true);
                } else if (plan_.timed) {
                    entry.outcome = cyxwiz::sqlplan::FormatSeconds(plan_.operator_seconds) + " in operators";
                    AddMessage("Timed run: " + std::to_string(plan_.ops.size()) + " operators, " +
                                   cyxwiz::sqlplan::FormatSeconds(plan_.operator_seconds) + " in operators (" +
                                   Seconds(run->seconds) + " in all).",
                               false);
                } else {
                    entry.outcome = std::to_string(plan_.ops.size()) + " operators";
                    AddMessage("Planned the query without running it: " + std::to_string(plan_.ops.size()) +
                                   " operators.",
                               false);
                }
                break;
            default:
                result_ = result.table;
                result_label_ = run->label;
                result_is_step_ = run->kind == RunKind::Step;
                result_seconds_ = run->seconds;
                result_grid_.SetTable(cyxwiz::DataTableFromArrow(*result_, result_label_, kResultRows));
                result_grid_.options.data_bars = false;
                result_grid_.options.rows_per_page = static_cast<int>(kResultRows);  // one page
                result_profile_ = std::move(run->profile);
                // Registered under its own name, so the step's saved output is untouched.
                result_dataset_ = "sqlwb_" + std::to_string(node_id_);
                cyxwiz::DataRegistry::Instance().RegisterArrowTable(result_, result_dataset_);
                if (result_is_step_) EvaluateChecks();
                if (bottom_tab_ == kMessages || bottom_tab_ == kPlan) bottom_tab_ = kResults;
                entry.outcome = sqlstep::FormatCount(result_->num_rows()) + " rows, " + Seconds(run->seconds);
                AddMessage(result_label_ + ": " + sqlstep::FormatCount(result_->num_rows()) + " rows, " +
                               std::to_string(result_->num_columns()) + " columns (" + Seconds(run->seconds) + ")",
                           false);
                break;
        }
    }
    history_.push_back(std::move(entry));
    if (history_.size() > kHistoryLimit) history_.erase(history_.begin());
    // A Guided preview runs the selected operation once the counts are in.
    if (preview_part_after_counts_ && run->kind == RunKind::Counts) {
        preview_part_after_counts_ = false;
        if (result.ok) {
            int part = GuidedPartIndex(guided_.Picked());
            if (part < 0) part = static_cast<int>(pane_.Outline().parts.size()) - 1;
            StartRun(RunKind::ToPart, static_cast<size_t>(std::max(0, part)), true);
        }
    }
}

void SqlWorkbenchDialog::EvaluateChecks() {
    // The step's Row Count Checks, run on the draft's result with the
    // operator the pipeline uses.
    check_outcomes_.clear();
    for (const auto& check : context_.checks) {
        CheckOutcome outcome;
        const auto name = check.parameters.find("check_name");
        outcome.name = name != check.parameters.end() && !name->second.empty() ? name->second : check.name;
        outcome.rule = sqlstep::DescribeRowCountRule(check.parameters);
        cyxwiz::RowCountCheckOperator op;
        std::string error;
        if (!op.Configure(check.parameters, error)) {
            outcome.detail = error;
        } else {
            const auto applied = op.Apply(result_);
            outcome.passed = applied.ok();
            outcome.detail = applied.ok() ? op.OutcomeSummary() : applied.status().message();
        }
        check_outcomes_.push_back(std::move(outcome));
    }
}

void SqlWorkbenchDialog::ExportResult(bool parquet) {
    if (!result_) return;
    const cyxwiz::FileDialogs::FilterList filters = {
        parquet ? cyxwiz::FileDialogs::Filter{"Parquet", "parquet"} : cyxwiz::FileDialogs::Filter{"CSV", "csv"}};
    const std::string name = step_name_ + (parquet ? ".parquet" : ".csv");
    const auto path = cyxwiz::FileDialogs::SaveFile("Export result", filters, nullptr, name.c_str());
    if (!path) return;
    auto& registry = cyxwiz::DataRegistry::Instance();
    const bool ok = parquet ? registry.ExportArrowToParquet(result_dataset_, *path)
                            : registry.ExportArrowToCSV(result_dataset_, *path);
    AddMessage(ok ? "Exported " + sqlstep::FormatCount(result_->num_rows()) + " rows to " + *path
                  : "Export to " + *path + " failed (see the Console).",
               !ok);
}

void SqlWorkbenchDialog::MarkPart(size_t index) {
    if (index >= pane_.Outline().parts.size()) return;
    const auto& part = pane_.Outline().parts[index];
    const int count = std::max(1, pane_.Editor().Doc().LineCount());
    const int first = std::clamp(part.first_line - 1, 0, count - 1);
    const int last = std::clamp(part.last_line - 1, first, count - 1);
    const cyxwiz::editor::Pos end{last, static_cast<int>(pane_.Editor().Doc().Line(last).size())};
    pane_.Editor().SetMarks({{{first, 0}, end, false}});
    cyxwiz::editor::Selection selection;
    selection.anchor = {first, 0};
    selection.head = end;
    pane_.Editor().Doc().SetSelections({selection});
    pane_.Editor().ScrollToCursor();
    marked_part_ = static_cast<int>(index);
}

void SqlWorkbenchDialog::RenderContent() {
    MLNode* node = CurrentNode();
    if (!node) {
        Dim("This step is no longer in the graph. Close the SQL Workbench.");
        return;
    }
    PollRun();

    // Inputs: on open, every 2 s, and when a pipeline run ends.
    const bool pipeline_running = node_editor_ && node_editor_->IsDataPipelineRunning();
    const double now = ImGui::GetTime();
    if (inputs_time_ < 0.0 || now - inputs_time_ > 2.0 || (pipeline_was_running_ && !pipeline_running)) {
        RefreshInputs();
        inputs_time_ = now;
    }
    pipeline_was_running_ = pipeline_running;

    pane_.Refresh();
    // The step's query changed elsewhere (Properties, undo) while the draft
    // still matches what was saved: follow it.
    const std::string step_query = Param(*node, "query");
    if (step_query != saved_query_ && pane_.Text() == saved_query_) {
        saved_query_ = step_query;
        pane_.Editor().SetReadOnly(false);
        pane_.Editor().SetText(saved_query_);
        pane_.Refresh();
        LoadGuided(*node);
    }

    // Ctrl+Enter runs the step, Ctrl+Shift+Enter the selection.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::GetIO().KeyCtrl &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
        StartRun(ImGui::GetIO().KeyShift ? RunKind::Selection : RunKind::Step);
    }

    const ui::Tokens& t = ui::CurrentTokens();
    RenderHeader(*node);
    if (mode_ == 0) RenderGuidedToolbar();
    else RenderToolbar();
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));

    const float status_h = ImGui::GetFrameHeight();
    const float body_h = std::max(200.0f, ImGui::GetContentRegionAvail().y - status_h - t.space_sm);
    if (mode_ == 0) {
        RenderGuided(body_h);
        RenderStatusBar();
        RenderCompare();
        return;
    }
    if (guided_.Active()) {
        // SQL mode on a guided step: the query is read-only.
        Coloured(ICON_FA_CIRCLE_INFO " Written from the guided operations (read-only).", t.info);
        ImGui::SameLine();
        if (!confirm_edit_as_sql_) {
            if (ui::LinkButton("Edit as SQL")) confirm_edit_as_sql_ = true;
        } else {
            Coloured("Editing turns this step into an SQL step; the operations go when you save.", t.warning);
            ImGui::SameLine();
            if (ui::LinkButton("Edit as SQL")) {
                confirm_edit_as_sql_ = false;
                guided_.Stop();
                pane_.Editor().SetReadOnly(false);
                AddMessage("The query is yours to edit; Save to step keeps it as an SQL step (Reset to saved undoes this).", false);
            }
            ImGui::SameLine();
            if (ui::LinkButton("Cancel")) confirm_edit_as_sql_ = false;
        }
    }
    const float content_h = body_h - (guided_.Active() ? ImGui::GetFrameHeightWithSpacing() : 0.0f);
    RenderSidePanel(content_h);
    ImGui::SameLine();
    ImGui::BeginGroup();
    // Editor, a splitter (drag to move), then the bottom panel.
    const float splitter_h = 6.0f;
    const float spacing_y = ImGui::GetStyle().ItemSpacing.y;
    const float usable = content_h - splitter_h - spacing_y * 2.0f;
    const float editor_h = std::floor(usable * editor_share_);
    pane_.Render(editor_h);
    {
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        ImGui::InvisibleButton("##wb_splitter", ImVec2(width, splitter_h));
        const bool active = ImGui::IsItemActive();
        if (ImGui::IsItemHovered() || active) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            const float y = at.y + splitter_h * 0.5f;
            ImGui::GetWindowDrawList()->AddLine(ImVec2(at.x, y), ImVec2(at.x + width, y),
                                                ui::ToU32(ui::CurrentTokens().border_soft), 2.0f);
        }
        if (active && ImGui::GetIO().MouseDelta.y != 0.0f && usable > 0.0f) {
            editor_share_ = std::clamp(editor_share_ + ImGui::GetIO().MouseDelta.y / usable, 0.12f, 0.88f);
            editor_share_set_by_user_ = true;
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            editor_share_ = editor_share_ > 0.3f ? 0.25f : 0.55f;
            editor_share_set_by_user_ = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Drag to resize; double-click to give the panel below more room");
    }
    RenderBottomPanel(usable - editor_h);
    ImGui::EndGroup();
    RenderStatusBar();
    RenderCompare();
}

void SqlWorkbenchDialog::RenderHeader(MLNode& node) {
    const ui::Tokens& t = ui::CurrentTokens();
    static const char* const kModes[] = {"Guided", "SQL"};
    ui::SegmentedControl("##wb_mode", kModes, 2, &mode_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Guided: build the step from operations. SQL: the query itself.");
    ImGui::SameLine();
    if (!context_.recipe_name.empty()) {
        Dim("recipe " + context_.recipe_name + "  " ICON_FA_ANGLE_RIGHT "  step");
    } else {
        Dim("step");
    }
    ImGui::SameLine();
    Coloured(node.name, t.text_bright);
    if (pane_.Text() != saved_query_) {
        ImGui::SameLine();
        Coloured(ICON_FA_CIRCLE " Edited, not saved to the step", t.warning);
    }
}

void SqlWorkbenchDialog::RenderToolbar() {
    const bool pipeline = node_editor_ && node_editor_->IsDataPipelineRunning();
    const bool running = run_ != nullptr || pipeline;
    const char* busy = run_ ? "A query is running" : pipeline ? "A pipeline run is in progress" : nullptr;
    const bool parts = pane_.Outline().structured && pane_.Outline().parts.size() > 1;
    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running, busy)) StartRun(RunKind::Step);
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("Run the whole query on the step's inputs (Ctrl+Enter)");
    ImGui::SameLine();
    if (ui::SecondaryButton("Run selection", !running, busy, ui::ButtonSize::Regular)) StartRun(RunKind::Selection);
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("Run the selected SQL (Ctrl+Shift+Enter)");
    ImGui::SameLine();
    if (ui::SecondaryButton("Run to part " ICON_FA_CHEVRON_DOWN, !running && parts,
                            running ? busy : "The query has no parts to stop at", ui::ButtonSize::Regular)) {
        ImGui::OpenPopup("##run_to_part");
    }
    if (ImGui::BeginPopup("##run_to_part")) {
        for (size_t i = 0; i < pane_.Outline().parts.size(); ++i) {
            const auto& part = pane_.Outline().parts[i];
            const std::string label = part.name + "  (lines " + std::to_string(part.first_line) + "-" +
                                      std::to_string(part.last_line) + ")";
            if (ImGui::MenuItem(label.c_str())) {
                MarkPart(i);
                StartRun(RunKind::ToPart, i);
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Count rows per part", !running && parts,
                            running ? busy : "The query has no parts to count", ui::ButtonSize::Regular)) {
        StartRun(RunKind::Counts);
    }
    ImGui::SameLine();
    if (ui::DangerButton(ICON_FA_STOP " Stop", run_ != nullptr, "No query is running", ui::ButtonSize::Regular)) {
        StopRun();
    }
    ImGui::SameLine();
    if (ui::GhostButton("Check", !running, busy, false, ui::ButtonSize::Regular)) StartRun(RunKind::Check);
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("Check the query against its inputs without running it");
    ImGui::SameLine();
    if (ui::GhostButton("Format", true, nullptr, false, ui::ButtonSize::Regular)) FormatDraft();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keywords in upper case, trailing spaces removed (Ctrl+Z undoes it)");
    ImGui::SameLine();
    if (ui::GhostButton("Compare with saved", pane_.Text() != saved_query_, "The draft is the saved query", compare_open_,
                        ui::ButtonSize::Regular)) {
        compare_open_ = !compare_open_;
    }
    ImGui::SameLine();
    if (ui::GhostButton("Explain plan", !running, busy, false, ui::ButtonSize::Regular)) {
        bottom_tab_ = kPlan;
        StartRun(RunKind::Explain);
    }
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("DuckDB's plan for the query, without running it");

    const bool edited = pane_.Text() != saved_query_;
    const float reset_w = ui::ButtonWidth("Reset to saved", ui::ButtonSize::Regular);
    const float save_w = ui::ButtonWidth("Save to step", ui::ButtonSize::Regular);
    AlignRight(reset_w + save_w + ImGui::GetStyle().ItemSpacing.x);
    if (ui::SecondaryButton("Reset to saved", edited, "The draft is the saved query", ui::ButtonSize::Regular)) Reset();
    ImGui::SameLine();
    if (ui::SecondaryButton("Save to step", edited, "Nothing to save", ui::ButtonSize::Regular)) Apply();
    if (ImGui::IsItemHovered() && edited) ImGui::SetTooltip("Write the draft into the step's query");
}

void SqlWorkbenchDialog::RenderSidePanel(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = std::clamp(ImGui::GetContentRegionAvail().x * 0.2f, 270.0f, 360.0f);
    if (BeginPanel("##wb_side", ImVec2(width, height), t.bg_panel)) {
        static const char* const kTabs[] = {"Inputs", "Parts", "History", "Snippets"};
        ui::SegmentedControl("##wb_side_tab", kTabs, 4, &side_tab_);
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        // Selection and hover in the theme's own tones (not the header colour).
        ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
        if (ImGui::BeginChild("##wb_side_body", ImGui::GetContentRegionAvail())) {
            switch (side_tab_) {
                case 0: RenderInputsTab(); break;
                case 1: RenderPartsTab(); break;
                case 2: RenderHistoryTab(); break;
                default: RenderSnippetsTab(); break;
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(3);
    }
    EndPanel();
}

void SqlWorkbenchDialog::RenderInputsTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (inputs_.empty()) {
        DimWrapped(context_.error.empty() ? "Nothing is connected to this step's input."
                                          : "The graph could not be read: " + context_.error);
    }
    bool missing = false;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        const auto& input = inputs_[i];
        ImGui::PushID(static_cast<int>(i));
        Coloured(input.alias, t.accent_text);
        ImGui::SameLine();
        Dim(input.table ? sqlstep::FormatCount(input.table->num_rows()) + " rows" : "not in memory");
        DimWrapped("from " + input.source);
        if (input.table) {
            const auto& facts = node_editor_->GetNodeRunFacts();
            const auto found = facts.find(input.node_id);
            if (open_table_ && found != facts.end() && ui::LinkButton("Open input in Table Viewer")) {
                open_table_(found->second.output_dataset, input.source);
            }
            for (const auto& field : input.table->schema()->fields()) {
                if (ImGui::Selectable(field->name().c_str(), false, ImGuiSelectableFlags_AllowOverlap)) {
                    pane_.Editor().Doc().Type(field->name());
                    pane_.Editor().RequestFocus();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Click to insert at the cursor");
                const std::string type = field->type()->ToString();
                AlignRight(ImGui::CalcTextSize(type.c_str()).x);
                Dim(type);
            }
        } else {
            missing = true;
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_sm));
        ImGui::PopID();
    }
    if (missing) {
        DimWrapped("Run the steps before this one to load the inputs.");
        const bool can = node_editor_ && !node_editor_->IsDataPipelineRunning() && !run_;
        if (ui::SecondaryButton("Load inputs", can, run_ ? "A query is running" : "A pipeline run is in progress")) {
            node_editor_->ExecuteDataPipelineThrough(node_id_, false, false);
        }
    }
}

void SqlWorkbenchDialog::RenderPartsTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool counted = !part_rows_.empty() && part_rows_draft_ == pane_.Text();
    if (ui::LinkButton("Count rows per part", !run_ && pane_.Outline().structured && pane_.Outline().parts.size() > 1)) {
        StartRun(RunKind::Counts);
    }
    if (!part_rows_.empty()) {
        ImGui::SameLine();
        Dim(counted ? Seconds(part_rows_seconds_) : "(counted on an older draft)");
    }
    DimWrapped("Rows after each part. Click a part to run the query up to it.");
    int64_t previous = -1;
    for (size_t i = 0; i < pane_.Outline().parts.size(); ++i) {
        const auto& part = pane_.Outline().parts[i];
        ImGui::PushID(static_cast<int>(i));
        const auto found = part_rows_.find(part.name);
        const bool known = found != part_rows_.end();
        std::string rows = known ? sqlstep::FormatCount(found->second) : "";
        if (known && previous >= 0 && found->second != previous) {
            const int64_t change = found->second - previous;
            rows += change > 0 ? "  +" + sqlstep::FormatCount(change) : "  -" + sqlstep::FormatCount(-change);
        }
        if (ImGui::Selectable("##part", marked_part_ == static_cast<int>(i), ImGuiSelectableFlags_AllowOverlap)) {
            MarkPart(i);
            StartRun(RunKind::ToPart, i);
        }
        ImGui::SameLine(0.0f, 0.0f);
        ImGui::SetCursorPosX(ImGui::GetStyle().WindowPadding.x);
        Coloured(part.name, t.accent_text);
        if (!rows.empty()) {
            AlignRight(ImGui::CalcTextSize(rows.c_str()).x);
            Coloured(rows, counted ? t.text : t.text_dim);
        }
        if (known) previous = found->second;
        ImGui::PopID();
    }
}

void SqlWorkbenchDialog::RenderHistoryTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (history_.empty()) {
        DimWrapped("Runs in this window appear here, newest first. Click one to bring its query back into the "
                   "editor (Ctrl+Z undoes it).");
        return;
    }
    for (size_t n = history_.size(); n-- > 0;) {
        const auto& entry = history_[n];
        ImGui::PushID(static_cast<int>(n));
        const ImVec2 start = ImGui::GetCursorPos();
        if (ImGui::Selectable("##history", false, ImGuiSelectableFlags_AllowOverlap,
                              ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 2.0f))) {
            auto& doc = pane_.Editor().Doc();
            const int last = std::max(0, doc.LineCount() - 1);
            doc.Replace({0, 0}, {last, static_cast<int>(doc.Line(last).size())}, entry.draft);
            AddMessage("Brought back the query of the " + entry.time + " " + entry.label + " run.", false);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bring this query back into the editor");
        const ImVec2 after = ImGui::GetCursorPos();
        ImGui::SetCursorPos(start);
        Dim(entry.time);
        ImGui::SameLine();
        Coloured(entry.label, t.text);
        std::string outcome = entry.outcome;
        if (outcome.size() > 60) outcome = outcome.substr(0, 60) + "...";
        Coloured(outcome, entry.error ? t.caution : t.text_dim);
        ImGui::SetCursorPos(after);
        ImGui::PopID();
    }
}

void SqlWorkbenchDialog::RenderSnippetsTab() {
    const ui::Tokens& t = ui::CurrentTokens();
    DimWrapped("Click to insert at the cursor. Change the table and column names to yours.");
    for (size_t i = 0; i < std::size(kSnippets); ++i) {
        const auto& snippet = kSnippets[i];
        ImGui::PushID(static_cast<int>(i));
        const ImVec2 start = ImGui::GetCursorPos();
        if (ImGui::Selectable("##snippet", false, ImGuiSelectableFlags_AllowOverlap,
                              ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 2.0f))) {
            pane_.Editor().Doc().Paste(snippet.text);
            pane_.Editor().RequestFocus();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", snippet.text);
        const ImVec2 after = ImGui::GetCursorPos();
        ImGui::SetCursorPos(start);
        Coloured(snippet.name, t.accent_text);
        Dim(snippet.hint);
        ImGui::SetCursorPos(after);
        ImGui::PopID();
    }
}

void SqlWorkbenchDialog::RenderBottomPanel(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##wb_bottom", ImVec2(ImGui::GetContentRegionAvail().x, height), t.bg_panel)) {
        int passed = 0;
        for (const auto& check : check_outcomes_) passed += check.passed ? 1 : 0;
        const std::string checks = check_outcomes_.empty()
                                       ? std::string("Checks")
                                       : "Checks " + std::to_string(passed) + "/" + std::to_string(check_outcomes_.size());
        const std::string messages = "Messages (" + std::to_string(messages_.size()) + ")";
        const char* tabs[kBottomTabs] = {"Results", "Column profile", "Rows removed", checks.c_str(), "Plan",
                                         messages.c_str()};
        ui::SegmentedControl("##wb_bottom_tab", tabs, kBottomTabs, &bottom_tab_);
        if (run_) {
            ImGui::SameLine();
            Coloured(ICON_FA_SPINNER " Running " + run_->label + "...", t.running);
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        if (ImGui::BeginChild("##wb_bottom_body", ImGui::GetContentRegionAvail())) {
            switch (bottom_tab_) {
                case kResults: RenderResults(); break;
                case kProfile: RenderProfile(); break;
                case kRemoved: RenderRemoved(); break;
                case kChecks: RenderChecks(); break;
                case kPlan: RenderPlan(); break;
                default: RenderMessages(); break;
            }
        }
        ImGui::EndChild();
    }
    EndPanel();
}

void SqlWorkbenchDialog::RenderResults() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!result_) {
        Dim("Run the step (Ctrl+Enter) or run to a part to see its rows here.");
        return;
    }
    Coloured(result_label_, t.text_bright);
    ImGui::SameLine();
    std::string facts = sqlstep::FormatCount(result_->num_rows()) + " rows, " +
                        std::to_string(result_->num_columns()) + " columns, " + Seconds(result_seconds_);
    if (result_->num_rows() > kResultRows) facts += "  (first " + sqlstep::FormatCount(kResultRows) + " shown)";
    Dim(facts);
    const float open_w = ImGui::CalcTextSize("Open in Table Viewer").x;
    const float copy_w = ui::ButtonWidth("Copy shown rows", ui::ButtonSize::Small);
    const float export_w = ui::ButtonWidth("Export " ICON_FA_CHEVRON_DOWN, ui::ButtonSize::Small);
    AlignRight(open_w + copy_w + export_w + ImGui::GetStyle().ItemSpacing.x * 2.0f);
    if (ui::SecondaryButton("Copy shown rows")) {
        // In the grid's order (sorted, filtered), tab-separated.
        const auto& headers = result_grid_.table->GetHeaders();
        std::string text;
        for (size_t c = 0; c < headers.size(); ++c) text += (c ? "\t" : "") + headers[c];
        text += "\n";
        const auto& rows = result_grid_.DisplayIndices();
        for (size_t r : rows) {
            for (size_t c = 0; c < headers.size(); ++c) text += (c ? "\t" : "") + result_grid_.CellText(r, c);
            text += "\n";
        }
        ImGui::SetClipboardText(text.c_str());
        AddMessage("Copied " + std::to_string(rows.size()) + " rows (tab-separated).", false);
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Export " ICON_FA_CHEVRON_DOWN)) ImGui::OpenPopup("##wb_export");
    if (ImGui::BeginPopup("##wb_export")) {
        if (ImGui::MenuItem("Parquet...")) ExportResult(true);
        if (ImGui::MenuItem("CSV...")) ExportResult(false);
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (open_table_ && ui::LinkButton("Open in Table Viewer")) {
        open_table_(result_dataset_, step_name_ + " " + result_label_);
    }
    // The Table Viewer's grid: sort, select, copy; Open in Table Viewer for more.
    result_grid_.Render("##wb_result_grid");
}

void SqlWorkbenchDialog::RenderProfile() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (result_profile_.columns.empty()) {
        Dim(result_profile_.ok() ? "Run the step or a part to profile its columns."
                                 : "The result could not be profiled: " + result_profile_.error);
        return;
    }
    Dim("Columns of " + result_label_ + ", over all " + sqlstep::FormatCount(result_ ? result_->num_rows() : 0) +
        " rows");
    if (ImGui::BeginTable("##wb_profile", 5,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp,
                          ImGui::GetContentRegionAvail())) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("column", ImGuiTableColumnFlags_WidthStretch, 0.9f);
        ImGui::TableSetupColumn("distinct", ImGuiTableColumnFlags_WidthStretch, 0.5f);
        ImGui::TableSetupColumn("missing / empty", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableSetupColumn("values", ImGuiTableColumnFlags_WidthStretch, 1.8f);
        ImGui::TableSetupColumn("repeat an earlier row", ImGuiTableColumnFlags_WidthStretch, 0.6f);
        ImGui::TableHeadersRow();
        for (const auto& column : result_profile_.columns) {
            const auto count = [](size_t n) { return sqlstep::FormatCount(static_cast<int64_t>(n)); };
            const size_t nulls = column.facts.rows - std::min(column.facts.rows, column.facts.non_null);
            // Rows whose value appeared in an earlier row.
            const size_t repeated = column.facts.non_null - std::min(column.facts.non_null, column.facts.distinct);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            Coloured(column.facts.name, t.text);
            Dim(column.sql_type);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(count(column.facts.distinct).c_str());
            ImGui::TableNextColumn();
            Coloured(count(nulls) + " / " + count(column.empty_text), nulls + column.empty_text > 0 ? t.warning : t.text_dim);
            ImGui::TableNextColumn();
            if (column.numeric) {
                ImGui::TextUnformatted(("min " + Number(column.min) + "   median " + Number(column.median) +
                                        "   mean " + Number(column.mean) + "   max " + Number(column.max))
                                           .c_str());
            } else if (repeated == 0 && column.facts.non_null > 0) {
                Dim("every value is different");
            } else {
                // The five most common values.
                std::string top;
                for (size_t i = 0; i < std::min<size_t>(column.top.size(), 5); ++i) {
                    const auto& [value, times] = column.top[i];
                    std::string shown = value.size() > 28 ? value.substr(0, 28) + "..." : value;
                    for (char& ch : shown) {
                        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
                    }
                    top += (top.empty() ? "" : "   ") + shown + " (" + count(times) + ")";
                }
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(top.c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::TableNextColumn();
            Coloured(count(repeated), repeated > 0 ? t.text : t.text_dim);
        }
        ImGui::EndTable();
    }
}

void SqlWorkbenchDialog::RenderRemoved() {
    const ui::Tokens& t = ui::CurrentTokens();
    const bool counted = !part_rows_.empty() && part_rows_draft_ == pane_.Text();
    if (!counted) {
        DimWrapped("Count the rows after each part first; then each part that removed rows is listed here.");
        if (ui::SecondaryButton("Count rows per part", !run_ && pane_.Outline().structured && pane_.Outline().parts.size() > 1,
                                "The query has no parts to count")) {
            StartRun(RunKind::Counts);
        }
        return;
    }
    DimWrapped("Rows a part removed, compared with the part before it. \"Show\" lists them in Results (both parts "
               "need the same columns).");
    bool any = false;
    for (size_t i = 1; i < pane_.Outline().parts.size(); ++i) {
        const auto before = part_rows_.find(pane_.Outline().parts[i - 1].name);
        const auto after = part_rows_.find(pane_.Outline().parts[i].name);
        if (before == part_rows_.end() || after == part_rows_.end() || after->second >= before->second) continue;
        any = true;
        ImGui::PushID(static_cast<int>(i));
        const int64_t removed = before->second - after->second;
        Coloured(pane_.Outline().parts[i].name, t.accent_text);
        ImGui::SameLine();
        ImGui::TextUnformatted(
            ("removed " + sqlstep::FormatCount(removed) + " of " + sqlstep::FormatCount(before->second)).c_str());
        if (!pane_.Outline().parts[i].description.empty()) {
            ImGui::SameLine();
            Dim(pane_.Outline().parts[i].description);
        }
        ImGui::SameLine();
        const std::string show = "Show the " + sqlstep::FormatCount(removed);
        if (ui::LinkButton(show.c_str(), !run_)) {
            bottom_tab_ = kResults;
            StartRun(RunKind::Removed, i);
        }
        ImGui::PopID();
    }
    if (!any) Dim("No part removed rows.");
}

void SqlWorkbenchDialog::RenderChecks() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (context_.checks.empty()) {
        DimWrapped("This step has no Row Count Checks after it. Add one in the graph to stop a run when a count is "
                   "wrong.");
        return;
    }
    if (check_outcomes_.empty()) {
        DimWrapped("Run the step (Ctrl+Enter): its checks then run on the draft's result, before you save it.");
    } else {
        Dim("The step's checks on the draft's result:");
    }
    for (size_t i = 0; i < context_.checks.size(); ++i) {
        const auto& check = context_.checks[i];
        const CheckOutcome* outcome = i < check_outcomes_.size() ? &check_outcomes_[i] : nullptr;
        const auto status = !outcome ? ui::Status::NotVerifiedYet
                                     : outcome->passed ? ui::Status::Verified : ui::Status::Failed;
        const auto style = ui::StatusStyleFor(status);
        Coloured(style.icon, style.colour);
        ImGui::SameLine();
        const auto name = check.parameters.find("check_name");
        ImGui::TextUnformatted(name != check.parameters.end() && !name->second.empty() ? name->second.c_str()
                                                                                       : check.name.c_str());
        ImGui::SameLine();
        Dim(sqlstep::DescribeRowCountRule(check.parameters));
        if (outcome && !outcome->detail.empty()) {
            ImGui::PushTextWrapPos(0.0f);
            Coloured("    " + outcome->detail, outcome->passed ? t.text_dim : t.caution);
            ImGui::PopTextWrapPos();
        }
    }
}

void SqlWorkbenchDialog::RenderMessages() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (messages_.empty()) {
        Dim("Runs, saves and errors appear here.");
        return;
    }
    for (const auto& message : messages_) {
        Dim(message.time);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, message.error ? t.caution : t.text);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(message.text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f) ImGui::SetScrollHereY(1.0f);
}

void SqlWorkbenchDialog::RenderStatusBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    Dim("DuckDB, read-only, no file access    Results must be flat columns    Ctrl+Enter run    Ctrl+Shift+Enter "
        "run selection");
    const auto head = pane_.Editor().Doc().Primary().head;
    std::string right = "Ln " + std::to_string(head.line + 1) + ", Col " + std::to_string(head.col + 1) + "    " +
                        std::to_string(pane_.Outline().line_count) + " lines";
    if (!pane_.Outline().structured) right += "    not split into parts";
    if (!syntax_state_.empty()) right += "    " + syntax_state_;
    AlignRight(ImGui::CalcTextSize(right.c_str()).x);
    Coloured(right, syntax_error_ ? t.caution : (pane_.Outline().structured ? t.text_dim : t.warning));
}

}  // namespace gui
