// SQL Workbench window (TOFIX146 step 4a, mockup v5 board 2).
#include "sql_workbench.h"

#include "../core/arrow_dataset.h"
#include "../core/data_registry.h"
#include "../core/sql_transform.h"
#include "editor_fonts.h"
#include "icons.h"
#include "ui_buttons.h"
#include "ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <thread>

namespace gui {

namespace {
namespace ui = cyxwiz::ui;
namespace sqlstep = cyxwiz::sqlstep;

constexpr int64_t kResultRows = 500;    // rows shown in the grid
constexpr size_t kResultCellChars = 200;

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

void Dim(const std::string& text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_dim);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

void Coloured(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(text.c_str());
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

// A panel one tone off the window, no outline.
bool BeginPanel(const char* id, const ImVec2& size, const ImVec4& background) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, background);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, ui::CurrentTokens().rounding_card);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ui::CurrentTokens().card_padding);
    return ImGui::BeginChild(id, size, ImGuiChildFlags_AlwaysUseWindowPadding);
}

void EndPanel() {
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
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
    double seconds = 0.0;
    std::thread thread;
};

SqlWorkbenchDialog::SqlWorkbenchDialog(MLNode* node)
    : NodeConfigDialog("SQL Workbench", node) {
    node_id_ = node ? node->id : -1;
    step_name_ = node ? node->name : std::string();
    title_ = "SQL Workbench - " + step_name_ + "###sql_workbench_" + std::to_string(node_id_);
    saved_query_ = node ? Param(*node, "query") : std::string();
    editor_.SetLanguage(cyxwiz::editor::Language::Sql);
    editor_.SetTabSize(4);
    editor_.SetText(saved_query_);
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
    saved_query_ = editor_.GetText();
    node->parameters["query"] = saved_query_;
    AddMessage("Saved the query to the step.", false);
}

void SqlWorkbenchDialog::Reset() {
    MLNode* node = CurrentNode();
    if (node) saved_query_ = Param(*node, "query");
    editor_.SetText(saved_query_);
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
}

void SqlWorkbenchDialog::StartRun(RunKind kind, size_t part_index) {
    if (run_) return;
    if (node_editor_ && node_editor_->IsDataPipelineRunning()) {
        AddMessage("A pipeline run is in progress; run again when it ends.", true);
        bottom_tab_ = 1;
        return;
    }
    std::vector<cyxwiz::SqlTransformInput> inputs;
    for (const auto& input : inputs_) {
        if (!input.table) {
            AddMessage("Input '" + input.alias + "' is not in memory. Use \"Load inputs\" first.", true);
            bottom_tab_ = 1;
            return;
        }
        inputs.push_back({input.alias, input.table});
    }
    if (inputs.empty()) {
        AddMessage("Nothing is connected to this step's input.", true);
        bottom_tab_ = 1;
        return;
    }
    const std::string draft = editor_.GetText();
    const auto outline = sqlstep::OutlineSqlQuery(draft);
    std::string query;
    std::string label;
    switch (kind) {
        case RunKind::Step:
            query = draft;
            label = "result";
            break;
        case RunKind::ToPart:
            query = sqlstep::QueryThroughPart(draft, outline, part_index);
            label = part_index < outline.parts.size() ? "part " + outline.parts[part_index].name : "result";
            break;
        case RunKind::Counts:
            query = sqlstep::PartCountsQuery(draft, outline);
            label = "rows after each part";
            if (query.empty()) {
                AddMessage("The query has no parts to count (no WITH, or it could not be split).", true);
                bottom_tab_ = 1;
                return;
            }
            break;
    }
    auto run = std::make_unique<Run>();
    run->kind = kind;
    run->label = label;
    run->draft = draft;
    Run* raw = run.get();
    raw->thread = std::thread([raw, query, inputs = std::move(inputs)]() {
        const auto start = std::chrono::steady_clock::now();
        raw->result = raw->transform.Run(query, inputs);
        raw->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        raw->done = true;
    });
    run_ = std::move(run);
    AddMessage("Running " + label + "...", false);
}

void SqlWorkbenchDialog::PollRun() {
    if (!run_ || !run_->done) return;
    run_->thread.join();
    std::unique_ptr<Run> run = std::move(run_);
    const auto& result = run->result;
    if (!result.ok) {
        AddMessage(result.cancelled ? "Stopped." : result.error, !result.cancelled);
        bottom_tab_ = 1;
        return;
    }
    if (run->kind == RunKind::Counts) {
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
        side_tab_ = 1;
        AddMessage("Counted the rows after each part (" + Seconds(run->seconds) + ").", false);
        return;
    }
    result_ = result.table;
    result_label_ = run->label;
    result_seconds_ = run->seconds;
    result_preview_ = cyxwiz::PreviewArrowRows(*result_, kResultRows, kResultCellChars);
    // Registered under its own name, so the step's saved output is untouched.
    result_dataset_ = "sqlwb_" + std::to_string(node_id_);
    cyxwiz::DataRegistry::Instance().RegisterArrowTable(result_, result_dataset_);
    bottom_tab_ = 0;
    AddMessage(result_label_ + ": " + sqlstep::FormatCount(result_->num_rows()) + " rows, " +
                   std::to_string(result_->num_columns()) + " columns (" + Seconds(run->seconds) + ")",
               false);
}

void SqlWorkbenchDialog::MarkPart(size_t index) {
    if (index >= outline_.parts.size()) return;
    const auto& part = outline_.parts[index];
    const int count = std::max(1, editor_.Doc().LineCount());
    const int first = std::clamp(part.first_line - 1, 0, count - 1);
    const int last = std::clamp(part.last_line - 1, first, count - 1);
    const cyxwiz::editor::Pos end{last, static_cast<int>(editor_.Doc().Line(last).size())};
    editor_.SetMarks({{{first, 0}, end, false}});
    cyxwiz::editor::Selection selection;
    selection.anchor = {first, 0};
    selection.head = end;
    editor_.Doc().SetSelections({selection});
    editor_.ScrollToCursor();
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

    if (editor_.Doc().Version() != outline_version_) {
        outline_version_ = editor_.Doc().Version();
        draft_ = editor_.GetText();
        outline_ = sqlstep::OutlineSqlQuery(draft_);
    }
    // The step's query changed elsewhere (Properties, undo) while the draft
    // still matches what was saved: follow it.
    const std::string step_query = Param(*node, "query");
    if (step_query != saved_query_ && draft_ == saved_query_) {
        saved_query_ = step_query;
        editor_.SetText(saved_query_);
        outline_version_ = ~0ull;
    }

    // Ctrl+Enter runs the step from anywhere in the window.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::GetIO().KeyCtrl &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
        StartRun(RunKind::Step);
    }

    const ui::Tokens& t = ui::CurrentTokens();
    RenderHeader(*node);
    RenderToolbar(*node);
    ImGui::Dummy(ImVec2(0.0f, t.space_xs));

    const float status_h = ImGui::GetFrameHeight();
    const float body_h = std::max(200.0f, ImGui::GetContentRegionAvail().y - status_h - t.space_sm);
    RenderSidePanel(body_h);
    ImGui::SameLine();
    ImGui::BeginGroup();
    const float editor_h = std::floor(body_h * 0.55f);
    RenderEditor(editor_h);
    RenderBottomPanel(body_h - editor_h - ImGui::GetStyle().ItemSpacing.y);
    ImGui::EndGroup();
    RenderStatusBar();
}

void SqlWorkbenchDialog::RenderHeader(MLNode& node) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!context_.recipe_name.empty()) {
        Dim("recipe " + context_.recipe_name + "  " ICON_FA_ANGLE_RIGHT "  step");
        ImGui::SameLine();
    } else {
        Dim("step");
        ImGui::SameLine();
    }
    Coloured(node.name, t.text_bright);
    const bool edited = draft_ != saved_query_;
    if (edited) {
        ImGui::SameLine();
        Coloured(ICON_FA_CIRCLE " Edited, not saved to the step", t.warning);
    }
}

void SqlWorkbenchDialog::RenderToolbar(MLNode& node) {
    (void)node;
    const bool pipeline = node_editor_ && node_editor_->IsDataPipelineRunning();
    const bool running = run_ != nullptr || pipeline;
    const char* busy = run_ ? "A query is running" : pipeline ? "A pipeline run is in progress" : nullptr;
    if (ui::PrimaryButton(ICON_FA_PLAY " Run step", !running, busy)) StartRun(RunKind::Step);
    if (ImGui::IsItemHovered() && !running) ImGui::SetTooltip("Run the whole query on the step's inputs (Ctrl+Enter)");
    ImGui::SameLine();
    if (ui::SecondaryButton("Run to part " ICON_FA_CHEVRON_DOWN, !running && outline_.structured &&
                                                                     outline_.parts.size() > 1,
                            running ? busy : "The query has no parts to stop at", ui::ButtonSize::Regular)) {
        ImGui::OpenPopup("##run_to_part");
    }
    if (ImGui::BeginPopup("##run_to_part")) {
        for (size_t i = 0; i < outline_.parts.size(); ++i) {
            const auto& part = outline_.parts[i];
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
    if (ui::SecondaryButton("Count rows per part", !running && outline_.structured && outline_.parts.size() > 1,
                            running ? busy : "The query has no parts to count", ui::ButtonSize::Regular)) {
        StartRun(RunKind::Counts);
    }
    ImGui::SameLine();
    if (ui::DangerButton(ICON_FA_STOP " Stop", run_ != nullptr, "No query is running", ui::ButtonSize::Regular)) {
        run_->transform.Interrupt();
    }

    const bool edited = draft_ != saved_query_;
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
    const float width = std::clamp(ImGui::GetContentRegionAvail().x * 0.2f, 240.0f, 340.0f);
    if (BeginPanel("##wb_side", ImVec2(width, height), t.bg_panel)) {
        static const char* const kTabs[] = {"Inputs", "Parts"};
        ui::SegmentedControl("##wb_side_tab", kTabs, 2, &side_tab_);
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        if (side_tab_ == 0) {
            if (inputs_.empty()) {
                ImGui::PushTextWrapPos(0.0f);
                Dim(context_.error.empty() ? "Nothing is connected to this step's input."
                                           : "The graph could not be read: " + context_.error);
                ImGui::PopTextWrapPos();
            }
            bool missing = false;
            for (size_t i = 0; i < inputs_.size(); ++i) {
                const auto& input = inputs_[i];
                ImGui::PushID(static_cast<int>(i));
                Coloured(input.alias, t.accent_text);
                ImGui::SameLine();
                Dim(input.table ? sqlstep::FormatCount(input.table->num_rows()) + " rows" : "not in memory");
                ImGui::PushTextWrapPos(0.0f);
                Dim("from " + input.source);
                ImGui::PopTextWrapPos();
                if (input.table) {
                    const auto facts = node_editor_->GetNodeRunFacts().find(input.node_id);
                    if (open_table_ && facts != node_editor_->GetNodeRunFacts().end() &&
                        ui::LinkButton("Open input in Table Viewer")) {
                        open_table_(facts->second.output_dataset, input.source);
                    }
                    for (const auto& field : input.table->schema()->fields()) {
                        if (ImGui::Selectable(field->name().c_str(), false, ImGuiSelectableFlags_AllowOverlap)) {
                            editor_.Doc().Type(field->name());
                            editor_.RequestFocus();
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
                ImGui::PushTextWrapPos(0.0f);
                Dim("Run the steps before this one to load the inputs.");
                ImGui::PopTextWrapPos();
                const bool can = node_editor_ && !node_editor_->IsDataPipelineRunning() && !run_;
                if (ui::SecondaryButton("Load inputs", can, run_ ? "A query is running" : "A pipeline run is in progress")) {
                    node_editor_->ExecuteDataPipelineThrough(node_id_, false, false);
                }
            }
        } else {
            const bool counted = !part_rows_.empty() && part_rows_draft_ == draft_;
            if (ui::LinkButton("Count rows per part", !run_ && outline_.structured && outline_.parts.size() > 1)) {
                StartRun(RunKind::Counts);
            }
            if (!part_rows_.empty()) {
                ImGui::SameLine();
                Dim(counted ? Seconds(part_rows_seconds_) : "(counted on an older draft)");
            }
            ImGui::PushTextWrapPos(0.0f);
            Dim("Rows after each part. Click a part to run the query up to it.");
            ImGui::PopTextWrapPos();
            int64_t previous = -1;
            for (size_t i = 0; i < outline_.parts.size(); ++i) {
                const auto& part = outline_.parts[i];
                ImGui::PushID(static_cast<int>(i));
                const auto found = part_rows_.find(part.name);
                const bool known = found != part_rows_.end();
                std::string rows = known ? sqlstep::FormatCount(found->second) : "";
                if (known && previous >= 0 && found->second != previous) {
                    const int64_t change = found->second - previous;
                    rows += change > 0 ? "  +" + sqlstep::FormatCount(change) : "  -" + sqlstep::FormatCount(-change);
                }
                if (ImGui::Selectable("##part", marked_part_ == static_cast<int>(i),
                                      ImGuiSelectableFlags_AllowOverlap)) {
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
    }
    EndPanel();
}

void SqlWorkbenchDialog::RenderEditor(float height) {
    ImFont* mono = cyxwiz::gui::GetCodeFont();
    if (mono) ImGui::PushFont(mono);
    editor_.Render("##wb_editor", ImVec2(ImGui::GetContentRegionAvail().x, height));
    if (mono) ImGui::PopFont();
}

void SqlWorkbenchDialog::RenderBottomPanel(float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (BeginPanel("##wb_bottom", ImVec2(ImGui::GetContentRegionAvail().x, height), t.bg_panel)) {
        const std::string messages = "Messages (" + std::to_string(messages_.size()) + ")";
        const char* tabs[] = {"Results", messages.c_str()};
        ui::SegmentedControl("##wb_bottom_tab", tabs, 2, &bottom_tab_);
        if (run_) {
            ImGui::SameLine();
            Coloured(ICON_FA_SPINNER " Running " + run_->label + "...", t.running);
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        if (bottom_tab_ == 0) RenderResults();
        else RenderMessages();
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
    if (open_table_) {
        AlignRight(ImGui::CalcTextSize("Open in Table Viewer").x);
        if (ui::LinkButton("Open in Table Viewer")) open_table_(result_dataset_, step_name_ + " " + result_label_);
    }
    const int columns = static_cast<int>(result_preview_.headers.size()) + 1;
    if (columns > 1 &&
        ImGui::BeginTable("##wb_results", std::min(columns, 64),
                          ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit,
                          ImGui::GetContentRegionAvail())) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed);
        for (int c = 0; c + 1 < std::min(columns, 64); ++c) {
            const auto& header = result_preview_.headers[static_cast<size_t>(c)];
            ImGui::TableSetupColumn(header.c_str(), ImGuiTableColumnFlags_WidthFixed,
                                    header == "text" ? 480.0f : 0.0f);
        }
        ImGui::TableHeadersRow();
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(result_preview_.rows.size()));
        while (clipper.Step()) {
            for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                Dim(std::to_string(r + 1));
                const auto& row = result_preview_.rows[static_cast<size_t>(r)];
                for (int c = 0; c + 1 < std::min(columns, 64); ++c) {
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(row[static_cast<size_t>(c)].c_str());
                }
            }
        }
        ImGui::EndTable();
    }
}

void SqlWorkbenchDialog::RenderMessages() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (messages_.empty()) {
        Dim("Runs, saves and errors appear here.");
        return;
    }
    if (ImGui::BeginChild("##wb_messages", ImGui::GetContentRegionAvail())) {
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
    ImGui::EndChild();
}

void SqlWorkbenchDialog::RenderStatusBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    Dim("DuckDB, read-only, no file access    Results must be flat columns");
    const auto head = editor_.Doc().Primary().head;
    std::string right = "Ln " + std::to_string(head.line + 1) + ", Col " + std::to_string(head.col + 1) + "    " +
                        std::to_string(outline_.line_count) + " lines";
    if (!outline_.structured) right += "    the query could not be split into parts";
    AlignRight(ImGui::CalcTextSize(right.c_str()).x);
    Coloured(right, outline_.structured ? t.text_dim : t.warning);
}

}  // namespace gui
