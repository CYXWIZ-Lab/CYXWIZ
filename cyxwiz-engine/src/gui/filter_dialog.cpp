// Row Filter dialog (TOFIX129 B-5, board 6): the condition as rows of
// column / operator / value, with columns picked by name from the step's
// input. Writes the `condition` the executor runs (core/filter_condition.h).
#include "node_config_dialog.h"
#include "node_editor.h"
#include "ui_buttons.h"
#include "ui_tokens.h"
#include "../core/sql_transform.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gui {

namespace {

std::optional<cyxwiz::FilterValueKind> KindOf(const arrow::DataType& type) {
    if (type.id() == arrow::Type::BOOL) return cyxwiz::FilterValueKind::TrueFalse;
    if (arrow::is_integer(type.id()) || arrow::is_floating(type.id())) return cyxwiz::FilterValueKind::Number;
    if (type.id() == arrow::Type::STRING || type.id() == arrow::Type::LARGE_STRING)
        return cyxwiz::FilterValueKind::Text;
    return std::nullopt;  // dates, lists, ...: the executor cannot compare them
}

// A typed value's kind while the columns are unknown.
cyxwiz::FilterValueKind GuessKind(const std::string& value) {
    if (value == "true" || value == "false") return cyxwiz::FilterValueKind::TrueFalse;
    char* end = nullptr;
    std::strtod(value.c_str(), &end);
    if (!value.empty() && end == value.c_str() + value.size()) return cyxwiz::FilterValueKind::Number;
    return cyxwiz::FilterValueKind::Text;
}

const char* KindWords(cyxwiz::FilterValueKind kind) {
    switch (kind) {
        case cyxwiz::FilterValueKind::Number: return "numbers";
        case cyxwiz::FilterValueKind::Text: return "text";
        case cyxwiz::FilterValueKind::TrueFalse: return "true/false";
    }
    return "";
}

constexpr cyxwiz::FilterOp kOps[] = {
    cyxwiz::FilterOp::Is,       cyxwiz::FilterOp::IsNot,   cyxwiz::FilterOp::MoreThan,
    cyxwiz::FilterOp::LessThan, cyxwiz::FilterOp::AtLeast, cyxwiz::FilterOp::AtMost,
};

}  // namespace

FilterDialog::FilterDialog(MLNode* node)
    : NodeConfigDialog("Row Filter", node)
{
    LoadFromNode();
}

void FilterDialog::LoadFromNode() {
    rows_.clear();
    written_ = false;
    written_text_.clear();
    std::string saved;
    if (node_) {
        const auto it = node_->parameters.find("condition");
        if (it != node_->parameters.end()) saved = it->second;
    }
    if (auto rows = cyxwiz::ParseFilterCondition(saved)) {
        rows_ = std::move(*rows);
    } else {
        written_ = true;
        written_text_ = saved;
    }
    if (!written_ && rows_.empty()) rows_.push_back({});
}

void FilterDialog::RefreshColumns() {
    columns_.clear();
    input_name_.clear();
    other_columns_ = 0;
    if (!node_editor_ || !node_) return;
    const auto context = node_editor_->GetDataStepContext(node_->id);
    if (context.inputs.empty()) return;
    const auto& input = context.inputs.front();
    input_name_ = input.name;
    const auto& facts = node_editor_->GetNodeRunFacts();
    const auto fact = facts.find(input.id);
    if (fact == facts.end() || fact->second.output_dataset.empty()) return;
    const auto source = cyxwiz::SqlInputForDataset("input", fact->second.output_dataset);
    const auto schema = source.table ? source.table->schema() : source.schema;
    if (!schema) return;
    for (const auto& field : schema->fields()) {
        if (const auto kind = KindOf(*field->type())) columns_.push_back({field->name(), *kind});
        else ++other_columns_;
    }
    // A saved or typed value takes its column's real type: 5 against a text
    // column is the text '5'.
    for (auto& row : rows_) {
        const Column* column = FindColumn(row.column);
        if (!column || column->kind == row.kind) continue;
        row.kind = column->kind;
        if (!cyxwiz::FilterOpAllowed(row.op, row.kind)) row.op = cyxwiz::FilterOp::Is;
        if (row.kind == cyxwiz::FilterValueKind::TrueFalse && row.value != "false") row.value = "true";
    }
}

const FilterDialog::Column* FilterDialog::FindColumn(const std::string& name) const {
    const auto it = std::find_if(columns_.begin(), columns_.end(), [&](const Column& c) { return c.name == name; });
    return it == columns_.end() ? nullptr : &*it;
}

std::string FilterDialog::Problem() const {
    if (written_) {
        if (written_text_.find_first_not_of(" \t\r\n") == std::string::npos) return "Write a condition";
        std::vector<cyxwiz::FilterConditionToken> tokens;
        std::string error;
        if (!cyxwiz::TokenizeFilterCondition(written_text_, tokens, error)) return error;
        return {};
    }
    for (const auto& row : rows_) {
        if (auto problem = cyxwiz::FilterRowProblem(row); !problem.empty()) return problem;
        if (!columns_.empty() && !FindColumn(row.column))
            return "\"" + row.column + "\" is not a column of the input";
    }
    return {};
}

bool FilterDialog::CanApply() const {
    return Problem().empty();
}

const char* FilterDialog::ApplyBlockedReason() const {
    blocked_reason_ = Problem();
    return blocked_reason_.empty() ? NodeConfigDialog::ApplyBlockedReason() : blocked_reason_.c_str();
}

void FilterDialog::Apply() {
    if (!node_ || !CanApply()) return;
    node_->parameters["condition"] = written_ ? written_text_ : cyxwiz::BuildFilterCondition(rows_);
    has_changes_ = false;
}

void FilterDialog::Reset() {
    if (!node_) return;
    node_->parameters = original_params_;
    LoadFromNode();
    has_changes_ = false;
}

void FilterDialog::RenderContent() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    // The input's columns: on open and every 2 s (a pipeline run adds them).
    const double now = ImGui::GetTime();
    if (columns_time_ < 0.0 || now - columns_time_ > 2.0) {
        RefreshColumns();
        columns_time_ = now;
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Keep the rows where");
    ImGui::PushTextWrapPos(0.0f);
    if (!columns_.empty()) {
        std::string from = std::to_string(columns_.size()) + " columns of \"" + input_name_ + "\"";
        if (other_columns_ > 0)
            from += " (" + std::to_string(other_columns_) + " more hold dates or lists, which a filter cannot compare)";
        ImGui::TextColored(t.text_dim, "%s", from.c_str());
    } else if (!input_name_.empty()) {
        ImGui::TextColored(t.text_dim, "Run the pipeline once to pick the columns of \"%s\" by name.",
                           input_name_.c_str());
    } else {
        ImGui::TextColored(t.text_dim, "Connect a table to the step's input to pick its columns by name.");
    }
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    if (written_) {
        RenderWrittenCondition();
    } else {
        RenderRows();
    }

    ImGui::Spacing();
    const std::string problem = Problem();
    if (!problem.empty()) ValidationMessage(problem);
}

void FilterDialog::RenderRows() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    const ImGuiStyle& style = ImGui::GetStyle();
    const float join_w = ImGui::CalcTextSize("where").x + style.FramePadding.x * 2.0f + ImGui::GetFrameHeight();
    const float remove_w = cyxwiz::ui::ButtonWidth("Remove", cyxwiz::ui::ButtonSize::Small);
    const float x0 = ImGui::GetCursorPosX();

    for (size_t i = 0; i < rows_.size(); ++i) {
        auto& row = rows_[i];
        ImGui::PushID(static_cast<int>(i));
        // Every row fits the width: the fields share what the join and Remove leave.
        const float rest = ImGui::GetContentRegionAvail().x - join_w - remove_w - style.ItemSpacing.x * 4.0f;
        const float column_w = std::max(80.0f, rest * 0.38f);
        const float op_w = std::max(80.0f, rest * 0.26f);
        const float value_w = std::max(60.0f, rest - column_w - op_w);

        if (i == 0) {
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_dim, "where");
            ImGui::SameLine(x0 + join_w + style.ItemSpacing.x);
        } else {
            const char* joins[] = {"and", "or"};
            int join = row.or_join ? 1 : 0;
            ImGui::SetNextItemWidth(join_w);
            if (ImGui::Combo("##join", &join, joins, 2)) {
                row.or_join = join == 1;
                has_changes_ = true;
            }
            ImGui::SameLine();
        }

        // Column, by name when the input's columns are known.
        ImGui::SetNextItemWidth(column_w);
        if (!columns_.empty()) {
            const char* preview = row.column.empty() ? "Pick a column" : row.column.c_str();
            if (ImGui::BeginCombo("##column", preview, ImGuiComboFlags_HeightLarge)) {
                for (const auto& column : columns_) {
                    const bool selected = column.name == row.column;
                    std::string label = column.name + "  (" + KindWords(column.kind) + ")";
                    if (ImGui::Selectable(label.c_str(), selected)) {
                        if (row.kind != column.kind) {
                            row.value = column.kind == cyxwiz::FilterValueKind::TrueFalse ? "true" : "";
                            if (!cyxwiz::FilterOpAllowed(row.op, column.kind)) row.op = cyxwiz::FilterOp::Is;
                        }
                        row.column = column.name;
                        row.kind = column.kind;
                        has_changes_ = true;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        } else {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s", row.column.c_str());
            if (ImGui::InputTextWithHint("##column", "column name", buf, sizeof(buf))) {
                row.column = buf;
                has_changes_ = true;
            }
        }
        ImGui::SameLine();

        // Operator, in words; text and true/false offer only is / is not.
        ImGui::SetNextItemWidth(op_w);
        if (ImGui::BeginCombo("##op", cyxwiz::FilterOpWords(row.op))) {
            for (const auto op : kOps) {
                if (!cyxwiz::FilterOpAllowed(op, row.kind)) continue;
                if (ImGui::Selectable(cyxwiz::FilterOpWords(op), op == row.op)) {
                    row.op = op;
                    has_changes_ = true;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();

        // Value.
        ImGui::SetNextItemWidth(value_w);
        const bool known = FindColumn(row.column) != nullptr;
        if (row.kind == cyxwiz::FilterValueKind::TrueFalse && known) {
            const char* values[] = {"true", "false"};
            int value = row.value == "false" ? 1 : 0;
            if (ImGui::Combo("##value", &value, values, 2)) {
                row.value = values[value];
                has_changes_ = true;
            }
        } else {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s", row.value.c_str());
            const char* hint = row.kind == cyxwiz::FilterValueKind::Number && known ? "a number" : "value";
            if (ImGui::InputTextWithHint("##value", hint, buf, sizeof(buf))) {
                row.value = buf;
                if (!known) row.kind = GuessKind(row.value);
                has_changes_ = true;
            }
        }
        ImGui::SameLine();

        if (cyxwiz::ui::DangerButton("Remove", rows_.size() > 1, "A filter needs one condition")) {
            rows_.erase(rows_.begin() + static_cast<std::ptrdiff_t>(i));
            has_changes_ = true;
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }

    ImGui::Spacing();
    if (cyxwiz::ui::SecondaryButton("Add condition")) {
        rows_.push_back({});
        has_changes_ = true;
    }

    // The text that runs (once every row can run), and how and/or combine.
    if (rows_.empty() || !Problem().empty()) return;
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Runs as");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(cyxwiz::BuildFilterCondition(rows_).c_str());
    const bool has_and = std::any_of(rows_.begin() + 1, rows_.end(), [](const auto& r) { return !r.or_join; });
    const bool has_or = std::any_of(rows_.begin() + 1, rows_.end(), [](const auto& r) { return r.or_join; });
    if (has_and && has_or)
        ImGui::TextColored(t.text_dim, "\"and\" binds first: a and b or c keeps rows matching both a and b, or c.");
    ImGui::PopTextWrapPos();
}

void FilterDialog::RenderWrittenCondition() {
    const auto& t = cyxwiz::ui::CurrentTokens();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_dim,
                       "This condition uses brackets, so it does not fit rows. Edit it as text: "
                       "column names, = != < > <= >=, 'text', numbers, true / false, and, or, ( ).");
    ImGui::PopTextWrapPos();
    char buf[2048];
    std::snprintf(buf, sizeof(buf), "%s", written_text_.c_str());
    if (ImGui::InputTextMultiline("##condition", buf, sizeof(buf),
                                  ImVec2(-1.0f, ImGui::GetTextLineHeightWithSpacing() * 5.0f))) {
        written_text_ = buf;
        has_changes_ = true;
    }
    if (cyxwiz::ui::LinkButton("Start over with rows")) {
        written_ = false;
        written_text_.clear();
        rows_.assign(1, {});
        has_changes_ = true;
    }
}

}  // namespace gui
