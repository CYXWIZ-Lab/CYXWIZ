#include "test_errors_by_value.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/parquet_backed_dataset.h"
#include "../icons.h"
#include "../ui_tokens.h"

#include <arrow/api.h>
#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

namespace {

std::string Measure(double x, bool classifier) {
    char buf[32];
    if (classifier) std::snprintf(buf, sizeof(buf), "%.1f%%", x * 100.0);
    else std::snprintf(buf, sizeof(buf), "%.4g", x);
    return buf;
}

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

}  // namespace

ErrorsByValueView::~ErrorsByValueView() { CancelTask(); }

void ErrorsByValueView::CancelTask() {
    if (task_) AsyncTaskManager::Instance().Cancel(task_);
    task_ = 0;
}

void ErrorsByValueView::SetRows(const std::shared_ptr<arrow::Table>& rows) {
    if (rows == rows_) return;
    CancelTask();
    rows_ = rows;
    row_errors_ = studio::RowErrorsFromPredictions(rows);
    column_.clear();
    slices_ = {};
    error_.clear();
    columns_.clear();
}

void ErrorsByValueView::SliceBy(const std::string& column) {
    CancelTask();
    column_ = column;
    slices_ = {};
    error_.clear();
    const std::string source = row_errors_.source_dataset;
    const std::vector<int64_t> row_ids = row_errors_.row_ids;
    const std::vector<double> errors = row_errors_.errors;
    struct Outcome {
        studio::ErrorSlices slices;
        std::string error;
    };
    auto outcome = std::make_shared<Outcome>();
    std::weak_ptr<int> alive = alive_;
    task_ = AsyncTaskManager::Instance().RunAsync(
        "Error by " + column,
        [source, column, row_ids, errors, outcome](LambdaTask& task) {
            studio::ColumnGatherer gatherer(row_ids);
            std::string error;
            bool ok = true;
            if (const auto memory = DataRegistry::Instance().GetArrowDataset(source)) {
                const auto table = memory->GetArrowTable();
                ok = table && gatherer.Add(table->GetColumnByName(column), error);
            } else if (const auto disk = DataRegistry::Instance().GetParquetBackedDataset(source)) {
                const int groups = disk->GetNumRowGroups();
                for (int g = 0; ok && g < groups; ++g) {
                    if (task.ShouldStop()) {
                        task.MarkFailed("Cancelled.");
                        return;
                    }
                    task.ReportProgress(static_cast<float>(g) / static_cast<float>(std::max(1, groups)),
                                        "row group " + std::to_string(g + 1) + " of " + std::to_string(groups));
                    ok = gatherer.Add(disk->ReadRowGroupColumn(g, column), error);
                }
            } else {
                ok = false;
                error = "'" + source + "' is no longer loaded; load it again to slice by its columns";
            }
            studio::GatheredColumn values;
            if (ok) ok = gatherer.Finish(values, error);
            if (!ok) {
                outcome->error = error.empty() ? "the column could not be read" : error;
                task.MarkFailed(outcome->error);
                return;
            }
            outcome->slices = studio::SliceErrors(values, errors);
        },
        nullptr,
        [this, alive, outcome](bool /*success*/, const std::string& /*message*/) {
            if (alive.expired()) return;
            task_ = 0;
            error_ = outcome->error == "Cancelled." ? std::string() : outcome->error;
            slices_ = outcome->slices;
        },
        alive_);
}

void ErrorsByValueView::Render() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!row_errors_.ok) {
        Wrapped(t.text_dim, "Not available: " + row_errors_.error + ". Test again to record the rows.");
        return;
    }
    // The tested table's columns that can be sliced (not lists or bytes), read
    // while it is loaded; the list fills in when it is loaded again.
    if (columns_.empty()) {
        std::shared_ptr<arrow::Schema> schema;
        if (const auto memory = DataRegistry::Instance().GetArrowDataset(row_errors_.source_dataset)) {
            if (const auto table = memory->GetArrowTable()) schema = table->schema();
        } else if (const auto disk = DataRegistry::Instance().GetParquetBackedDataset(row_errors_.source_dataset)) {
            schema = disk->GetSchema();
        }
        if (schema)
            for (const auto& field : schema->fields())
                if (!arrow::is_nested(field->type()->id()) && field->type()->id() != arrow::Type::BINARY &&
                    field->type()->id() != arrow::Type::LARGE_BINARY)
                    columns_.push_back(field->name());
    }
    if (columns_.empty()) {
        Wrapped(t.text_dim, "The tested table '" + row_errors_.source_dataset +
                                "' is not loaded; load it (its Data Input) to slice the error by its columns.");
        return;
    }
    Wrapped(t.text_dim, "The error of the test rows grouped by the values of a column of '" + row_errors_.source_dataset +
                            "', joined by each row's dataset row.");
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "Column");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::min(280.0f, ImGui::GetContentRegionAvail().x * 0.6f));
    if (ImGui::BeginCombo("##slice_column", column_.empty() ? "Choose a column" : column_.c_str())) {
        for (const auto& c : columns_)
            if (ImGui::Selectable(c.c_str(), c == column_)) SliceBy(c);
        ImGui::EndCombo();
    }
    if (column_.empty()) return;
    if (task_) {
        ImGui::TextColored(t.text_dim, "%s Reading '%s'...", ICON_FA_SPINNER, column_.c_str());
        return;
    }
    if (!error_.empty()) {
        Wrapped(t.error, error_);
        return;
    }
    const bool classifier = row_errors_.classifier;
    if (ImGui::BeginTable("##by_value", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings)) {
        ImGui::TableSetupColumn(slices_.binned ? "range" : "value");
        ImGui::TableSetupColumn("test rows");
        ImGui::TableSetupColumn(classifier ? "wrong" : "mean |error|");
        ImGui::TableHeadersRow();
        for (const auto& s : slices_.slices) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(s.value.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%lld", static_cast<long long>(s.rows));
            ImGui::TableNextColumn();
            // Above the overall error by a quarter or more stands out.
            ImGui::TextColored(s.error > slices_.overall * 1.25 && s.error > 0.0 ? t.warning : t.text, "%s",
                               Measure(s.error, classifier).c_str());
        }
        ImGui::EndTable();
    }
    Wrapped(t.text_faint, "Overall " + Measure(slices_.overall, classifier) + " over " +
                              std::to_string(row_errors_.errors.size()) + " test rows." +
                              (slices_.binned ? " Numbers are grouped in ranges of about equal rows." : "") +
                              " A value with few rows can differ by chance.");
}

}  // namespace cyxwiz
