#include "query_editor.h"

#include "../../core/arrow_data_table.h"
#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/session_query_service.h"
#include "../plot/plot_window.h"

#include <spdlog/spdlog.h>

#include <cstring>

namespace cyxwiz {

namespace {

bool IsNumber(const std::shared_ptr<arrow::DataType>& t) {
    return arrow::is_integer(t->id()) || arrow::is_floating(t->id());
}

constexpr const char* kPlaceholder = "SELECT * FROM \"your dataset\" LIMIT 100";

}  // namespace

QueryEditor::QueryEditor() {
    std::snprintf(query_buffer_, sizeof(query_buffer_), "%s", kPlaceholder);
}

QueryEditor::~QueryEditor() {
    if (task_id_) SessionQueryService::Instance().Cancel(task_id_);
}

void QueryEditor::SetActiveDataset(const std::string& dataset_name) {
    // Queries name the table as people call it (the Data Input node's name).
    const auto entry = DatasetCatalog::Instance().Resolve(dataset_name);
    current_dataset_ = entry ? entry->Shown() : dataset_name;
    // A fresh editor names the picked dataset.
    if (std::strcmp(query_buffer_, kPlaceholder) == 0 && !dataset_name.empty()) {
        const std::string q = "SELECT * FROM " + SessionQueryEngine::QuoteIdentifier(current_dataset_) + " LIMIT 100";
        std::snprintf(query_buffer_, sizeof(query_buffer_), "%s", q.c_str());
    }
}

void QueryEditor::SetQuery(const std::string& sql) {
    std::snprintf(query_buffer_, sizeof(query_buffer_), "%s", sql.c_str());
    last_error_ = sql.size() >= sizeof(query_buffer_) ? "The query is too long for the editor; it was cut." : std::string();
}

std::string QueryEditor::ExampleTable() const {
    if (!current_dataset_.empty()) return current_dataset_;
    const auto names = SessionQueryService::Instance().QueryableNames();
    return names.empty() ? std::string("your dataset") : names.front();
}

bool QueryEditor::ExecuteQuery() {
    if (task_id_) return false;
    const std::string sql = query_buffer_;
    if (sql.find_first_not_of(" \t\r\n") == std::string::npos) {
        last_error_ = "Write a query first.";
        return false;
    }
    last_error_.clear();
    save_note_.clear();
    running_sql_ = sql;
    started_at_ = ImGui::GetTime();
    QueryRequest request;
    request.sql = sql;
    request.row_limit = kDisplayRows;
    request.label = "Data Studio query";
    std::weak_ptr<int> alive = alive_;
    task_id_ = SessionQueryService::Instance().Submit(std::move(request), [this, alive, sql](const QueryResult& r) {
        if (alive.expired()) return;
        task_id_ = 0;
        if (!r.ok) {
            last_error_ = r.error;
            return;
        }
        last_result_ = r;
        result_sql_ = sql;
        query_history_.push_back(sql);
        if (query_history_.size() > kMaxHistory) query_history_.erase(query_history_.begin());
        spdlog::info("[Data Studio] Query: {} rows{} in {:.0f} ms", r.table ? r.table->num_rows() : 0,
                     r.truncated ? " (display limit)" : "", r.elapsed_ms);
    }, alive_);
    return task_id_ != 0;
}

void QueryEditor::CancelQuery() {
    if (task_id_) SessionQueryService::Instance().Cancel(task_id_);
}

bool QueryEditor::SaveResultAsDataset(const std::string& dataset_name) {
    if (result_sql_.empty() || dataset_name.empty()) return false;
    QueryRequest request;
    request.sql = result_sql_;
    request.label = "Save query result as " + dataset_name;
    std::weak_ptr<int> alive = alive_;
    save_note_ = "Saving...";
    SessionQueryService::Instance().Submit(std::move(request), [this, alive, dataset_name](const QueryResult& r) {
        if (alive.expired()) return;
        if (!r.ok || !r.table) {
            save_note_ = "Not saved: " + r.error;
            return;
        }
        DataRegistry::Instance().RegisterArrowTable(r.table, dataset_name);
        save_note_ = "Saved as '" + dataset_name + "' (" + std::to_string(r.table->num_rows()) +
                     " rows): pick it in Dataset or name it in a query.";
    }, alive_);
    return true;
}

void QueryEditor::OpenPlot() {
    if (!last_result_.table) return;
    if (!plot_window_) plot_window_ = std::make_unique<plot::PlotWindow>("data_studio_query_plot");
    // A first plot that fits the result: a category and a number as bars,
    // two numbers as a scatter, else a histogram of the first number.
    const auto schema = last_result_.table->schema();
    int text_col = -1, num_a = -1, num_b = -1;
    for (int i = 0; i < schema->num_fields(); ++i) {
        const auto& t = schema->field(i)->type();
        if (IsNumber(t)) {
            if (num_a < 0) num_a = i;
            else if (num_b < 0) num_b = i;
        } else if (text_col < 0) {
            text_col = i;
        }
    }
    plot::PlotSpec spec;
    if (text_col >= 0 && num_a >= 0) {
        spec.kind = plot::Kind::Bar;
        spec.x_column = schema->field(text_col)->name();
        spec.y_columns = {schema->field(num_a)->name()};
    } else if (num_a >= 0 && num_b >= 0) {
        spec.kind = plot::Kind::Scatter;
        spec.x_column = schema->field(num_a)->name();
        spec.y_columns = {schema->field(num_b)->name()};
    } else if (num_a >= 0) {
        spec.kind = plot::Kind::Histogram;
        spec.x_column = schema->field(num_a)->name();
    }
    plot_window_->SetSpec(spec);
    plot_window_->SetArrowTable("Query result", last_result_.table);
    plot_window_->visible = true;
}

void QueryEditor::RenderWindows() {
    if (plot_window_) plot_window_->Render();
}

}  // namespace cyxwiz
