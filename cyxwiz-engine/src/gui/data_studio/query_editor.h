#pragma once

// Data Studio Explore > Query (TOFIX134 P3.3; TOFIX148 step 3): scratch SQL
// over the datasets by their real names in the SQL Workbench's editor pane
// (highlighting, completion with the picked dataset's columns, find,
// Format), run by the session query service in the background (Task View,
// Cancel); the result in the table grid, opened in the Table Viewer or the
// Plot window, added to a dashboard, saved as a dataset, or turned into an
// SQL step of the graph.

#include "../../core/session_query_engine.h"
#include "../sql_editor_pane.h"
#include "../table_grid.h"

#include <imgui.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class DataTable;
namespace plot {
class PlotWindow;
}

class QueryEditor {
public:
    QueryEditor();
    ~QueryEditor();

    void Render();
    // The Plot window this tab opens (call every frame).
    void RenderWindows();

    // The dataset picked in the Data Studio header: the examples name it.
    void SetActiveDataset(const std::string& dataset_name);

    // Puts this SQL in the editor (it replaces what was there).
    void SetQuery(const std::string& sql);

    // Runs the query in the editor (Ctrl+Enter, Run).
    bool ExecuteQuery();
    void CancelQuery();
    bool IsRunning() const { return task_id_ != 0; }

    // Runs the query again without the display limit and registers the
    // result as a dataset (it then appears in the picker).
    bool SaveResultAsDataset(const std::string& dataset_name);

    std::function<void(std::shared_ptr<DataTable>)> on_open_table;  // Table Viewer (MainWindow)

    // Turn into SQL step (the panel answers from the graph): why the last
    // result's query cannot become a step ("" when it can), and making it.
    // `inputs` are the tables the query read.
    std::function<std::string(const std::vector<std::string>& inputs)> step_blocker;
    std::function<std::string(const std::string& sql, const std::vector<std::string>& inputs)> on_turn_into_step;

private:
    void RenderEditor();
    void RenderResult();
    void RenderExamples();
    void OpenPlot();
    std::string ExampleTable() const;

    ::gui::SqlEditorPane pane_{"data_studio_query"};
    std::string default_query_;     // what the editor holds until someone types (follows the picked dataset)
    std::string columns_for_;       // the dataset completion offers columns of
    bool error_details_ = false;    // the whole error, not its first line
    std::string step_note_;         // what Turn into SQL step did
    std::string current_dataset_;   // as people call it (the examples, Add to Dashboard)
    std::string active_registry_;   // the picked dataset's catalog name
    std::string last_error_;
    std::string running_sql_;
    uint64_t task_id_ = 0;
    double started_at_ = 0;
    QueryResult last_result_;
    TableGrid result_grid_;  // last_result_ in the Table Viewer's grid
    std::string result_sql_;
    std::vector<std::string> query_history_;
    char save_name_[256] = "query_result";
    std::string save_note_;
    std::unique_ptr<plot::PlotWindow> plot_window_;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);  // results come back only while this tab exists

    static constexpr size_t kDisplayRows = 100000;
    static constexpr size_t kMaxHistory = 50;
};

}  // namespace cyxwiz
