#pragma once

// Data Studio Query tab (TOFIX134 P3.3, approved board 14): SQL over the
// datasets by their real names, run by the session query service in the
// background (Task View, Cancel), results as an Arrow table shown here and
// opened in the Table Viewer or the Plot window, or saved as a dataset.

#include "../../core/session_query_engine.h"

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

private:
    void RenderEditor();
    void RenderResult();
    void RenderExamples();
    void OpenPlot();
    std::string ExampleTable() const;

    char query_buffer_[8192] = {};
    std::string current_dataset_;
    std::string last_error_;
    std::string running_sql_;
    uint64_t task_id_ = 0;
    double started_at_ = 0;
    QueryResult last_result_;
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
