#pragma once

#include "../panel.h"
#include "../query/cyxql.h"
#include <vector>
#include <string>
#include <deque>

// Forward declaration
namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * Query Console Panel - Interactive CyxQL query interface
 *
 * Features:
 * - Query input with multi-line editing
 * - Syntax highlighting (basic)
 * - Query history with navigation
 * - Results table display
 * - Example queries dropdown
 * - Error display with line/column info
 */
class QueryConsolePanel : public Panel {
public:
    QueryConsolePanel();

    void Render() override;

    // Set the node editor to query against
    void SetNodeEditor(gui::NodeEditor* editor) { node_editor_ = editor; }

private:
    // UI Sections
    void RenderToolbar();
    void RenderQueryInput();
    void RenderResults();
    void RenderHistory();
    void RenderExamples();

    // Query execution
    void ExecuteQuery();
    void ExecuteQuery(const std::string& query);

    // History management
    void AddToHistory(const std::string& query);
    void NavigateHistoryUp();
    void NavigateHistoryDown();
    void ClearHistory();

    // Helpers
    std::string FormatResultValue(const query::ResultValue& value);
    void CopyResultsToClipboard();

    // Node editor reference
    gui::NodeEditor* node_editor_ = nullptr;

    // Query input
    static constexpr size_t QUERY_BUFFER_SIZE = 4096;
    char query_buffer_[QUERY_BUFFER_SIZE] = "";

    // Current result
    query::QueryResult current_result_;
    bool has_result_ = false;

    // Query history
    std::deque<std::string> history_;
    int history_index_ = -1;
    static constexpr size_t MAX_HISTORY = 50;

    // UI state
    bool show_examples_ = false;
    bool show_history_ = false;
    bool auto_execute_ = false;
    int selected_row_ = -1;

    // Example queries
    struct ExampleQuery {
        const char* name;
        const char* query;
        const char* description;
    };

    static const std::vector<ExampleQuery> example_queries_;
};

} // namespace cyxwiz
