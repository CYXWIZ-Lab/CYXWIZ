#pragma once

// State of the SQL step Properties view (TOFIX146 board 1). Rendered by
// Properties::RenderSqlStep in properties_sql_step.cpp; one view follows the
// selected SQL step.

#include "../data/arrow_data_table.h"
#include "code_editor.h"
#include "node_editor.h"
#include "properties.h"

#include <string>
#include <vector>

namespace gui {

struct Properties::SqlStepView {
    int node_id = -1;
    cyxwiz::CodeEditor editor;
    std::string loaded_query;  // the query text the editor last loaded or wrote
    int mode = 1;              // 0 Guided, 1 SQL
    int picked_part = -1;
    bool details_open = false;
    bool about_open = false;

    // Neighbours in the executable graph, refreshed every 2 s and after a run.
    NodeEditor::DataStepContext context;
    double context_time = -1.0;
    bool was_running = false;

    // First rows of the output, read from the registry once per run.
    std::string preview_key;  // dataset + finish time
    cyxwiz::ArrowPreview preview;
    std::vector<std::string> input_columns;
    std::string input_columns_key;
};

}  // namespace gui
