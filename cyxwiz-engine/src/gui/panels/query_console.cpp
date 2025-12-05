#include "query_console.h"
#include "../node_editor.h"
#include "../IconsFontAwesome6.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

// Example queries for the dropdown
const std::vector<QueryConsolePanel::ExampleQuery> QueryConsolePanel::example_queries_ = {
    {
        "List all nodes",
        "MATCH (n) RETURN n",
        "Returns all nodes in the graph"
    },
    {
        "List all Dense layers",
        "MATCH (n:Dense) RETURN n.name, n.units",
        "Find all Dense layers and show their names and units"
    },
    {
        "List all Conv2D layers",
        "MATCH (n:Conv2D) RETURN n.name, n.filters, n.kernel_size",
        "Find all Conv2D layers with their parameters"
    },
    {
        "Count nodes by type",
        "MATCH (n) RETURN type(n) AS type, count(n) AS count",
        "Count how many nodes of each type exist"
    },
    {
        "Find input to output path",
        "MATCH (a:DatasetInput)-[*]->(b:Output) RETURN a.name, b.name",
        "Find paths from DatasetInput to Output nodes"
    },
    {
        "Find residual connections",
        "MATCH (a)-[r:ResidualSkip]->(b) RETURN a.name, b.name",
        "Find all residual/skip connections"
    },
    {
        "Find Conv-BN-ReLU sequences",
        "MATCH (c:Conv2D)-[:TensorFlow]->(bn:BatchNorm)-[:TensorFlow]->(r:ReLU)\nRETURN c.name, bn.name, r.name",
        "Find Conv2D -> BatchNorm -> ReLU patterns"
    },
    {
        "Find Dense layers > 256 units",
        "MATCH (n:Dense) WHERE n.units > 256 RETURN n.name, n.units",
        "Filter Dense layers by number of units"
    },
    {
        "Find Dropout layers",
        "MATCH (n:Dropout) RETURN n.name, n.rate",
        "List all Dropout layers with their rates"
    },
    {
        "Find attention layers",
        "MATCH (n:MultiHeadAttention) RETURN n.name, n.d_model, n.num_heads",
        "Find all Multi-Head Attention layers"
    },
    {
        "Get node properties",
        "MATCH (n) WHERE n.name CONTAINS \"Dense\" RETURN n.name, properties(n)",
        "Get all properties of nodes with 'Dense' in the name"
    },
    {
        "List optimizers",
        "MATCH (n:Adam) RETURN n.name, n.learning_rate\nUNION MATCH (n:SGD) RETURN n.name, n.learning_rate",
        "Find all optimizer nodes"
    }
};

QueryConsolePanel::QueryConsolePanel()
    : Panel("Query Console", false)  // Hidden by default
{
    std::memset(query_buffer_, 0, sizeof(query_buffer_));

    // Set default query
    std::strcpy(query_buffer_, "MATCH (n) RETURN n");
}

void QueryConsolePanel::Render() {
    if (!visible_) return;

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar)) {
        focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

        RenderToolbar();

        // Split view: query input on top, results below
        float available_height = ImGui::GetContentRegionAvail().y;
        float input_height = 100.0f;
        float results_height = available_height - input_height - ImGui::GetStyle().ItemSpacing.y;

        // Query input section
        ImGui::BeginChild("QueryInput", ImVec2(0, input_height), true);
        RenderQueryInput();
        ImGui::EndChild();

        // Results section
        ImGui::BeginChild("Results", ImVec2(0, results_height), true);
        RenderResults();
        ImGui::EndChild();
    }
    ImGui::End();

    // Render popups
    if (show_examples_) {
        RenderExamples();
    }
    if (show_history_) {
        RenderHistory();
    }
}

void QueryConsolePanel::RenderToolbar() {
    if (ImGui::BeginMenuBar()) {
        // Run button
        if (ImGui::MenuItem(ICON_FA_PLAY " Run", "Ctrl+Enter")) {
            ExecuteQuery();
        }

        ImGui::Separator();

        // Examples dropdown
        if (ImGui::MenuItem(ICON_FA_BOOK " Examples")) {
            show_examples_ = true;
        }

        // History dropdown
        if (ImGui::MenuItem(ICON_FA_CLOCK_ROTATE_LEFT " History")) {
            show_history_ = true;
        }

        ImGui::Separator();

        // Clear button
        if (ImGui::MenuItem(ICON_FA_TRASH " Clear")) {
            std::memset(query_buffer_, 0, sizeof(query_buffer_));
            has_result_ = false;
        }

        // Copy results
        if (has_result_ && current_result_.success && !current_result_.rows.empty()) {
            if (ImGui::MenuItem(ICON_FA_COPY " Copy Results")) {
                CopyResultsToClipboard();
            }
        }

        ImGui::Separator();

        // Status
        if (has_result_) {
            if (current_result_.success) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f),
                                   "%zu rows in %.2f ms",
                                   current_result_.rows.size(),
                                   current_result_.executionTimeMs);
            } else {
                ImGui::TextColored(ImVec4(0.8f, 0.4f, 0.4f, 1.0f), "Error");
            }
        }

        ImGui::EndMenuBar();
    }
}

void QueryConsolePanel::RenderQueryInput() {
    // Multiline text input
    ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;

    ImGui::Text(ICON_FA_CODE " Query:");
    ImGui::SameLine();
    ImGui::TextDisabled("(Ctrl+Enter to execute)");

    float input_height = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeight();

    if (ImGui::InputTextMultiline("##query",
                                   query_buffer_,
                                   QUERY_BUFFER_SIZE,
                                   ImVec2(-1, input_height),
                                   flags)) {
        // Query changed
    }

    // Keyboard shortcuts
    if (ImGui::IsItemFocused()) {
        // Ctrl+Enter to execute
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
            ExecuteQuery();
        }

        // Up/Down for history navigation
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            NavigateHistoryUp();
        }
        if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            NavigateHistoryDown();
        }
    }
}

void QueryConsolePanel::RenderResults() {
    if (!has_result_) {
        ImGui::TextDisabled("Execute a query to see results");
        return;
    }

    if (!current_result_.success) {
        // Show error
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s %s", ICON_FA_TRIANGLE_EXCLAMATION, current_result_.error.c_str());
        ImGui::PopStyleColor();

        if (current_result_.errorLine > 0) {
            ImGui::TextDisabled("at line %d, column %d",
                               current_result_.errorLine,
                               current_result_.errorColumn);
        }
        return;
    }

    // Show statistics for modification queries
    if (current_result_.nodesCreated > 0 ||
        current_result_.nodesDeleted > 0 ||
        current_result_.linksCreated > 0 ||
        current_result_.linksDeleted > 0 ||
        current_result_.propertiesSet > 0) {

        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), ICON_FA_CHECK " Query executed successfully");

        if (current_result_.nodesCreated > 0) {
            ImGui::Text("  Nodes created: %d", current_result_.nodesCreated);
        }
        if (current_result_.nodesDeleted > 0) {
            ImGui::Text("  Nodes deleted: %d", current_result_.nodesDeleted);
        }
        if (current_result_.linksCreated > 0) {
            ImGui::Text("  Links created: %d", current_result_.linksCreated);
        }
        if (current_result_.linksDeleted > 0) {
            ImGui::Text("  Links deleted: %d", current_result_.linksDeleted);
        }
        if (current_result_.propertiesSet > 0) {
            ImGui::Text("  Properties set: %d", current_result_.propertiesSet);
        }
        return;
    }

    // Show results table
    if (current_result_.columns.empty() || current_result_.rows.empty()) {
        ImGui::TextDisabled("(no results)");
        return;
    }

    // Results table
    ImGuiTableFlags table_flags = ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_Resizable |
                                   ImGuiTableFlags_ScrollX |
                                   ImGuiTableFlags_ScrollY |
                                   ImGuiTableFlags_SizingStretchProp;

    if (ImGui::BeginTable("Results", static_cast<int>(current_result_.columns.size()), table_flags)) {
        // Headers
        for (const auto& col : current_result_.columns) {
            ImGui::TableSetupColumn(col.c_str());
        }
        ImGui::TableHeadersRow();

        // Rows
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(current_result_.rows.size()));

        while (clipper.Step()) {
            for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; row_idx++) {
                const auto& row = current_result_.rows[row_idx];

                ImGui::TableNextRow();

                // Row selection
                bool is_selected = (selected_row_ == row_idx);

                for (size_t col_idx = 0; col_idx < current_result_.columns.size(); col_idx++) {
                    ImGui::TableSetColumnIndex(static_cast<int>(col_idx));

                    const auto& col_name = current_result_.columns[col_idx];
                    auto value = row.get(col_name);
                    std::string text = FormatResultValue(value);

                    // Selectable for the first column
                    if (col_idx == 0) {
                        if (ImGui::Selectable(text.c_str(), is_selected,
                                              ImGuiSelectableFlags_SpanAllColumns)) {
                            selected_row_ = row_idx;
                        }
                    } else {
                        ImGui::TextUnformatted(text.c_str());
                    }
                }
            }
        }

        ImGui::EndTable();
    }

    // Row count
    ImGui::TextDisabled("%zu row(s)", current_result_.rows.size());
}

void QueryConsolePanel::RenderHistory() {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Query History", &show_history_)) {
        if (history_.empty()) {
            ImGui::TextDisabled("No query history");
        } else {
            if (ImGui::Button(ICON_FA_TRASH " Clear History")) {
                ClearHistory();
            }

            ImGui::Separator();

            for (int i = static_cast<int>(history_.size()) - 1; i >= 0; i--) {
                ImGui::PushID(i);

                // Truncate long queries for display
                std::string display = history_[i];
                if (display.length() > 60) {
                    display = display.substr(0, 57) + "...";
                }

                // Replace newlines
                std::replace(display.begin(), display.end(), '\n', ' ');

                if (ImGui::Selectable(display.c_str())) {
                    // Copy to query buffer
                    std::strncpy(query_buffer_, history_[i].c_str(), QUERY_BUFFER_SIZE - 1);
                    show_history_ = false;
                }

                // Tooltip with full query
                if (ImGui::IsItemHovered() && history_[i].length() > 60) {
                    ImGui::SetTooltip("%s", history_[i].c_str());
                }

                ImGui::PopID();
            }
        }
    }
    ImGui::End();
}

void QueryConsolePanel::RenderExamples() {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Example Queries", &show_examples_)) {
        ImGui::TextWrapped("Click an example to load it into the query editor.");
        ImGui::Separator();

        for (const auto& example : example_queries_) {
            ImGui::PushID(example.name);

            // Collapsing header for each example
            if (ImGui::CollapsingHeader(example.name)) {
                ImGui::Indent();

                // Description
                ImGui::TextDisabled("%s", example.description);
                ImGui::Spacing();

                // Query text
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.9f, 1.0f, 1.0f));
                ImGui::TextWrapped("%s", example.query);
                ImGui::PopStyleColor();

                ImGui::Spacing();

                // Use button
                if (ImGui::Button(ICON_FA_COPY " Use This Query")) {
                    std::strncpy(query_buffer_, example.query, QUERY_BUFFER_SIZE - 1);
                    show_examples_ = false;
                }

                ImGui::SameLine();

                // Run button
                if (ImGui::Button(ICON_FA_PLAY " Run")) {
                    ExecuteQuery(example.query);
                    show_examples_ = false;
                }

                ImGui::Unindent();
            }

            ImGui::PopID();
        }
    }
    ImGui::End();
}

void QueryConsolePanel::ExecuteQuery() {
    ExecuteQuery(query_buffer_);
}

void QueryConsolePanel::ExecuteQuery(const std::string& query) {
    if (!node_editor_) {
        current_result_ = query::QueryResult::makeError("No node editor connected");
        has_result_ = true;
        return;
    }

    if (query.empty()) {
        current_result_ = query::QueryResult::makeError("Empty query");
        has_result_ = true;
        return;
    }

    // Add to history
    AddToHistory(query);

    // Execute
    spdlog::info("Executing CyxQL query: {}", query);
    current_result_ = query::runQuery(*node_editor_, query);
    has_result_ = true;
    selected_row_ = -1;

    if (current_result_.success) {
        spdlog::info("Query returned {} rows in {:.2f}ms",
                     current_result_.rows.size(),
                     current_result_.executionTimeMs);
    } else {
        spdlog::warn("Query failed: {}", current_result_.error);
    }
}

void QueryConsolePanel::AddToHistory(const std::string& query) {
    // Don't add duplicates consecutively
    if (!history_.empty() && history_.back() == query) {
        return;
    }

    history_.push_back(query);

    // Trim if too long
    while (history_.size() > MAX_HISTORY) {
        history_.pop_front();
    }

    // Reset history index
    history_index_ = -1;
}

void QueryConsolePanel::NavigateHistoryUp() {
    if (history_.empty()) return;

    if (history_index_ < 0) {
        history_index_ = static_cast<int>(history_.size()) - 1;
    } else if (history_index_ > 0) {
        history_index_--;
    }

    std::strncpy(query_buffer_, history_[history_index_].c_str(), QUERY_BUFFER_SIZE - 1);
}

void QueryConsolePanel::NavigateHistoryDown() {
    if (history_.empty() || history_index_ < 0) return;

    if (history_index_ < static_cast<int>(history_.size()) - 1) {
        history_index_++;
        std::strncpy(query_buffer_, history_[history_index_].c_str(), QUERY_BUFFER_SIZE - 1);
    } else {
        // Clear buffer at end of history
        history_index_ = -1;
        std::memset(query_buffer_, 0, sizeof(query_buffer_));
    }
}

void QueryConsolePanel::ClearHistory() {
    history_.clear();
    history_index_ = -1;
}

std::string QueryConsolePanel::FormatResultValue(const query::ResultValue& value) {
    switch (value.type) {
        case query::ResultValue::Type::Null:
            return "(null)";

        case query::ResultValue::Type::Bool:
            return std::get<bool>(value.scalar) ? "true" : "false";

        case query::ResultValue::Type::Int:
            return std::to_string(std::get<int64_t>(value.scalar));

        case query::ResultValue::Type::Float: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4g", std::get<double>(value.scalar));
            return buf;
        }

        case query::ResultValue::Type::String:
            return std::get<std::string>(value.scalar);

        case query::ResultValue::Type::Node: {
            // Get node name if possible
            if (node_editor_) {
                for (const auto& node : node_editor_->GetNodes()) {
                    if (node.id == value.nodeId) {
                        return node.name;
                    }
                }
            }
            return "Node(" + std::to_string(value.nodeId) + ")";
        }

        case query::ResultValue::Type::Link:
            return "Link(" + std::to_string(value.linkId) + ")";

        case query::ResultValue::Type::Path: {
            std::string s = "[";
            for (size_t i = 0; i < value.pathNodeIds.size(); i++) {
                if (i > 0) s += "->";
                s += std::to_string(value.pathNodeIds[i]);
            }
            s += "]";
            return s;
        }

        case query::ResultValue::Type::List: {
            std::string s = "[";
            for (size_t i = 0; i < value.list.size(); i++) {
                if (i > 0) s += ", ";
                s += FormatResultValue(value.list[i]);
            }
            s += "]";
            return s;
        }

        case query::ResultValue::Type::Map: {
            std::string s = "{";
            bool first = true;
            for (const auto& [k, v] : value.map) {
                if (!first) s += ", ";
                s += k + ": " + FormatResultValue(v);
                first = false;
            }
            s += "}";
            return s;
        }

        default:
            return "?";
    }
}

void QueryConsolePanel::CopyResultsToClipboard() {
    if (!current_result_.success || current_result_.rows.empty()) return;

    std::string text;

    // Header
    for (size_t i = 0; i < current_result_.columns.size(); i++) {
        if (i > 0) text += "\t";
        text += current_result_.columns[i];
    }
    text += "\n";

    // Rows
    for (const auto& row : current_result_.rows) {
        for (size_t i = 0; i < current_result_.columns.size(); i++) {
            if (i > 0) text += "\t";
            text += FormatResultValue(row.get(current_result_.columns[i]));
        }
        text += "\n";
    }

    ImGui::SetClipboardText(text.c_str());
    spdlog::info("Copied {} rows to clipboard", current_result_.rows.size());
}

} // namespace cyxwiz
