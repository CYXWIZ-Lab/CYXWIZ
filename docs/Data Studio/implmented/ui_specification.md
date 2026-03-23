# Data Studio UI Specification

**Document Version:** 1.0
**Date:** 2026-03-19
**Purpose:** Comprehensive UI/UX specification for Data Studio Panel implementation
**Based on:** Use case mockups in `CyxWiz_DataStudio_UseCases.html`

---

## 1. Color Palette

### Primary Colors (from use case mockups)

```cpp
// Color constants for Data Studio UI
namespace DataStudio {
    namespace Colors {
        // Base colors
        constexpr ImVec4 Navy      = ImVec4(0.051f, 0.106f, 0.165f, 1.0f); // #0d1b2a
        constexpr ImVec4 Navy2     = ImVec4(0.102f, 0.184f, 0.271f, 1.0f); // #1a2f45
        constexpr ImVec4 Background = ImVec4(0.027f, 0.067f, 0.110f, 1.0f); // #07111c
        constexpr ImVec4 Surface   = ImVec4(0.059f, 0.122f, 0.180f, 1.0f); // #0f1f2e
        constexpr ImVec4 Card      = ImVec4(0.075f, 0.133f, 0.200f, 1.0f); // #132233
        constexpr ImVec4 Border    = ImVec4(0.118f, 0.204f, 0.314f, 1.0f); // #1e3450

        // Accent colors
        constexpr ImVec4 Blue      = ImVec4(0.055f, 0.498f, 0.761f, 1.0f); // #0e7fc2
        constexpr ImVec4 Teal      = ImVec4(0.000f, 0.722f, 0.663f, 1.0f); // #00b8a9
        constexpr ImVec4 Gold      = ImVec4(0.941f, 0.647f, 0.000f, 1.0f); // #f0a500
        constexpr ImVec4 Red       = ImVec4(0.902f, 0.224f, 0.275f, 1.0f); // #e63946
        constexpr ImVec4 Green     = ImVec4(0.176f, 0.776f, 0.325f, 1.0f); // #2dc653
        constexpr ImVec4 Purple    = ImVec4(0.486f, 0.302f, 1.000f, 1.0f); // #7c4dff

        // Text colors
        constexpr ImVec4 Text      = ImVec4(0.804f, 0.847f, 0.902f, 1.0f); // #cdd8e6
        constexpr ImVec4 Muted     = ImVec4(0.420f, 0.510f, 0.600f, 1.0f); // #6b8299
        constexpr ImVec4 White     = ImVec4(0.918f, 0.949f, 1.000f, 1.0f); // #eaf2ff
    }
}
```

### Node Category Colors

| Category | Color | Hex | ImVec4 |
|----------|-------|-----|--------|
| Input | Blue | #0e7fc2 | `(0.055f, 0.498f, 0.761f, 1.0f)` |
| Tabular/Clean | Gold | #f0a500 | `(0.941f, 0.647f, 0.000f, 1.0f)` |
| Text | Green | #2dc653 | `(0.176f, 0.776f, 0.325f, 1.0f)` |
| Time-Series | Purple | #7c4dff | `(0.486f, 0.302f, 1.000f, 1.0f)` |
| Feature Eng. | Cyan/Teal | #00b8a9 | `(0.000f, 0.722f, 0.663f, 1.0f)` |
| Analyze | Teal | #00b8a9 | `(0.000f, 0.722f, 0.663f, 1.0f)` |
| Output | Green | #2dc653 | `(0.176f, 0.776f, 0.325f, 1.0f)` |

---

## 2. DataStudioPanel Layout

### 2.1 Overall Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Studio                                            [X]      │ ← Window title bar
├─────────────────────────────────────────────────────────────────┤
│  [Pipeline] [Analysis] [Visualization] [Query]                  │ ← Tab bar (sticky)
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Content Area - varies by active tab]                          │
│                                                                  │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tab Bar Styling

```cpp
// Tab bar rendering
void DataStudioPanel::RenderTabBar() {
    ImGui::PushStyleColor(ImGuiCol_Tab, Colors::Surface);
    ImGui::PushStyleColor(ImGuiCol_TabActive, Colors::Card);
    ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.1f, 0.2f, 0.3f, 1.0f));

    if (ImGui::BeginTabBar("DataStudioTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("🔧 Pipeline")) {
            active_tab_ = Tab::Pipeline;
            RenderPipelineTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("📊 Analysis")) {
            active_tab_ = Tab::Analysis;
            RenderAnalysisTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("📈 Visualization")) {
            active_tab_ = Tab::Visualization;
            RenderVisualizationTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("🦆 Query")) {
            active_tab_ = Tab::Query;
            RenderQueryTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::PopStyleColor(3);
}
```

**Tab Design:**
- Font: DM Sans 14px
- Padding: 16px horizontal, 12px vertical
- Active tab: Underline with Teal color (3px thick)
- Hover: Text color changes from Muted to White
- Icons: FontAwesome 16px

---

## 3. Pipeline Canvas (Primary View)

### 3.1 Toolbar

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE CANVAS — cleaning_pipeline.json                       │ ← Title (JetBrains Mono 11px, Muted)
│  [▶ Run Pipeline] [💾 Save] [{ } Code] [Clear]                │ ← Buttons
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
void PipelineCanvas::ShowToolbar() {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::Navy2);
    ImGui::BeginChild("Toolbar", ImVec2(0, 44), true, ImGuiWindowFlags_NoScrollbar);

    // Title
    ImGui::PushFont(JetBrainsMonoFont);
    ImGui::TextColored(Colors::Muted, "PIPELINE CANVAS — %s", pipeline_name_.c_str());
    ImGui::PopFont();

    ImGui::SameLine(ImGui::GetWindowWidth() - 400); // Right-align buttons

    // Run button (highlighted)
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.722f, 0.663f, 0.15f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.722f, 0.663f, 0.25f));
    ImGui::PushStyleColor(ImGuiCol_Border, Colors::Teal);
    if (ImGui::Button("▶ Run Pipeline")) {
        RunPipeline();
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    if (ImGui::Button("💾 Save")) {
        SavePipeline();
    }

    ImGui::SameLine();
    if (ImGui::Button("{ } Code")) {
        ShowGeneratedCode();
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        ClearPipeline();
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}
```

**Button Styling:**
- Background: `rgba(255,255,255,0.06)` (normal), Teal (Run button)
- Border: `Colors::Border` (1px)
- Border radius: 6px
- Padding: 4px horizontal, 12px vertical
- Font: JetBrains Mono 11px, semi-bold

### 3.2 Canvas Background

**Dot Grid Pattern:**
```cpp
void PipelineCanvas::RenderCanvasBackground() {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();

    // Dot grid (24px spacing)
    const float grid_size = 24.0f;
    const ImU32 dot_color = ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.03f));

    for (float x = fmodf(scroll_offset_.x, grid_size); x < canvas_size.x; x += grid_size) {
        for (float y = fmodf(scroll_offset_.y, grid_size); y < canvas_size.y; y += grid_size) {
            draw_list->AddCircleFilled(
                ImVec2(canvas_pos.x + x, canvas_pos.y + y),
                1.0f,  // radius
                dot_color
            );
        }
    }
}
```

### 3.3 Node Visual Design

**Node Structure:**
```
┌─────────────────────┐
│ INPUT               │ ← Category badge (color-coded, 9px uppercase)
├─────────────────────┤
│  📁 File Input      │ ← Icon + name (Syne 12px bold)
├─────────────────────┤
│ dataset: props.csv  │ ← Parameter (11px)
│ ✓ 80,412 rows       │ ← Status (JetBrains Mono 10px)
└─────────────────────┘
● Input port (left)    ● Output port (right)
```

**Implementation:**
```cpp
void PipelineCanvas::RenderNode(const DataPipelineNode& node) {
    ImNodes::BeginNode(node.id);

    // Top colored bar (category indicator)
    ImVec4 category_color = GetCategoryColor(node.type);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
    draw_list->AddRectFilled(
        node_pos,
        ImVec2(node_pos.x + 150, node_pos.y + 2),
        ImGui::ColorConvertFloat4ToU32(category_color)
    );

    // Category label
    ImGui::PushFont(JetBrainsMonoFont);
    ImGui::PushStyleColor(ImGuiCol_Text, category_color);
    ImGui::TextUnformatted(GetCategoryName(node.type).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Separator();

    // Node name with icon
    ImGui::PushFont(SyneBoldFont);
    ImGui::Text("%s %s", GetNodeIcon(node.type).c_str(), node.name.c_str());
    ImGui::PopFont();

    ImGui::Separator();

    // Status
    ImGui::PushFont(JetBrainsMonoFont);
    if (node.executed && !node.has_error) {
        ImGui::TextColored(Colors::Green, "✓ %s", node.status_message.c_str());
    } else if (node.has_error) {
        ImGui::TextColored(Colors::Red, "✗ %s", node.error_message.c_str());
    } else if (is_running_ && current_executing_node_ == node.id) {
        ImGui::TextColored(Colors::Teal, "● Running...");
    } else {
        ImGui::TextColored(Colors::Muted, "⏳ Waiting...");
    }
    ImGui::PopFont();

    // Input/output pins
    ImNodes::BeginInputAttribute(node.inputs[0].id);
    ImGui::Text("●"); // Input port
    ImNodes::EndInputAttribute();

    ImNodes::BeginOutputAttribute(node.outputs[0].id);
    ImGui::SameLine(130);
    ImGui::Text("●"); // Output port
    ImNodes::EndOutputAttribute();

    ImNodes::EndNode();
}
```

**Node Dimensions:**
- Min width: 120px
- Max width: 150px
- Padding: 12px horizontal, 16px vertical
- Border: 1.5px, `Colors::Border` (normal), category color (active)
- Border radius: 10px
- Hover effect: Border → Teal, transform translateY(-2px)

**Node Status Colors:**
- ⚪ **Gray** (pending): `Colors::Border`
- 🔵 **Blue** (running): `Colors::Teal` with pulsing shadow
- 🟢 **Green** (success): `Colors::Green`
- 🔴 **Red** (error): `Colors::Red`
- 🟡 **Yellow** (warning): `Colors::Gold`

### 3.4 Node Connections (Links)

**Arrow Style:**
```cpp
void PipelineCanvas::RenderLinks() {
    for (const auto& link : links_) {
        // Gradient from border color to teal
        ImU32 color_start = ImGui::ColorConvertFloat4ToU32(Colors::Border);
        ImU32 color_end = ImGui::ColorConvertFloat4ToU32(Colors::Teal);

        ImNodes::PushColorStyle(ImNodesCol_Link, color_end);
        ImNodes::Link(link.id, link.from_pin, link.to_pin);
        ImNodes::PopColorStyle();

        // Draw arrow at end
        ImVec2 link_end = GetLinkEndPosition(link);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddTriangleFilled(
            ImVec2(link_end.x - 6, link_end.y - 3),
            ImVec2(link_end.x - 6, link_end.y + 3),
            ImVec2(link_end.x, link_end.y),
            ImGui::ColorConvertFloat4ToU32(Colors::Teal)
        );
    }
}
```

---

## 4. Execution Log (Terminal-Style)

### 4.1 Terminal Design

```
┌─────────────────────────────────────────────────────────────────┐
│ 🔴 🟡 🟢  Pipeline Execution Log                               │ ← Title bar with macOS-style dots
├─────────────────────────────────────────────────────────────────┤
│  [00:00.00] CloudImport: loading sensor_logs.parquet...        │
│  [00:02.84] CloudImport: ✓ 2,592,000 rows loaded               │
│  [00:04.91] TSWindow: windowing per machine_id (500 machines)  │
│  [00:19.44] TSWindow: ✓ 86,080 windows created [86080, 60, 6]  │
│  [00:28.67] ✓ PIPELINE COMPLETE — 28.67s total                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
class ExecutionLog {
public:
    struct LogEntry {
        float timestamp;
        std::string message;
        LogLevel level; // Info, Success, Error
    };

    void Render() {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.012f, 0.051f, 0.082f, 1.0f)); // #030d15
        ImGui::BeginChild("ExecutionLog", ImVec2(0, 200), true);

        // Title bar with dots
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        draw_list->AddCircleFilled(ImVec2(pos.x + 10, pos.y + 12), 5, IM_COL32(255, 95, 87, 255));
        draw_list->AddCircleFilled(ImVec2(pos.x + 30, pos.y + 12), 5, IM_COL32(254, 188, 46, 255));
        draw_list->AddCircleFilled(ImVec2(pos.x + 50, pos.y + 12), 5, IM_COL32(40, 200, 64, 255));

        ImGui::SetCursorPosX(70);
        ImGui::PushFont(JetBrainsMonoFont);
        ImGui::TextColored(Colors::Muted, "Pipeline Execution Log");
        ImGui::PopFont();

        ImGui::Separator();
        ImGui::Spacing();

        // Log entries
        ImGui::PushFont(JetBrainsMonoFont);
        for (const auto& entry : log_entries_) {
            ImVec4 color = Colors::Text;
            if (entry.level == LogLevel::Success) color = Colors::Green;
            else if (entry.level == LogLevel::Error) color = Colors::Red;

            ImGui::TextColored(Colors::Teal, "[%06.2f]", entry.timestamp);
            ImGui::SameLine();
            ImGui::TextColored(color, "%s", entry.message.c_str());
        }
        ImGui::PopFont();

        // Auto-scroll to bottom
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
            ImGui::SetScrollHereY(1.0f);

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    void AddEntry(float timestamp, const std::string& message, LogLevel level = LogLevel::Info) {
        log_entries_.push_back({timestamp, message, level});
    }

private:
    std::vector<LogEntry> log_entries_;
};
```

**Log Entry Format:**
- Timestamp: `[00:00.00]` in Teal
- Message: Text in White (normal), Green (success), Red (error)
- Font: JetBrains Mono 12px
- Line height: 2.0 (double-spaced for readability)

---

## 5. Statistics Cards

### 5.1 Card Design

```
┌─────────────────────┐
│ TOTAL ROWS          │ ← Label (JetBrains Mono 10px, uppercase, Muted)
│ 80,412              │ ← Value (Syne 28px bold, White)
│ Raw, unprocessed    │ ← Change indicator (12px, color-coded)
└─────────────────────┘
```

**Implementation:**
```cpp
void DataStudioPanel::RenderStatCard(const std::string& label,
                                     const std::string& value,
                                     const std::string& change,
                                     StatChangeType change_type) {
    ImGui::BeginChild(label.c_str(), ImVec2(200, 100), true);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::Card);
    ImGui::PushStyleColor(ImGuiCol_Border, Colors::Border);

    // Label
    ImGui::PushFont(JetBrainsMonoFont);
    ImGui::TextColored(Colors::Muted, "%s", label.c_str());
    ImGui::PopFont();

    ImGui::Spacing();

    // Value
    ImGui::PushFont(SyneBoldFont);
    ImVec4 value_color = Colors::White;
    if (change_type == StatChangeType::Good) value_color = Colors::Green;
    else if (change_type == StatChangeType::Bad) value_color = Colors::Red;
    else if (change_type == StatChangeType::Warn) value_color = Colors::Gold;

    ImGui::TextColored(value_color, "%s", value.c_str());
    ImGui::PopFont();

    // Change indicator
    ImVec4 change_color = Colors::Muted;
    const char* icon = "";
    if (change_type == StatChangeType::Good) {
        change_color = Colors::Green;
        icon = "↓";
    } else if (change_type == StatChangeType::Bad) {
        change_color = Colors::Red;
        icon = "↑";
    } else if (change_type == StatChangeType::Warn) {
        change_color = Colors::Gold;
        icon = "";
    }

    ImGui::TextColored(change_color, "%s %s", icon, change.c_str());

    ImGui::PopStyleColor(2);
    ImGui::EndChild();
}
```

**Card Grid Layout:**
```cpp
// 4 cards in a row
ImGui::Columns(4, nullptr, false);
RenderStatCard("Total Rows", "74,908", "6.9% from raw", StatChangeType::Good);
ImGui::NextColumn();
RenderStatCard("Null Rate", "0.1%", "from 5.2%", StatChangeType::Good);
ImGui::NextColumn();
RenderStatCard("Feature Columns", "31", "+13 from one-hot", StatChangeType::Warn);
ImGui::NextColumn();
RenderStatCard("Pipeline Time", "6.2s", "Under 8s target ✓", StatChangeType::Good);
ImGui::Columns(1);
```

---

## 6. Analysis Tab Components

### 6.1 Bar Chart (Null Rate Visualization)

```cpp
class BarChart {
public:
    void Render(const std::vector<BarData>& data) {
        for (const auto& bar : data) {
            // Label
            ImGui::PushFont(JetBrainsMonoFont);
            ImGui::TextColored(Colors::Muted, "%s", bar.label.c_str());
            ImGui::SameLine(120);

            // Progress bar track
            ImGui::PushStyleColor(ImGuiCol_FrameBg, Colors::Border);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, GetBarColor(bar.value));
            ImGui::ProgressBar(bar.value, ImVec2(300, 8), "");
            ImGui::PopStyleColor(2);

            ImGui::SameLine();
            ImGui::TextColored(Colors::Muted, "%.1f%%", bar.value * 100);
            ImGui::PopFont();
        }
    }

private:
    ImVec4 GetBarColor(float value) {
        if (value < 0.01f) return Colors::Green;
        if (value < 0.05f) return Colors::Gold;
        return Colors::Red;
    }
};
```

### 6.2 Distribution Histogram (ImPlot)

```cpp
void DataStudioAnalyzer::RenderDistribution() {
    if (ImPlot::BeginPlot("Price Distribution", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("Price Range", "Frequency", 0, 0);

        // Bar chart data
        std::vector<double> bins = {85000, 200000, 350000, 500000, 650000,
                                   800000, 1000000, 1200000, 1500000, 1950000};
        std::vector<double> counts = {120, 850, 4200, 18400, 22100,
                                     15800, 9200, 3400, 720, 118};

        ImPlot::SetNextFillStyle(Colors::Teal);
        ImPlot::PlotBars("Properties", bins.data(), counts.data(), bins.size(), 50000);

        ImPlot::EndPlot();
    }
}
```

### 6.3 Finding Cards

```
┌─────────────────────────────────────────────────────────────────┐
│ 🔴 CRITICAL: Machine M017: 847 windows with temp > 130°C       │ ← Red left border
├─────────────────────────────────────────────────────────────────┤
│ 🟡 WARNING: 3.2% of windows contain 3-sigma exceedance         │ ← Gold left border
├─────────────────────────────────────────────────────────────────┤
│ 🔵 INFO: Normal operating range: 60-85°C, 0.5-2.1 m/s²         │ ← Blue left border
├─────────────────────────────────────────────────────────────────┤
│ ✅ SUCCESS: Dataset suitable for unsupervised training          │ ← Green left border
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
enum class FindingLevel { Critical, Warning, Info, Success };

void DataStudioAnalyzer::RenderFinding(const std::string& message, FindingLevel level) {
    ImVec4 border_color = Colors::Blue;
    const char* icon = "🔵";

    switch (level) {
        case FindingLevel::Critical:
            border_color = Colors::Red;
            icon = "🔴";
            break;
        case FindingLevel::Warning:
            border_color = Colors::Gold;
            icon = "🟡";
            break;
        case FindingLevel::Success:
            border_color = Colors::Green;
            icon = "✅";
            break;
    }

    // Draw left border
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    draw_list->AddRectFilled(
        pos,
        ImVec2(pos.x + 3, pos.y + 50),
        ImGui::ColorConvertFloat4ToU32(border_color)
    );

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 15);

    ImGui::BeginChild(("finding_" + message).c_str(), ImVec2(0, 50), true);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, Colors::Surface);
    ImGui::PushStyleColor(ImGuiCol_Border, Colors::Border);

    ImGui::Text("%s", icon);
    ImGui::SameLine();
    ImGui::TextWrapped("%s", message.c_str());

    ImGui::PopStyleColor(2);
    ImGui::EndChild();
}
```

---

## 7. DuckDB Query Editor

### 7.1 Query Input Area

```
┌─────────────────────────────────────────────────────────────────┐
│ 🦆 DuckDB Query Editor                                         │
├─────────────────────────────────────────────────────────────────┤
│ -- Run directly in the Query Editor tab                        │
│ SELECT city, COUNT(*) AS listings,                             │
│        ROUND(AVG(price), 0) AS avg_price                        │
│ FROM properties_cleaned                                         │
│ GROUP BY city                                                   │
│ ORDER BY listings DESC                                          │
│ LIMIT 10;                                                       │
│                                                                  │
│ [Execute Query (Ctrl+Enter)]  [Clear]  [Save Query]            │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
void DataStudioQueryEditor::RenderQueryInput() {
    ImGui::Text("🦆 DuckDB Query Editor");
    ImGui::Separator();

    // Multi-line text input with syntax highlighting (optional)
    ImGui::PushFont(JetBrainsMonoFont);
    ImGui::InputTextMultiline("##query", query_buffer_, sizeof(query_buffer_),
                               ImVec2(-1, 200),
                               ImGuiInputTextFlags_AllowTabInput);
    ImGui::PopFont();

    // Buttons
    if (ImGui::Button("Execute Query (Ctrl+Enter)") ||
        (ImGui::IsKeyPressed(ImGuiKey_Enter) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))) {
        ExecuteQuery();
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        query_buffer_[0] = '\0';
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Query")) {
        SaveQuery();
    }
}
```

### 7.2 Query Results Table

```cpp
void DataStudioQueryEditor::RenderResults() {
    if (!has_results_) return;

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Query Result — %.2fs · %zu rows", query_time_ms_ / 1000.0f, result_count_);

    if (ImGui::BeginTable("QueryResults", result_columns_.size(),
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable)) {

        // Headers
        for (const auto& col : result_columns_) {
            ImGui::TableSetupColumn(col.c_str());
        }
        ImGui::TableHeadersRow();

        // Rows
        ImGui::PushFont(JetBrainsMonoFont);
        for (const auto& row : result_rows_) {
            ImGui::TableNextRow();
            for (size_t i = 0; i < row.size(); ++i) {
                ImGui::TableSetColumnIndex(i);
                ImGui::TextUnformatted(row[i].c_str());
            }
        }
        ImGui::PopFont();

        ImGui::EndTable();
    }
}
```

---

## 8. Deploy Box

### 8.1 Visual Design

```
┌─────────────────────────────────────────────────────────────────┐
│ 🚀  Deploy to Node Editor                      [→ Send] button  │
│     Clicking this sends the cleaned dataset directly to the      │
│     Model Builder's DataInput node. Shape and types are          │
│     automatically populated.                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
void DataStudioPanel::RenderDeployBox() {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.722f, 0.663f, 0.08f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.722f, 0.663f, 0.3f));

    ImGui::BeginChild("DeployBox", ImVec2(0, 100), true);

    // Icon
    ImGui::SetCursorPosX(20);
    ImGui::SetCursorPosY(30);
    ImGui::Text("🚀");
    ImGui::SameLine();

    // Title and description
    ImGui::BeginGroup();
    ImGui::PushFont(SyneBoldFont);
    ImGui::Text("Deploy to Node Editor");
    ImGui::PopFont();

    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 500);
    ImGui::TextColored(Colors::Muted,
        "Clicking this sends the cleaned dataset directly to the Model Builder's "
        "DataInput node. Shape and types are automatically populated.");
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();

    // Button (right-aligned)
    ImGui::SameLine(ImGui::GetWindowWidth() - 200);
    ImGui::SetCursorPosY(30);

    ImGui::PushStyleColor(ImGuiCol_Button, Colors::Teal);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.9f, 0.8f, 1.0f));
    ImGui::PushFont(SyneBoldFont);
    if (ImGui::Button("→ Send to Node Editor", ImVec2(180, 40))) {
        OnDeployToNodeEditor();
    }
    ImGui::PopFont();
    ImGui::PopStyleColor(2);

    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}
```

---

## 9. Context Menu (Right-Click on Canvas)

### 9.1 Node Category Structure

```
Add Node ▶
  Input ▶
    📁 File Input
    ☁️ Cloud Input
    🗄️ SQL Input
    🌐 API Input
  Tabular ▶
    🧹 Remove Duplicates
    🔧 Fill Missing
    🎯 Filter Rows
    🔄 Type Cast
    ➕ Select Columns
    ➖ Drop Columns
    ✏️ Rename Columns
    ⬆️ Sort Rows
    🔀 Merge Datasets
  Text ▶
    🧼 Text Clean
    ✂️ Text Tokenize
    📝 Text Normalize
    🔢 Text Vectorize
  Time-Series ▶
    ⏱️ TS Window
    📊 TS Features
    ✂️ TS Split
    🔄 TS Resample
  Feature Engineering ▶
    📏 Standard Scale
    📐 Min-Max Scale
    💪 Robust Scale
    1️⃣ One-Hot Encode
    #️⃣ Label Encode
    📦 Bin Column
    ✖️ Polynomial Features
    🎯 PCA
    📉 Truncated SVD
  Analyze ▶
    📊 Descriptive Stats
    🔗 Correlation
    ❓ Missing Value Report
    🔍 Outlier Detection
    ✂️ Train/Val Split
  Output ▶
    💾 Save Dataset
    📤 Export File
    🚀 Deploy to Node Editor
─────────────────
📋 Paste
🔲 Select All
🗑️ Clear Pipeline
```

**Implementation:**
```cpp
void PipelineCanvas::ShowContextMenu() {
    if (ImGui::BeginPopupContextItem("CanvasContextMenu")) {
        if (ImGui::BeginMenu("Add Node")) {
            if (ImGui::BeginMenu("Input")) {
                if (ImGui::MenuItem("📁 File Input")) AddNode(DataNodeType::FileInput);
                if (ImGui::MenuItem("☁️ Cloud Input")) AddNode(DataNodeType::CloudInput);
                if (ImGui::MenuItem("🗄️ SQL Input")) AddNode(DataNodeType::SQLInput);
                if (ImGui::MenuItem("🌐 API Input")) AddNode(DataNodeType::APIInput);
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Tabular")) {
                if (ImGui::MenuItem("🧹 Remove Duplicates")) AddNode(DataNodeType::RemoveDuplicates);
                if (ImGui::MenuItem("🔧 Fill Missing")) AddNode(DataNodeType::FillMissing);
                if (ImGui::MenuItem("🎯 Filter Rows")) AddNode(DataNodeType::FilterRows);
                // ... more items
                ImGui::EndMenu();
            }

            // ... other categories

            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::MenuItem("📋 Paste")) PasteNodes();
        if (ImGui::MenuItem("🔲 Select All")) SelectAllNodes();
        if (ImGui::MenuItem("🗑️ Clear Pipeline")) ClearPipeline();

        ImGui::EndPopup();
    }
}
```

---

## 10. Font Configuration

### 10.1 Required Fonts

```cpp
// In Application initialization
ImGuiIO& io = ImGui::GetIO();

// Regular fonts
io.Fonts->AddFontFromFileTTF("resources/fonts/DMSans-Regular.ttf", 15.0f);
io.Fonts->AddFontFromFileTTF("resources/fonts/DMSans-Medium.ttf", 15.0f);

// Monospace font (JetBrains Mono)
ImFont* jetbrains_mono = io.Fonts->AddFontFromFileTTF(
    "resources/fonts/JetBrainsMono-Regular.ttf", 12.0f);

// Bold headings (Syne)
ImFont* syne_bold = io.Fonts->AddFontFromFileTTF(
    "resources/fonts/Syne-Bold.ttf", 20.0f);
ImFont* syne_extrabold = io.Fonts->AddFontFromFileTTF(
    "resources/fonts/Syne-ExtraBold.ttf", 28.0f);

io.Fonts->Build();
```

### 10.2 Font Usage Map

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Node name | Syne | 12px | Bold |
| Node category | JetBrains Mono | 9px | Semi-bold, uppercase |
| Node status | JetBrains Mono | 10px | Regular |
| Toolbar title | JetBrains Mono | 11px | Semi-bold |
| Button text | JetBrains Mono | 11px | Semi-bold |
| Tab labels | DM Sans | 14px | Medium |
| Stat card label | JetBrains Mono | 10px | Semi-bold, uppercase |
| Stat card value | Syne | 28px | Extra-bold |
| Execution log | JetBrains Mono | 12px | Regular |
| Code block | JetBrains Mono | 12px | Regular |
| Table headers | JetBrains Mono | 10px | Semi-bold, uppercase |
| Table data | JetBrains Mono | 11px | Regular |

---

## 11. Animation & Interaction

### 11.1 Node Hover Effect

```cpp
// In RenderNode()
if (ImNodes::IsNodeHovered(node.id)) {
    // Translate up by 2px
    ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node.id);
    ImNodes::SetNodeScreenSpacePos(node.id, ImVec2(node_pos.x, node_pos.y - 2));

    // Change border to Teal
    ImNodes::PushColorStyle(ImNodesCol_NodeOutline, Colors::Teal);
}
```

### 11.2 Running Node Pulse Animation

```cpp
void PipelineCanvas::RenderRunningNodePulse(int node_id) {
    if (current_executing_node_ != node_id) return;

    float time = ImGui::GetTime();
    float pulse = 0.5f + 0.5f * sinf(time * 4.0f); // 4 Hz pulse

    ImVec4 glow_color = ImVec4(
        Colors::Teal.x,
        Colors::Teal.y,
        Colors::Teal.z,
        0.4f * pulse
    );

    // Draw glow around node
    ImVec2 node_pos = ImNodes::GetNodeScreenSpacePos(node_id);
    ImVec2 node_size = ImNodes::GetNodeDimensions(node_id);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRect(
        ImVec2(node_pos.x - 4, node_pos.y - 4),
        ImVec2(node_pos.x + node_size.x + 4, node_pos.y + node_size.y + 4),
        ImGui::ColorConvertFloat4ToU32(glow_color),
        10.0f,  // rounding
        0,      // flags
        3.0f    // thickness
    );
}
```

### 11.3 Progress Bar (Pipeline Execution)

```cpp
void PipelineCanvas::ShowExecutionProgress() {
    if (!is_running_) return;

    float progress = (float)completed_nodes_ / (float)total_nodes_;

    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, 40),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::Begin("PipelineProgress", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Running Pipeline... %d/%d nodes", completed_nodes_, total_nodes_);
    ImGui::ProgressBar(progress, ImVec2(400, 0));

    if (ImGui::Button("Stop")) {
        StopPipeline();
    }

    ImGui::End();
}
```

---

## 12. Responsive Behavior

### 12.1 Window Minimum Size

```cpp
// In DataStudioPanel constructor
min_window_size_ = ImVec2(800, 600);
```

### 12.2 Canvas Scrolling

```cpp
// Enable infinite canvas scrolling
ImGui::BeginChild("PipelineCanvas", ImVec2(0, 0), true,
                  ImGuiWindowFlags_HorizontalScrollbar);

// Middle mouse button drag to pan
if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
    ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
    scroll_offset_.x += delta.x;
    scroll_offset_.y += delta.y;
    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
}

// Mouse wheel zoom
if (ImGui::GetIO().MouseWheel != 0.0f && ImGui::IsWindowHovered()) {
    float zoom_delta = ImGui::GetIO().MouseWheel * 0.1f;
    canvas_zoom_ = std::clamp(canvas_zoom_ + zoom_delta, 0.5f, 2.0f);
}

ImGui::EndChild();
```

---

## 13. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Execute query (in Query tab) |
| `Ctrl + S` | Save pipeline |
| `Ctrl + N` | Clear pipeline (with confirmation) |
| `Ctrl + R` | Run pipeline |
| `Delete` | Delete selected nodes |
| `Ctrl + C` | Copy selected nodes |
| `Ctrl + V` | Paste nodes |
| `Ctrl + Z` | Undo (TODO: implement undo stack) |
| `Ctrl + Shift + Z` | Redo |
| `F` | Frame selected nodes (zoom to fit) |
| `A` | Select all nodes |
| `Escape` | Deselect all nodes |

**Implementation:**
```cpp
void PipelineCanvas::HandleKeyboardShortcuts() {
    if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) {
        if (ImGui::IsKeyPressed(ImGuiKey_S)) SavePipeline();
        if (ImGui::IsKeyPressed(ImGuiKey_R)) RunPipeline();
        if (ImGui::IsKeyPressed(ImGuiKey_C)) CopySelectedNodes();
        if (ImGui::IsKeyPressed(ImGuiKey_V)) PasteNodes();
        if (ImGui::IsKeyPressed(ImGuiKey_Z)) Undo();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        DeleteSelectedNodes();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_F)) {
        FrameSelectedNodes();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_A)) {
        SelectAllNodes();
    }

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        DeselectAllNodes();
    }
}
```

---

## 14. Summary

This UI specification provides:

1. ✅ **Complete color palette** with exact RGB/hex values
2. ✅ **Layout specifications** for all major components
3. ✅ **ImGui implementation examples** for each UI element
4. ✅ **Font configuration** with size/weight mappings
5. ✅ **Animation and interaction patterns**
6. ✅ **Keyboard shortcut mappings**

**Next Steps:**
- Implement Phase 0 (DataRegistry Arrow extension) as foundation
- Begin UI implementation starting with PipelineCanvas
- Test visual fidelity against use case mockups
- Iterate on node styling and canvas interactions

**Implementation Priority:**
1. PipelineCanvas with toolbar and nodes (Week 1-2)
2. Execution log and progress tracking (Week 2)
3. Analysis tab components (Week 3)
4. Query editor (Week 4)
5. Deploy box and Node Editor integration (Week 5)

---

**Document Version:** 1.0
**Last Updated:** 2026-03-19
**Status:** ✅ Ready for Implementation
