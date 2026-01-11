# CyxWiz Script Editor Architecture

## Overview

A professional, Jupyter-like script editor for CyxWiz with live code execution, cell-based editing, inline plotting, markdown support, debugging capabilities, and seamless integration with the ML pipeline.

## Goals

1. **Professional Code Editing** - Syntax highlighting, auto-completion, error indicators
2. **Cell-Based Interface** - Cell-based editing with markdown and code cells (Jupyter-style)
3. **Inline Plotting** - Display matplotlib/implot charts directly in the editor
4. **Robust Debugger** - Breakpoints, step-through, variable inspection
5. **Live Execution** - Run cells individually or entire scripts
6. **Rich Output** - Support text, tables, plots, images inline
7. **Integration** - Connect with Node Editor, Training Pipeline, Data Registry

---

## Current Implementation Analysis

### Existing ScriptEditorPanel (`script_editor.h/.cpp`)

The current implementation already provides:

| Feature | Status | Location |
|---------|--------|----------|
| Multi-tab editing | Done | `ScriptEditorPanel::tabs_` |
| Python syntax highlighting | Done | ImGuiColorTextEdit integration |
| Section execution (`%%`) | Done | `RunCurrentSection()` |
| Multiple themes | Done | Monokai, Dracula, OneDark, GitHub, etc. |
| Code minimap | Done | `RenderMinimap()` |
| Async file loading | Done | `OpenFileAsync()` |
| Python engine integration | Done | `ScriptingEngine` pointer |
| Sandbox security | Done | `Security` menu |
| File format | Done | `.cyx` files |

### What Needs Enhancement

| Feature | Status | Priority |
|---------|--------|----------|
| Cell-based editing | Missing | High |
| Inline output display | Missing | High |
| Inline plotting (matplotlib) | Missing | High |
| Markdown cells | Missing | Medium |
| Debugger integration | Missing | Medium |
| Variable explorer | Missing | Medium |
| Auto-completion (Jedi) | Missing | Low |

---

## Architecture Diagram

```
+------------------------------------------------------------------+
|                    CyxWiz Script Editor (Enhanced)                |
+------------------------------------------------------------------+
|  +------------------+  +------------------+  +------------------+ |
|  |   File Tabs      |  |   Cell Actions   |  |   Run Controls   | |
|  | script1.cyx      |  | Add/Delete/Move  |  | Run/Stop/Debug   | |
|  +------------------+  +------------------+  +------------------+ |
+------------------------------------------------------------------+
|                                                                    |
|  +--------------------------------------------------------------+ |
|  |                    Cell-Based Editor View                     | |
|  |  +--------------------------------------------------------+  | |
|  |  | [%%markdown]                                In [*]:    |  | |
|  |  | # Training Script                                      |  | |
|  |  | This script trains a neural network...                 |  | |
|  |  +--------------------------------------------------------+  | |
|  |  | [%%code] [Run Cell] [Debug]                In [1]:     |  | |
|  |  | import cyxwiz as cyx                                   |  | |
|  |  | model = cyx.get_current_model()                        |  | |
|  |  +--------------------------------------------------------+  | |
|  |  | [Output]                                   Out [1]:    |  | |
|  |  | Model loaded: Sequential(3 layers)                     |  | |
|  |  +--------------------------------------------------------+  | |
|  |  | [%%code] [Run Cell] [Debug]                In [2]:     |  | |
|  |  | import matplotlib.pyplot as plt                        |  | |
|  |  | plt.plot(loss_history)                                 |  | |
|  |  | plt.show()                                             |  | |
|  |  +--------------------------------------------------------+  | |
|  |  | [Plot Output]                              Out [2]:    |  | |
|  |  | +--------------------------------------------------+   |  | |
|  |  | |          [Inline Matplotlib Plot]                |   |  | |
|  |  | +--------------------------------------------------+   |  | |
|  |  +--------------------------------------------------------+  | |
|  +--------------------------------------------------------------+ |
|                                                                    |
+------------------------------------------------------------------+
|  +------------------------+  +--------------------------------+   |
|  |   Variable Explorer    |  |   Debug Panel                  |   |
|  | - model: Sequential    |  | Breakpoints | Call Stack       |   |
|  | - loss: 0.0234         |  | Line 5      | train()          |   |
|  +------------------------+  +--------------------------------+   |
+------------------------------------------------------------------+
```

---

## File Format

### Enhanced .cyx Format

The `.cyx` format is extended to support cells while remaining backward-compatible:

```python
# CyxWiz Script v0.3.0
# Cell markers: %%code, %%markdown, %%raw

%%markdown
# Training Script

This notebook demonstrates training a neural network.

%%code
import cyxwiz as cyx
import matplotlib.pyplot as plt

model = cyx.get_current_model()
print(f"Model: {model}")

%%code
# Train the model
history = model.train(epochs=10)

# Plot results inline
plt.figure(figsize=(8, 4))
plt.plot(history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()  # This will render inline

%%markdown
## Results

The model achieved a final loss of **0.023**.
```

### Cell Markers

| Marker | Description |
|--------|-------------|
| `%%code` | Python code cell (executable) |
| `%%markdown` | Markdown documentation cell |
| `%%raw` | Raw text cell (no execution/rendering) |
| `%%` | Legacy section marker (treated as code) |

---

## Implementation Plan

### Phase 1: Cell-Based Architecture (Week 1)

#### 1.1 Cell Data Structures

**File:** `cyxwiz-engine/src/scripting/cell.h`

```cpp
enum class CellType { Code, Markdown, Raw };
enum class CellState { Idle, Queued, Running, Success, Error };

struct CellOutput {
    enum Type { Text, Error, Image, Plot, Table, Stream };
    Type type;
    std::string data;
    std::string mime_type;
    GLuint texture_id = 0;  // For images/plots
    int width = 0, height = 0;
};

struct Cell {
    std::string id;
    CellType type = CellType::Code;
    std::string source;
    std::vector<CellOutput> outputs;
    int execution_count = 0;
    CellState state = CellState::Idle;
    bool collapsed = false;
    bool output_collapsed = false;

    // Editor state
    TextEditor editor;  // ImGuiColorTextEdit
    float editor_height = 100.0f;
};
```

#### 1.2 Cell Manager

**File:** `cyxwiz-engine/src/scripting/cell_manager.h`

```cpp
class CellManager {
public:
    // Cell operations
    int AddCell(CellType type, int position = -1);
    void DeleteCell(int index);
    void MoveCell(int from, int to);
    void DuplicateCell(int index);
    void MergeCells(int first, int second);
    void SplitCell(int index, int line);
    void ChangeCellType(int index, CellType type);

    // Execution
    void RunCell(int index);
    void RunAllCells();
    void RunCellsAbove(int index);
    void RunCellsBelow(int index);
    void InterruptExecution();

    // Output
    void ClearCellOutput(int index);
    void ClearAllOutputs();

    // Serialization
    std::string SerializeToCyx() const;
    bool ParseFromCyx(const std::string& content);

private:
    std::vector<Cell> cells_;
    int execution_counter_ = 0;
    int running_cell_ = -1;
};
```

#### 1.3 Modify ScriptEditorPanel

**Changes to:** `cyxwiz-engine/src/gui/panels/script_editor.h`

```cpp
// Add to EditorTab struct:
struct EditorTab {
    // ... existing fields ...

    // Cell-based mode
    bool cell_mode = false;           // True for cell-based editing
    CellManager cell_manager;         // Cell management
    int selected_cell = -1;           // Currently selected cell
    int editing_cell = -1;            // Cell being edited

    // Output display
    std::vector<GLuint> plot_textures;  // Cached plot textures
};

// Add new methods:
void RenderCellBasedEditor();
void RenderCell(Cell& cell, int index);
void RenderCodeCell(Cell& cell, int index);
void RenderMarkdownCell(Cell& cell, int index);
void RenderCellOutput(const CellOutput& output);
void RenderCellToolbar(int index);
void HandleCellKeyboardShortcuts();
```

### Phase 2: Inline Output Display (Week 1-2)

#### 2.1 Output Renderer

**File:** `cyxwiz-engine/src/gui/panels/output_renderer.h`

```cpp
class OutputRenderer {
public:
    // Text output
    static void RenderText(const std::string& text, bool wrap = true);
    static void RenderError(const std::string& error);
    static void RenderStream(const std::string& stream, const std::string& name);

    // Rich output
    static void RenderImage(GLuint texture_id, int width, int height);
    static void RenderTable(const std::vector<std::vector<std::string>>& data,
                           const std::vector<std::string>& headers);
    static void RenderMarkdown(const std::string& markdown);

    // Plot output (ImPlot integration)
    static void RenderPlot(GLuint texture_id, int width, int height);

    // Utilities
    static GLuint CreateTextureFromPNG(const unsigned char* data, size_t size);
    static GLuint CreateTextureFromRGBA(const unsigned char* data, int w, int h);
};
```

#### 2.2 Modify Render Functions

**In:** `script_editor.cpp`

```cpp
void ScriptEditorPanel::RenderCellBasedEditor() {
    auto& tab = tabs_[active_tab_index_];
    if (!tab->cell_mode) return;

    float available_height = ImGui::GetContentRegionAvail().y;

    ImGui::BeginChild("##cells_view", ImVec2(0, available_height), false);

    auto& cells = tab->cell_manager.GetCells();
    for (int i = 0; i < static_cast<int>(cells.size()); i++) {
        RenderCell(cells[i], i);
    }

    // Add cell button at bottom
    if (ImGui::Button(ICON_FA_PLUS " Add Cell")) {
        tab->cell_manager.AddCell(CellType::Code);
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::RenderCell(Cell& cell, int index) {
    ImGui::PushID(index);

    // Cell container with border
    ImGui::BeginGroup();

    // Cell toolbar
    RenderCellToolbar(index);

    // Cell content based on type
    switch (cell.type) {
        case CellType::Code:
            RenderCodeCell(cell, index);
            break;
        case CellType::Markdown:
            RenderMarkdownCell(cell, index);
            break;
        case CellType::Raw:
            // Render as plain text
            ImGui::TextWrapped("%s", cell.source.c_str());
            break;
    }

    // Output display
    if (!cell.outputs.empty() && !cell.output_collapsed) {
        ImGui::Separator();
        for (const auto& output : cell.outputs) {
            RenderCellOutput(output);
        }
    }

    ImGui::EndGroup();

    // Cell border
    ImVec2 min = ImGui::GetItemRectMin();
    ImVec2 max = ImGui::GetItemRectMax();
    ImU32 border_color = (tab->selected_cell == index)
        ? IM_COL32(0, 150, 255, 255)   // Blue for selected
        : IM_COL32(80, 80, 80, 255);   // Gray for others
    ImGui::GetWindowDrawList()->AddRect(min, max, border_color, 4.0f, 0, 2.0f);

    ImGui::PopID();
    ImGui::Spacing();
}
```

### Phase 3: Inline Plotting (Week 2)

#### 3.1 Matplotlib Capture

The key to inline plotting is capturing matplotlib figures as PNG images:

**In Python Engine:** `python_engine.cpp`

```cpp
// Setup matplotlib to save figures to buffer
void PythonEngine::SetupMatplotlibCapture() {
    Execute(R"(
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import io
import base64
import matplotlib.pyplot as plt

# Store original show()
_original_show = plt.show

def _capture_show():
    """Capture current figure as base64 PNG"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()

    # Signal to C++ that we have a plot
    __cyxwiz_plot_output__(img_base64)

plt.show = _capture_show
)");
}
```

**C++ Callback for Plot Output:**

```cpp
// Register callback to receive plot data
void PythonEngine::RegisterPlotCallback(std::function<void(const std::string&)> callback) {
    plot_callback_ = callback;

    // Register Python function
    auto cyxwiz_module = py::module_::import("cyxwiz");
    cyxwiz_module.def("__cyxwiz_plot_output__", [this](const std::string& base64_png) {
        if (plot_callback_) {
            plot_callback_(base64_png);
        }
    });
}
```

#### 3.2 Plot Texture Creation

```cpp
// In OutputRenderer
GLuint OutputRenderer::CreateTextureFromBase64PNG(const std::string& base64_data) {
    // Decode base64
    std::vector<unsigned char> png_data = Base64Decode(base64_data);

    // Load PNG using stb_image
    int width, height, channels;
    unsigned char* pixels = stbi_load_from_memory(
        png_data.data(), static_cast<int>(png_data.size()),
        &width, &height, &channels, 4);

    if (!pixels) return 0;

    // Create OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    stbi_image_free(pixels);
    return texture;
}

void OutputRenderer::RenderPlot(GLuint texture_id, int width, int height) {
    if (texture_id == 0) return;

    // Scale to fit
    float max_width = ImGui::GetContentRegionAvail().x - 20.0f;
    float scale = std::min(1.0f, max_width / width);

    ImVec2 size(width * scale, height * scale);

    ImGui::Image((ImTextureID)(intptr_t)texture_id, size);

    // Context menu for plot
    if (ImGui::BeginPopupContextItem("##plot_context")) {
        if (ImGui::MenuItem(ICON_FA_COPY " Copy Plot")) {
            // Copy to clipboard
        }
        if (ImGui::MenuItem(ICON_FA_DOWNLOAD " Save As...")) {
            // Save dialog
        }
        if (ImGui::MenuItem(ICON_FA_EXPAND " Open in Window")) {
            // Open in PlotWindow
        }
        ImGui::EndPopup();
    }
}
```

### Phase 4: Markdown Rendering (Week 2)

#### 4.1 Simple Markdown Renderer

For Phase 1, implement basic markdown without external dependencies:

```cpp
void OutputRenderer::RenderMarkdown(const std::string& markdown) {
    std::istringstream stream(markdown);
    std::string line;

    while (std::getline(stream, line)) {
        // Headers
        if (line.starts_with("### ")) {
            ImGui::PushFont(/* header3 font */);
            ImGui::TextWrapped("%s", line.substr(4).c_str());
            ImGui::PopFont();
        }
        else if (line.starts_with("## ")) {
            ImGui::PushFont(/* header2 font */);
            ImGui::TextWrapped("%s", line.substr(3).c_str());
            ImGui::PopFont();
        }
        else if (line.starts_with("# ")) {
            ImGui::PushFont(/* header1 font */);
            ImGui::TextWrapped("%s", line.substr(2).c_str());
            ImGui::PopFont();
        }
        // Bold
        else if (line.find("**") != std::string::npos) {
            RenderMarkdownWithFormatting(line);
        }
        // Code
        else if (line.starts_with("```")) {
            // Skip code blocks for now
        }
        // Lists
        else if (line.starts_with("- ") || line.starts_with("* ")) {
            ImGui::Bullet();
            ImGui::SameLine();
            ImGui::TextWrapped("%s", line.substr(2).c_str());
        }
        // Regular text
        else {
            ImGui::TextWrapped("%s", line.c_str());
        }
    }
}
```

### Phase 5: Cell Keyboard Shortcuts (Week 2)

```cpp
void ScriptEditorPanel::HandleCellKeyboardShortcuts() {
    if (!tab->cell_mode) return;

    ImGuiIO& io = ImGui::GetIO();
    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;

    // Run cell: Ctrl+Enter
    if (ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (tab->selected_cell >= 0) {
            tab->cell_manager.RunCell(tab->selected_cell);
        }
    }

    // Run cell and select next: Shift+Enter
    if (!ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (tab->selected_cell >= 0) {
            tab->cell_manager.RunCell(tab->selected_cell);
            tab->selected_cell = std::min(tab->selected_cell + 1,
                                          static_cast<int>(tab->cell_manager.GetCells().size()) - 1);
        }
    }

    // Add cell above: A (in command mode)
    if (!ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_A) && tab->editing_cell < 0) {
        tab->cell_manager.AddCell(CellType::Code, tab->selected_cell);
    }

    // Add cell below: B (in command mode)
    if (!ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_B) && tab->editing_cell < 0) {
        tab->cell_manager.AddCell(CellType::Code, tab->selected_cell + 1);
        tab->selected_cell++;
    }

    // Delete cell: D,D (double press) or Ctrl+Shift+D
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_D)) {
        if (tab->selected_cell >= 0) {
            tab->cell_manager.DeleteCell(tab->selected_cell);
            tab->selected_cell = std::min(tab->selected_cell,
                                          static_cast<int>(tab->cell_manager.GetCells().size()) - 1);
        }
    }

    // Convert to markdown: M (in command mode)
    if (!ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_M) && tab->editing_cell < 0) {
        if (tab->selected_cell >= 0) {
            tab->cell_manager.ChangeCellType(tab->selected_cell, CellType::Markdown);
        }
    }

    // Convert to code: Y (in command mode)
    if (!ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Y) && tab->editing_cell < 0) {
        if (tab->selected_cell >= 0) {
            tab->cell_manager.ChangeCellType(tab->selected_cell, CellType::Code);
        }
    }

    // Move cell up: Ctrl+Up
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        if (tab->selected_cell > 0) {
            tab->cell_manager.MoveCell(tab->selected_cell, tab->selected_cell - 1);
            tab->selected_cell--;
        }
    }

    // Move cell down: Ctrl+Down
    if (ctrl && ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        if (tab->selected_cell < static_cast<int>(tab->cell_manager.GetCells().size()) - 1) {
            tab->cell_manager.MoveCell(tab->selected_cell, tab->selected_cell + 1);
            tab->selected_cell++;
        }
    }

    // Enter edit mode: Enter
    if (!ctrl && !shift && ImGui::IsKeyPressed(ImGuiKey_Enter) && tab->editing_cell < 0) {
        tab->editing_cell = tab->selected_cell;
    }

    // Exit edit mode: Escape
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && tab->editing_cell >= 0) {
        tab->editing_cell = -1;
    }
}
```

---

## Keyboard Shortcuts Reference

| Shortcut | Action | Mode |
|----------|--------|------|
| Ctrl+Enter | Run current cell | Edit/Command |
| Shift+Enter | Run cell and select next | Edit/Command |
| Ctrl+Shift+Enter | Run all cells | Edit/Command |
| A | Insert cell above | Command |
| B | Insert cell below | Command |
| Ctrl+Shift+D | Delete cell | Edit/Command |
| M | Convert to markdown | Command |
| Y | Convert to code | Command |
| Ctrl+Up | Move cell up | Edit/Command |
| Ctrl+Down | Move cell down | Edit/Command |
| Enter | Enter edit mode | Command |
| Escape | Exit edit mode | Edit |
| Ctrl+S | Save | Any |
| F5 | Run all | Any |
| Shift+F5 | Stop execution | Any |

---

## Integration Points

### 1. Node Editor Integration

```cpp
// In cyxwiz Python module
cyx.def("get_current_model", []() {
    return NodeEditor::Instance().GetCompiledModel();
});

cyx.def("get_selected_nodes", []() {
    return NodeEditor::Instance().GetSelectedNodes();
});

cyx.def("update_node_param", [](int node_id, const std::string& param, py::object value) {
    NodeEditor::Instance().UpdateNodeParameter(node_id, param, value);
});
```

### 2. Training Pipeline Integration

```cpp
cyx.def("get_training_state", []() {
    return TrainingExecutor::Instance().GetState();
});

cyx.def("get_metrics", []() {
    return TrainingExecutor::Instance().GetMetrics();
});

cyx.def("plot_training_history", []() {
    auto history = TrainingExecutor::Instance().GetHistory();
    // Return as dict for matplotlib
    return py::dict(
        "loss"_a = history.loss,
        "accuracy"_a = history.accuracy,
        "val_loss"_a = history.val_loss,
        "val_accuracy"_a = history.val_accuracy
    );
});
```

### 3. Data Registry Integration

```cpp
cyx.def("get_dataset", [](const std::string& name) {
    return DataRegistry::Instance().GetDataset(name);
});

cyx.def("list_datasets", []() {
    return DataRegistry::Instance().GetLoadedDatasets();
});

cyx.def("preview_dataset", [](const std::string& name, int n) {
    return DataRegistry::Instance().Preview(name, n);
});
```

---

## File Structure

```
cyxwiz-engine/src/
├── scripting/
│   ├── cell.h                    # Cell data structures
│   ├── cell_manager.h            # Cell operations
│   ├── cell_manager.cpp
│   ├── python_engine.h           # (existing, enhance)
│   └── python_engine.cpp         # (existing, add plot capture)
├── gui/
│   └── panels/
│       ├── script_editor.h       # (existing, enhance with cell mode)
│       ├── script_editor.cpp     # (existing, add cell rendering)
│       ├── output_renderer.h     # Output rendering utilities
│       ├── output_renderer.cpp
│       ├── variable_explorer.h   # Variable inspection panel
│       └── variable_explorer.cpp
└── resources/
    └── scripts/
        └── examples/
            ├── getting_started.cyx
            ├── training_basics.cyx
            └── plotting_example.cyx
```

---

## Implementation Timeline

### Week 1: Core Cell System
- [ ] Cell data structures (`cell.h`)
- [ ] Cell manager with CRUD operations
- [ ] Cell parsing from `.cyx` format
- [ ] Cell serialization to `.cyx` format
- [ ] Basic cell rendering in ScriptEditorPanel
- [ ] Cell selection and navigation

### Week 2: Output & Plotting
- [ ] OutputRenderer class
- [ ] Text/Error output display
- [ ] Matplotlib capture setup in Python
- [ ] Base64 PNG to OpenGL texture
- [ ] Inline plot rendering
- [ ] Plot context menu (copy/save)

### Week 3: Markdown & Polish
- [ ] Basic markdown rendering
- [ ] Cell toolbar (run/stop/clear)
- [ ] Cell collapse/expand
- [ ] Output collapse/expand
- [ ] Keyboard shortcuts
- [ ] Mode switching (cell/plain)

### Week 4: Debugger (Optional)
- [ ] debugpy integration
- [ ] Breakpoint management
- [ ] Step controls
- [ ] Variable explorer panel

---

## Testing Checklist

| Test | Description |
|------|-------------|
| Cell parsing | Parse .cyx with %%code, %%markdown markers |
| Cell execution | Run code cell, capture output |
| Inline plot | Execute `plt.show()`, see plot inline |
| Cell navigation | Arrow keys, A/B to add cells |
| Mode switch | Toggle between cell and plain mode |
| Save/Load | Save cells to .cyx, reload correctly |
| Large output | Handle long text output with scrolling |
| Error display | Show Python errors with traceback |

---

## Dependencies

### Already Integrated
- ImGuiColorTextEdit (syntax highlighting)
- ImPlot (for native plotting)
- Python/pybind11 (scripting engine)
- stb_image (image loading)

### Required Python Packages
```
matplotlib>=3.5.0
numpy>=1.20.0
pandas>=1.3.0 (optional, for DataFrame display)
```

### Optional (Future)
- cmark (advanced markdown parsing)
- Jedi (auto-completion)
- debugpy (debugging)

---

## Notes

1. **Backward Compatibility**: Plain `.cyx` files without cell markers are treated as single code cell.

2. **Cell Mode Toggle**: Users can switch between cell mode and plain text mode via View menu.

3. **Performance**: Large outputs (>1000 lines) are truncated with "Show more..." button.

4. **Plot Caching**: Plot textures are cached and reused until cell is re-executed.

5. **Thread Safety**: Plot callbacks use mutex to safely pass data from Python thread to main thread.

  Jupyter-Like Script Editor - Complete Implementation

  1. Cell-Based Editor (from previous session)

  - Cell mode toggle: View > Notebook Mode or Ctrl+Shift+N
  - Cell types: Code (Python) and Markdown
  - Jupyter keyboard shortcuts: Shift+Enter (run), A/B (add above/below), D,D (delete), M/Y (type toggle)
  - Visual UI: Selection border, execution counters, toolbar buttons

  2. Matplotlib Capture System (this session)

  Files Modified:
  - scripting_engine.h - Added CapturedPlot struct and plots field to ExecutionResult
  - scripting_engine.cpp - Added Python code to intercept plt.show() and capture figures as PNG
  - cell.h - Added image_data field for raw PNG bytes
  - cell_manager.cpp - Handle plot outputs in completion callback
  - output_renderer.cpp - Create OpenGL textures from PNG data on-demand

  How it works:
  1. When Python code calls plt.show(), our custom function intercepts it
  2. Captures all matplotlib figures as PNG bytes using BytesIO
  3. Sends PNG data back to C++ via callback
  4. Output renderer decodes PNG with stb_image and creates OpenGL texture
  5. Plot displays inline in cell output area

  Example usage:
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(0, 10, 100)
  plt.plot(x, np.sin(x))
  plt.title("Sine Wave")
  plt.show()  # Plot appears inline below cell

  Key Implementation Details:

  - Non-blocking: Uses Agg backend so matplotlib doesn't open windows
  - Multiple figures: Captures all open figures when plt.show() is called
  - Title capture: Extracts figure title or subplot title for display
  - Thread-safe: Uses mutex-protected queues for plot data
  - Memory efficient: Stores raw PNG bytes, creates textures on-demand

    Debug Features for Traditional Script Editor:

  1. Breakpoint Gutter (left side of editor)
    - Click empty circle to add breakpoint (turns red)
    - Click red circle to remove breakpoint
    - Yellow arrow shows current debug line when paused
  2. Keyboard Shortcuts:
    - F5 - Start/Continue debuggings
    - Shift+F5 - Stop debugging
    - F9 - Toggle breakpoint at cursor
    - F10 - Step over
    - F11 - Step into
    - Shift+F11 - Step out
  3. Debug Toolbar - Appears when debugging is active, showing Continue, Step Over, Step Into, Step Out, and Stop
  buttons