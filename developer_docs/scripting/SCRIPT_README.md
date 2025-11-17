# CyxWiz Scripting System - Architecture & Design Document

**Author**: CTO/Senior Architect
**Date**: 2025-11-16
**Status**: Design Phase
**Priority**: HIGH

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Design Goals & Vision](#design-goals--vision)
3. [System Architecture](#system-architecture)
4. [Component Design](#component-design)
5. [Security Architecture](#security-architecture)
6. [Integration Points](#integration-points)
7. [Implementation Phases](#implementation-phases)
8. [Technology Stack & Libraries](#technology-stack--libraries)
9. [Code Examples & API Design](#code-examples--api-design)
10. [Performance Considerations](#performance-considerations)
11. [Open Questions & Future Enhancements](#open-questions--future-enhancements)

---

## Executive Summary

The CyxWiz Scripting System transforms CyxWiz Engine from a pure visual IDE into a **hybrid visual/code** ML development environment. This feature provides:

1. **Command Window** - MATLAB/IPython-style REPL for interactive experimentation
2. **Script Editor** - Multi-tab code editor with syntax highlighting and section execution
3. **Bidirectional Conversion** - Seamless translation between visual node graphs and `.cyx` script files
4. **Multi-Format Support** - Native handling of CSV, Excel, HDF5, and text files
5. **Sandboxed Execution** - Secure Python runtime with resource limits and permission controls

This bridges the gap between visual learners (node editor) and code-first developers (scripting), enabling users to choose the workflow that suits them best - or mix both approaches dynamically.

---

## Design Goals & Vision

### Primary Goals

1. **Workflow Flexibility**: Enable users to work visually OR with code, switching seamlessly
2. **MATLAB-Like UX**: Familiar interface for ML practitioners transitioning from MATLAB/Python
3. **Security First**: Prevent malicious scripts while maintaining full ML capabilities
4. **Performance**: Execute GPU-accelerated operations without Python overhead where possible
5. **Project Integration**: Scripts as first-class project assets (like datasets and models)

### Non-Goals

1. **Not a General IDE**: We're not building VSCode - focus on ML workflow only
2. **Not Full Python**: Sandbox restricts filesystem, network, and dangerous imports
3. **Not Language Agnostic**: Python-first; other languages deferred to Phase 3+

### Success Metrics

- Users can build a complete ML training pipeline using ONLY the command window
- Node graphs auto-generate clean, readable `.cyx` code
- Scripts execute 90%+ as fast as equivalent C++ backend calls (minimal Python overhead)
- Zero successful sandbox escapes in security testing

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CyxWiz Engine (Desktop)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Command      │  │ Script       │  │ Node         │         │
│  │ Window       │◄─┤ Editor       │◄─┤ Editor       │         │
│  │ (REPL)       │  │ (Multi-Tab)  │  │ (Visual)     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                   ┌────────▼──────────┐                         │
│                   │ ScriptingEngine   │                         │
│                   │ (C++ Orchestrator)│                         │
│                   └────────┬──────────┘                         │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│    ┌────▼─────┐   ┌────────▼──────┐   ┌──────▼──────┐          │
│    │ Python   │   │ NodeScript    │   │ FileFormat  │          │
│    │ Sandbox  │   │ Converter     │   │ Handlers    │          │
│    │ Executor │   │ (Bi-directional)│ │ (CSV/HDF5)  │          │
│    └────┬─────┘   └───────────────┘   └─────────────┘          │
│         │                                                       │
├─────────┼───────────────────────────────────────────────────────┤
│         │    cyxwiz-backend (Shared DLL/SO)                    │
│    ┌────▼─────────────────────────────────────────────┐        │
│    │ pycyxwiz (pybind11 Python Bindings)              │        │
│    │ - Tensor, Device, Optimizer, Model, Layer APIs   │        │
│    └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Command Execution

```
User types: f:> x = Tensor.random([10, 10])
     │
     ▼
CommandWindow::OnEnter()
     │
     ▼
ScriptingEngine::ExecuteCommand(command, context)
     │
     ├─→ Parse command (syntax check)
     │
     ├─→ Security validation (sandboxing rules)
     │
     ├─→ PythonSandbox::Execute(command, globals, locals)
     │        │
     │        ├─→ RestrictedPython compile (AST transformation)
     │        │
     │        └─→ pybind11 → cyxwiz-backend → ArrayFire (GPU)
     │
     ├─→ Capture result + stdout/stderr
     │
     └─→ CommandWindow::AppendOutput(result, LogLevel)
```

### Data Flow: Node to Script Conversion

```
User clicks "Generate Script" on node graph
     │
     ▼
NodeEditor::ExportToScript()
     │
     ▼
NodeScriptConverter::NodeGraphToScript(graph)
     │
     ├─→ Topological sort (execution order)
     │
     ├─→ For each node:
     │   ├─ Map NodeType → CyxWiz API calls
     │   ├─ Resolve input/output variable names
     │   └─ Generate Python statements
     │
     ├─→ Add section markers (%%)
     │
     └─→ Write to .cyx file + open in ScriptEditor
```

### Data Flow: Script to Node Conversion

```
User clicks "Create Nodes" in ScriptEditor
     │
     ▼
ScriptEditor::ImportToNodeGraph()
     │
     ▼
NodeScriptConverter::ScriptToNodeGraph(script_text)
     │
     ├─→ Parse Python AST
     │
     ├─→ Pattern matching:
     │   ├─ Tensor operations → Tensor nodes
     │   ├─ Layer instantiation → Layer nodes
     │   ├─ Model training → Training nodes
     │   └─ Data loading → Data nodes
     │
     ├─→ Infer connections (variable dependencies)
     │
     └─→ NodeEditor::CreateNodesFromAST(nodes, edges)
```

---

## Component Design

### 1. CommandWindow (REPL Panel)

**Purpose**: Interactive Python shell for rapid experimentation

**Design Pattern**: Interpreter + Observer Pattern

**File**: `cyxwiz-engine/src/gui/panels/command_window.h`

```cpp
#pragma once

#include "../panel.h"
#include "../../scripting/scripting_engine.h"
#include <vector>
#include <string>
#include <deque>

namespace cyxwiz {

class CommandWindow : public Panel {
public:
    CommandWindow(std::shared_ptr<scripting::ScriptingEngine> engine);
    ~CommandWindow() override = default;

    void Render() override;
    void HandleKeyboardShortcuts() override;

    // API for programmatic control
    void ExecuteCommand(const std::string& command);
    void AppendOutput(const std::string& output, LogLevel level);
    void Clear();

private:
    // UI rendering
    void RenderPrompt();
    void RenderHistory();
    void RenderAutoComplete();

    // Input handling
    void OnEnterPressed();
    void OnTabPressed();
    void OnUpArrow();
    void OnDownArrow();

    // Auto-completion
    void UpdateAutoComplete();
    std::vector<std::string> GetCompletionCandidates(const std::string& prefix);

    // Data members
    std::shared_ptr<scripting::ScriptingEngine> engine_;

    // Command history
    char input_buffer_[1024];
    std::deque<std::string> command_history_;
    int history_position_;
    size_t max_history_;

    // Output display
    struct OutputEntry {
        std::string text;
        LogLevel level;
        float timestamp;
    };
    std::vector<OutputEntry> output_buffer_;
    size_t max_output_lines_;

    // Auto-completion
    std::vector<std::string> autocomplete_candidates_;
    int autocomplete_selection_;
    bool show_autocomplete_;

    // UI state
    bool scroll_to_bottom_;
    bool focus_input_;
};

} // namespace cyxwiz
```

**Key Features**:
- Command history (up/down arrows) with persistence
- Tab completion using Python introspection
- Multi-line command support (backslash continuation)
- Variable inspector (show all defined variables)
- Output formatting (MATLAB-style `ans =` for unnamed results)

**MATLAB Compatibility Layer**:
```python
# Map MATLAB syntax to CyxWiz
# User types: A = rand(10, 10)
# Internally translates to: A = Tensor.random([10, 10])
```

---

### 2. ScriptEditor (Multi-Tab Code Editor)

**Purpose**: Full-featured code editor for `.cyx` scripts

**Design Pattern**: Strategy Pattern (file formats), Command Pattern (undo/redo)

**File**: `cyxwiz-engine/src/gui/panels/script_editor.h`

```cpp
#pragma once

#include "../panel.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <ImGuiColorTextEdit/TextEditor.h>
#include <memory>
#include <vector>
#include <filesystem>

namespace cyxwiz {

class ScriptEditor : public Panel {
public:
    ScriptEditor(std::shared_ptr<scripting::ScriptingEngine> engine);
    ~ScriptEditor() override = default;

    void Render() override;
    void HandleKeyboardShortcuts() override;

    // File operations
    void NewFile(const std::string& default_name = "untitled.cyx");
    void OpenFile(const std::filesystem::path& path);
    void SaveFile();
    void SaveFileAs(const std::filesystem::path& path);
    void CloseFile(int tab_index);

    // Execution
    void RunScript();
    void RunSelectedSection();
    void RunCurrentLine();
    void StopExecution();

    // Import/Export
    void ImportFromNodeGraph();
    void ExportToNodeGraph();

private:
    struct EditorTab {
        std::string title;
        std::filesystem::path filepath;
        TextEditor editor;
        bool is_modified;
        bool is_new_file;
        FileType file_type;

        EditorTab(const std::string& name, FileType type);
    };

    // UI rendering
    void RenderMenuBar();
    void RenderTabBar();
    void RenderActiveEditor();
    void RenderStatusBar();
    void RenderSectionMarkers();

    // Menu actions
    void ShowFileMenu();
    void ShowEditMenu();
    void ShowRunMenu();
    void ShowViewMenu();

    // Section management
    std::vector<SectionMarker> ParseSections(const std::string& text);
    SectionMarker* GetCurrentSection();
    void HighlightSection(const SectionMarker& section);

    // Syntax highlighting
    void ConfigureSyntaxHighlighting(TextEditor& editor, FileType type);
    TextEditor::LanguageDefinition CreateCyxLanguageDefinition();

    // Data members
    std::shared_ptr<scripting::ScriptingEngine> engine_;
    std::vector<std::unique_ptr<EditorTab>> tabs_;
    int active_tab_index_;

    // Execution state
    bool is_executing_;
    std::atomic<bool> execution_stop_requested_;

    // UI state
    bool show_line_numbers_;
    bool show_whitespace_;
    bool auto_indent_;
    int font_size_;
};

enum class FileType {
    CYX,      // CyxWiz script
    Python,   // .py
    Text,     // .txt
    CSV,      // .csv
    Excel,    // .xlsx
    HDF5      // .hdf5, .h5
};

struct SectionMarker {
    int start_line;
    int end_line;
    std::string title; // Optional title after %%
};

} // namespace cyxwiz
```

**Key Features**:
- Multi-tab interface with unsaved changes indicator (*)
- Section execution using `%%` delimiters (MATLAB/Jupyter style)
- Syntax highlighting for Python + custom `.cyx` extensions
- Breakpoint support (for future debugger integration)
- File format handlers for CSV/Excel/HDF5 (view-only initially)

**Text Editor Component**: **ImGuiColorTextEdit**
- Repository: https://github.com/BalazsJako/ImGuiColorTextEdit
- License: MIT (compatible)
- Integration: Add as vcpkg dependency or manual submodule
- Features: Syntax highlighting, undo/redo, breakpoints, error markers

**Section Execution Example**:
```python
# File: train_mnist.cyx

%% Data Loading
data = load_dataset("mnist")
X_train, y_train = data.split()

%% Model Definition
model = Sequential([
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

%% Training
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(X_train, y_train, epochs=10)
```

User can:
- Run entire file: Executes all sections
- Run selected section: Place cursor in section, press Ctrl+Enter
- Run current line: Ctrl+Shift+Enter

---

### 3. ScriptingEngine (Core Orchestrator)

**Purpose**: Central coordinator for all scripting operations

**Design Pattern**: Facade Pattern (unified interface), Singleton (shared state)

**File**: `cyxwiz-engine/src/scripting/scripting_engine.h`

```cpp
#pragma once

#include "python_engine.h"
#include "python_sandbox.h"
#include "node_script_converter.h"
#include "file_format_handler.h"
#include <memory>
#include <functional>
#include <map>

namespace scripting {

// Execution result
struct ExecutionResult {
    bool success;
    std::string output;
    std::string error;
    std::map<std::string, std::string> variables; // name -> repr
};

// Execution context
struct ExecutionContext {
    std::map<std::string, pybind11::object> globals;
    std::map<std::string, pybind11::object> locals;
    std::filesystem::path working_directory;
    size_t max_execution_time_ms; // 0 = no limit
    size_t max_memory_mb;          // 0 = no limit
};

// Callback types
using OutputCallback = std::function<void(const std::string&, LogLevel)>;
using ProgressCallback = std::function<void(float progress, const std::string& status)>;

class ScriptingEngine {
public:
    ScriptingEngine();
    ~ScriptingEngine();

    // Initialization
    bool Initialize(const std::filesystem::path& python_home);
    void Shutdown();

    // Command execution (REPL)
    ExecutionResult ExecuteCommand(const std::string& command);
    ExecutionResult ExecuteCommand(const std::string& command, ExecutionContext& context);

    // Script execution
    ExecutionResult ExecuteScript(const std::string& script_text);
    ExecutionResult ExecuteFile(const std::filesystem::path& filepath);
    ExecutionResult ExecuteSection(const std::string& script_text, int start_line, int end_line);

    // Stop execution
    void StopExecution();

    // Context management
    ExecutionContext& GetGlobalContext();
    void ResetContext();
    std::vector<std::string> GetDefinedVariables();
    std::string GetVariableRepr(const std::string& name);

    // Auto-completion
    std::vector<std::string> GetCompletions(const std::string& prefix);
    std::string GetDocstring(const std::string& symbol);

    // Callbacks
    void SetOutputCallback(OutputCallback callback);
    void SetProgressCallback(ProgressCallback callback);

    // Node <-> Script conversion
    NodeScriptConverter* GetConverter() { return converter_.get(); }

    // File format handlers
    FileFormatHandler* GetFileHandler(FileType type);

private:
    // Components
    std::unique_ptr<PythonEngine> python_engine_;
    std::unique_ptr<PythonSandbox> sandbox_;
    std::unique_ptr<NodeScriptConverter> converter_;
    std::map<FileType, std::unique_ptr<FileFormatHandler>> file_handlers_;

    // Execution state
    ExecutionContext global_context_;
    std::atomic<bool> is_executing_;
    std::atomic<bool> stop_requested_;

    // Callbacks
    OutputCallback output_callback_;
    ProgressCallback progress_callback_;

    // Helper methods
    void InitializeBuiltins();
    void RedirectStdout();
    void RestoreStdout();
};

} // namespace scripting
```

**Responsibilities**:
1. Manages Python interpreter lifecycle
2. Coordinates sandboxing and security
3. Routes execution to appropriate handler (REPL vs script vs section)
4. Maintains global execution context (variables persist across commands)
5. Provides introspection for auto-completion

---

### 4. PythonSandbox (Security Layer)

**Purpose**: Secure execution environment for untrusted scripts

**Design Pattern**: Decorator Pattern (wraps Python execution)

**File**: `cyxwiz-engine/src/scripting/python_sandbox.h`

```cpp
#pragma once

#include <pybind11/embed.h>
#include <string>
#include <vector>
#include <set>
#include <chrono>

namespace scripting {

// Security policy
struct SandboxPolicy {
    // Module restrictions
    std::set<std::string> allowed_imports;
    std::set<std::string> blocked_imports;

    // Filesystem access
    bool allow_file_read;
    bool allow_file_write;
    std::vector<std::filesystem::path> allowed_directories;

    // Network access
    bool allow_network;
    std::vector<std::string> allowed_hosts;

    // Resource limits
    size_t max_execution_time_ms;
    size_t max_memory_mb;
    size_t max_cpu_time_ms;

    // Dangerous operations
    bool allow_subprocess;
    bool allow_eval;
    bool allow_exec;
    bool allow_compile;

    // Default policy factory
    static SandboxPolicy CreateDefault();
    static SandboxPolicy CreateRestrictive();
    static SandboxPolicy CreatePermissive();
};

class PythonSandbox {
public:
    PythonSandbox(const SandboxPolicy& policy);
    ~PythonSandbox();

    // Execute with sandboxing
    pybind11::object Execute(
        const std::string& code,
        pybind11::dict globals,
        pybind11::dict locals
    );

    // Policy management
    void SetPolicy(const SandboxPolicy& policy);
    SandboxPolicy GetPolicy() const { return policy_; }

    // Monitoring
    struct ExecutionStats {
        std::chrono::milliseconds execution_time;
        size_t memory_used_mb;
        size_t cpu_time_ms;
        std::vector<std::string> imports_attempted;
        std::vector<std::string> files_accessed;
    };

    ExecutionStats GetLastExecutionStats() const { return last_stats_; }

private:
    // Sandboxing implementation
    void SetupRestrictedBuiltins();
    void SetupImportHook();
    void SetupFileAccessHook();
    void SetupResourceMonitoring();

    // Security validation
    bool ValidateImport(const std::string& module_name);
    bool ValidateFileAccess(const std::filesystem::path& path, bool is_write);

    // Resource monitoring
    void StartResourceMonitoring();
    void StopResourceMonitoring();
    void CheckResourceLimits();

    // Data members
    SandboxPolicy policy_;
    ExecutionStats last_stats_;
    std::chrono::steady_clock::time_point execution_start_;

    // Python objects
    pybind11::dict restricted_builtins_;
    pybind11::object import_hook_;
    pybind11::object file_hook_;
};

} // namespace scripting
```

**Security Implementation Strategy**:

1. **Layer 1: AST Transformation** (RestrictedPython-inspired)
   - Parse Python AST before execution
   - Rewrite dangerous constructs:
     - `__import__()` → Custom import hook
     - `open()` → Sandboxed file access
     - `eval()`, `exec()` → Blocked by default
   - Inject resource tracking code

2. **Layer 2: Restricted Builtins**
   ```python
   safe_builtins = {
       'abs': abs, 'all': all, 'any': any, 'len': len,
       'max': max, 'min': min, 'sum': sum,
       'range': range, 'enumerate': enumerate,
       'print': sandboxed_print,  # Redirect to console
       'open': sandboxed_open,    # Filesystem restrictions
       '__import__': sandboxed_import,  # Module whitelist
       # Block: eval, exec, compile, __builtins__
   }
   ```

3. **Layer 3: Import Hook**
   ```python
   # Whitelist for ML work
   ALLOWED_IMPORTS = {
       'pycyxwiz',    # Our backend
       'numpy',       # Numerical computing
       'math',        # Standard library
       'json',        # Data serialization
       'datetime',    # Time handling
       # BLOCKED: os, sys, subprocess, socket, ctypes, etc.
   }
   ```

4. **Layer 4: Resource Monitoring**
   - Timeout mechanism: Python signal handlers (SIGALRM on Unix, thread-based on Windows)
   - Memory tracking: `tracemalloc` module
   - CPU time: `resource.getrusage()` (Unix) or Windows equivalents

**CVE Mitigations** (based on RestrictedPython vulnerabilities):
- CVE-2025-22153: Block `try/except*` clauses via AST validation
- CVE-2024-47532: Remove `AttributeError.obj` from exception handling
- CVE-2023-37271: Comprehensive audit of all object introspection paths

**Known Limitations**:
- Cannot prevent all timing attacks
- CPython-specific (won't work on PyPy)
- Determined attackers may find escapes - defense in depth required
- Performance overhead: ~10-30% due to monitoring

**Recommended Additional Layers**:
- Run Engine in least-privileged user account
- Container isolation (Docker) for Server Node execution
- Filesystem sandboxing (chroot/AppContainer)

---

### 5. NodeScriptConverter (Bidirectional Conversion)

**Purpose**: Translate between visual node graphs and Python scripts

**Design Pattern**: Visitor Pattern (node traversal), Strategy Pattern (node-to-code mapping)

**File**: `cyxwiz-engine/src/scripting/node_script_converter.h`

```cpp
#pragma once

#include "../gui/node_editor.h"
#include <string>
#include <vector>
#include <map>

namespace scripting {

// Forward declarations
struct NodeGraph;
struct Node;
struct Edge;

class NodeScriptConverter {
public:
    NodeScriptConverter();
    ~NodeScriptConverter();

    // Node Graph → Script
    std::string NodeGraphToScript(const NodeGraph& graph);
    std::string NodeToCode(const Node& node);

    // Script → Node Graph
    NodeGraph ScriptToNodeGraph(const std::string& script);
    std::vector<Node> ParseScriptToNodes(const std::string& script);

    // Configuration
    void SetCodeStyle(CodeStyle style);
    void SetCommentVerbosity(CommentVerbosity level);

private:
    // Node → Code strategies
    struct NodeCodeGenerator {
        virtual ~NodeCodeGenerator() = default;
        virtual std::string Generate(const Node& node) = 0;
        virtual std::vector<std::string> GetDependencies(const Node& node) = 0;
    };

    std::map<NodeType, std::unique_ptr<NodeCodeGenerator>> generators_;

    // Code → Node parsers
    struct CodeNodeParser {
        virtual ~CodeNodeParser() = default;
        virtual std::optional<Node> Parse(const pybind11::ast& statement) = 0;
    };

    std::vector<std::unique_ptr<CodeNodeParser>> parsers_;

    // Topological sorting
    std::vector<Node*> TopologicalSort(const NodeGraph& graph);

    // Variable name generation
    std::string GenerateVariableName(const Node& node, const std::string& output_name);
    std::map<std::string, std::string> node_to_var_map_;

    // Code formatting
    std::string FormatCode(const std::string& code, CodeStyle style);
    std::string AddComments(const std::string& code, const Node& node, CommentVerbosity level);

    // Configuration
    CodeStyle code_style_;
    CommentVerbosity comment_verbosity_;
};

enum class CodeStyle {
    Compact,    // Minimal whitespace
    Readable,   // Balanced
    Verbose     // Maximum comments
};

enum class CommentVerbosity {
    None,       // No auto-generated comments
    Minimal,    // Only section headers
    Full        // Comment each node
};

// Example node type → code generators

class TensorOpGenerator : public NodeCodeGenerator {
public:
    std::string Generate(const Node& node) override {
        // Example: MatMul node → "C = A @ B"
        std::string A = node.inputs[0].variable_name;
        std::string B = node.inputs[1].variable_name;
        std::string C = node.outputs[0].variable_name;
        return fmt::format("{} = {} @ {}", C, A, B);
    }
};

class LayerGenerator : public NodeCodeGenerator {
public:
    std::string Generate(const Node& node) override {
        // Example: Dense layer → "layer1 = Dense(128, activation='relu')"
        int units = node.properties["units"].as_int();
        std::string activation = node.properties["activation"].as_string();
        return fmt::format("{} = Dense({}, activation='{}')",
            node.outputs[0].variable_name, units, activation);
    }
};

} // namespace scripting
```

**Node Type Mappings**:

| Node Type | Generated Code |
|-----------|---------------|
| Tensor Input | `X = Tensor.load("data.npy")` |
| Dense Layer | `layer1 = Dense(128, activation="relu")` |
| Conv2D Layer | `conv1 = Conv2D(32, kernel_size=3, padding=1)` |
| MatMul | `C = A @ B` |
| Activation | `out = relu(x)` |
| Loss | `loss = mse_loss(y_pred, y_true)` |
| Optimizer | `optimizer = Adam(lr=0.001)` |
| Training Loop | `model.fit(X, y, epochs=10, batch_size=32)` |

**Script → Node Parsing**:

Use Python `ast` module to parse script:
```python
import ast

tree = ast.parse(script_text)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        # Detect pattern: x = Dense(...)
        if isinstance(node.value, ast.Call):
            if node.value.func.id == "Dense":
                # Create Dense layer node
```

**Challenges**:
1. **Ambiguity**: Multiple valid node graphs for same script
2. **Comments**: How to preserve user comments during round-trip?
3. **Control Flow**: Nodes don't support if/else/loops - how to represent?

**Solutions**:
1. Prefer canonical representations
2. Store comments as node metadata
3. Represent control flow as "Meta Nodes" (e.g., Loop node wrapping subgraph)

---

### 6. FileFormatHandler (Multi-Format Support)

**Purpose**: Read/write various data file formats

**Design Pattern**: Strategy Pattern + Factory Pattern

**File**: `cyxwiz-engine/src/scripting/file_format_handler.h`

```cpp
#pragma once

#include <filesystem>
#include <vector>
#include <map>
#include <variant>

namespace scripting {

// Unified data structure
struct DataTable {
    std::vector<std::string> column_names;
    std::vector<std::vector<std::variant<int, double, std::string>>> rows;
    std::map<std::string, std::string> metadata;
};

// Base interface
class FileFormatHandler {
public:
    virtual ~FileFormatHandler() = default;

    virtual DataTable Read(const std::filesystem::path& path) = 0;
    virtual void Write(const std::filesystem::path& path, const DataTable& data) = 0;
    virtual bool CanHandle(const std::filesystem::path& path) = 0;

    virtual std::string GetFormatName() const = 0;
    virtual std::vector<std::string> GetSupportedExtensions() const = 0;
};

// CSV Handler
class CSVHandler : public FileFormatHandler {
public:
    DataTable Read(const std::filesystem::path& path) override;
    void Write(const std::filesystem::path& path, const DataTable& data) override;
    bool CanHandle(const std::filesystem::path& path) override;

    std::string GetFormatName() const override { return "CSV"; }
    std::vector<std::string> GetSupportedExtensions() const override {
        return {".csv", ".tsv"};
    }

    // CSV-specific options
    void SetDelimiter(char delim) { delimiter_ = delim; }
    void SetHasHeader(bool has_header) { has_header_ = has_header; }

private:
    char delimiter_ = ',';
    bool has_header_ = true;
};

// Excel Handler (using xlnt or openpyxl via Python)
class ExcelHandler : public FileFormatHandler {
public:
    DataTable Read(const std::filesystem::path& path) override;
    void Write(const std::filesystem::path& path, const DataTable& data) override;
    bool CanHandle(const std::filesystem::path& path) override;

    std::string GetFormatName() const override { return "Excel"; }
    std::vector<std::string> GetSupportedExtensions() const override {
        return {".xlsx", ".xls"};
    }

    // Excel-specific
    void SetSheetName(const std::string& name) { sheet_name_ = name; }
    std::vector<std::string> GetSheetNames(const std::filesystem::path& path);

private:
    std::string sheet_name_ = "Sheet1";
};

// HDF5 Handler (using HighFive library)
class HDF5Handler : public FileFormatHandler {
public:
    DataTable Read(const std::filesystem::path& path) override;
    void Write(const std::filesystem::path& path, const DataTable& data) override;
    bool CanHandle(const std::filesystem::path& path) override;

    std::string GetFormatName() const override { return "HDF5"; }
    std::vector<std::string> GetSupportedExtensions() const override {
        return {".hdf5", ".h5", ".hdf"};
    }

    // HDF5-specific
    void SetDatasetPath(const std::string& path) { dataset_path_ = path; }
    std::vector<std::string> GetDatasetPaths(const std::filesystem::path& path);

    // Direct tensor loading (bypass DataTable for efficiency)
    cyxwiz::Tensor ReadTensor(const std::filesystem::path& path, const std::string& dataset);
    void WriteTensor(const std::filesystem::path& path, const std::string& dataset, const cyxwiz::Tensor& tensor);

private:
    std::string dataset_path_ = "/data";
};

// Factory
class FileFormatFactory {
public:
    static FileFormatFactory& Instance();

    void RegisterHandler(std::unique_ptr<FileFormatHandler> handler);
    FileFormatHandler* GetHandler(const std::filesystem::path& path);
    std::vector<std::string> GetSupportedExtensions();

private:
    std::vector<std::unique_ptr<FileFormatHandler>> handlers_;
};

} // namespace scripting
```

**Library Dependencies**:

1. **CSV**: Custom implementation (simple) or `csv-parser` library
2. **Excel**: `xlnt` (C++) or delegate to Python `openpyxl`
3. **HDF5**: **HighFive** (header-only C++ wrapper)
   - Repo: https://github.com/BlueBrain/HighFive
   - Add to vcpkg.json: `"highfive": "^2.8.0"`

**Integration with ScriptEditor**:

When user opens `.csv`/`.xlsx`/`.hdf5` file:
1. Detect format via extension
2. Load data using appropriate handler
3. Display in read-only table view (ImGui table)
4. Provide "Export to Tensor" button to convert to CyxWiz tensor

---

## Security Architecture

### Threat Model

**Trusted**:
- CyxWiz Engine executable
- cyxwiz-backend DLL
- System Python installation
- User's local files in project directory

**Untrusted**:
- User scripts (.cyx files)
- Downloaded scripts from community
- Server Node job scripts (future)

**Attack Vectors**:
1. **Code Injection**: Malicious script modifies Engine or backend
2. **Data Exfiltration**: Script reads sensitive files, sends over network
3. **Denial of Service**: Infinite loop or memory bomb crashes Engine
4. **Privilege Escalation**: Script gains OS-level permissions
5. **Sandbox Escape**: Script bypasses Python restrictions to run arbitrary code

### Security Layers (Defense in Depth)

```
┌─────────────────────────────────────────────────────────┐
│ Layer 5: OS-Level (Optional)                           │
│ - User account sandboxing (least privilege)            │
│ - Container isolation (Docker on Server Nodes)         │
│ - Filesystem permissions                               │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Process Isolation                             │
│ - Engine runs in separate process from scripts?        │
│ - Future: Execute scripts in subprocess                │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ Layer 3: Resource Limits (PythonSandbox)               │
│ - Execution timeout (SIGALRM/threading)                │
│ - Memory limit (tracemalloc)                           │
│ - CPU time limit (resource.getrusage)                  │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Capability Restrictions (PythonSandbox)       │
│ - Import whitelist (custom __import__ hook)            │
│ - Filesystem restrictions (sandboxed open())           │
│ - No network access (block socket module)              │
│ - No subprocess (block subprocess, os.system)          │
└─────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Code Validation (AST Transformation)          │
│ - Parse AST before execution                           │
│ - Detect dangerous patterns                            │
│ - Rewrite builtins (eval → error)                      │
│ - Block introspection abuse                            │
└─────────────────────────────────────────────────────────┘
```

### Default Sandbox Policy

```cpp
SandboxPolicy SandboxPolicy::CreateDefault() {
    SandboxPolicy policy;

    // Allowed modules (ML-focused whitelist)
    policy.allowed_imports = {
        "pycyxwiz",   // Our backend
        "numpy",      // Numerical computing
        "math",       // Standard math
        "random",     // Random numbers
        "json",       // Data serialization
        "datetime",   // Time handling
        "collections", // Data structures
        "itertools",  // Iteration tools
        "functools",  // Functional programming
    };

    // Blocked modules (dangerous)
    policy.blocked_imports = {
        "os", "sys", "subprocess", "multiprocessing",
        "socket", "urllib", "http", "requests",
        "ctypes", "cffi", "__builtin__", "__builtins__",
        "importlib", "pkgutil", "runpy",
    };

    // Filesystem
    policy.allow_file_read = true;
    policy.allow_file_write = true;
    policy.allowed_directories = {
        GetProjectDirectory(),  // Only project folder
    };

    // Network: DISABLED
    policy.allow_network = false;

    // Resource limits
    policy.max_execution_time_ms = 60000;  // 60 seconds
    policy.max_memory_mb = 2048;           // 2 GB
    policy.max_cpu_time_ms = 50000;        // 50 seconds CPU

    // Dangerous operations: DISABLED
    policy.allow_subprocess = false;
    policy.allow_eval = false;
    policy.allow_exec = false;
    policy.allow_compile = false;

    return policy;
}
```

### User Permissions System (Future Enhancement)

Allow users to selectively enable capabilities:

```cpp
// UI: Preferences → Scripting → Security
[x] Allow scripts to read files in project directory
[ ] Allow scripts to write files
[ ] Allow network access (required for downloading datasets)
[ ] Allow subprocess execution (required for calling external tools)

Resource Limits:
  Max execution time: [60] seconds
  Max memory usage:   [2048] MB
```

When user enables risky permission:
1. Show security warning dialog
2. Require explicit confirmation
3. Log permission grants for audit trail

---

## Integration Points

### 1. Integration with Existing Python Engine

**Current State**:
- `PythonEngine` class handles interpreter initialization
- Basic `ExecuteScript()` and `ExecuteFile()` methods
- No sandboxing, context management, or REPL support

**Integration Strategy**:
1. **Preserve Existing API**: Keep `PythonEngine` for backward compatibility
2. **Extend with ScriptingEngine**: New `ScriptingEngine` wraps `PythonEngine`
3. **Migrate Gradually**: Existing code continues to use `PythonEngine`, new features use `ScriptingEngine`

```cpp
// Old code (still works)
PythonEngine engine;
engine.ExecuteScript("print('hello')");

// New code (enhanced features)
ScriptingEngine script_engine;
script_engine.Initialize();
auto result = script_engine.ExecuteCommand("x = Tensor.random([10, 10])");
```

**File Updates**:
- `python_engine.h/cpp`: No changes (or minor refactoring)
- `scripting_engine.h/cpp`: New file, uses `PythonEngine` internally
- `python_sandbox.h/cpp`: New file, wraps Python execution

---

### 2. Integration with Asset Browser

**Current State**:
- Asset Browser displays folders: Datasets, Models, Scripts, etc.
- Double-clicking asset opens it (TODO: implement handlers)

**Integration**:

**File**: `cyxwiz-engine/src/gui/panels/asset_browser.cpp`

```cpp
void AssetBrowserPanel::OnAssetDoubleClick(AssetItem* item) {
    if (item->type == AssetType::Script) {
        // Open in script editor
        auto script_editor = GetMainWindow()->GetScriptEditor();
        script_editor->OpenFile(item->path);
        script_editor->Show();
    }
    else if (item->type == AssetType::Dataset) {
        // Detect file format and open accordingly
        auto ext = item->path.extension();
        if (ext == ".csv" || ext == ".xlsx" || ext == ".hdf5") {
            auto script_editor = GetMainWindow()->GetScriptEditor();
            script_editor->OpenFile(item->path);  // View in table mode
        }
    }
    // ... other asset types
}
```

**Project Structure**:
```
MyProject/
├── project.cyxproj       # Project metadata
├── assets/
│   ├── datasets/
│   │   ├── mnist.hdf5
│   │   └── train.csv
│   ├── models/
│   │   └── my_model.h5
│   ├── scripts/         # <-- Scripts stored here
│   │   ├── preprocess.cyx
│   │   ├── train.cyx
│   │   └── evaluate.cyx
│   └── checkpoints/
└── graphs/
    └── main_graph.cyxgraph
```

---

### 3. Integration with Node Editor

**Current State**:
- Node Editor renders node graph (placeholder implementation)
- TODO: Integrate ImNodes library

**Integration**:

Add menu options to Node Editor:

```cpp
void NodeEditor::RenderMenuBar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Export to Script...")) {
                ExportToScript();
            }
            if (ImGui::MenuItem("Import from Script...")) {
                ImportFromScript();
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
}

void NodeEditor::ExportToScript() {
    auto converter = GetScriptingEngine()->GetConverter();
    std::string script = converter->NodeGraphToScript(current_graph_);

    // Save to file
    auto filepath = FileDialog::SaveFile("Save Script", ".cyx");
    if (!filepath.empty()) {
        std::ofstream file(filepath);
        file << script;
        file.close();

        // Open in script editor
        GetScriptEditor()->OpenFile(filepath);
    }
}

void NodeEditor::ImportFromScript() {
    auto filepath = FileDialog::OpenFile("Open Script", ".cyx");
    if (!filepath.empty()) {
        std::ifstream file(filepath);
        std::string script((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        auto converter = GetScriptingEngine()->GetConverter();
        NodeGraph graph = converter->ScriptToNodeGraph(script);

        // Replace current graph
        current_graph_ = graph;
        RenderGraph();
    }
}
```

**Workflow Example**:
1. User builds visual graph with nodes
2. Clicks "Export to Script"
3. Script editor opens with generated `.cyx` code
4. User modifies script (adds custom logic)
5. Clicks "Import to Node Editor" (if desired)
6. Graph updates to reflect script changes

---

### 4. Integration with Main Window

**File**: `cyxwiz-engine/src/gui/main_window.h`

Add new panels:

```cpp
class MainWindow {
private:
    std::unique_ptr<CommandWindow> command_window_;
    std::unique_ptr<ScriptEditor> script_editor_;
    std::shared_ptr<ScriptingEngine> scripting_engine_;  // Shared across panels
};
```

**Constructor**:
```cpp
MainWindow::MainWindow() {
    // Initialize scripting engine (shared resource)
    scripting_engine_ = std::make_shared<ScriptingEngine>();
    scripting_engine_->Initialize();

    // Create panels
    command_window_ = std::make_unique<CommandWindow>(scripting_engine_);
    script_editor_ = std::make_unique<ScriptEditor>(scripting_engine_);

    // ... other panels
}
```

**Render**:
```cpp
void MainWindow::Render() {
    RenderDockSpace();

    if (command_window_->IsVisible()) command_window_->Render();
    if (script_editor_->IsVisible()) script_editor_->Render();

    // ... other panels
}
```

**Menu Bar Integration**:
```cpp
void MainWindow::RenderMenuBar() {
    if (ImGui::BeginMenu("View")) {
        if (ImGui::MenuItem("Command Window", "Ctrl+`", command_window_->IsVisible())) {
            command_window_->Toggle();
        }
        if (ImGui::MenuItem("Script Editor", "Ctrl+E", script_editor_->IsVisible())) {
            script_editor_->Toggle();
        }
        // ... other panels
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Script")) {
        if (ImGui::MenuItem("New Script", "Ctrl+N")) {
            script_editor_->NewFile();
            script_editor_->Show();
        }
        if (ImGui::MenuItem("Open Script...", "Ctrl+O")) {
            auto path = FileDialog::OpenFile("Open Script", ".cyx");
            if (!path.empty()) {
                script_editor_->OpenFile(path);
                script_editor_->Show();
            }
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Context")) {
            scripting_engine_->ResetContext();
            command_window_->AppendOutput("Context reset", LogLevel::Info);
        }
        ImGui::EndMenu();
    }
}
```

---

### 5. Integration with Project System

**Project File Format** (`.cyxproj`):

```json
{
  "name": "MyMLProject",
  "version": "1.0.0",
  "created": "2025-11-16T00:00:00Z",
  "assets": {
    "datasets": ["assets/datasets/mnist.hdf5"],
    "models": ["assets/models/resnet.h5"],
    "scripts": [
      "assets/scripts/preprocess.cyx",
      "assets/scripts/train.cyx"
    ],
    "graphs": ["graphs/main_graph.cyxgraph"]
  },
  "settings": {
    "default_device": "CUDA",
    "python_path": "C:/Python311",
    "script_execution_policy": "sandbox_default"
  },
  "script_startup": "assets/scripts/startup.cyx"  // Auto-run on project load
}
```

**Startup Script** (`startup.cyx`):

Automatically executed when project loads (like MATLAB's `startup.m`):

```python
# startup.cyx - Auto-executed when project loads

# Set default device
import pycyxwiz as cyx
cyx.set_default_device("CUDA")

# Load common datasets
mnist = load_dataset("assets/datasets/mnist.hdf5")

# Define helper functions
def plot_sample(idx):
    img = mnist.images[idx]
    label = mnist.labels[idx]
    plot(img, title=f"Label: {label}")

print("Project loaded. Type 'help()' for available commands.")
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 weeks)

**Goal**: Basic REPL functionality

**Tasks**:
1. Extend `PythonEngine` with context management
2. Implement `CommandWindow` panel (basic REPL, no auto-complete)
3. Basic output capture (stdout/stderr redirection)
4. Integrate with `MainWindow` docking system
5. Command history (up/down arrows)

**Deliverable**: Users can type Python commands and see results

**Dependencies**:
- Existing `PythonEngine` class
- ImGui (already integrated)
- pybind11 (already integrated)

**Example Test**:
```
f:> import pycyxwiz as cyx
f:> cyx.initialize()
f:> x = cyx.Tensor.random([10, 10])
f:> print(x.shape())
[10, 10]
```

---

### Phase 2: Script Editor (3-4 weeks)

**Goal**: Multi-tab code editor with execution

**Tasks**:
1. Integrate **ImGuiColorTextEdit** library
   - Add to vcpkg.json or submodule
   - Create wrapper class for ImGui integration
2. Implement `ScriptEditor` panel
   - Multi-tab interface
   - File open/save/close
   - Syntax highlighting for Python
3. Section execution (parse `%%` markers)
4. Output routing to Console or CommandWindow
5. Basic file format handlers (CSV, text)

**Deliverable**: Users can write, save, and execute `.cyx` scripts

**Dependencies**:
- ImGuiColorTextEdit: https://github.com/BalazsJako/ImGuiColorTextEdit
- Phase 1 completion

**Example Test**:
Create `test.cyx`:
```python
%% Load Data
data = load_csv("data.csv")

%% Train Model
model = Sequential([Dense(128)])
model.fit(data)

%% Evaluate
print("Accuracy:", model.evaluate())
```
Execute entire file → all sections run in order

---

### Phase 3: Security & Sandboxing (2-3 weeks)

**Goal**: Secure execution environment

**Tasks**:
1. Implement `PythonSandbox` class
   - AST transformation (inspect and rewrite code)
   - Restricted builtins (custom `__builtins__`)
   - Import hook (whitelist validation)
   - File access hook (directory restrictions)
2. Resource monitoring
   - Timeout mechanism (signal handlers / threading)
   - Memory tracking (`tracemalloc`)
   - CPU time tracking (`resource` module)
3. Security testing
   - Attempt sandbox escapes
   - Test resource limits
   - Validate import restrictions

**Deliverable**: Scripts run in sandboxed environment with enforced limits

**Dependencies**:
- Phase 1 & 2 completion
- Python `ast`, `tracemalloc`, `signal` modules

**Example Test**:
```python
# This should be BLOCKED
import os
os.system("rm -rf /")  # Error: 'os' module not allowed

# This should TIMEOUT
while True:
    pass  # Error: Execution time limit exceeded (60s)

# This should be ALLOWED
import pycyxwiz as cyx
x = cyx.Tensor.random([1000, 1000])  # OK
```

---

### Phase 4: Node-Script Conversion (3-4 weeks)

**Goal**: Bidirectional node ↔ script translation

**Tasks**:
1. Implement `NodeScriptConverter` class
2. Node → Code generators for common node types
   - Tensor operations (MatMul, Add, etc.)
   - Layers (Dense, Conv2D, etc.)
   - Training nodes (Optimizer, Loss, etc.)
3. Code → Node parsers
   - Python AST parsing
   - Pattern matching (detect CyxWiz API calls)
   - Graph reconstruction (infer connections)
4. Integration with Node Editor
   - "Export to Script" button
   - "Import from Script" button
5. Round-trip testing (ensure lossless conversion)

**Deliverable**: Users can convert between node graphs and scripts

**Dependencies**:
- Phase 1-3 completion
- Node Editor implementation (ImNodes integration)

**Example Test**:
Visual graph:
```
[Input] → [Dense(128)] → [ReLU] → [Dense(10)] → [Softmax] → [Output]
```

Generates script:
```python
# Auto-generated from node graph
x = input_tensor
layer1 = Dense(128)(x)
act1 = relu(layer1)
layer2 = Dense(10)(act1)
output = softmax(layer2)
```

Re-import script → reconstructs identical graph

---

### Phase 5: Advanced Features (2-3 weeks)

**Goal**: Polish and power-user features

**Tasks**:
1. Auto-completion in CommandWindow
   - Python introspection (`dir()`, `help()`)
   - CyxWiz API suggestions
   - Dropdown menu (ImGui combo)
2. File format handlers
   - Excel support (via `xlnt` or Python `openpyxl`)
   - HDF5 support (via **HighFive** library)
   - Table viewer for data files
3. Startup scripts (auto-run on project load)
4. Script templates (File → New → From Template)
5. Code snippets (Ctrl+Space → insert common patterns)
6. Documentation tooltips (hover over function → show docstring)

**Deliverable**: Professional scripting experience

**Dependencies**:
- Phase 1-4 completion
- HighFive library: https://github.com/BlueBrain/HighFive
- Optional: xlnt library for Excel

**Example Test**:
- Type `Tensor.r` → Auto-complete suggests `Tensor.random()`
- Open `data.hdf5` → Table view shows dataset contents
- Create project → `startup.cyx` auto-runs, loads common functions

---

### Phase 6: Optimization & Testing (1-2 weeks)

**Goal**: Performance optimization and comprehensive testing

**Tasks**:
1. Performance profiling
   - Measure Python overhead vs native C++
   - Optimize hot paths (e.g., avoid redundant copies)
2. Memory leak detection
   - Python reference counting audit
   - pybind11 lifetime management
3. Cross-platform testing
   - Windows, macOS, Linux
   - Python 3.8, 3.9, 3.10, 3.11
4. Security audit
   - Penetration testing (sandbox escapes)
   - Code review (security best practices)
5. User acceptance testing
   - Internal dogfooding
   - External beta testers

**Deliverable**: Production-ready scripting system

---

## Technology Stack & Libraries

### Core Dependencies (Already in Project)

| Library | Purpose | Status |
|---------|---------|--------|
| pybind11 | Python/C++ bindings | ✅ Integrated |
| ImGui | GUI framework | ✅ Integrated |
| spdlog | Logging | ✅ Integrated |
| fmt | String formatting | ✅ Integrated |

### New Dependencies (To Add)

| Library | Purpose | Integration | License |
|---------|---------|-------------|---------|
| **ImGuiColorTextEdit** | Syntax highlighting editor | vcpkg or submodule | MIT |
| **HighFive** | HDF5 C++ wrapper | vcpkg | Boost 1.0 |
| Python `ast` | AST parsing | Built-in (Python stdlib) | PSF |
| Python `tracemalloc` | Memory tracking | Built-in (Python 3.4+) | PSF |
| Python `signal` | Timeout mechanism | Built-in (Unix/Windows) | PSF |

### Optional Dependencies (Future)

| Library | Purpose | Priority |
|---------|---------|----------|
| xlnt | Excel file support | Medium |
| csv-parser | Fast CSV parsing | Low (can use custom) |
| Python `openpyxl` | Excel via Python | Medium |

### vcpkg.json Updates

Add to `vcpkg.json`:

```json
{
  "dependencies": [
    "imgui[docking,glfw-binding,opengl3-binding]",
    "grpc",
    "spdlog",
    "fmt",
    "nlohmann-json",
    "pybind11",
    "highfive"  // <-- ADD THIS
  ]
}
```

For ImGuiColorTextEdit (not in vcpkg):

**Option 1: Git Submodule**
```bash
cd cyxwiz-engine/external
git submodule add https://github.com/BalazsJako/ImGuiColorTextEdit
```

**Option 2: Manual Copy**
- Download release: https://github.com/BalazsJako/ImGuiColorTextEdit/releases
- Copy `TextEditor.h` and `TextEditor.cpp` to `cyxwiz-engine/src/external/`

Add to CMakeLists.txt:
```cmake
# In cyxwiz-engine/CMakeLists.txt
add_executable(cyxwiz-engine
    src/main.cpp
    # ... other sources
    src/external/TextEditor.cpp  # <-- ADD
)
```

---

## Code Examples & API Design

### Example 1: CommandWindow Usage

```cpp
// In main_window.cpp
void MainWindow::Initialize() {
    // Create shared scripting engine
    scripting_engine_ = std::make_shared<ScriptingEngine>();
    scripting_engine_->Initialize("/path/to/python");

    // Create command window with callbacks
    command_window_ = std::make_unique<CommandWindow>(scripting_engine_);

    // Set output callback to also log to file
    scripting_engine_->SetOutputCallback([this](const std::string& msg, LogLevel level) {
        command_window_->AppendOutput(msg, level);
        console_->AddLog(msg, level);  // Also log to console
    });
}
```

User interaction:
```
f:> import pycyxwiz as cyx
f:> cyx.initialize()
[INFO] CyxWiz backend initialized (version 1.0.0)
[INFO] Device: NVIDIA GeForce RTX 3080 (CUDA)

f:> x = cyx.Tensor.random([1000, 1000])
f:> y = cyx.Tensor.random([1000, 1000])
f:> z = x @ y
f:> print(z.shape())
[1000, 1000]

f:> help(cyx.Tensor.random)
Tensor.random(shape: List[int], dtype: DataType = Float32) -> Tensor
    Create a tensor with random values from uniform distribution [0, 1).

    Args:
        shape: Dimensions of the tensor
        dtype: Data type (Float32, Float64, Int32, etc.)

    Returns:
        Tensor with random values
```

---

### Example 2: Script Editor Section Execution

```cpp
// In script_editor.cpp
void ScriptEditor::RunSelectedSection() {
    auto& editor = tabs_[active_tab_index_]->editor;
    int current_line = editor.GetCursorPosition().mLine;

    // Find section containing current line
    auto sections = ParseSections(editor.GetText());
    SectionMarker* section = nullptr;
    for (auto& sec : sections) {
        if (current_line >= sec.start_line && current_line <= sec.end_line) {
            section = &sec;
            break;
        }
    }

    if (!section) {
        spdlog::warn("Cursor not in any section");
        return;
    }

    // Extract section text
    std::string section_text = editor.GetTextInRange(section->start_line, section->end_line);

    // Execute
    is_executing_ = true;
    auto result = engine_->ExecuteSection(section_text, section->start_line, section->end_line);
    is_executing_ = false;

    if (result.success) {
        spdlog::info("Section executed successfully");
        if (!result.output.empty()) {
            console_->AddInfo(result.output);
        }
    } else {
        spdlog::error("Section execution failed: {}", result.error);
        console_->AddError(result.error);

        // Highlight error line
        if (result.error_line >= 0) {
            editor.SetErrorMarkers({{result.error_line, result.error}});
        }
    }
}
```

---

### Example 3: Sandboxed Execution

```cpp
// In python_sandbox.cpp
pybind11::object PythonSandbox::Execute(
    const std::string& code,
    pybind11::dict globals,
    pybind11::dict locals
) {
    // Step 1: Parse and validate AST
    pybind11::object ast_module = pybind11::module_::import("ast");
    pybind11::object tree = ast_module.attr("parse")(code);

    // Check for dangerous constructs
    for (auto node : ast_module.attr("walk")(tree)) {
        std::string node_type = pybind11::str(node.attr("__class__").attr("__name__"));

        if (node_type == "Import" || node_type == "ImportFrom") {
            // Validate import
            pybind11::list names = node.attr("names");
            for (auto name_node : names) {
                std::string module_name = pybind11::str(name_node.attr("name"));
                if (!ValidateImport(module_name)) {
                    throw std::runtime_error("Import not allowed: " + module_name);
                }
            }
        }

        if (!policy_.allow_eval && node_type == "Call") {
            // Check for eval/exec calls
            auto func = node.attr("func");
            if (pybind11::hasattr(func, "id")) {
                std::string func_name = pybind11::str(func.attr("id"));
                if (func_name == "eval" || func_name == "exec" || func_name == "compile") {
                    throw std::runtime_error("Function not allowed: " + func_name);
                }
            }
        }
    }

    // Step 2: Replace __builtins__ with restricted version
    globals["__builtins__"] = restricted_builtins_;

    // Step 3: Start resource monitoring
    StartResourceMonitoring();

    // Step 4: Execute with timeout
    pybind11::object result;
    try {
        // On Unix: use signal.alarm() for timeout
        // On Windows: use threading.Timer
        #ifdef _WIN32
            result = ExecuteWithThreadTimeout(code, globals, locals);
        #else
            result = ExecuteWithSignalTimeout(code, globals, locals);
        #endif
    } catch (...) {
        StopResourceMonitoring();
        throw;
    }

    // Step 5: Check resource limits
    StopResourceMonitoring();
    CheckResourceLimits();

    return result;
}
```

---

### Example 4: Node to Script Conversion

```cpp
// In node_script_converter.cpp
std::string NodeScriptConverter::NodeGraphToScript(const NodeGraph& graph) {
    std::ostringstream code;

    // Header
    code << "# Auto-generated script from CyxWiz node graph\n";
    code << "# Created: " << GetCurrentTimestamp() << "\n\n";
    code << "import pycyxwiz as cyx\n\n";

    // Topological sort (ensure dependencies execute before dependents)
    auto sorted_nodes = TopologicalSort(graph);

    // Group nodes by type for better organization
    std::map<std::string, std::vector<Node*>> sections;
    for (auto* node : sorted_nodes) {
        sections[node->section].push_back(node);
    }

    // Generate code by section
    for (auto& [section_name, nodes] : sections) {
        if (!section_name.empty()) {
            code << "%% " << section_name << "\n";
        }

        for (auto* node : nodes) {
            // Get code generator for this node type
            auto it = generators_.find(node->type);
            if (it != generators_.end()) {
                std::string node_code = it->second->Generate(*node);
                code << node_code << "\n";

                // Add comment if verbosity is high
                if (comment_verbosity_ >= CommentVerbosity::Full) {
                    code << "  # Node ID: " << node->id << "\n";
                }
            } else {
                spdlog::warn("No code generator for node type: {}", node->type);
            }
        }
        code << "\n";
    }

    return code.str();
}
```

Example output:
```python
# Auto-generated script from CyxWiz node graph
# Created: 2025-11-16 15:30:00

import pycyxwiz as cyx

%% Data Loading
data = cyx.load_dataset("mnist.hdf5")
X_train, y_train = data.split(train=0.8)

%% Model Architecture
layer1 = cyx.Dense(128, activation="relu")
layer2 = cyx.Dense(64, activation="relu")
layer3 = cyx.Dense(10, activation="softmax")

%% Training
optimizer = cyx.Adam(learning_rate=0.001)
model = cyx.Sequential([layer1, layer2, layer3])
model.compile(optimizer=optimizer, loss="categorical_crossentropy")
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

%% Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

---

## Performance Considerations

### Python Overhead Analysis

**Baseline**: Native C++ call to ArrayFire
```cpp
af::array A = af::randu(1000, 1000);
af::array B = af::randu(1000, 1000);
af::array C = af::matmul(A, B);
// Time: ~0.5ms (NVIDIA RTX 3080)
```

**Via Python/pybind11**:
```python
A = cyx.Tensor.random([1000, 1000])
B = cyx.Tensor.random([1000, 1000])
C = A @ B
# Time: ~0.6ms (overhead: ~20%)
```

**Analysis**:
- pybind11 overhead: ~0.1ms per call
- Acceptable for ML workloads (operations are GPU-bound)
- For tight loops, encourage batching or vectorization

### Optimization Strategies

1. **Batch Operations** (encourage in documentation):
   ```python
   # Bad: Loop with Python overhead
   result = []
   for i in range(1000):
       result.append(A[i] * B[i])

   # Good: Vectorized operation
   result = A * B  # Single GPU kernel
   ```

2. **Lazy Evaluation** (future enhancement):
   ```python
   # Build computation graph without executing
   C = A @ B  # Just records operation
   D = C + E  # Still not executed
   result = D.eval()  # Execute entire graph at once
   ```

3. **JIT Compilation** (future enhancement):
   - Use Python `numba` for JIT-compiling hot loops
   - Compile frequently-used scripts to native code

4. **Minimize Python ↔ C++ Transfers**:
   - Keep tensors on GPU
   - Only transfer results back to Python when necessary

---

## Open Questions & Future Enhancements

### Open Questions (Need Decisions)

1. **Multi-Threading**:
   - Should scripts run in background thread to avoid blocking UI?
   - Pro: Responsive UI during long-running scripts
   - Con: Thread safety complexity (pybind11 GIL management)
   - **Recommendation**: Start with main thread, add threading in Phase 6

2. **Debugger Integration**:
   - Should we support breakpoints and step-through debugging?
   - Requires Python `pdb` module or custom debugger
   - **Recommendation**: Defer to Phase 7+ (nice-to-have)

3. **Package Management**:
   - Should users be able to install Python packages (pip)?
   - Security risk: arbitrary code execution
   - **Recommendation**: Curated package whitelist (numpy, scipy, matplotlib)

4. **Cloud Execution**:
   - Should scripts be able to submit jobs to Server Nodes?
   - Enables distributed computing from scripts
   - **Recommendation**: Phase 8+ (requires gRPC integration)

5. **Version Control Integration**:
   - Should script editor have Git integration?
   - Show diff, commit, blame annotations
   - **Recommendation**: Out of scope (use external tools)

### Future Enhancements (Roadmap)

**Phase 7: Debugger (4-6 weeks)**
- Breakpoint support in ScriptEditor
- Step-through execution (step over, step into, continue)
- Variable inspector (watch expressions)
- Call stack visualization
- Integration with Python `pdb` or custom debugger

**Phase 8: Cloud Integration (3-4 weeks)**
- Submit scripts to Server Nodes for execution
- Monitor remote execution progress
- Retrieve results asynchronously
- Cost estimation (CYXWIZ token usage)

**Phase 9: Collaborative Editing (6-8 weeks)**
- Real-time collaborative script editing (like Google Docs)
- User cursors and selections visible
- WebRTC or WebSocket synchronization
- Conflict resolution (operational transforms)

**Phase 10: Package Marketplace (8-10 weeks)**
- Community-contributed script packages
- NFT-based licensing (on Solana)
- Verified publisher system
- Dependency management (like npm/pip)
- Automatic updates

**Phase 11: Visual Scripting (8-12 weeks)**
- Hybrid node/code editor (inline code in nodes)
- Visual loops, conditionals (flowchart-style)
- Auto-convert control flow to visual nodes
- Best of both worlds: visual structure + code flexibility

**Phase 12: Jupyter Notebook Integration (4-6 weeks)**
- Import/export Jupyter `.ipynb` files
- Render markdown cells
- Support for rich output (plots, tables, HTML)
- Two-way sync with JupyterLab

---

## Success Criteria & Testing

### Acceptance Criteria

- [ ] User can execute Python commands in CommandWindow and see results
- [ ] User can write and execute `.cyx` scripts with section markers
- [ ] Scripts run in sandboxed environment (imports restricted, timeout enforced)
- [ ] User can convert node graph to script and back (round-trip)
- [ ] CSV, Excel, HDF5 files open in table viewer
- [ ] Auto-completion works in CommandWindow (Tab key)
- [ ] Syntax highlighting works for Python and `.cyx` extensions
- [ ] Multiple editor tabs can be open simultaneously
- [ ] Unsaved changes are indicated with `*` in tab title
- [ ] Scripts can access `pycyxwiz` module and perform tensor operations
- [ ] No memory leaks after 1000+ command executions
- [ ] No successful sandbox escapes in penetration testing

### Performance Benchmarks

- [ ] Matrix multiplication (1000x1000): < 1ms overhead vs native C++
- [ ] Command execution latency: < 50ms from Enter press to result display
- [ ] Script editor: Syntax highlighting updates < 16ms (60 FPS)
- [ ] Auto-completion: Suggestions appear < 100ms after Tab press
- [ ] File open: < 500ms for 10,000-line script file

### Security Testing

- [ ] Attempt to import `os` module → blocked
- [ ] Attempt to call `eval()` → blocked
- [ ] Infinite loop → terminated after timeout
- [ ] Memory bomb (allocate 10GB) → stopped at limit
- [ ] Read file outside project directory → access denied
- [ ] Network socket creation → blocked
- [ ] Subprocess execution → blocked

---

## Conclusion

The CyxWiz Scripting System is a comprehensive feature that transforms the Engine from a visual-only tool into a flexible hybrid environment. By combining:

1. **CommandWindow** - Interactive experimentation (MATLAB-style REPL)
2. **ScriptEditor** - Full code editing with syntax highlighting
3. **PythonSandbox** - Secure execution with resource limits
4. **NodeScriptConverter** - Seamless visual ↔ code translation
5. **FileFormatHandler** - Multi-format data support

...we enable users to work in the way that suits them best, while maintaining security, performance, and integration with the broader CyxWiz ecosystem.

The phased implementation plan (6 phases over ~16-20 weeks) ensures we deliver value incrementally while managing technical risk. The security-first architecture protects users from malicious scripts, and the performance optimizations ensure Python doesn't become a bottleneck for GPU-accelerated ML workloads.

This design balances **innovation** (bidirectional conversion, sandboxing) with **practicality** (leveraging proven libraries like ImGuiColorTextEdit and HighFive), and **security** (defense-in-depth) with **usability** (familiar MATLAB/IPython interface).

---

**Next Steps**:

1. **Review this document** with the team
2. **Approve Phase 1 scope** and begin implementation
3. **Set up project tracking** (GitHub issues/milestones)
4. **Assign developers** to CommandWindow and ScriptingEngine
5. **Begin dependency integration** (ImGuiColorTextEdit, HighFive)

**Questions or Feedback**: Contact the CTO/Architect

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Authors**: CTO/Senior Architect
