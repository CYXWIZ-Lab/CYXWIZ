# Asset Browser System Architecture

## Overview

The Asset Browser provides a dual-view system (Filter View vs Directory View) similar to Visual Studio's Solution Explorer. It integrates with the project system to manage all project assets with proper working directory context.

---

## Core Components

### 1. ProjectManager (New - Singleton)

**Location:** `cyxwiz-engine/src/core/project_manager.h/cpp`

**Purpose:** Global project state management and working directory tracking.

```cpp
class ProjectManager {
public:
    static ProjectManager& Instance();

    // Project lifecycle
    bool CreateProject(const std::string& name, const std::string& location);
    bool OpenProject(const std::string& cyxwiz_file_path);
    void CloseProject();
    bool SaveProject();

    // Accessors
    bool HasActiveProject() const;
    const std::string& GetProjectRoot() const;      // Absolute path to project dir
    const std::string& GetProjectName() const;
    const ProjectConfig& GetConfig() const;

    // Path utilities (all return absolute paths)
    std::string GetScriptsPath() const;             // {root}/scripts/
    std::string GetModelsPath() const;              // {root}/models/
    std::string GetDatasetsPath() const;            // {root}/datasets/
    std::string GetCheckpointsPath() const;         // {root}/checkpoints/
    std::string GetExportsPath() const;             // {root}/exports/
    std::string GetPluginsPath() const;             // {root}/plugins/

    // Resolve relative path to absolute
    std::string ResolveAssetPath(const std::string& relative_path) const;

    // Callbacks for state changes
    using ProjectCallback = std::function<void(const std::string& project_root)>;
    void SetOnProjectOpened(ProjectCallback callback);
    void SetOnProjectClosed(ProjectCallback callback);

private:
    std::string project_root_;
    std::string project_name_;
    ProjectConfig config_;
    ProjectCallback on_opened_;
    ProjectCallback on_closed_;
};

struct ProjectConfig {
    std::string name;
    std::string version;
    std::time_t created;
    std::string description;
    std::vector<std::string> recent_files;
    // Filter definitions (custom filters can be added)
    std::map<std::string, std::vector<std::string>> filters; // filter_name -> extensions
};
```

---

### 2. AssetBrowserPanel (Enhanced)

**Location:** `cyxwiz-engine/src/gui/panels/asset_browser.h/cpp`

**UI Layout:**
```
+-----------------------------------------------+
| [Filter Icon] [Folder Icon] | [Search...    ] |  <- Toolbar
| [Refresh]                                     |
+-----------------------------------------------+
| FILTER VIEW (when active):                    |
| > Scripts                                     |
|   - train.py                                  |
|   - utils.py                                  |
| > Models                                      |
|   - model_v1.h5                               |
| > Datasets                                    |
|   - train_data.csv                            |
| > Checkpoints                                 |
| > Exports                                     |
| > Plugins                                     |
+-----------------------------------------------+
| DIRECTORY VIEW (when active):                 |
| > scripts/                                    |
|   - train.py                                  |
|   - utils.py                                  |
| > models/                                     |
|   - model_v1.h5                               |
| > datasets/                                   |
+-----------------------------------------------+
| Status: 15 items | Project: MyProject         |  <- Status bar
+-----------------------------------------------+
```

**Enhanced Class Structure:**

```cpp
enum class AssetViewMode {
    FilterView,     // Group by asset type (like VS filters)
    DirectoryView   // Show actual filesystem structure
};

enum class AssetType {
    Script,         // .py, .cyx
    Model,          // .h5, .onnx, .pt, .safetensors
    Dataset,        // .csv, .json, .parquet, .h5
    Checkpoint,     // .ckpt, .pt
    Export,         // .onnx, .gguf, .lora
    Plugin,         // .dll, .so, .dylib
    Folder,         // Directory
    Unknown         // Unrecognized file
};

struct AssetItem {
    std::string name;               // Display name
    std::string relative_path;      // Path relative to project root
    std::string absolute_path;      // Full filesystem path
    AssetType type;
    bool is_directory;
    bool is_expanded = false;
    std::vector<std::unique_ptr<AssetItem>> children;

    // For filter view - which filter this belongs to
    std::string filter_category;
};

class AssetBrowserPanel : public Panel {
public:
    AssetBrowserPanel();
    void Render() override;

    // Project integration
    void SetProjectRoot(const std::string& root);
    void Refresh();                 // Rescan filesystem
    void Clear();                   // Clear all assets

    // View mode
    void SetViewMode(AssetViewMode mode);
    AssetViewMode GetViewMode() const;

    // Callbacks for asset operations
    using AssetCallback = std::function<void(const AssetItem&)>;
    void SetOnAssetDoubleClick(AssetCallback callback);
    void SetOnAssetDeleted(AssetCallback callback);

private:
    // UI Rendering
    void RenderToolbar();
    void RenderSearchBar();
    void RenderFilterView();
    void RenderDirectoryView();
    void RenderAssetNode(AssetItem& item, int depth = 0);
    void RenderContextMenu();
    void RenderStatusBar();

    // Asset operations
    void ScanProjectDirectory();
    void BuildFilterTree();
    void BuildDirectoryTree();
    AssetType DetermineAssetType(const std::string& path);
    std::string GetAssetIcon(AssetType type);

    // Context menu actions
    void CreateNewScript();
    void DeleteSelectedAsset();
    void RenameSelectedAsset();
    void UnloadSelectedAsset();
    void OpenInExplorer();
    void OpenInTerminal();
    void ShowDebugInfo();

    // Search/Filter
    void FilterAssets(const std::string& query);
    bool MatchesSearch(const AssetItem& item, const std::string& query);

    // State
    std::string project_root_;
    AssetViewMode view_mode_ = AssetViewMode::FilterView;
    char search_buffer_[256] = {0};

    // Asset trees
    std::unique_ptr<AssetItem> filter_root_;      // For filter view
    std::unique_ptr<AssetItem> directory_root_;   // For directory view

    // Selection
    AssetItem* selected_item_ = nullptr;
    AssetItem* context_menu_item_ = nullptr;

    // Dialogs
    bool show_rename_dialog_ = false;
    bool show_delete_confirm_ = false;
    bool show_new_script_dialog_ = false;
    char rename_buffer_[256] = {0};
    char new_script_name_[256] = {0};

    // Callbacks
    AssetCallback on_double_click_;
    AssetCallback on_deleted_;
};
```

---

### 3. Filter Definitions

**Default Filters (matching project directories):**

| Filter Name | Extensions | Directory |
|-------------|------------|-----------|
| Scripts | .py, .cyx | scripts/ |
| Models | .h5, .onnx, .pt, .safetensors, .bin | models/ |
| Datasets | .csv, .json, .parquet, .h5, .arrow | datasets/ |
| Checkpoints | .ckpt, .pt, .checkpoint | checkpoints/ |
| Exports | .onnx, .gguf, .lora, .safetensors | exports/ |
| Plugins | .dll, .so, .dylib | plugins/ |

---

### 4. Context Menu Actions

**Right-click menu items:**

```
+------------------------+
| New Script...     Ctrl+N |
|--------------------------|
| Open                     |
| Open With...             |
|--------------------------|
| Rename              F2   |
| Delete            Delete |
|--------------------------|
| Debug Info               |
| Unload from Project      |
|--------------------------|
| Open in Explorer         |
| Open in Terminal         |
+------------------------+
```

**Action Implementations:**

| Action | Behavior |
|--------|----------|
| **New Script** | Opens dialog to create .py or .cyx file in scripts/ |
| **Delete** | Confirmation dialog, then filesystem delete |
| **Rename** | Inline rename with Enter to confirm, Esc to cancel |
| **Debug Info** | Shows file path, size, modified date, type |
| **Unload** | Removes from project (doesn't delete file) |
| **Open in Explorer** | `explorer.exe /select,{path}` on Windows |
| **Open in Terminal** | Opens cmd/powershell in directory |

---

### 5. Integration Points

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Toolbar       │────>│  ProjectManager  │<────│  Asset Browser  │
│  (Create/Open)  │     │   (Singleton)    │     │    (Panel)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               │ Callbacks
                               ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Script Editor  │<────│   MainWindow     │────>│   Node Editor   │
│  (Open .py)     │     │  (Coordinates)   │     │  (Drag Models)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**Callback Flow:**

1. **Toolbar** creates/opens project → calls `ProjectManager::CreateProject()` or `OpenProject()`
2. **ProjectManager** sets project root → fires `on_opened_` callback
3. **MainWindow** receives callback → calls `AssetBrowser::SetProjectRoot()` and `Refresh()`
4. **AssetBrowser** scans filesystem → builds filter/directory trees
5. User double-clicks asset → **AssetBrowser** fires `on_double_click_` callback
6. **MainWindow** receives callback → opens in appropriate panel (Script Editor, Table Viewer, etc.)

---

### 6. File Operations

**Path Resolution (always absolute):**

```cpp
// In any component that needs to access a file:
auto& pm = ProjectManager::Instance();
if (pm.HasActiveProject()) {
    std::string script_path = pm.ResolveAssetPath("scripts/train.py");
    // Returns: "C:/Projects/MyProject/scripts/train.py"
}
```

**Import File Flow:**

```cpp
// When user imports a file from outside project:
void ImportFile(const std::string& external_path) {
    auto& pm = ProjectManager::Instance();
    AssetType type = DetermineAssetType(external_path);

    std::string target_dir;
    switch (type) {
        case AssetType::Script:  target_dir = pm.GetScriptsPath(); break;
        case AssetType::Model:   target_dir = pm.GetModelsPath(); break;
        case AssetType::Dataset: target_dir = pm.GetDatasetsPath(); break;
        // ...
    }

    std::filesystem::copy(external_path, target_dir);
    asset_browser_->Refresh();
}
```

---

### 7. Implementation Order

**Phase 1: Core Infrastructure**
1. Create `ProjectManager` singleton
2. Update `Toolbar` to use ProjectManager for create/open
3. Parse `.cyxwiz` project file on open

**Phase 2: Asset Browser Enhancement**
4. Add view mode toggle (Filter/Directory)
5. Implement `ScanProjectDirectory()` filesystem scanning
6. Build filter tree from scanned files
7. Build directory tree from scanned files

**Phase 3: UI Polish**
8. Add toolbar with view toggle, search, refresh buttons
9. Add status bar with item count
10. Implement search filtering

**Phase 4: Context Menu**
11. Implement right-click context menu
12. Add "New Script" dialog
13. Add "Rename" inline editing
14. Add "Delete" with confirmation
15. Add "Open in Explorer/Terminal"

**Phase 5: Integration**
16. Connect double-click to Script Editor
17. Add drag-drop support to Node Editor
18. Implement file watching for auto-refresh

---

### 8. File Structure After Implementation

```
cyxwiz-engine/
├── src/
│   ├── core/
│   │   ├── project_manager.h      [NEW]
│   │   └── project_manager.cpp    [NEW]
│   ├── gui/
│   │   ├── panels/
│   │   │   ├── asset_browser.h    [ENHANCED]
│   │   │   ├── asset_browser.cpp  [ENHANCED]
│   │   │   ├── toolbar.h          [MODIFIED]
│   │   │   └── toolbar.cpp        [MODIFIED]
│   │   └── main_window.cpp        [MODIFIED - add callbacks]
│   └── application.cpp            [MODIFIED - init ProjectManager]
```

---

### 9. UI Mockup - Filter View

```
┌─ Asset Browser ─────────────────────────────────┐
│ [F] [D]  │  🔍 Search assets...      │ [↻]    │
├───────────────────────────────────────────────────┤
│ ▼ 📜 Scripts (3)                                  │
│   ├─ 📄 train.py                                  │
│   ├─ 📄 evaluate.py                               │
│   └─ 📄 utils.py                                  │
│ ▶ 🧠 Models (1)                                   │
│ ▶ 📊 Datasets (2)                                 │
│ ▶ 💾 Checkpoints (0)                              │
│ ▶ 📦 Exports (0)                                  │
│ ▶ 🔌 Plugins (0)                                  │
├───────────────────────────────────────────────────┤
│ 6 items │ MyProject                               │
└───────────────────────────────────────────────────┘
```

### 10. UI Mockup - Directory View

```
┌─ Asset Browser ─────────────────────────────────┐
│ [F] [D]  │  🔍 Search assets...      │ [↻]    │
├───────────────────────────────────────────────────┤
│ ▼ 📁 scripts/                                     │
│   ├─ 📄 train.py                                  │
│   ├─ 📄 evaluate.py                               │
│   └─ 📄 utils.py                                  │
│ ▼ 📁 models/                                      │
│   └─ 📄 model_v1.h5                               │
│ ▼ 📁 datasets/                                    │
│   ├─ 📄 train.csv                                 │
│   └─ 📄 test.csv                                  │
│ ▶ 📁 checkpoints/                                 │
│ ▶ 📁 exports/                                     │
├───────────────────────────────────────────────────┤
│ 6 items │ C:/Projects/MyProject                   │
└───────────────────────────────────────────────────┘
```

---

### 11. Key Technical Decisions

1. **Absolute paths internally** - All paths stored and used are absolute for reliability
2. **Relative paths in UI** - Display relative paths for readability
3. **Filter = Virtual grouping** - Filters don't correspond to folders, they categorize by file type
4. **Directory = Real filesystem** - Directory view shows actual folder structure
5. **Lazy loading** - Only scan directories when expanded (for large projects)
6. **No file watching initially** - Use manual refresh button; add file watcher in future

---

### 12. Panel Consolidation

**Deprecated Panels:**

The enhanced Asset Browser replaces the need for standalone panels:

| Deprecated Panel | Replacement |
|------------------|-------------|
| **Dataset Manager** | Asset Browser → Datasets filter |

**Rationale:**
- Dataset Manager functionality is fully covered by the "Datasets" filter in Asset Browser
- Double-clicking a dataset file opens the appropriate viewer (CSV viewer, JSON viewer, etc.)
- Context menu provides all dataset operations (delete, rename, debug info)
- Reduces UI complexity and consolidates asset management in one place

**Migration:**
1. Remove `DatasetManager` panel class from `cyxwiz-engine/src/gui/panels/`
2. Remove registration in `MainWindow::RegisterPanelsWithSidebar()`
3. Remove from sidebar panel list
4. Ensure double-click on dataset files triggers appropriate viewer

---

This design provides a solid foundation for a professional asset management system that integrates well with the existing CyxWiz Engine architecture.
