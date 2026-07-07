# 39) Phase 9 design register (project session, Python environment, and registry hygiene contracts)

## 39.1 Objective
Document lifecycle contracts that bind project state, Python environment provisioning, UI/session boundaries, and dataset registry hygiene into a coherent session model. This section focuses on what is guaranteed by implementation and what is intentionally relaxed for compatibility.

## 39.2 ASCII session lifecycle

```text
Start page / menu action
  -> CreateProject / OpenProject / ContinueWithoutProject
  -> ProjectManager state transition (ActiveProject)
  -> UI subscription callbacks fire
     -> OnProjectOpened:
        - restore persisted settings/layout roots
        - asset browser refresh
        - optional python reload
  -> data load + training preparation
  -> training / edits / node changes
  -> user-triggered close project or app shutdown
     -> toolbar pre-save hook (settings)
     -> OnProjectClosed:
        - python reload on boundary
        - asset browser clear
        - ClearAllTabularDatasets
        - next-project layout reset flag
  -> Process repeats with isolated registry state
```

## 39.3 ProjectManager contract model

### 39.3.1 Core state contract
- `ProjectManager` is a global singleton owning:
  - `project_root_`, `project_name_`, `project_file_path_`
  - `ProjectConfig` (filters, open scripts, editor settings, recent files)
  - persistent recent-project list.
- `HasActiveProject()` is derived directly from non-empty `project_root_`.
- Path utility contract:
  - when no active project, helpers return empty strings,
  - when active, return absolute project child paths (`scripts`, `models`, `datasets`, `checkpoints`, `exports`, `plugins`, `layout.ini`).

### 39.3.2 Session transitions
- Create:
  - closes active project first,
  - creates directory structure,
  - initializes default config (`version`, timestamps, description, default filters),
  - writes `.cyxwiz`,
  - queues venv creation,
  - adds to recent list,
  - fires `on_opened_`.
- Open:
  - closes active project first,
  - validates `.cyxwiz` file,
  - reads and applies config,
  - computes project root/name from file path,
  - initializes default filters if empty,
  - updates recent list,
  - fires `on_opened_`,
  - runs venv bootstrap for legacy folders lacking `python_env.json` (async).
- Close:
  - clears all in-memory project state and config,
  - fires `on_closed_` with old root.
- Save / SaveAs:
  - Save: writes `.cyxwiz` after potentially normalizing `python_env`,
  - SaveAs: clones project directory recursively, updates name/path, updates `.cyxwiz`,
    refreshes python env path if needed, writes new file, and reopens event callback to new root.

### 39.3.3 Callback contract
- `SetOnProjectOpened(cb)` -> invoked after successful create/open.
- `SetOnProjectClosed(cb)` -> invoked once close clears state.
- `SetOnProjectVenvReady(cb)` -> invoked when async venv creation succeeds for a project.
- These callbacks are stored on the singleton and consumed by UI setup code at startup.

## 39.4 UI project contract with MainWindow

### 39.4.1 Callback registration contract
`MainWindow` registers all three project callbacks in constructor:
- open callback -> `MainWindow::OnProjectOpened`
- close callback -> `MainWindow::OnProjectClosed`
- venv-ready callback -> `MainWindow::OnProjectVenvReady`

### 39.4.2 Open contract (on boundary enter)
`OnProjectOpened(project_root)`:
- logs open event,
- calls `LoadProjectSettings`:
  - restores script editor theme/scale/tab sizing/settings,
  - applies app theme and UI scale,
  - loads layout scaffold,
  - reopens persisted open script list (path-resolved relative to project),
  - applies active script index.
- if scripting is initialized, reloads python for project context.
- updates `asset_browser_` root and refreshes.

### 39.4.3 Close contract (on boundary exit)
`OnProjectClosed(project_root)`:
- reloads python context if needed after boundary,
- clears asset browser,
- calls `DataRegistry::Instance().ClearAllTabularDatasets()`,
- sets `first_time_layout_ = true` for default dock layout next render.

### 39.4.4 Venv-ready contract
`OnProjectVenvReady(project_root)`:
- ignores callback if no active project or stale root mismatch,
- if scripting is initialized, triggers python reload for the active project.

### 39.4.5 Close action wiring
`toolbar_file_menu.cpp` close and save actions enforce the ordering:
- save settings callback when available before project close,
- call `ProjectManager::Instance().CloseProject()` as the authoritative close event.

### 39.4.6 New project/open flow contract
- Start page and toolbar flows all route through:
  - `ProjectManager::CreateProject(...)`
  - `ProjectManager::OpenProject(path)`
- New project action also writes initial `.cyxwiz` and resolves selected path into startup project result state.

## 39.5 Python environment lifecycle contract

### 39.5.1 Provisioning rules
- `ProjectManager` supports:
  - legacy detection via `python_env.json`,
  - default interpreter probe/writer helpers,
  - venv path normalization for relative file entries.
- Project create/open can trigger async venv creation (`AsyncTaskManager::RunAsync`) with:
  - primary task creating venv with system Python,
  - completion callback:
    - on success calls `NotifyProjectVenvReady`,
    - on failure logs error.

### 39.5.2 Interpreter/path invariants
- `MaybeUpdateProjectPythonEnv` stores interpreter path in project (`python_env.json`) only when relative path is valid under project root.
- Existing outside-root custom interpreter is preserved.
- SaveAs migration updates `python_env.json` when env path moved into new root or from old root context.

## 39.6 Project scope vs data scope contract

### 39.6.1 Registry surface split
`DataRegistry` stores multiple dataset categories:
- classic in-memory `datasets_` (legacy typed datasets),
- columnar `arrow_datasets_`,
- disk-backed `parquet_backed_datasets_`,
- image/audio/text metadata entries.

### 39.6.2 Project boundary cleanup
- On project close boundary, `ClearAllTabularDatasets()` explicitly clears:
  - `arrow_datasets_`,
  - `parquet_backed_datasets_`,
  - `image_dataset_entries_`,
  - `audio_dataset_entries_`,
  - `text_dataset_entries_`.
- It does not clear `datasets_` and not all preprocessing metadata maps.
- This is a deliberate scoped cleanup contract tuned to known auto-generated collision classes.

### 39.6.3 Why this exists
- Prevents stale tabular and metadata entries from one project from colliding with names generated in the next project (especially auto-named datasets),
- keeps training dispatch deterministic when entering `IsArrowDataset` / `IsImageDataset` / `IsTextDataset` routing.

## 39.7 Project + runtime boundary risks
- Partial hygiene scope:
  - dataset maps are split by storage backend; only tabular/metadata families are guaranteed cleared at close.
  - any non-tabular legacy map entries may persist across boundaries unless separately unloaded.
- This keeps compatibility with existing training paths but is a maintenance risk if new loaders grow and rely on full global clear.
- Shutdown safety still depends on main-thread ordered shutdown from control-plane contracts documented in Sections 38 and 34.

## 39.8 ASCII boundary map

```text
ProjectManager::CloseProject()
  -> state clear (root/name/file/config)
  -> on_closed callback
       MainWindow::OnProjectClosed
          -> script engine reload guard
          -> asset_browser clear
          -> DataRegistry::ClearAllTabularDatasets
          -> first_time_layout_ = true
```

```text
Open flow
  StartPage::OpenProject(path)
    -> ProjectManager::OpenProject(path)
      -> on_project_opened callback
         -> MainWindow::OnProjectOpened
            -> Settings restore + asset refresh + python reload
```

## 39.9 Evidence anchors

| Contract area | Evidence |
|---|---|
| Project singleton lifecycle contracts | `cyxwiz-engine/src/core/project_manager.h` and `cyxwiz-engine/src/core/project_manager.cpp` |
| Callback wiring and open/close/venv handlers | `cyxwiz-engine/src/gui/main_window.cpp` |
| Project action entry points (file menu, start page, project dialogs) | `cyxwiz-engine/src/gui/panels/toolbar_file_menu.cpp`, `cyxwiz-engine/src/gui/dialogs/start_page.cpp`, `cyxwiz-engine/src/gui/dialogs/project_selection_dialog.cpp` |
| Registry hygiene contract | `cyxwiz-engine/src/core/data_registry.h`, `cyxwiz-engine/src/core/data_registry_utils.cpp`, `cyxwiz-engine/src/core/data_registry_core.cpp` |
| Python env file and async bootstrap path | `cyxwiz-engine/src/core/project_manager.cpp` |
| Session close pre-save flow | `cyxwiz-engine/src/gui/panels/toolbar.h`, `cyxwiz-engine/src/gui/panels/toolbar_file_menu.cpp`, `cyxwiz-engine/src/gui/main_window.cpp` |

## 39.10 Lean guardrail assessment
- Essential retained: one singleton project domain, one explicit close/open contract, and boundary-level dataset cleanup.
- Optional complexity to challenge:
  - widen session cleanup to a typed session scope object if registry growth makes global clear semantics insufficient.
- No new abstractions were added; this register stays documentation-only with explicit contracts from existing code.

