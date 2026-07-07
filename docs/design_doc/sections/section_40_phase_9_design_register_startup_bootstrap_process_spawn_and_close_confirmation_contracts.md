# 40) Phase 9 design register (startup bootstrap, process spawn/restart, and close-confirmation contracts)

## 40.1 Objective
Define the cross-cutting contracts that gate how the engine starts, how new engine windows are created, how projects/graphs are bootstrapped into the main UI, and how close safety is enforced before final shutdown.

This section connects:
- process launch (native entrypoint),
- app-level startup phase transitions,
- start-page and wizard gates,
- main-window activation conditions,
- and close confirmation behavior.

## 40.2 ASCII startup and exit sequence

```text
main(argc, argv)
  -> set CYXWIZ_LAUNCH_CWD
  -> set cwd to executable directory
  -> init logging + backend + plugin search paths
  -> construct CyxWizApp
      -> ProcessCommandLine()
      -> Initialize()
          -> glfw+imgui/implot/imnodes context
          -> python scan (non-initialized)
          -> if no python configured -> PythonSetupWizard active
          -> else -> StartPage active
  -> app.Run() loop
      -> wizard page OR start page OR main window render
      -> async task drain + job_manager update
      -> close confirmation state machine
  -> app.Shutdown() / _exit(0)
```

```text
WindowManager::LaunchWindow(project)
  -> resolve executable path
  -> spawn new process with [exe, --project, path]
  -> new process returns to ProcessCommandLine
  -> resolves and opens project on startup
```

## 40.3 Process entrypoint contract

### 40.3.1 Launch-time environment contract
`main.cpp` sets:
- `CYXWIZ_LAUNCH_CWD` env variable with launching process cwd,
- working directory to executable directory (best effort),
- logger sinks (file `engine_log.txt` + console),
- backend/device discovery, and
- plugin search-path configuration before entering application run loop.

This gives deterministic relative path behavior for first-run project resolution and CLI parsing.

### 40.3.2 Command-line parse contract
`CyxWizApp::ProcessCommandLine` scans argv from index 1 onward and sets `startup_project_path_` to the first resolvable project-style argument (`ResolveProjectArg`).

Current behavior:
- treats any non-option argument position as candidate project path,
- uses `ProjectManager::ResolveProjectFilePath` (file or folder inference),
- ignores parser failures silently and keeps scanning subsequent args.

`--project` emitted by `WindowManager` is therefore effectively a positional flag consumed as “not a path,” with the real path expected in the following token.

## 40.4 Boot contract state machine

### 40.4.1 Python provisioning gate
- `Initialize()` always performs a startup Python scan (`ScanForPython`) but does not initialize runtime Python engine.
- If no system python is configured:
  - creates `PythonSetupWizard`,
  - keeps rendering pipeline in wizard phase until completion/cancel.
- If configured:
  - creates `StartPage`,
  - waits for explicit user selection (`project`, `example graph`, `continue without`).

### 40.4.2 Wizard contract
`PythonSetupWizard::Render` remains active until it returns `Completed` or `Cancelled`.
- Completed:
  - `CyxWizApp::Render` transitions from wizard to `StartPage`,
  - `python_configured_ = true`.
- Cancelled:
  - app flags `glfwSetWindowShouldClose(window_, GLFW_TRUE)` and exits loop.

### 40.4.3 Start page contract
`start_page_` is rendered when Python is configured and no project selected.

Result transitions:
- `ProjectSelected` -> `startup_project_path_` set, `startup_graph_path_` cleared.
- `ExampleGraphSelected` -> `startup_graph_path_` set, project path cleared.
- `ContinueWithout` -> both paths cleared, but `project_selected_ = true`.
- `Exit` -> close request.

### 40.4.4 Main-window activation contract
Creation condition:
- `python_configured_ == true`,
- `project_selected_ == true`,
- `main_window_ == nullptr`.

Activation steps:
- instantiate `gui::MainWindow`,
- call `OpenStartupProjectIfRequested`,
- call `OpenStartupGraphIfRequested`,
- update title,
- initialize `grpc_client_` + `job_manager_`,
- inject networking and logging flags into UI,
- register exit callback.

This is the boundary where startup project selection moves from boot artifacts to live runtime state.

## 40.5 Multi-window and restart contracts

### 40.5.1 New window contract
`WindowManager::LaunchWindow(project_path)`:
- validates project path exists,
- resolves executable path,
- spawns detached child process (Windows `CreateProcessA`, Unix `fork+execv`),
- passes `--project <path>` args.

### 40.5.2 Dialog-based window contract
`WindowManager::LaunchWindowWithDialog()` launches the same executable with no args; startup follows normal start page path.

### 40.5.3 Restart contract
`WindowManager::RestartEngine(project_path)` calls launch + caller is responsible for closing current process.

`toolbar_file_menu` restart action uses this path and then exits current app via callback if successful.

### 40.5.4 Open gap
`SwitchProject` is TODO:
- logs that automatic close is not yet implemented,
- launches new window only.

This is an explicit incomplete automation area in process lifecycle.

## 40.6 Close-safety confirmation contract
`CyxWizApp::Run` checks `glfwWindowShouldClose` and runs gate checks in order:
1. if running script -> show close confirmation popup (`HandleCloseConfirmation`),
2. else if unsaved files -> show unsaved popup (`HandleUnsavedConfirmation`),
3. else if dataset memory loaded -> show dataset unload popup (`HandleDataLoadedConfirmation`),
4. else proceed to break loop.

`HandleDataLoadedConfirmation` gives explicit data-loss warning and offers unload-before-close path (`DataRegistry::UnloadAll`).

These gates enforce user intention before `main_window_->PrepareForShutdown` and final teardown.

## 40.7 Close confirmation popup contracts

### 40.7.1 Script running
Buttons:
- Stop Script & Close,
- Force Close,
- Cancel.

### 40.7.2 Unsaved files
Buttons:
- Save All & Close,
- Discard & Close (force-close mode),
- Cancel.

### 40.7.3 Loaded data
Buttons:
- Unload & Close,
- Cancel.

Close is only allowed when one path returns to loop termination (`running_ = false` or close flag).

## 40.8 Lean guardrail assessment
- Gate split is explicit and user-visible; avoids silent exit with script/data/user-work loss.
- Process spawn is intentionally simple (`CreateProcessA`/`fork+execv`) and stateless from app side.
- Deferred opportunities:
  - robust argument parser with named flags for graphs/projects,
  - complete `SwitchProject` transactional close sequence.

## 40.9 Evidence anchors

| Layer | Evidence |
|---|---|
| Entrypoint and boot init | `cyxwiz-engine/src/main.cpp` |
| App command-line/project parsing | `cyxwiz-engine/src/application.cpp` (`ResolveProjectArg`, `ProcessCommandLine`) |
| Startup phase transitions | `cyxwiz-engine/src/application.cpp` (`Initialize`, `Render`, `OpenStartupProjectIfRequested`, `OpenStartupGraphIfRequested`) |
| Wizard behavior | `cyxwiz-engine/src/gui/dialogs/python_setup_wizard.h` and `.cpp` |
| Start-page transition contract | `cyxwiz-engine/src/gui/dialogs/start_page.h` and `.cpp` |
| Multi-window process spawn | `cyxwiz-engine/src/core/window_manager.h` and `.cpp` |
| Close gating + unload contract | `cyxwiz-engine/src/application.h` and `application.cpp` (`ShouldPreventClose`, `HasUnsavedWork`, `HasLoadedData`, confirmation handlers) |
| Restart action wiring | `cyxwiz-engine/src/gui/panels/toolbar_file_menu.cpp` |

