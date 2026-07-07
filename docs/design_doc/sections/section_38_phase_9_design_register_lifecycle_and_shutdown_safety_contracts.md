# 38) Phase 9 design register (lifecycle and shutdown safety contracts)

## 38.1 Objective
Define the lifecycle contracts for:
- process startup,
- runtime stop/teardown hooks,
- object ownership release order,
- and finalization constraints that prevent cross-subsystem crashes during close.

This section is specifically about making the system safe when users or the host close the app during active background work.

## 38.2 ASCII lifecycle overview

```text
CyxWizApp constructor
  -> Initialize() (GLFW + ImGui contexts + Python scan + startup/start page)
  -> main loop
     -> Render/Update tick
     -> task callbacks drained on main thread
     -> training/task panel visibility checks
  -> Shutdown()
      -> stop script logging path
      -> reset main_window_
      -> cleanup OpenGL/ImGui contexts
      -> destroy window/system resources
      -> _exit(0)

MainWindow destructor (nested lifecycle):
  if TrainingManager.IsTrainingActive()
      StopTraining()
      WaitForTrainingStop()
  destroy plot/test python-owned widgets first
  destroy scripting_engine_ later
```

## 38.3 Startup contract

### 38.3.1 Boot phases
- `CyxWizApp` bootstrap does:
  - native context initialization (GLFW/OpenGL),
  - ImGui/ImPlot/ImNodes context setup,
  - Python scan only (no initialization),
  - Python/wizard/start-page gating.
- On Python configured + project selection:
  - `main_window_` is created,
  - startup project/graph are optionally loaded,
  - networking components are injected into UI,
  - console sink is attached to GUI console.

### 38.3.2 MainWindow construction contract
- `MainWindow` owns rich UI and cross-subsystem references.
- Key subsystems created and wired in constructor:
  - `training_plot_panel_`,
  - `scripting_engine_`,
  - Python integrations (`RegisterTrainingDashboard` etc),
  - viewport and node editor wiring.
- `PrepareForShutdown` hook exists to pre-stop non-UI background actions if close flows occur.

## 38.4 Shutdown contract

### 38.4.1 App-level shutdown sequence
- `CyxWizApp::Shutdown()` is the canonical path and performs explicit teardown:
  - removes `ConsoleSink` before window/scripting panel teardown,
  - resets network handles (`job_manager_`, `grpc_client_`),
  - resets `main_window_` (full UI destruction),
  - calls OpenGL resource cleanup (`TextureManager::DeleteAllTextures()`),
  - destroys ImGui/ImNodes/ImPlot backends/contexts,
  - destroys GLFW window and terminates GLFW,
  - logs completion and calls `_exit(0)` to avoid fragile static destruction order.

### 38.4.2 Why `_exit(0)` is used
- App intentionally avoids C++ global/static teardown for subsystems that may still need live dependencies.
- This avoids post-destruction logging/GL usage issues from singletons with nontrivial destructors.
- Shutdown policy trades graceful static cleanup for deterministic process exit after critical explicit cleanup.

## 38.5 MainWindow destructor ownership contract

```text
MainWindow::~MainWindow
  if (TrainingManager.IsTrainingActive())
    StopTraining()
    WaitForTrainingStop()
  reset(plot-test + training_plot_panel)   // before scripting_engine
  reset(script-bound panels)               // command_window / script_editor / variable_explorer / plot_output / startup_script_manager
  reset(scripting_engine_)
  reset(other panels)                      // remaining UI modules
```

Important invariants:
- Plot panels that depend on `PlotManager`/Python are released before the scripting engine.
- Training panel is removed before the scripting engine to avoid dangling global-pointer ownership from embedded scripts.
- Logging makes the order explicit and auditable.

## 38.6 Pre-shutdown guard contract

### 38.6.1 MainWindow::PrepareForShutdown
- Called in app shutdown path before final object destruction.
- Active behaviors stopped in deterministic order:
  - running scripts,
  - P2P model downloads,
  - P2P monitoring loops,
  - running inference server.
- This keeps background subsystems from depending on UI that is about to be destroyed.

### 38.6.2 Training stop contract
- Toolbar stop callback directly calls `TrainingManager::StopTraining()`.
- Training manager stop path cancels executor and async task.
- MainWindow destructor additionally waits for worker thread completion before panel destruction.

## 38.7 Project close + registry hygiene contract
- On project close:
  - `DataRegistry::Instance().ClearAllTabularDatasets()` to avoid stale dataset IDs crossing projects,
  - dock layout reset to first-run state for new project.
- This is a lifecycle boundary between project-scoped data and UI state.

## 38.8 Task-system lifecycle intersection
- `CyxWizApp::Update()` drains async completion callbacks each frame.
- `MainWindow::Render()` also drains callbacks (`ProcessCompletedCallbacks`) before UI frame draw; this is a repeated main-thread pump for task completion side effects.
- Because this pump is continuous while main window lives, completion callbacks remain deterministic to UI.

## 38.9 Close-risk model and required guarantees
- If close arrives during active async work, explicit stop/pump guards plus deterministic teardown order prevent:
  - panel callbacks firing into destroyed objects,
  - python-bound widgets touching deallocated runtime,
  - rendering system use after context teardown.
- Remaining risk under current design:
  - `_exit(0)` prevents normal singleton/dtor cleanup outside the explicit path.
  - therefore all safety-critical cleanup must be explicit in `Shutdown()` / `MainWindow::~MainWindow()`.

## 38.10 Evidence anchors

| Layer | Evidence |
|---|---|
| App lifecycle and shutdown policy | `cyxwiz-engine/src/application.h` and `cyxwiz-engine/src/application.cpp` |
| MainWindow shutdown ordering | `cyxwiz-engine/src/gui/main_window.cpp` |
| Training lifecycle guard in shutdown | `cyxwiz-engine/src/core/training_manager.h` and `cyxwiz-engine/src/core/training_manager.cpp` |
| Async task callback pumping | `cyxwiz-engine/src/core/async_task_manager.h`, `cyxwiz-engine/src/core/async_task_manager.cpp`, `cyxwiz-engine/src/application.cpp`, `cyxwiz-engine/src/gui/main_window.cpp` |
| Data hygiene on project close | `cyxwiz-engine/src/gui/main_window.cpp` |

## 38.11 Lean guardrail assessment
- Essential kept in design:
  - explicit ownership order over implicit RAII,
  - explicit close guards for scripts/tasks/training,
  - no hidden cross-module shutdown dependencies.
- Non-essential removal candidate:
  - if full static-safe destruction is ever needed, replace `_exit(0)` with explicit singleton registry shutdown orchestration (higher complexity, currently intentional boundary).
