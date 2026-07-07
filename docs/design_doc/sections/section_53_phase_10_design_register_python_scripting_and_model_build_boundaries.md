# 53) Phase 10 design register (Python scripting and model-build boundary contracts)

## 53.1 Objective

This section closes the remaining gap in the engine design by documenting:
- whether Python scripts can be the runtime owner of model building,
- where Python-generated code fits relative to the native C++ training pipeline,
- and how data/control can cross from node graph to Python script runtime without breaking engine invariants.

## 53.2 Three execution planes (canonical view)

```text
User action on Node Editor
  -> [TRAIN button] -> StartTrainingFromGraph -> compile gate -> materialize -> C++ executor
     -> BuildExecutableFromConfig / BuildGraphExecutableFromConfig -> TrainingExecutor::Train
     -> TrainingPlotPanel + task system + checkpoints + artifacts

User action on Node Editor
  -> [Generate Python code / Code menu] -> GenerateCodeForFramework -> code string
     -> ScriptEditor panel OR file export
     -> (user-run script only) -> ScriptWindow / external python runtime
     -> no implicit pass into TrainingExecutor

User action on Node Editor (RL graph)
  -> [Train RL] -> OnStartRLTraining
     -> RLScriptGenerator::Generate -> Python script text
     -> ScriptingEngine::ExecuteScriptAsync
     -> Python runtime with pycyxwiz RL bridge -> model.save()
     -> optional metric stream back to Dashboard via rl_update_metric / callbacks
```

## 53.3 Ownership matrix

| Plane | Primary owner | What it owns | What it consumes | What it returns | Does it produce a C++ executable model? |
|---|---|---|---|---|---|
| C++ training plane | `MainWindow`, `GraphTrainingLauncher`, `TrainingExecutor`, `PipelineMaterializer`, `ModelBuilder` | Executable runtime model contract, dataset materialization, executor lifecycle | `std::vector<MLNode>`, `std::vector<NodeLink>` | Trained model artifacts, run summary, dashboard metrics | Yes (`BuildExecutableFromConfig` / `BuildGraphExecutableFromConfig`) |
| Script generation plane | `NodeEditor` + codegen backends + `PluginNodeRegistry` | Portable Python text for chosen framework | Topologically sorted graph | `.py` source shown in Script Editor and optionally exported to disk | No (no engine registration path from generated script text) |
| RL Python execution plane | `NodeEditor::OnStartRLTraining`, `RLScriptGenerator`, `ScriptingEngine`, pycyxwiz bridge | Script lifecycle, command flags, runtime metrics stream | RL config + reward/filter params + MJCF/obs config | Script output, logs, saved policy file from Python | No (model is trained by Python libraries, e.g., stable-baselines3) |

## 53.4 Native model-build path in detail

```text
StartTrainingFromGraph(nodes, links)
  -> BuildCompileResult(nodes, links)
  -> (if compile_result_success_)
       GraphCompiler::Compile(nodes, links)
       -> PipelineMaterializer::Materialize(config)
       -> TrainingExecutor::Start/Train
         -> BuildExecutableFromConfig(config)
             -> BuildGraphExecutableFromConfig(config) if graph ops present
             -> BuildSequentialFromConfig(config) fallback/legacy path
         -> run loop + callbacks + stop/pause
```

This is the engine-native graph-to-model path:
- deterministic from graph nodes to `TrainingConfiguration`,
- deterministic training runtime semantics from executor,
- no dependency on user-authored Python for model graph execution.

## 53.5 Script generation/export path and its boundary

```text
UI request: Generate Python / Generate Keras / Generate PyCyxWiz
  -> NodeEditor::GenerateCodeForFramework
     -> GeneratePyTorchCode / GenerateTensorFlowCode / GenerateKerasCode / GeneratePyCyxWizCode
     -> script text + headers
     -> ScriptEditorPanel (preview) or export to .py file
```

Design boundary:
- this path is a *design-time artifact export path*.
- it does not alter training executor contracts unless the user manually runs that generated script outside the native training flow.

## 53.6 Python execution runtime plane (scripting engine contracts)

### 53.6.1 Preconditions and environment
- `ScriptingEngine::ExecuteScriptAsync` and command execution require initialized Python.
- `ScriptingEngine::EnsurePythonInitialized` requires an active project and will block when no project is loaded.
- On script execution, `ScriptingEngine::ExecuteWithStreaming` sets:
  - Python working directory -> project root,
  - `sys.path` entries -> project root and scripts directory (if exists).

### 53.6.2 Async model
- Console / assistant commands use `ExecuteCommandAsync` and completion via `GetCommandResult`.
- Script execution uses `ExecuteScriptAsync` with `ScriptWorker` and completion via `ExecutionResult`.
- Command/script cancellation is asynchronous flag-based (`shared_cancel_flag_`) with Python trace checks.

### 53.6.3 Output and plotting
- Script output is collected into internal queues (`GetPendingOutput`, `GetPendingPlots`).
- Output callbacks can stream text to Command Window / script execution consumers.

## 53.7 RL Python training contract (where this is truly Python-run model training)

### 53.7.1 Trigger contract
- RL button is rendered only when RL nodes are present.
- Pressing `Train RL` calls `NodeEditor::OnStartRLTraining`.
- It is explicitly blocked if:
  - another RL script is running,
  - `ScriptingEngine` is missing,
  - a generic script is already running.

### 53.7.2 Runtime contract
- Read RL config from `RLTraining` + `MuJoCoPlant` + optional reward/filter nodes.
- Generate full script text with `RLScriptGenerator::Generate(config, reward_params, obs_filter_params, save_path)`.
- Set dashboard script state and execute generated script asynchronously.
- On stop, call `pycyxwiz.rl_set_stop(True)` then `StopScript`.

```text
OnStartRLTraining
  -> extract RLTraining / MuJoCo / Reward / Observation params
  -> script = RLScriptGenerator::Generate(...)
  -> show RL TrainingDashboardPanel
  -> setup_script sets pycyxwiz rl_set_stop(False) + rl_set_paused(False)
  -> ScriptingEngine::SetCompletionCallback
  -> ScriptingEngine::ExecuteScriptAsync(script)
```

### 53.7.3 Python<->C++ RL bridge contract
- C++ injects dashboard pointer lazily: `ScriptingEngine::EnsureTrainingDashboardRegistered` -> `cyxwiz_plotting.set_training_plot_panel`.
- `pycyxwiz` side exposes:
  - `rl_update_metric(name, value)`
  - `rl_should_stop()`
  - `rl_is_paused()`
  - `rl_set_stop(bool)`
  - `rl_set_paused(bool)`
- C++ side consumes stop/pause via script globals and completion callback.

### 53.7.4 RL runtime outputs
- Training script controls its own model, optimizer, and policy graph (inside Python runtime).
- C++ receives only control/result side effects:
  - metric events,
  - cancellation/stop status,
  - textual output and completion status.
- Trained policy file persistence is performed by Python (`model.save(...)`).

## 53.8 Command console and startup script boundary (adjacent to, but separate from, training control)

```text
User types command / python expression in Command Window
  -> StartAsyncCommand(command)
     -> ScriptingEngine::ExecuteCommandAsync
     -> timeout wrapper + trace interrupt
     -> result returned to panel
  -> no automatic integration into ModelBuilder/Executor path
```

Startup scripts flow is similarly explicit:

```text
StartupScriptManager::ExecuteAll
  -> read file list
  -> scripting_engine_->ExecuteScript(script_content)
  -> output to CommandWindow only
```

## 53.9 What is NOT done by Python scripts in current engine

- Python scripts do not directly instantiate or register `TrainingExecutor` from graph nodes.
- Generated framework scripts are not the default execution transport for normal (non-RL) training.
- `ExecuteScriptAsync` does not convert arbitrary script output back into internal `ModelBuilder` contracts.

## 53.10 Risks and boundary health
- Script and C++ execution paths can diverge semantically (e.g., custom generated logic vs graph contracts).
- Error handling and lifecycle visibility differ between planes:
  - C++ plane returns structured training trace and run summaries,
  - Python plane relies on script result + custom callback conventions.
- RL bridge uses global function-state style (`rl_set_stop`, `rl_set_paused`) and should be kept minimal and namespaced.

## 53.11 Evidence anchors

| Domain | Evidence files |
|---|---|
| Native training control path | `cyxwiz-engine/src/gui/main_window.cpp`, `cyxwiz-engine/src/core/training_executor.cpp`, `cyxwiz-engine/src/core/model_builder.cpp`, `cyxwiz-engine/src/core/pipeline_materializer.cpp`, `cyxwiz-engine/src/core/graph_compiler.*` |
| Code generation path | `cyxwiz-engine/src/gui/node_editor.h`, `cyxwiz-engine/src/gui/node_editor_codegen.cpp`, `cyxwiz-engine/src/gui/node_editor_io.cpp`, `cyxwiz-engine/src/core/node_executors` |
| RL script generation path | `cyxwiz-engine/src/gui/node_editor.cpp`, `cyxwiz-engine/src/core/rl_script_generator.cpp`, `cyxwiz-engine/src/core/rl_training_executor.h` |
| Script engine + runtime contract | `cyxwiz-engine/src/scripting/scripting_engine.h`, `cyxwiz-engine/src/scripting/scripting_engine.cpp`, `cyxwiz-engine/src/gui/panels/command_window.cpp`, `cyxwiz-engine/src/scripting/startup_script_manager.cpp`, `cyxwiz-engine/src/scripting/cell_manager.cpp` |
| Python bridge contracts | `cyxwiz-engine/src/gui/panels/training_plot_panel_global.cpp`, `cyxwiz-engine/src/gui/panels/training_plot_panel.h`, `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp`, `cyxwiz-engine/src/python/plot_bindings.cpp`, `cyxwiz-backend/python/bindings.cpp` |
| Python runtime diagnostics and env boundary | `cyxwiz-engine/src/scripting/python_engine.cpp`, `cyxwiz-engine/src/scripting/python_sandbox.cpp`, `cyxwiz-engine/src/core/project_manager.*` |
