# M-01) Python scripting ownership boundary (explicitly not the native training executor path)

## M-01.1 Question
Can Python scripts generated/executed by the engine directly build and own the same graph-to-training runtime path used by the C++ trainer?

## M-01.2 Answer (current engine)
No. Python script execution is an adjacent execution plane:

```text
Graph training UI flow
  User presses Train
    -> Graph compile -> materialize -> TrainingExecutor -> native model/training artifacts

Python generation flow
  User presses Generate Python
    -> code text shown/exported
    -> no automatic graph executor registration

RL flow
  User presses Train RL
    -> RLScriptGenerator emits script
    -> ScriptingEngine executes script asynchronously
    -> pycyxwiz metrics/control bridge only
    -> policy artifacts persisted by Python runtime
```

## M-01.3 What Python scripts explicitly do NOT do in current engine
- They do not instantiate or register `TrainingExecutor` from graph nodes.
- Generated framework scripts are not the default transport for normal non-RL training.
- `ExecuteScriptAsync` does not transform arbitrary script output back into `ModelBuilder` contracts.

## M-01.4 Why this matters for engine invariants
- Prevents script text from silently bypassing compile/materialization/runtime contracts.
- Keeps callback, checkpoint, pause/stop, and trace contracts stable in native training plane.
- Contains unsafe surface growth by leaving Python as:
  - design-time export path,
  - optional external runtime,
  - or dedicated RL runtime with explicit bridge functions.

## M-01.5 Evidence anchors
- `cyxwiz-engine/src/core/graph_compiler.h:305` (TrainingConfiguration and compile contract)
- `cyxwiz-engine/src/core/graph_compiler.h:501` (GraphCompiler API)
- `cyxwiz-engine/src/core/graph_compiler.cpp:1375` (Build() / validation path)
- `cyxwiz-engine/src/core/training_manager.h:106` (Trainer entrypoint contracts)
- `cyxwiz-engine/src/core/training_manager.cpp:189` (StartTraining() creates TrainingExecutor from compiled config)
- `cyxwiz-engine/src/core/training_executor.h:111` (TrainingExecutor constructor contract)
- `cyxwiz-engine/src/core/training_executor.cpp:201` (TrainingExecutor::Train uses BuildExecutableFromConfig)
- `cyxwiz-engine/src/gui/node_editor.h:1138` (`GeneratePythonCode()` dispatch)
- `cyxwiz-engine/src/gui/node_editor.h:1235` (`GenerateRLPyCyxWizCode()`)
- `cyxwiz-engine/src/gui/node_editor.cpp:1217` (RL train button dispatch boundary in node editor)
- `cyxwiz-engine/src/gui/node_editor.cpp:4519` (OnStartRLTraining path starts RL scripting flow)
- `cyxwiz-engine/src/gui/node_editor.cpp:4620` (Script executor entry point in RL flow)
- `cyxwiz-engine/src/gui/node_editor_codegen.cpp:88` (code generation entrypoint)
- `cyxwiz-engine/src/gui/node_editor_codegen.cpp:199` (RL generator selection)
- `cyxwiz-engine/src/core/rl_script_generator.cpp:20` (script emission path)
- `cyxwiz-engine/src/scripting/scripting_engine.cpp:950` (ExecuteScriptAsync)
- `cyxwiz-engine/src/gui/main_window.cpp:609` (Node editor receives scripting engine)
