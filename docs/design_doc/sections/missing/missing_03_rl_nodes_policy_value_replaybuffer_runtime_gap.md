# M-03) RL graph node runtime path (`PolicyNetwork/ValueNetwork/ReplayBuffer`) gap

## M-03.1 Question
Can `PolicyNetwork`, `ValueNetwork`, `ReplayBuffer`, and related RL nodes execute through the native C++ graph executor path?

## M-03.2 Current runtime partition

```text
Standard trainer flow (non-RL)
  graph compile
    -> pipeline capabilities gate
    -> materialize
    -> TrainingExecutor::Train
    -> native model + checkpoints

RL trainer flow
  RL button path
    -> RLTrainingConfig + env/reward/filter nodes
    -> RLScriptGenerator::Generate
    -> ScriptingEngine::ExecuteScriptAsync
    -> pycyxwiz bridge callbacks
```

## M-03.3 Effective behavior today
- RL-focused nodes are represented in compiler/pipeline capability metadata.
- In runtime-capability tables, RL training/control nodes are currently treated as unsupported / fail-closed for the standard pipeline execution path.
- `Train RL` path is explicitly a separate UI->scripted path and does not require `TrainingExecutor` graph dispatch.
- `RLTrainingExecutor` exists but is not the default production route for node-editor RL flows in current flow.

## M-03.4 Implication
- RL nodes are not “native executor-native” in the normal training pipeline by current architecture.
- A complete RL execution design requires:
  - explicit runtime contract for RL node materialization, or
  - documented hard boundary that RL is script-driven only.

## M-03.5 Evidence anchors
- `cyxwiz-engine/src/gui/node_editor.cpp:1217` (RL control button dispatch)
- `cyxwiz-engine/src/gui/node_editor.cpp:4519` (OnStartRLTraining entrypoint)
- `cyxwiz-engine/src/gui/node_editor.cpp:4591` (RLScriptGenerator::Generate callsite)
- `cyxwiz-engine/src/gui/node_editor.cpp:4620` (scripting engine executes generated script)
- `cyxwiz-engine/src/core/rl_script_generator.cpp:20` (script generation for RL training)
- `cyxwiz-engine/src/core/rl_script_generator.cpp:289` (completion of script emission)
- `cyxwiz-engine/src/core/rl_training_executor.h:72` (RLTrainingExecutor type exists)
- `cyxwiz-engine/src/core/rl_training_executor.cpp:9` (RLTrainingExecutor lifecycle methods)
- `cyxwiz-engine/src/core/graph_executor.h:7` (documented separate RL executor architecture note)
- `cyxwiz-engine/src/core/graph_compiler.cpp:965` (RL sketch detection)
- `cyxwiz-engine/src/core/graph_compiler.cpp:2517` (graph compile route for RL sketches)
- `cyxwiz-engine/src/core/graph_compiler.cpp:4174` (RL-related node catalog mapping and RL node handling)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:239` (ReplayBuffer unsupported in PipelineExecutor)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:243` (PolicyNetwork unsupported in PipelineExecutor)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:245` (ValueNetwork unsupported in PipelineExecutor)
- `cyxwiz-engine/src/core/training_manager.cpp:189` (native training uses TrainingExecutor path)
- `cyxwiz-engine/src/core/training_executor.cpp:201` (non-RL executor contract)
