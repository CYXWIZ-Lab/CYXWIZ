# M-08) `GraphExecutableModel` consumption path gap in runtime execution

## M-08.1 Question
Does `GraphExecutableModel` get fully consumed by the native trainer in all execution flows, or does legacy sequential execution remain active?

## M-08.2 Observed behavior

```text
StartTrainingFromGraph
  -> GraphCompiler::Compile
  -> BuildExecutableFromConfig(config)
     -> if graph-compatible and non-trivial -> BuildGraphExecutableFromConfig
     -> else -> BuildSequentialFromConfig fallback path
  -> executor Train() through selected model executable and batcher path
```

## M-08.3 Gap evidence
- The compiled graph plan object exists and is populated.
- The executor construction path can still route through legacy sequential model building.
- This means graph form and plan availability does not by itself guarantee end-to-end graph-only execution.

## M-08.4 Architectural interpretation
- Graph execution path and plan construction are implemented, but the system still uses compatibility behavior for unimplemented or mixed paths.
- Stability policy is effectively: prefer native graph path where supported, fallback to legacy while preserving feature coverage.

## M-08.5 Required clarification for release-readiness
- Define a strict matrix:
  - operators guaranteed to go through graph executable, and
  - operators that must follow fallback sequential execution.
- Add migration rule for deprecating fallback dependencies.

## M-08.6 Evidence anchors
- `docs/design_doc/sections/section_53_phase_10_design_register_python_scripting_and_model_build_boundaries.md:1` (existing boundary context)
- `cyxwiz-engine/src/core/model_builder.h:42` (`BuildSequentialFromConfig` / `BuildExecutableFromConfig`)
- `cyxwiz-engine/src/core/model_builder.h:45` (`BuildGraphExecutableFromConfig`)
- `cyxwiz-engine/src/core/model_builder.cpp:1355` (`BuildExecutableFromConfig` primary graph path)
- `cyxwiz-engine/src/core/model_builder.cpp:1360` (fallback to sequential on non-graph-capable config)
- `cyxwiz-engine/src/core/model_builder.cpp:1374` (`BuildGraphExecutableFromConfig` entrypoint)
- `cyxwiz-engine/src/core/model_builder.cpp:1387` (GraphExecutableModel::CanRunLinearPlan gating)
- `cyxwiz-engine/src/core/model_builder.cpp:1392` (fallback when graph linear plan missing)
- `cyxwiz-engine/src/core/model_builder.cpp:1401` (GraphExecutable path error recovery)
- `cyxwiz-engine/src/core/model_builder.cpp:1425` (graph model + optimizer path)
- `cyxwiz-engine/src/core/model_builder.cpp:1429` (constructed model emitted in GraphExecutableModel)
- `cyxwiz-engine/src/core/graph_executable_model.h:25` (GraphExecutableModel type contract)
- `cyxwiz-engine/src/core/graph_executable_model.cpp:373` (`BuildLinearPlan` requirement check)
- `cyxwiz-engine/src/core/training_executor.cpp:201` (`BuildExecutableFromConfig` called at initialize)
- `cyxwiz-engine/src/core/training_executor.cpp:268` (comment on mixed legacy/modern execution flow)
- `cyxwiz-engine/src/core/training_executor.cpp:272` (legacy dataset path still available)
- `cyxwiz-engine/src/core/training_executor.cpp:335` (legacy dataset-only execution branch)
