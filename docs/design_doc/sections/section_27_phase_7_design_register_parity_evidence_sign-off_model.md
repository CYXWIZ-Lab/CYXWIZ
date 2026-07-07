## 27) Phase 7 design register (parity evidence + sign-off model)

### 27.1 Objective
lock documentation/runtime parity through source-level evidence and explicit acceptance criteria.

### 27.2 Evidence matrix (source of truth)

```text
Concern                                | File(s)                                  | Evidence output
--------------------------------------|------------------------------------------|-----------------------------
Compile DAG validation                  | core/graph_compiler.cpp                   | CompileResult.issues, is_valid
Training configuration assembly          | core/graph_compiler.cpp                   | TrainingConfiguration fields
Pin role extraction                     | core/graph_compiler.cpp                   | CompiledGraphPlan role ids
Shape and loss checks                   | core/graph_compiler.cpp                   | is_valid + issue messages
Backend placement reports                | core/graph_compiler.cpp                   | BackendPlacement records
Execution model selection                | core/model_builder.cpp                    | BuildSequential/BuildGraph executable
Batching contract by source              | core/training_batcher_setup.cpp           | Arrow/Parquet/External builders
Runtime mode dispatch                     | core/training_executor.cpp                 | mode-specific construction
Runtime loop lifecycle                    | core/training_executor.cpp                 | train/pause/resume/stop paths
Materializer source resolution             | core/pipeline_materializer.cpp            | Apply source-kind + operators
Materializer aliasing + capabilities        | core/pipeline_runtime_capabilities.cpp    | capability sets + aliases
UI compile launch path                     | gui/main_window.cpp                       | CompileGraphAndReport
Training orchestration                     | gui/main_window.cpp, core/training_manager.cpp | StartTrainingFromGraph
GUI progress feedback                      | gui/main_window.cpp                       | callback-based updates
Preflight gate                             | core/preflight_validator.cpp               | Validate + summaries
Failure paths                               | all above                                 | issue/warning/error fields
```

### 27.3 Parity acceptance checklist
- For each new feature or node:
  - verify compile gate entry in `graph_compiler.cpp`.
  - verify preflight behavior in `preflight_validator.cpp`.
  - verify materialization behavior in `pipeline_materializer.cpp` if data ops are used.
  - verify training path in `training_executor.cpp` and `model_builder.cpp`.
  - verify UI launch and callback behavior in `main_window.cpp`.
- Add issue-level evidence for any blocked flow:
  - compile issue id,
  - preflight issue id,
  - runtime outcome id.

### 27.4 Evidence quality rules
- no claim in this document should remain unlinked for more than one phase.
- any row marked `U` in runtime-critical matrices must have evidence note and owner.
- blockers must include:
  - detection layer,
  - failing code path,
  - recovery recommendation.

### 27.5 Suggested review cadence
- before release: validate the full evidence matrix in one cycle.
- before node graduation: update evidence rows and bump phase status.
- every two architecture edits: refresh this section and phase scorecard.

### 27.6 Completion criteria for Phase 7
- every claim in phases 0-6 has corresponding file evidence row.
- no unresolved `U` rows in active runtime-critical families.
- each blocker has a recovery path and phase owner.
- `track_design.md` has one consistent version header with status.
