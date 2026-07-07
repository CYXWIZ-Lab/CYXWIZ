## 25) Phase 6 design register (compute layer + frontend/backend synchronization)

### 25.1 Compute layer objective
Define the compute/runtime boundary as a first-class design object:
- what executes on CPU/CUDA/ArrayFire,
- how backend placement is selected and reported,
- what happens when backend probes fail.

### 25.2 Compute-layer boundary map

```text
Compute request
  -> GraphCompiler/backend probe
       -> TrainingConfiguration.backendPlacement
          -> IExecutableModel + backend operator selection
             -> TrainingExecutor
                -> backend runtime call path
```

### 25.3 Backend placement behavior
From existing compile/executor behavior:
- CPU path is baseline fallback.
- CUDA/ArrayFire placement is accepted where availability and model compatibility allow.
- mixed placements are possible with warnings when reuse or dispatch constraints exist.
- unavailable placement must not silently continue as random behavior; it is reported and reduced to valid supported path.

### 25.4 Frontend-to-runtime synchronization
UI and backend synchronization is event-driven through compile and start pipelines:
- compile button builds and stores compiled artifacts.
- compile report updates node-level pin/error display data.
- start button reads stored artifacts and executes launch if preconditions pass.
- asynchronous training thread updates UI through callback emissions.
- stop/pause states modify executor state and are reflected in UI with deterministic flags.

ASCII handoff:

```text
Node Editor Edits
      |
      v
MainWindow::CompileGraphAndReport
      |
      v
CompileResult (issues + plan + config)
      |
      v
Compile popup + editor pin/state hints
      |
      v
MainWindow::StartTrainingFromGraph
      |
      +--> PreflightValidator
      +--> BuildCompileResult (UI status refresh)
      +--> TrainingManager::StartTraining
                |
                v
      TrainingExecutor worker task (async)
                |
                v
      callbacks -> UI metric/progress panels
```

### 25.5 Frontend synchronization risk controls
- stale compile artifacts are invalidated when node graph changes.
- executor start requires graph and config consistency with latest compile result.
- compile errors block launch even if graph appears visually edited later.
- all user-facing blockers must include source and recovery hint.

### 25.6 Computer/graph runtime interaction
The graph model and compute model interact via shared ids:
- node ids from editor map into compiled node ids,
- pin ids map into `CompiledGraphPlan` role ids,
- tensor cache in GraphExecutableModel indexes by pin/layer key.

This avoids re-parsing full graph during execution and keeps execution replay deterministic.

### 25.7 Scheduler and control thread model
- start creates async task/work item managed by TrainingManager.
- training loop runs independent of UI thread.
- state callbacks are marshaled to UI layer.
- stop/pause writes cancellation flags in manager/executor shared state.

### 25.8 Phase 6 completion criteria
- backend placement failure states are explicit and non-ambiguous in compile reports.
- UI launch status cannot diverge from latest compile artifacts.
- compute path includes deterministic placement fallback policy.
- pause/resume/stop events are consistent in UI and executor state.
