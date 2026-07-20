# Current CyxWiz Runtime Baseline

This document records the present architecture facts that BackendExtern must
not break. It is a design baseline, not a claim that the proposed components
already exist.

## Native graph and training plane

Normal graph training is C++-owned:

```text
Node Editor
  -> GraphCompiler::Compile
  -> PipelineMaterializer::Materialize
  -> TrainingExecutor
  -> BuildExecutableFromConfig / BuildGraphExecutableFromConfig
  -> native model, callbacks, checkpoints, dashboard
```

The design register describes this as the canonical native model-build path.
Generated PyTorch/Keras/PyCyxWiz code is an export/design-time path, not an
input to `TrainingExecutor`. BackendExtern must add an explicit new execution
route; it must not silently redirect native graphs.

## Existing Python plane

`scripting::ScriptingEngine` wraps `PythonEngine` and supports synchronous and
asynchronous commands/scripts, output capture, plot capture, cancellation,
timeouts, and optional sandbox behavior. It uses an embedded interpreter.

The Python interpreter design currently selects one project/global Python
environment, initializes lazily, and requires an Engine restart to switch an
already initialized interpreter. The project environment is a Python 3.12
venv by design. The scripting plane is useful for user code and the existing
RL script route, but it is not suitable as the stable host for multiple heavy
framework runtimes in a long-lived GUI process.

## Existing plugin plane

The Engine already has `PluginManager`, a manifest/API-version gate, permission
checks, lifecycle states, crash-isolated callbacks, and registries for nodes,
panels, data providers, analytics providers, and training hooks. This is a
good way to surface provider nodes and panels, but it does not currently offer
a versioned external-worker protocol or managed framework environments.

## Existing compute plane

`cyxwiz-backend` owns native tensors and optional ArrayFire acceleration.
ArrayFire is selected at runtime where implemented; other operations may use a
CPU path or provide a recorded fallback reason. Backend placement observations
already exist and should be extended in concept, not replaced, for external
worker results.

## Consequences for BackendExtern

1. Do not add PyTorch/JAX conditionals throughout native layer/operator code.
2. Do not run curated PyTorch/JAX/Flax models through `ScriptingEngine` in the
   Engine process.
3. Reuse task, project, plugin, diagnostics, and graph validation conventions
   where they fit.
4. Define a new, explicit `external` node execution route with fail-closed
   compiler/materializer behavior until a provider is installed and valid.

## Evidence anchors

- `cyxwiz-engine/src/scripting/scripting_engine.h/.cpp`
- `docs/python_interpreter_design.md`
- `cyxwiz-engine/src/core/graph_compiler.*`
- `cyxwiz-engine/src/core/pipeline_materializer.*`
- `cyxwiz-engine/src/core/training_executor.*`
- `cyxwiz-engine/src/plugin/*`
- `cyxwiz-backend/README.md`
- `docs/design_doc/sections/section_53_phase_10_design_register_python_scripting_and_model_build_boundaries.md`
- `docs/design_doc/sections/section_49_phase_10_design_register_plugin_extensibility_and_compatibility_contracts.md`

