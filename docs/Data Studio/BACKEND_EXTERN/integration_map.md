# BackendExtern Engine Integration Map

## Purpose

This map prevents implementation from scattering external-runtime behavior
through unrelated Engine code. It names the intended integration seams and the
changes that are explicitly forbidden.

## Core service boundary

Proposed focused module ownership:

```text
cyxwiz-engine/src/backend_extern/
  backend_extern_types.h          protocol-neutral value types
  runtime_catalog.h/.cpp          installed/available runtime metadata
  worker_manager.h/.cpp           process and IPC lifecycle
  run_registry.h/.cpp             run state and provenance
  artifact_store.h/.cpp           approved artifact paths and hashes
  protocol_validator.h/.cpp       request/event/result schema checks
  external_run_executor.h/.cpp    graph-to-service adapter only
```

Do not put this code in `ScriptingEngine`, `TrainingExecutor`, or general
tensor/layer files. Those components have different ownership and lifecycle
contracts.

## Existing seam -> proposed responsibility

| Existing area | BackendExtern use | Must not change |
| --- | --- | --- |
| Project manager | Resolve project paths and persist runtime/model provenance | Existing project Python venv selection semantics |
| Task manager/UI task flow | Surface progress, cancellation, completion, and errors | Native task ownership or UI-thread rules |
| Graph compiler | Validate external node/provider/pin schema | Native node compilation route |
| Pipeline materializer | Produce a narrow external run plan | Native model materialization rules |
| Node executor factory | Route recognized external nodes to `ExternalRunExecutor` | Existing executor routing for native nodes |
| Plugin manager/registry | Register optional official provider nodes/panels | Core service lifecycle and protocol authority |
| Support bundle/debugger | Add sanitized external run records | Existing ArrayFire placement records |
| Data Studio | Produce validated table artifacts and render results | Existing data ownership/model training assumptions |

## Initialization and shutdown order

```text
Engine startup
  -> project/config services available
  -> BackendExternService creates catalog and read-only state
  -> optional providers register node/panel metadata
  -> no worker starts yet

First external run
  -> validate graph and runtime
  -> create task/run record
  -> start worker

Project close / Engine shutdown
  -> reject new external runs
  -> request cancellation for active runs
  -> bounded worker termination if needed
  -> flush sanitized provenance
  -> destroy BackendExternService
```

Workers are demand-started. No framework import, GPU initialization, or Hub
network activity may occur at normal Engine startup.

## Graph execution boundary

The first release should use an explicit `ExternalRunExecutor` that performs:

```text
validated native input artifact
  -> RunRequest
  -> BackendExternService::Run
  -> validated result artifact
  -> native Data Studio artifact/result
```

It must not make arbitrary external nodes appear as `SequentialModel` layers.
External model execution and native layer execution are separate model classes
with a typed artifact boundary.

