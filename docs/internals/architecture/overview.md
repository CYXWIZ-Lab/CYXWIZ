---
title: Architecture overview
description: The CyxWiz system in C4-style diagrams, from the people and systems down to the Engine's components, with the sequence of a training run.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - ../code-layout.md
  - threading.md
  - plugin-system.md
  - ../protocol.md
  - ../../concepts/ecosystem.md
---

This page draws CyxWiz at three levels, the way the C4 model does: the
system in its surroundings, the programs it is made of, and the components
inside the Engine. It ends with what happens when you press **Train**. Each
diagram names real targets and folders, so you can go from a box to the
code. The user-level picture is [The CyxWiz ecosystem](../../concepts/ecosystem.md).

## Level 1: people and systems

```mermaid
flowchart LR
  user([Data scientist])
  author([Plugin author])
  operator([Node operator])

  engine[CyxWiz Engine<br/>desktop app]
  node[CyxWiz Server Node<br/>compute worker]
  server[CyxWiz server<br/>sign-in, node registry,<br/>reservations, jobs]
  cloud[CyxCloud<br/>dataset storage]
  releases[Release host<br/>installers, GPU packs]
  python[Python 3.12]

  user --> engine
  author -->|plugins and Python nodes| engine
  operator --> node
  engine -->|sign-in, reserve a node| server
  engine -->|job, dataset files, metrics| node
  engine -->|upload and browse datasets| cloud
  node -->|register, heartbeat, report| server
  node -->|fetch dataset files| cloud
  engine --> python
  releases -->|install, packs| engine
```

- The **Engine** is where a model is built, trained, tested and exported. It
  works fully offline; the network parts are optional.
- The **Server Node** runs jobs other people send. It trains with the same
  compile-and-train code as the Engine.
- The **CyxWiz server** is the central service: it signs users and nodes in,
  keeps the registry of nodes, reserves a node for an Engine and tracks jobs.
  Its source is private; this repository has only its clients and the
  [protocol](../protocol.md) it speaks.
- **CyxCloud** stores datasets in its own repository and service; the Engine
  and the node reach it through its gateway.

## Level 2: programs and libraries

```mermaid
flowchart TB
  subgraph PC[User's computer]
    E[cyxwiz-engine]
    B1[(cyxwiz-backend<br/>shared library)]
    TC1[cyxwiz-training-core]
    PY[pycyxwiz + embedded Python]
    PL[Plugins: C++ libraries<br/>and Python node bundles]
    E --- TC1 --- B1
    E --- PY --- B1
    E --- PL
  end
  subgraph NODE[Worker machine]
    D[cyxwiz-server-daemon]
    G[cyxwiz-server-gui<br/>GUI or TUI]
    TC2[cyxwiz-training-core]
    B2[(cyxwiz-backend)]
    G <-->|DaemonService, IPC on localhost| D
    D --- TC2 --- B2
  end
  S[CyxWiz server<br/>gRPC + HTTP sign-in API]
  C[CyxCloud gateway]
  E -->|JobService, NodeDiscoveryService,<br/>JobReservationService| S
  E <-->|JobExecutionService stream| D
  E -->|DataStreamService| C
  D -->|NodeService, JobStatusService,<br/>JobReservationService| S
  S -->|NodeService.AssignJob| D
  D -->|DataService| C
```

| Program | What it is | Built from |
| --- | --- | --- |
| `cyxwiz-engine` | The desktop app: GLFW window, OpenGL 3.3 (2.1 on macOS), Dear ImGui with ImNodes, ImPlot and ImPlot3D | `cyxwiz-engine/` |
| `cyxwiz-backend` | Shared library: tensors on ArrayFire, layers, losses, optimizers, algorithms, devices, neural providers | `cyxwiz-backend/` |
| `cyxwiz-training-core` | Static library shared by the Engine and the node: compiler, executors, batchers, checkpoints | `cmake/CyxWizTrainingCore.cmake` |
| `pycyxwiz` | pybind11 module over the backend, used by scripts and Python nodes | `cyxwiz-backend/python/` |
| `cyxwiz-server-daemon` | The worker process: job execution, deployment, inference, terminal, HTTP API, IPC | `cyxwiz-server-node/` |
| `cyxwiz-server-gui` | The operator's front end, a GUI or a terminal UI, talking to the daemon over IPC | `cyxwiz-server-node/` |
| `cyxwiz-installer` | Installs the Engine and manages GPU packs | `cmake/CyxWizInstallerTargets.cmake` |

## Level 3: inside the Engine

```mermaid
flowchart LR
  subgraph gui[gui/]
    MW[MainWindow and menus]
    NE[NodeEditor]
    PN[Panels: Properties, Console,<br/>Training Dashboard, Data Studio, ...]
    SD[Studio Debugger panel]
  end
  subgraph core[core/]
    PM[ProjectManager]
    DR[DataRegistry and DatasetCatalog]
    NR[NodeMetadataRegistry]
    GC[GraphCompiler]
    PE[PipelineExecutor]
    TM[TrainingManager and TrainingExecutor]
    AT[AsyncTaskManager]
    DS[DebugSessionManager and traces]
  end
  subgraph other[other folders]
    SC[scripting/: Python engine]
    PG[plugin/: PluginManager and registries]
    NW[network/: gRPC clients]
    AU[auth/: sign-in client]
  end
  MW --> NE
  NE --> NR
  NE --> GC
  NE --> PE
  MW --> TM
  TM --> GC
  TM --> DR
  PE --> DR
  PN --> DR
  SD --> DS
  DS --> GC
  PG --> NR
  SC --> DR
  MW --> NW
  NW --> AU
  NE -.-> AT
  PE -.-> AT
  TM -.-> AT
```

- **NodeMetadataRegistry** (`core/node_metadata_registry.cpp`) is the one
  list of node types. The Nodes panel, the search palette and the context
  menu read it; plugins and Python bundles add to it through the extension
  registry. See [Add a node](../add-a-node.md).
- **GraphCompiler** turns the canvas graph into a `TrainingConfiguration`
  with a list of issues at Error, Warning or Info level. The live compile
  (`core/live_graph_compile.h`) reruns it shortly after each edit.
- **DataRegistry** holds every loaded dataset by name: in-memory Arrow tables,
  Parquet-backed tables on disk, image, audio, text and sparse sets. The
  DatasetCatalog announces changes to subscribers on the UI thread.
- **PipelineExecutor** runs the data steps of a graph (Data Pipeline mode,
  Plot nodes, the steps before training) and records the names it
  registers so they can be released.
- **TrainingManager** owns at most one training run; **TrainingExecutor**
  runs it on its own thread.
- **AsyncTaskManager** is the worker pool behind Task View. See
  [Threading](threading.md).

How a graph runs is explained for users in
[How graphs run](../../concepts/how-graphs-run.md) and
[The data pipeline](../../concepts/data-pipeline.md).

## A training run, step by step

```mermaid
sequenceDiagram
  actor U as User
  participant MW as MainWindow (UI thread)
  participant GC as GraphCompiler
  participant DR as DataRegistry
  participant TM as TrainingManager
  participant TX as TrainingExecutor (training thread)
  participant TD as Training Dashboard

  U->>MW: Train (F5)
  MW->>MW: Data Inputs not loaded? load them first, then come back
  MW->>GC: BuildCompileResult(nodes, links)
  GC-->>MW: issues (Error / Warning / Info)
  alt any Error
    MW-->>U: Cannot start training (compile report)
  else ready
    MW->>GC: Compile again for the run's configuration
    MW->>DR: resolve the datasets, run the data steps before the split
    MW->>TM: StartTrainingRecipe(recipe, config, epochs, batch size)
    TM->>TX: start the training thread (and a Task View entry)
    loop every epoch and batch
      TX->>TX: GPU lease, batcher, forward, loss, backward, optimizer step
      TX-->>TD: metrics (polled on the UI thread)
    end
    TX->>TX: checkpoints, early stopping, final model
    TX-->>MW: end callback (drained on the UI thread)
  end
```

The gate is the same one the **Compile** button uses, so a graph the report
calls ready can start. A run on a GPU holds that device's lease
(`GpuRunLease`, `core/run_execution_device.h`) for its whole life, so two
runs never share one GPU; the second waits.

## Start-up and shut-down

At start (`main.cpp`), the Engine takes the packaged-runtime lock, sets the
library search path to the active runtime, opens `engine_log.txt` (the last
five launches are kept), loads the route-qualification evidence, initialises
the backend with ArrayFire's back-end discovery limited to the active
runtime, applies the saved device choice, lists the devices, then creates
`CyxWizApp`. The plugin search paths are set next (`plugins/` beside the
program, the user's plugin folder and the active plugin packs), and the
plugins set to load at start-up are loaded. `CyxWizApp::Initialize`
creates the window and ImGui contexts, applies the appearance, starts the
Python scan on a worker and shows the start page or opens the project.

At exit, `CyxWizApp::Shutdown` stops scripts and servers, shuts the task pool
down with a five-second drain, destroys the window state, then `main.cpp`
shuts the plugins down, unloads them, and shuts the backend down last.

## Related

- [Code layout](../code-layout.md)
- [Threading and background work](threading.md)
- [Plugin system](plugin-system.md)
- [The protocol](../protocol.md)
- [Compute back ends and neural providers](compute-back-ends.md)
