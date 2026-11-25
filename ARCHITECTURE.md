# Architecture

A short map of the CyxWiz source for new contributors: what the parts are, where the code starts, which way the dependencies go, and the rules the code keeps. The detail, with diagrams, is in [docs/internals/](docs/internals/index.md).

## Bird's-eye view

CyxWiz is a desktop IDE for building, training and testing machine-learning models as node graphs. A graph of nodes (data steps, layers, a loss, an optimizer) is compiled into a training configuration and run on ArrayFire, on the CPU or a GPU. The same backend library runs jobs on a Server Node for remote training, with datasets in CyxCloud. See [The CyxWiz ecosystem](docs/concepts/ecosystem.md).

## The parts

| Part | Folder | What it is |
| --- | --- | --- |
| Engine | `cyxwiz-engine/` | The desktop app (C++20, Dear ImGui): main window, node editor, Data Studio, training, Python scripting, plugin host, installer |
| Backend | `cyxwiz-backend/` | The shared library: tensors, layers, models, losses, optimizers, algorithms, device providers; its public headers in `include/cyxwiz/` are the plugin SDK API; `python/` builds the `pycyxwiz` module |
| Protocol | `cyxwiz-protocol/proto/` | The protobuf / gRPC messages between the Engine, the Server Node, the central server and CyxCloud |
| Server Node | `cyxwiz-server-node/` | The compute worker: a daemon, plus a GUI and a TUI that talk to it over local IPC |
| Plugins | `plugins/` | Optional extensions built on the SDK (simulation, image nodes, samples, the plugin template) |

CyxWiz Cloud (CyxCloud) is a separate repository. The central server (node registry, scheduling, sign-in) is a service CYXWIZ runs and is not in this repository.

## Entry points

- `cyxwiz-engine/src/main.cpp`, then `CyxWizApp` in `cyxwiz-engine/src/application.cpp`; the window is `MainWindow` in `cyxwiz-engine/src/gui/main_window.cpp`.
- `cyxwiz-server-node/src/daemon_main.cpp` (daemon) and `gui_main.cpp` (GUI and TUI).
- `cyxwiz-backend/python/bindings.cpp` (`pycyxwiz`).
- The installer: `cyxwiz-engine/src/backend_pack_manager_main.cpp`.

Start reading the Engine at `MainWindow` (what exists), `NodeMetadataRegistry` in `cyxwiz-engine/src/core/node_metadata_registry.cpp` (every node type), `GraphCompiler` in `cyxwiz-engine/src/core/graph_compiler.cpp` (graph to training configuration) and `DataRegistry` in `cyxwiz-engine/src/core/data_registry.cpp` (loaded data).

## Dependency direction

```text
cyxwiz-engine ------+
                    +--> cyxwiz-backend   (compute: no GUI, no network)
cyxwiz-server-node -+
        |
        +-----------> cyxwiz-protocol <----- CyxCloud, central server

plugins --> the SDK (cyxwiz-backend/include/cyxwiz + the Engine's plugin interfaces)
```

The backend never depends on the Engine. Protocol types carry data between processes and do not become a second domain model.

## Invariants

- **One source per fact.** A node type is defined once, in the metadata registry; the Nodes panel, search, context menu, Properties and the node factory read it. Menus, the Command Palette and shortcuts come from one action table.
- **Compute lives in the backend,** written once on ArrayFire and device resident; the CPU fallback is ArrayFire's CPU back end. LibTorch is only the test oracle.
- **The UI thread never blocks.** Long work runs as an `AsyncTaskManager` task on a snapshot taken on the UI thread.
- **Saved files stay readable.** `NodeType` numbers are written into `.cyxgraph` files, so the enum is append only. Retired nodes are refused with their replacement named, not translated.
- **Nodes are real operations** in one of four pipeline bands: data preparation, partitioning, phase-aware iteration, model.
- **The screen tells the truth:** placement, fallback and blocked nodes are shown with their reason; nothing falls back silently.
- **Colours come from the theme tokens,** and widgets from the shared `cyxwiz::ui` kit.

## Where to go next

- [Architecture overview](docs/internals/architecture/overview.md): context, containers and components, with diagrams
- [Threading and async tasks](docs/internals/architecture/threading.md)
- [Plugin system](docs/internals/architecture/plugin-system.md)
- [Compute back ends](docs/internals/architecture/compute-back-ends.md)
- [Code layout](docs/internals/code-layout.md)
- [Protocol](docs/internals/protocol.md)
- [Tests](docs/internals/testing.md)
- [Decision records](docs/internals/decisions/README.md)
