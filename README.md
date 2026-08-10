# CyxWiz

CyxWiz is a native C++20, visual-first engineering environment for building, inspecting, and running machine-learning and data workflows. It combines a node graph, dataset tooling, training controls, scripting, model packaging, and worker-side services around one shared computation backend.

> **Project status:** active pre-release development. Local graph editing, data workflows, training, testing, scripting, and several inference paths are implemented and exercised by the test suite. Device-placement truth, exact checkpoint resume, and end-to-end distributed execution still require production hardening. Do not treat the current tree as a production-ready distributed training service.

CyxWiz is source-available under the [CYXWIZ Commercial Source and Evaluation License](LICENSE), not an open-source licence.

## What CyxWiz is

Machine-learning projects often separate data preparation, model design, training code, experiment monitoring, model files, and remote compute into unrelated tools. CyxWiz is designed to keep those stages connected through an inspectable project workspace and a saved graph.

The desktop **Engine** is the main user application. A user can inspect project assets, preview and profile datasets, compose a workflow from nodes, configure training, run supported workflows, monitor tasks and metrics, execute Python-based scripts, and package a trained model. The graph is not only a drawing: the Engine validates its contracts and translates supported nodes into data materialization, batching, model, loss, optimizer, training, testing, and inference operations.

The **Backend** owns computation. It provides tensors, layers, activations, losses, optimizers, data primitives, evaluation utilities, device selection, and ArrayFire execution for the public Engine runtime. The Engine must call this shared backend rather than maintain a second implementation of mathematical operations in GUI code.

The **Server Node** is the worker-side application. It exposes daemon and graphical modes for local hardware reporting, job lifecycle, deployment, files, metrics, and supported worker execution through the shared protocol and backend. The external orchestration service is CyxCloud, maintained in a separate repository. End-to-end distributed training is still being hardened and is not presented as production-ready here.

CyxWiz is visual-first rather than visual-only. The Script Editor supports Python and `.cyx` cell documents, while the plugin boundary allows optional integrations to remain outside the mandatory core.

## Typical Engine workflow

1. **Create or open a project workspace.** The workspace gives graphs, scripts, datasets, artifacts, checkpoints, and models an explicit project context.
2. **Add and inspect data.** Use the Asset Browser and Data Input/Data Studio surfaces to preview, profile, query, visualize, or prepare supported datasets. Data Studio uses Arrow-oriented data paths and DuckDB-backed analysis where applicable.
3. **Build a graph.** Connect data, preprocessing, model, training, evaluation, visualization, and export nodes in CyxWiz Studio. Save the graph as a `.cyxgraph` file.
4. **Validate and compile.** The graph compiler checks topology, pins, shapes, roles, labels or targets when required, and supported runtime contracts before execution.
5. **Materialize and execute.** The Engine prepares datasets and batches, constructs supported backend operations, and runs data pipelines, training, testing, or inference through task-managed execution paths.
6. **Observe the run.** Task progress, logs, the Training Dashboard, metrics, checkpoints, and device-placement evidence describe what the runtime actually did.
7. **Preserve the result.** Save checkpoints during development and package supported graphs, weights, configuration, assets, and history in the native `.cyxmodel` format.

Available nodes, import/export formats, and execution backends depend on the build configuration and the runtime capabilities discovered on the machine. A visible node or selected GPU preference is not by itself proof that every associated execution path is implemented or GPU-resident.

## Architecture at a glance

```text
Project workspace
  datasets | scripts | .cyxgraph files | checkpoints | artifacts
       |
       v
CyxWiz Engine
  Asset Browser / Data Studio / Node Editor / Script Editor
       |
       v
Graph validation and compilation
       |
       +--> data materialization and batching
       +--> model, loss, optimizer, training and testing contracts
       +--> task, metric, checkpoint and runtime-placement evidence
       |
       v
cyxwiz-backend
  C++ primitives + ArrayFire CPU/CUDA/oneAPI/OpenCL execution for Engine runs

CyxWiz Engine -------- cyxwiz-protocol -------- Server Node
                                                   |
                                                   +--> cyxwiz-backend
                                                   +--> external CyxCloud orchestration

Optional plugins --> explicit Engine/plugin contracts
```

This separation is deliberate: the Engine owns interaction and orchestration, the Backend owns computation, the Protocol owns process contracts, the Server Node owns worker lifecycle, and plugins own optional integrations.

## Core project formats

| Format | Purpose |
| --- | --- |
| `.cyxgraph` | Saved JSON graph containing nodes, links, parameters, and workflow metadata |
| `.cyx` | Python-compatible CyxWiz script document that may contain executable cells |
| `.cyxmodel` | Native model package containing supported graph, weights, configuration, metadata, training history, and related assets |
| Project directory | Workspace boundary for user assets, graphs, notes, generated artifacts, checkpoints, and models |

## Current capability scope

| Area | Current position |
| --- | --- |
| Desktop workflow | Dockable Engine UI, project assets, unified node canvas, properties, tasks, logs, Data Studio, scripting, and training views are implemented and under active hardening |
| Data workflows | Preview, profiling, query, visualization, conversion, preprocessing, materialization, and pipeline execution paths exist for supported tabular and time-series cases; large and irregular datasets remain an important production test surface |
| Graph execution | Validation, compilation, local training, testing, inference, checkpoint, and model-package paths are exercised by automated tests, but support varies by node family and problem contract |
| Computation | The C++ backend contains neural-network, evaluation, signal, time-series, text, and selected classical-ML primitives; consistent device-placement truth across all operations remains a priority |
| Scripting | Embedded Python, `.py`/`.cyx` editing, cell execution, completion, debugging, and graph-to-code surfaces are present when Python support is available |
| Model lifecycle | Native `.cyxmodel` inspection, packaging, import/export, checkpoints, and selected external-format integrations exist; exact continuation and some conversion paths are incomplete |
| Server Node | GUI and daemon targets provide worker services and hardware/job reporting; distributed orchestration and training are pre-release |
| Plugins | Assistant, image, logging, and simulation integrations are optional or experimental and are not required core behavior |

The repository is therefore best understood as a working pre-release ML engineering platform, not yet as a finished distributed training product. The current development priority is making execution contracts, computation placement, recovery, observability, and real-dataset workflows production-reliable.

## Repository components

| Path | Purpose |
| --- | --- |
| `cyxwiz-engine/` | Desktop GUI, node editor, Data Studio, training dashboard, scripting, and local orchestration |
| `cyxwiz-backend/` | Tensor, model, layer, loss, optimizer, data-loader, and device abstractions |
| `cyxwiz-protocol/` | Protobuf and gRPC contracts shared by CyxWiz processes |
| `cyxwiz-server-node/` | Compute-worker application and runtime services |
| `plugins/` | Optional integrations; experimental plugins are disabled by default |
| `tests/` | Backend unit tests and focused benchmarks |
| `examples/` | Example scripts, graphs, and integration samples |
| `docs/` | Active, verified public technical documentation |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/`, `vcpkg.json`, `vcpkg-ports/` | Root build definition, platform presets, package discovery, and project-specific packages |

The orchestration and web service are maintained separately in [CyxCloud](https://github.com/CYXWIZ-Lab/cyxcloud). A Central Server implementation is not part of this checkout.

See [Project structure](docs/project-structure.md) for ownership and dependency boundaries.

## Prerequisites

- CMake 3.20 or newer
- a C++20 compiler
- Git and vcpkg
- ArrayFire for Engine execution. Engine builds fail at CMake configure time
  when ArrayFire is missing.
- platform graphics development libraries required by GLFW/OpenGL

The CMake presets expect the `vcpkg` directory at the repository root. Optional Python scripting looks for Python 3.12 or 3.13 and pybind11. CUDA, OpenCL, ONNX Runtime, GGUF/llama.cpp, LibTorch, and MuJoCo are capability-dependent integrations; their presence must not be inferred solely from a GUI preference.

See [INSTALL.md](INSTALL.md) for platform setup and runtime-library guidance.

## Clone and dependencies

```powershell
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
.\setup.bat
```

On Linux or macOS, run `./setup.sh` instead. The public Engine build requires a separate ArrayFire installation. Backend-only native builds without ArrayFire are a reduced development/test configuration, not the GUI runtime path.

The setup script validates the local toolchain and clones/bootstraps vcpkg when necessary; it does not install the dependency manifest into a second global tree. The first CMake configure for a selected build directory validates the manifest and restores packages into that build tree. A populated vcpkg binary cache avoids rebuilding dependency source, although restoring or extracting large cached packages can still take several minutes.

## CMake build structure

The supported CMake entry point is the root [CMakeLists.txt](CMakeLists.txt). Run configure commands from the repository root: `-S .` tells CMake to read that file. The root build resolves shared dependencies and options, then delegates ownership through `add_subdirectory`:

```text
CMakeLists.txt                         root options, dependencies, output paths
├── cyxwiz-protocol/CMakeLists.txt     protobuf and gRPC contracts
├── cyxwiz-backend/CMakeLists.txt      computation backend library
├── cyxwiz-engine/CMakeLists.txt       Engine executable and Engine tests
├── cyxwiz-server-node/CMakeLists.txt  Server Node GUI, daemon, and tests
├── tests/CMakeLists.txt               shared backend tests and benchmarks
└── plugins/**/CMakeLists.txt          explicitly enabled or example plugins
```

Supporting build files are:

- [CMakePresets.json](CMakePresets.json), which defines supported platform/configuration presets and the direct-preset build directory;
- `cmake/Find*.cmake`, which locates dependencies not supplied through standard package configuration;
- [vcpkg.json](vcpkg.json), which is the normal dependency manifest;
- [vcpkg-ci.json](vcpkg-ci.json), which is the broader dependency-validation manifest;
- `vcpkg-ports/`, which contains project-specific vcpkg overlays.

Do not configure a component directory such as `cyxwiz-engine/` by itself. Component files consume targets and package decisions established by the root build.

## Build

### Wrapper scripts

The root wrappers are the simplest supported build interface:

```powershell
# Windows: focused Debug Engine build with eight parallel jobs
.\build.bat --debug --engine -j 8

# Build both Engine and Server Node in Release mode
.\build.bat -j 8

# Build only the Server Node GUI and daemon
.\build.bat --server-node -j 8
```

```bash
# Linux or macOS
./build.sh --debug --engine -j 8
./build.sh --server-node -j 8
```

| Wrapper option | Effect |
| --- | --- |
| `--debug` | Build Debug instead of the default Release configuration |
| `--engine` | Configure the Engine and build only `cyxwiz-engine` |
| `--server-node` | Configure the Server Node and build `cyxwiz-server-gui` plus `cyxwiz-server-daemon` |
| `--build-dir PATH` | Reuse a compatible CMake object tree or choose a custom build directory |
| `--clean` | Delete that wrapper build tree before configuring; use only when a clean rebuild is necessary |
| `-j N` | Run up to `N` compilation jobs in parallel; Windows defaults to 8 and Unix-like systems auto-detect |

Run `build.bat --help` or `./build.sh --help` for the command summary. Wrapper builds use platform/configuration-specific trees such as `build/windows-debug` and `build/windows-release`.

An advanced cached verification can reuse the direct-preset `build` tree:

```powershell
.\build.bat --debug --engine --build-dir build -j 8
```

Only reuse a build tree configured with the same source directory, generator, architecture, and compatible toolchain. Do not add `--clean` when the purpose is to preserve compiled objects.

### Direct CMake and fast iteration

CMake has two distinct phases:

1. **Configure** writes a build tree and stores `-DNAME=VALUE` cache options.
2. **Build** compiles one target or all enabled targets from that tree.

Configure a focused Engine development tree once:

```powershell
cmake --preset windows-debug `
  -DCYXWIZ_BUILD_ENGINE=ON `
  -DCYXWIZ_BUILD_SERVER_NODE=OFF `
  -DCYXWIZ_BUILD_TESTS=OFF `
  -DCYXWIZ_ENABLE_ONNX=OFF `
  -DCYXWIZ_ENABLE_GGUF=OFF `
  -DCYXWIZ_ENABLE_PYTORCH=OFF `
  -DCYXWIZ_ENABLE_NCCL=OFF `
  -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=OFF `
  -DCYXWIZ_BUILD_MUJOCO_PLUGIN=OFF
```

Then use the short incremental command after source edits:

```powershell
cmake --build --preset windows-debug --config Debug --target cyxwiz-engine --parallel 8
```

This is the normal fast edit/build loop: do not pass `--clean`, and do not build every target when only the Engine changed. Reconfigure when changing `-D` options, the toolchain, or dependencies; CMake normally detects ordinary `CMakeLists.txt` changes itself.

The vcpkg binary cache and the CMake object tree solve different problems: the vcpkg cache prevents third-party dependencies from being compiled again, while the configured `build/` tree preserves CyxWiz object files for an incremental target build. Preserve both when validating instructions without a from-scratch build.

`--parallel 8` (equivalent to `-j 8`) allows eight build jobs. `--parallel 1`, `-j 1`, or raw MSBuild `/m:1` forces a serial build: it can help diagnose errors or reduce memory use, but it is not a fast-build setting. Choose a parallel count appropriate for available CPU cores and memory.

For a focused Release Engine build, use the same pattern with `windows-release`, `--config Release`, and the `cyxwiz-engine` target. Equivalent `linux-debug`, `linux-release`, `macos-debug`, and `macos-release` presets are provided. The Android preset builds the backend only.

### Common CMake options

Options are passed during configure as `-DOPTION=ON` or `-DOPTION=OFF`.

| Option | Default | Purpose |
| --- | --- | --- |
| `CYXWIZ_BUILD_ENGINE` | `ON` | Include the desktop Engine |
| `CYXWIZ_BUILD_SERVER_NODE` | `ON` | Include the Server Node |
| `CYXWIZ_BUILD_TESTS` | `ON` | Include test targets; disable only for a focused edit loop |
| `CYXWIZ_ENABLE_PYTHON` | `ON` | Enable embedded Python scripting when Python is available |
| `CYXWIZ_ENABLE_CUDA` | `ON` | Enable ArrayFire CUDA capability paths when dependencies are available |
| `CYXWIZ_ENABLE_OPENCL` | `ON` | Enable ArrayFire OpenCL capability paths |
| `CYXWIZ_ENABLE_NCCL` | `ON` | Request the optional NCCL distributed GPU backend when its prerequisites are found |
| `CYXWIZ_ENABLE_ONNX` | `ON` | Enable ONNX Runtime discovery and integration |
| `CYXWIZ_ENABLE_GGUF` | `ON` | Enable llama.cpp/GGUF discovery and integration |
| `CYXWIZ_ENABLE_PYTORCH` | `ON` | Enable LibTorch discovery and integration |
| `CYXWIZ_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer on supported toolchains |
| `CYXWIZ_ENABLE_TRACY` | `OFF` | Enable optional Engine profiling when Tracy is installed |
| `CYXWIZ_BUILD_ASSISTANT_PLUGIN` | `OFF` | Build the experimental assistant plugin |
| `CYXWIZ_BUILD_MUJOCO_PLUGIN` | `OFF` | Build the optional MuJoCo plugin |

An `ON` capability request does not prove that a dependency, backend, or physical device was selected at runtime. Check CMake's configure summary and CyxWiz runtime placement evidence.

After configuring, inspect all cached CyxWiz options with:

```powershell
cmake -LAH -N build | Select-String CYXWIZ_
```

The optional MuJoCo plugin requires a separate MuJoCo installation and explicit opt-in:

```powershell
cmake --preset windows-release -DCYXWIZ_BUILD_MUJOCO_PLUGIN=ON
```

## Run

Direct Windows preset builds use `build/bin/<Configuration>`:

```powershell
.\build\bin\Debug\cyxwiz-engine.exe
.\build\bin\Debug\cyxwiz-server-gui.exe
.\build\bin\Debug\cyxwiz-server-daemon.exe
```

Root wrapper builds use their own trees:

```powershell
.\build\windows-debug\bin\Debug\cyxwiz-engine.exe
.\build\windows-release\bin\Release\cyxwiz-engine.exe
```

Linux and macOS generators normally place executables under the selected build tree's `bin/` directory without a configuration subdirectory.

Runtime configuration templates live in `config/` and the component resource directories. Never commit real credentials, API keys, tokens, generated checkpoints, datasets, or machine-specific runtime state.

## Test

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

The Catch2 test executable is normally `build/bin/Debug/cyxwiz-tests.exe` on Windows.

Tests prove only the paths and device placements they explicitly exercise. Hardware acceleration must be verified through runtime placement evidence and device monitoring; selecting “GPU” in the interface alone is not proof that every operation executed on a GPU.

## Development workflow

Before opening a change:

1. Keep the change scoped to one responsibility.
2. Preserve component boundaries and extend the shared backend/API instead of duplicating computation in the GUI.
3. Add or update tests for changed behavior.
4. Build the affected targets and run the relevant tests.
5. Document only behavior verified in the current tree.
6. Keep generated, private, licensed third-party, and machine-local files out of Git.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for coding, testing, commit, and review standards. Report security issues using [SECURITY.md](SECURITY.md).

## Current limitations

- computation placement is not yet uniformly guaranteed across every backend operation;
- distributed training is not yet a verified production workflow;
- checkpoint restoration and exact continuation need further lifecycle work;
- several GUI and plugin surfaces expose capabilities whose runtime support varies;
- cross-platform release validation is incomplete.

These limitations are tracked internally and are stated here so public documentation does not over-promise.

## Documentation

- [Installation](INSTALL.md)
- [Contributing](CONTRIBUTING.md)
- [Project structure](docs/project-structure.md)
- [Examples](examples/README.md)
- [Backend overview](cyxwiz-backend/README.md)
- [Server Node overview](cyxwiz-server-node/README.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)

## Licence

Copyright (c) 2026 CYXWIZ COMPUTER SYSTEMS L.L.C - S.P.C.

Use is governed by the [CYXWIZ Commercial Source and Evaluation License](LICENSE). Production, commercial, hosted-service, redistribution, and competitive use require separate written authorisation.
