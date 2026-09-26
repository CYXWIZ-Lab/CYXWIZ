# CyxWiz

CyxWiz is a native C++20, visual-first environment for building, inspecting, and running machine-learning and data workflows. It combines a node graph, dataset tooling, training controls, scripting, model packaging, and worker-side services around one shared computation backend.

> **Project status:** active pre-release development. Local graph editing, data workflows, training, testing, scripting, and several inference paths are implemented and exercised by the test suite. Device-placement truth, exact checkpoint resume, and end-to-end distributed execution still require production hardening. Do not treat the current tree as a production-ready distributed training service.

CyxWiz is source-available under the [CYXWIZ Commercial Source and Evaluation License](LICENSE), not an open-source licence.

## Contents

- [What CyxWiz is](#what-cyxwiz-is)
- [Install a pre-release](#install-a-pre-release)
- [Use GPUs](#use-gpus)
- [Quick start](#quick-start)
- [Build from source](#build-from-source)
- [Run and test a source build](#run-and-test-a-source-build)
- [Architecture](#architecture)
- [Current capability scope](#current-capability-scope)
- [Current limitations](#current-limitations)
- [Contributing and documentation](#contributing-and-documentation)
- [Licence](#licence)

## What CyxWiz is

Machine-learning projects often split data preparation, model design, training code, experiment monitoring, model files, and remote compute across unrelated tools. CyxWiz keeps those stages connected through a project workspace and a saved graph that the Engine validates and executes.

- **CyxWiz Engine** is the desktop application: project assets, Data Studio (preview, profile, query, visualize, prepare datasets), the node editor, training and testing with a live dashboard, checkpoints, the Script Editor, and `.cyxmodel` packaging. The graph is not only a drawing: the Engine checks its contracts and turns supported nodes into data materialization, batching, model, loss, optimizer, training, testing, and inference operations.
- **cyxwiz-backend** owns computation: tensors, layers, activations, losses, optimizers, data primitives, evaluation utilities, device selection, and ArrayFire execution on CPU, CUDA, OpenCL, and oneAPI.
- **CyxWiz Server Node** is the worker-side application (daemon and GUI) for hardware reporting, job lifecycle, deployment, files, and metrics through the shared protocol. The orchestration service, CyxCloud, lives in a [separate repository](https://github.com/CYXWIZ-Lab/cyxcloud). End-to-end distributed training is still being hardened.

CyxWiz is visual-first rather than visual-only: the Script Editor supports Python and `.cyx` cell documents, and optional integrations stay behind a plugin boundary.

### Project formats

| Format | Purpose |
| --- | --- |
| Project directory | Workspace for datasets, graphs, scripts, notes, generated artifacts, checkpoints, and models |
| `.cyxgraph` | Saved JSON graph: nodes, links, parameters, and workflow metadata |
| `.cyx` | Python-compatible CyxWiz script document that may contain executable cells |
| `.cyxmodel` | Native model package: supported graph, weights, configuration, metadata, training history, and related assets |

## Install a pre-release

Signed alpha installers are published as pre-releases on the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases). Each release has a small **setup** program per platform. The setup downloads and verifies the signed installer, and the installer installs the Engine and any compute packs you select; it downloads only what you choose.

| Platform | Setup download | Engine (CPU base) | Optional GPU compute packs |
| --- | --- | --- | --- |
| Windows x64 | `cyxwiz-setup-windows-x64.zip` | Yes | CUDA, OpenCL, oneAPI |
| Linux x86_64 | `cyxwiz-setup-ubuntu-x64.tar.gz` | Yes | Not yet |
| macOS Intel | `cyxwiz-setup-macos-x64.tar.gz` | Yes | Not yet (OpenCL planned) |
| macOS Apple Silicon | `cyxwiz-setup-macos-arm64.tar.gz` | Yes | Not yet |

Release builds are produced on Windows Server 2022, Ubuntu 22.04, and macOS 15 runners; older operating systems are untested.

**Windows**

1. Download and extract `cyxwiz-setup-windows-x64.zip`, then run `cyxwiz-setup.exe`.
2. Choose what to install. Per-user installs go to `%LOCALAPPDATA%\Programs\CyxWiz`, and the installer only accepts an empty folder or an existing CyxWiz install.
3. Start CyxWiz from the Start Menu. **CyxWiz Installer** (same Start Menu folder) modifies, repairs, or uninstalls it; it is also listed under Settings > Apps. Uninstalling removes the install folder and keeps your Engine settings and data in `%LOCALAPPDATA%\CyxWiz`.

The Microsoft Visual C++ runtime ships with CyxWiz; no separate redistributable is needed.

**Linux and macOS**

```bash
# <platform> is ubuntu-x64, macos-x64 or macos-arm64
mkdir cyxwiz-setup
tar -xzf cyxwiz-setup-<platform>.tar.gz -C cyxwiz-setup
./cyxwiz-setup/cyxwiz-setup
```

The binaries are not Apple-notarized. If macOS blocks the setup, allow it under **System Settings > Privacy & Security**, or remove the quarantine flag with `xattr -dr com.apple.quarantine cyxwiz-setup`.

**Verify downloads.** Release metadata and packs are Ed25519-signed and checked by the setup and installer. Windows binaries are not Authenticode-signed; compare files against the release's `SHA256SUMS.txt` if you download them manually.

**Which release?** Use the newest alpha. Alphas 1.0.3 and 1.0.4 have a Windows uninstall defect described in their release notes; if you still have one of them, back up `%LOCALAPPDATA%\CyxWiz` before uninstalling it.

## Use GPUs

The Engine always includes the ArrayFire **CPU** runtime. GPU support comes from optional **compute packs** that add ArrayFire's official GPU runtimes beside the Engine; nothing is installed system-wide.

| Pack (Windows x64) | GPUs | Approximate download |
| --- | --- | --- |
| OpenCL | NVIDIA, Intel (AMD untested) | 7 MB |
| oneAPI | Intel GPUs and CPUs | 280 MB |
| CUDA | NVIDIA | 1.7 GB |

1. Install a current driver for your GPU.
2. Select the pack in the installer (or add it later through **CyxWiz Installer**).
3. In the Engine, open **Preferences > Devices**. Each physical device appears as a card with its compute routes. Click **Verify** on a route, or **Verify all**.

Verification runs a short qualification for each route in an isolated process on your hardware and records a measured speed; training uses only verified routes, and a failed route shows its reason. A full machine typically verifies in about two minutes. On Intel GPUs the Engine recommends OpenCL; oneAPI remains available (oneAPI is known to fail on Intel UHD 630).

A GPU pack is published only after its routes pass on real hardware. The Windows packs are verified on NVIDIA GeForce (CUDA, OpenCL) and Intel UHD 630 / Iris Xe (OpenCL, oneAPI). Linux and macOS installs are CPU-only until their packs can be verified the same way.

## Quick start

1. **Create or open a project.** The workspace gives graphs, scripts, datasets, checkpoints, and models an explicit context.
2. **Add data.** Use the Asset Browser and Data Studio to preview, profile, query, visualize, or prepare supported datasets (Arrow-based paths, DuckDB-backed analysis).
3. **Build a graph.** Connect data, preprocessing, model, training, evaluation, visualization, and export nodes, then save it as a `.cyxgraph`.
4. **Validate and compile.** The compiler checks topology, pins, shapes, roles, targets, and runtime contracts before anything runs.
5. **Train or test.** The Engine materializes datasets and batches and runs the graph on the selected device; follow progress in Tasks, logs, and the Training Dashboard.
6. **Keep the result.** Checkpoints are saved during training; export a `.cyxmodel` package with **Export Model**.

The [examples](examples/README.md) folder contains sample graphs, scripts, and datasets. Available nodes, formats, and execution backends depend on the build and on what the runtime discovers on your machine; a visible node or a GPU preference alone does not prove that a path is implemented or GPU-resident.

## Build from source

Builds use CMake presets and vcpkg for dependencies; ArrayFire 3.10 is installed separately. The setup scripts check the toolchain and clone and bootstrap vcpkg in the repository root; the first configure then installs dependencies into the build tree, which can take a long time on a cold cache.

### Requirements

| | Windows | Linux | macOS |
| --- | --- | --- | --- |
| Compiler | Visual Studio 2026 with **Desktop development with C++** | GCC or Clang with C++20 (`build-essential`) | Xcode Command Line Tools |
| Build tools | CMake 3.20+ (Visual Studio generator) | CMake 3.20+, Ninja | CMake 3.20+, Ninja |
| ArrayFire 3.10 | [Windows installer](https://arrayfire.com/download/) | `ArrayFire.sh` installer, e.g. into `/opt` | `brew install arrayfire` |
| Python (optional scripting) | 3.12 or 3.13 | 3.12 or 3.13 | 3.12 or 3.13 |

**Linux packages** (Ubuntu 22.04 names, as used by CI):

```bash
sudo apt-get install -y autoconf autoconf-archive automake build-essential cmake curl \
  libgl1-mesa-dev libgtk-3-dev libtool libwayland-dev libxext-dev libxi-dev \
  libxinerama-dev libxkbcommon-dev libxcursor-dev libxrandr-dev ninja-build patchelf pkg-config
curl -LO https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
chmod +x ArrayFire.sh && sudo ./ArrayFire.sh --include-subdir --prefix=/opt
```

**macOS packages:**

```bash
xcode-select --install
brew install arrayfire autoconf autoconf-archive automake bison cmake libtool ninja pkg-config
```

### Build

**Windows** (from a *Developer Command Prompt for VS 18 2026*):

```powershell
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
.\setup.bat
.\build.bat --engine -j 8
```

**Linux and macOS:**

```bash
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
chmod +x setup.sh build.sh
./setup.sh
./build.sh --engine -j 8
```

| Wrapper option | Effect |
| --- | --- |
| `--debug` | Debug instead of the default Release configuration |
| `--engine` | Build only the Engine (`cyxwiz-engine`) |
| `--server-node` | Build only the Server Node (`cyxwiz-server-gui`, `cyxwiz-server-daemon`) |
| `--build-dir PATH` | Use a custom or existing compatible build tree |
| `--clean` | Delete the build tree first (only when a clean rebuild is needed) |
| `-j N` | Parallel jobs; Windows defaults to 8, Linux/macOS auto-detect |

With no component option, both the Engine and the Server Node are built. Wrapper builds use `build/<platform>-<configuration>`, for example `build/windows-release`. If CMake cannot find ArrayFire, pass `-DArrayFire_DIR=<ArrayFire>/cmake` (Windows) or the directory containing `ArrayFireConfig.cmake`.

For direct CMake presets, a fast edit/build loop, every build option, backend-only builds, and troubleshooting, see [INSTALL.md](INSTALL.md).

## Run and test a source build

```powershell
# Windows wrapper build
.\build\windows-release\bin\Release\cyxwiz-engine.exe
```

```bash
# Linux / macOS wrapper build
./build/linux-release/bin/cyxwiz-engine     # or build/macos-release/bin/cyxwiz-engine
```

ArrayFire's libraries must be on the library search path at run time (on Windows, `af.dll` and the backend DLLs you use; the ArrayFire installer can add them to `PATH`). The Server Node runs as `cyxwiz-server-gui` or `cyxwiz-server-daemon` from the same `bin` folder.

Tests use Catch2 and CTest:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

Tests prove only the paths and device placements they exercise. Hardware acceleration is shown by runtime placement evidence and verified device routes, not by a GPU preference in the interface.

Runtime configuration templates live in `config/` and the component resource folders. Never commit credentials, API keys, tokens, checkpoints, datasets, or machine-specific state.

## Architecture

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
  C++ primitives + ArrayFire CPU/CUDA/OpenCL/oneAPI execution

CyxWiz Engine -------- cyxwiz-protocol -------- Server Node
                                                   |
                                                   +--> cyxwiz-backend
                                                   +--> external CyxCloud orchestration

Optional plugins --> explicit Engine/plugin contracts
```

The Engine owns interaction and orchestration, the Backend owns computation, the Protocol owns process contracts, the Server Node owns worker lifecycle, and plugins own optional integrations. The Engine calls the shared backend rather than reimplementing mathematics in GUI code.

| Path | Purpose |
| --- | --- |
| `cyxwiz-engine/` | Desktop GUI, node editor, Data Studio, training dashboard, scripting, installer, and local orchestration |
| `cyxwiz-backend/` | Tensor, model, layer, loss, optimizer, data-loader, and device abstractions |
| `cyxwiz-protocol/` | Protobuf and gRPC contracts shared by CyxWiz processes |
| `cyxwiz-server-node/` | Compute-worker application and runtime services |
| `plugins/` | Optional integrations; experimental plugins are disabled by default |
| `redist/` | Release packaging, signing contracts, setup bootstrapper, and their tests |
| `tests/` | Backend unit tests, smoke probes, and focused benchmarks |
| `examples/` | Example projects, graphs, scripts, and integration samples |
| `docs/` | Public technical documentation |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/`, `vcpkg.json`, `vcpkg-ports/` | Build definition, presets, package discovery, and project-specific packages |

See [Project structure](docs/project-structure.md) for ownership and dependency boundaries.

## Current capability scope

| Area | Current position |
| --- | --- |
| Desktop workflow | Dockable Engine UI, project assets, unified node canvas, properties, tasks, logs, Data Studio, scripting, and training views are implemented and under active hardening |
| Data workflows | Preview, profiling, query, visualization, conversion, preprocessing, materialization, and pipeline execution exist for supported tabular and time-series cases; large and irregular datasets remain an important test surface |
| Graph execution | Validation, compilation, local training, testing, inference, checkpoint, and model-package paths are exercised by automated tests; support varies by node family and problem contract |
| Computation | Neural-network, evaluation, signal, time-series, text, and selected classical-ML primitives; ArrayFire CPU/CUDA/OpenCL/oneAPI routes are qualified per machine before training uses them |
| Scripting | Embedded Python, `.py`/`.cyx` editing, cell execution, completion, debugging, and graph-to-code surfaces when Python support is built in |
| Model lifecycle | `.cyxmodel` inspection, packaging, import/export, checkpoints, and selected external formats; exact continuation and some conversions are incomplete |
| Server Node | GUI and daemon provide worker services and hardware/job reporting; distributed orchestration and training are pre-release |
| Plugins | Assistant, image, logging, and simulation integrations are optional or experimental |

## Current limitations

- computation placement is not yet uniformly guaranteed across every backend operation;
- distributed training is not yet a verified production workflow;
- checkpoint restoration and exact continuation need further lifecycle work;
- several GUI and plugin surfaces expose capabilities whose runtime support varies;
- GPU compute packs are published for Windows only; Linux and macOS installs are CPU-only;
- current release builds do not include embedded Python scripting; build from source with Python 3.12 or 3.13 to use it;
- release binaries are not Authenticode-signed or Apple-notarized.

These limitations are stated so public documentation does not over-promise.

## Contributing and documentation

Keep changes scoped, preserve component boundaries (extend the shared backend instead of duplicating computation in the GUI), add tests for changed behavior, and document only behavior verified in the current tree. Read [CONTRIBUTING.md](CONTRIBUTING.md) for coding, testing, commit, and review standards, and report security issues through [SECURITY.md](SECURITY.md).

- [Building from source (detailed)](INSTALL.md)
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
