# CyxWiz

CyxWiz is a C++20 visual environment for building, inspecting, and running machine-learning and data workflows. This repository contains the desktop Engine, shared computation backend, network protocol, and Server Node worker.

> **Project status:** active pre-release development. Local graph editing, data workflows, training, testing, scripting, and several inference paths are implemented and exercised by the test suite. Device-placement truth, exact checkpoint resume, and end-to-end distributed execution still require production hardening. Do not treat the current tree as a production-ready distributed training service.

CyxWiz is source-available under the [CYXWIZ Commercial Source and Evaluation License](LICENSE), not an open-source licence.

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
| `cmake/`, `vcpkg-ports/`, `vcpkg-triplets/` | Build integration and project-specific package definitions |

The orchestration and web service are maintained separately in [CyxCloud](https://github.com/CYXWIZ-Lab/cyxcloud). A Central Server implementation is not part of this checkout.

See [Project structure](docs/project-structure.md) for ownership and dependency boundaries.

## Prerequisites

- CMake 3.20 or newer
- a C++20 compiler
- Git and vcpkg
- ArrayFire for accelerated backend execution
- platform graphics development libraries required by GLFW/OpenGL

The CMake presets expect the `vcpkg` directory at the repository root. Optional Python scripting looks for Python 3.12 or 3.13 and pybind11. CUDA, OpenCL, ONNX Runtime, GGUF/llama.cpp, LibTorch, and MuJoCo are capability-dependent integrations; their presence must not be inferred solely from a GUI preference.

See [INSTALL.md](INSTALL.md) for platform setup and runtime-library guidance.

## Clone and dependencies

```powershell
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
.\setup.bat
```

On Linux or macOS, run `./setup.sh` instead. The backend can compile without ArrayFire, but accelerated CPU/CUDA/OpenCL execution requires a separate ArrayFire installation.

## Build

The wrapper scripts build the Engine and Server Node.

```powershell
# Windows
.\build.bat
.\build.bat --debug
.\build.bat --engine
.\build.bat --server-node
```

```bash
# Linux or macOS
./build.sh
./build.sh --debug
./build.sh --engine
./build.sh --server-node
```

Run either script with `--help` for all options. You can also use CMake directly:

```powershell
cmake --preset windows-debug
cmake --build --preset windows-debug
```

For a smaller baseline build, disable integrations that are not installed:

```powershell
cmake --preset windows-debug -DCYXWIZ_ENABLE_ONNX=OFF -DCYXWIZ_ENABLE_GGUF=OFF -DCYXWIZ_ENABLE_PYTORCH=OFF -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=OFF -DCYXWIZ_BUILD_MUJOCO_PLUGIN=OFF
cmake --build --preset windows-debug
```

Equivalent `linux-debug`, `linux-release`, `macos-debug`, and `macos-release` presets are provided. The Android preset builds the backend only.

The optional MuJoCo plugin requires a separate MuJoCo installation and explicit opt-in:

```bash
cmake --preset windows-release -DCYXWIZ_BUILD_MUJOCO_PLUGIN=ON
```

## Run

With the default Windows build layout:

```powershell
.\build\bin\Release\cyxwiz-engine.exe
.\build\bin\Release\cyxwiz-server-gui.exe
.\build\bin\Release\cyxwiz-server-daemon.exe
```

For a Debug build, replace `Release` with `Debug`. Linux and macOS generators may use `build/bin` without a configuration subdirectory.

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
