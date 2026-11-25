---
title: Build CyxWiz from source
description: Set up the toolchain, configure with a preset, build the Engine or the server node, and keep incremental builds fast.
type: how-to
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - code-layout.md
  - testing.md
  - architecture/compute-back-ends.md
---

Use this when you want to change CyxWiz and run your own build. The full
reference, with every package name and error message, is
[INSTALL.md](../../INSTALL.md) at the repository root; this page is the
short path through it.

## Before you start

- Git and CMake 3.20 or newer (3.27 or newer to build the NVIDIA provider
  module).
- A C++20 compiler: Visual Studio 2026 with **Desktop development with C++**
  on Windows, GCC or Clang on Linux, the Xcode Command Line Tools on macOS.
  Linux and macOS also need Ninja.
- **ArrayFire 3.10**, installed separately. Every build that includes the
  backend needs it; configure fails without it. Only the standalone
  installer build (`CYXWIZ_BUILD_INSTALLER_ONLY=ON`) works without it.
- Optional: Python 3.12 or 3.13 for scripting (3.12 preferred), the CUDA
  Toolkit for the NVIDIA provider, the Intel oneAPI DPC++ compiler for the
  oneAPI provider, LibTorch for the parity oracle.
- Disk space and time for the first dependency build: vcpkg compiles gRPC,
  Arrow, ImGui and the rest once, then reuses its binary cache.

## Steps

1. Clone and set up vcpkg. The setup script checks the toolchain and
   bootstraps vcpkg in the repository root:

   ```powershell
   git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
   cd CYXWIZ
   .\setup.bat
   ```

   On Linux and macOS run `chmod +x setup.sh build.sh && ./setup.sh`. On
   Windows, work in a **Developer Command Prompt for VS 18 2026** so the
   presets find the compiler.

2. Build with the wrapper script. It configures a preset and builds:

   ```powershell
   .\build.bat --debug --engine -j 8
   ```

   `./build.sh --debug --engine -j 8` on Linux and macOS. `--server-node`
   builds `cyxwiz-server-gui` and `cyxwiz-server-daemon` instead; no flag
   builds both in Release. The build tree is `build/<platform>-<config>`,
   for example `build/windows-debug`.

3. Run it. Windows multi-configuration builds put the programs in
   `bin/Debug` or `bin/Release` inside the build tree; Linux and macOS in
   `bin/`. Start `cyxwiz-engine` from there.

4. Optional: if CMake cannot find ArrayFire, pass the folder that holds
   `ArrayFireConfig.cmake`, for example
   `-DArrayFire_DIR="C:\Program Files\ArrayFire\v3\cmake"`, or on Linux
   `<prefix>/share/ArrayFire/cmake`. At run time ArrayFire's libraries must be
   on the library search path.

## Configure for fast edit-build cycles

Configure a focused tree once with the presets (`windows-debug`,
`windows-release`, `linux-debug`, `linux-release`, `macos-debug`,
`macos-release`):

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_TESTS=OFF `
  -DCYXWIZ_ENABLE_ONNX=OFF -DCYXWIZ_ENABLE_GGUF=OFF -DCYXWIZ_ENABLE_PYTORCH=OFF `
  -DCYXWIZ_ENABLE_NCCL=OFF -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=OFF -DCYXWIZ_BUILD_MUJOCO_PLUGIN=OFF
```

Then rebuild only the target you changed:

```powershell
cmake --build --preset windows-debug --config Debug --target cyxwiz-engine --parallel 8
```

- Configure from the repository root only. A component folder such as
  `cyxwiz-engine/` uses targets the root build defines and cannot be
  configured alone.
- Reconfigure only when you change a `-D` option, the toolchain or the
  dependencies; CMake notices ordinary `CMakeLists.txt` edits itself.
- Keep one build tree per worktree or branch. Two checkouts sharing a tree
  leads to a build that reports success while running old code.
- After you change a widely included header, watch for crashes from stale
  objects; a clean rebuild of the affected target settles it.
- Do not judge a build by a piped command's exit status
  (`cmake --build ... | tail`): the pipe returns the last command's status,
  not CMake's.

## Build options

| Option | Default | What it does |
| --- | --- | --- |
| `CYXWIZ_BUILD_ENGINE` | ON | The desktop Engine |
| `CYXWIZ_BUILD_SERVER_NODE` | ON | The server node (daemon and GUI/TUI client) |
| `CYXWIZ_BUILD_TESTS` | ON | Every test target |
| `CYXWIZ_BUILD_INSTALLER_ONLY` | OFF | Only the installer and the GPU pack manager |
| `CYXWIZ_ENABLE_PYTHON` | ON | Embedded Python scripting and `pycyxwiz` when Python 3.12/3.13 is found |
| `CYXWIZ_REQUIRE_PYTHON` | OFF | Fail configure unless Python 3.12 scripting is available |
| `CYXWIZ_ENABLE_CUDA`, `CYXWIZ_ENABLE_OPENCL` | ON | ArrayFire CUDA and OpenCL capability paths |
| `CYXWIZ_ENABLE_NVIDIA_DNN_PROVIDER` | OFF | Build the `cyxwiz-nvidia-provider` module (needs the CUDA Toolkit) |
| `CYXWIZ_ENABLE_OPENCL_DNN_PROVIDER`, `CYXWIZ_ENABLE_OPENCL_CLBLAST` | OFF | The OpenCL neural provider, and CLBlast GEMMs for it |
| `CYXWIZ_ENABLE_ONEAPI_DNN_PROVIDER` | OFF | The oneAPI provider; set `CYXWIZ_ONEAPI_COMPILER_ROOT` to the DPC++ compiler |
| `CYXWIZ_ENABLE_NCCL` | ON | NCCL when the CUDA Toolkit and NCCL are found |
| `CYXWIZ_ENABLE_ONNX`, `CYXWIZ_ENABLE_GGUF` | ON | ONNX Runtime and llama.cpp/GGUF integration |
| `CYXWIZ_ENABLE_PYTORCH` | ON | LibTorch for the parity oracle and the node's TorchScript import |
| `CYXWIZ_ENABLE_SENTENCEPIECE`, `CYXWIZ_ENABLE_HTML_PARSER` | OFF | SentencePiece tokenizer, Lexbor HTML parsing in Text Clean |
| `CYXWIZ_ENABLE_TRACY` | OFF | Tracy profiling when Tracy is installed |
| `CYXWIZ_ENABLE_ASAN` | OFF | AddressSanitizer |
| `CYXWIZ_BUILD_MUJOCO_PLUGIN` | OFF | The MuJoCo plugin (downloads a pinned MuJoCo, or set `MUJOCO_DIR`) |
| `CYXWIZ_BUILD_ASSISTANT_PLUGIN` | OFF | The experimental assistant plugin |
| `CYXWIZ_ORACLE_DEVICES` | `cpu` | Devices the oracle tests register for (`cpu;cuda;opencl;oneapi`) |

An `ON` request does not prove a dependency or device is used at run time.
Read CMake's configure summary, and check **Preferences > Devices** in the
running Engine. List every cached option with `cmake -LAH -N build`.

## If something goes wrong

- **Configure fails**: read the first CMake error, not the summary. Check
  the vcpkg toolchain path exists (run the setup script again), turn off
  optional integrations you do not have, and check the compiler matches the
  preset.
- **"CMake was unable to find a build program corresponding to "Ninja""**
  right after `Running vcpkg install - failed`: a vcpkg library failed to
  build. Scroll up to `error: building <library>` and read the log it names.
- **GCC 15 or 16 errors in Abseil or ONNX Runtime**: build through
  `build.sh`, which applies the project's overlay ports and Linux triplet, or
  pass `-DVCPKG_OVERLAY_PORTS=$PWD/vcpkg-ports
  -DVCPKG_OVERLAY_TRIPLETS=$PWD/vcpkg-triplets` as absolute paths.

More cases are in [INSTALL.md](../../INSTALL.md#troubleshooting).

## Related

- [Code layout](code-layout.md)
- [Tests and checks](testing.md)
- [Compute back ends and neural providers](architecture/compute-back-ends.md)
