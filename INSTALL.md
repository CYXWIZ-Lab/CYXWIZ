# Building CyxWiz from source

CyxWiz is pre-release software. Signed alpha installers are published on the [Releases page](https://github.com/CYXWIZ-Lab/CYXWIZ/releases); see [Install a pre-release](README.md#install-a-pre-release). This guide is the detailed reference for building from source. The [README](README.md#build-from-source) has the short version.

## Requirements

- Git
- CMake 3.20 or newer
- a C++20 compiler: Visual Studio 2026 with **Desktop development with C++** on Windows, GCC or Clang on Linux, Xcode Command Line Tools on macOS
- Ninja on Linux and macOS (the presets use it)
- platform OpenGL/GLFW development libraries
- ArrayFire 3.10 for the desktop Engine runtime
- enough disk space and time for the first vcpkg dependency build

Python scripting is optional; CMake accepts Python 3.12 or 3.13 (3.12 is preferred).

ArrayFire is required when `CYXWIZ_BUILD_ENGINE=ON`: install it separately and expose its CMake package to the build. A backend-only build with `CYXWIZ_BUILD_ENGINE=OFF` may compile without ArrayFire as a reduced native development/test configuration, but that is not the public Engine runtime path and does not provide ArrayFire CPU/CUDA/OpenCL/oneAPI placement.

ONNX Runtime, llama.cpp/GGUF, LibTorch, the assistant plugin, and MuJoCo are optional integrations. Disable the ones you do not have.

### Linux packages

Ubuntu 22.04 package names, as used by CI:

```bash
sudo apt-get install -y autoconf autoconf-archive automake build-essential cmake curl \
  libgl1-mesa-dev libgtk-3-dev libtool libwayland-dev libxext-dev libxi-dev \
  libxinerama-dev libxkbcommon-dev libxcursor-dev libxrandr-dev ninja-build patchelf pkg-config
curl -LO https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh
chmod +x ArrayFire.sh && sudo ./ArrayFire.sh --include-subdir --prefix=/opt
```

### macOS packages

```bash
xcode-select --install
brew install arrayfire autoconf autoconf-archive automake bison cmake libtool ninja pkg-config
```

### Windows

Install Visual Studio 2026 with the C++ desktop workload and the ArrayFire 3.10 Windows installer. Run the build from a *Developer Command Prompt for VS 18 2026* so the presets find the toolchain.

## Clone and set up

```powershell
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
.\setup.bat          # Linux/macOS: chmod +x setup.sh build.sh && ./setup.sh
```

The setup script checks the toolchain and clones and bootstraps vcpkg in the repository root when necessary; it does not install the dependency manifest into a second global tree. The first CMake configure for a build directory validates the manifest and restores packages into that build tree. A populated vcpkg binary cache avoids rebuilding dependency sources, although restoring large cached packages can still take several minutes.

## Wrapper scripts

The root wrappers are the simplest build interface:

```powershell
# Windows: Debug Engine build with eight parallel jobs
.\build.bat --debug --engine -j 8

# Engine and Server Node in Release
.\build.bat -j 8

# Server Node GUI and daemon only
.\build.bat --server-node -j 8
```

```bash
# Linux or macOS
./build.sh --debug --engine -j 8
./build.sh --server-node -j 8
```

| Option | Effect |
| --- | --- |
| `--debug` | Debug instead of the default Release configuration |
| `--engine` | Configure the Engine and build only `cyxwiz-engine` |
| `--server-node` | Configure the Server Node and build `cyxwiz-server-gui` plus `cyxwiz-server-daemon` |
| `--build-dir PATH` | Reuse a compatible CMake tree or choose a custom build directory |
| `--clean` | Delete the wrapper build tree before configuring; only when a clean rebuild is necessary |
| `-j N` | Up to `N` parallel jobs; Windows defaults to 8, Linux/macOS auto-detect |

Run `build.bat --help` or `./build.sh --help` for the summary. Wrapper builds use `build/<platform>-<configuration>`, such as `build/windows-debug`, `build/linux-release`, or `build/macos-debug`.

A cached verification can reuse the direct-preset `build` tree:

```powershell
.\build.bat --debug --engine --build-dir build -j 8
```

Only reuse a build tree configured with the same source directory, generator, architecture, and a compatible toolchain. Do not add `--clean` when the goal is to keep compiled objects.

## CMake build structure

The supported entry point is the root [CMakeLists.txt](CMakeLists.txt). Configure from the repository root (`-S .`); the root build resolves shared dependencies and options, then delegates through `add_subdirectory`:

```text
CMakeLists.txt                         root options, dependencies, output paths
├── cyxwiz-protocol/CMakeLists.txt     protobuf and gRPC contracts
├── cyxwiz-backend/CMakeLists.txt      computation backend library
├── cyxwiz-engine/CMakeLists.txt       Engine executable and Engine tests
├── cyxwiz-server-node/CMakeLists.txt  Server Node GUI, daemon, and tests
├── tests/CMakeLists.txt               shared backend tests and benchmarks
└── plugins/**/CMakeLists.txt          explicitly enabled or example plugins
```

Supporting files:

- [CMakePresets.json](CMakePresets.json): platform/configuration presets (`windows-debug`, `windows-release`, `linux-debug`, `linux-release`, `macos-debug`, `macos-release`, `android-release`) and the direct-preset build directory `build`;
- `cmake/Find*.cmake`: dependency discovery not covered by standard package configuration;
- [vcpkg.json](vcpkg.json): the normal dependency manifest;
- [vcpkg-ci.json](vcpkg-ci.json): the broader dependency-validation manifest used by CI;
- `vcpkg-ports/`: project-specific vcpkg overlays.

Do not configure a component directory such as `cyxwiz-engine/` by itself; component files consume targets and decisions made by the root build.

## Direct CMake and fast iteration

CMake has two phases: **configure** writes a build tree and stores `-DNAME=VALUE` options; **build** compiles targets from that tree.

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

Then rebuild incrementally after edits:

```powershell
cmake --build --preset windows-debug --config Debug --target cyxwiz-engine --parallel 8
```

Do not pass `--clean`, and do not build every target when only the Engine changed. Reconfigure when changing `-D` options, the toolchain, or dependencies; CMake detects ordinary `CMakeLists.txt` changes itself.

The vcpkg binary cache and the CMake object tree solve different problems: the cache keeps third-party dependencies from being compiled again, and the configured `build/` tree keeps CyxWiz object files for incremental builds. Preserve both when you only want to re-verify.

`--parallel N` (or `-j N`) sets the number of build jobs. `--parallel 1` or MSBuild `/m:1` builds serially, which helps diagnose errors or limit memory, but is slow. Pick a count that suits your CPU cores and memory.

For a Release Engine, use `windows-release`, `--config Release`, and the `cyxwiz-engine` target. The Linux and macOS presets work the same way with Ninja (no `--config`). The Android preset builds the backend only.

Other useful builds:

```powershell
# Minimal capability build
cmake --preset windows-debug -DCYXWIZ_ENABLE_ONNX=OFF -DCYXWIZ_ENABLE_GGUF=OFF -DCYXWIZ_ENABLE_PYTORCH=OFF -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=OFF -DCYXWIZ_BUILD_MUJOCO_PLUGIN=OFF
cmake --build --preset windows-debug

# Backend-only native development without the Engine GUI
cmake --preset windows-debug -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
```

## Build options

Pass options at configure time as `-DOPTION=ON` or `-DOPTION=OFF`.

| Option | Default | Purpose |
| --- | --- | --- |
| `CYXWIZ_BUILD_ENGINE` | `ON` | Include the desktop Engine |
| `CYXWIZ_BUILD_SERVER_NODE` | `ON` | Include the Server Node |
| `CYXWIZ_BUILD_TESTS` | `ON` | Include test targets; disable only for a focused edit loop |
| `CYXWIZ_ENABLE_PYTHON` | `ON` | Embedded Python scripting when Python 3.12/3.13 is found |
| `CYXWIZ_ENABLE_CUDA` | `ON` | ArrayFire CUDA capability paths when dependencies are available |
| `CYXWIZ_ENABLE_OPENCL` | `ON` | ArrayFire OpenCL capability paths |
| `CYXWIZ_ENABLE_NCCL` | `ON` | Optional NCCL distributed GPU backend when its prerequisites are found |
| `CYXWIZ_ENABLE_ONNX` | `ON` | ONNX Runtime discovery and integration |
| `CYXWIZ_ENABLE_GGUF` | `ON` | llama.cpp/GGUF discovery and integration |
| `CYXWIZ_ENABLE_PYTORCH` | `ON` | LibTorch discovery and integration |
| `CYXWIZ_ENABLE_ASAN` | `OFF` | AddressSanitizer on supported toolchains |
| `CYXWIZ_ENABLE_TRACY` | `OFF` | Optional Engine profiling when Tracy is installed |
| `CYXWIZ_BUILD_ASSISTANT_PLUGIN` | `OFF` | Experimental assistant plugin |
| `CYXWIZ_BUILD_MUJOCO_PLUGIN` | `OFF` | Optional MuJoCo plugin (needs a separate MuJoCo installation) |

An `ON` capability request does not prove that a dependency, backend, or device is used at run time; check CMake's configure summary and the Engine's verified device routes. List every cached CyxWiz option with:

```powershell
cmake -LAH -N build | Select-String CYXWIZ_      # Linux/macOS: cmake -LAH -N build | grep CYXWIZ_
```

## ArrayFire discovery

If CMake cannot find ArrayFire, pass the directory that contains `ArrayFireConfig.cmake`:

```powershell
cmake --preset windows-debug -DArrayFire_DIR="C:\Program Files\ArrayFire\v3\cmake"
```

At run time, ArrayFire's libraries and the backend libraries you use must be on the platform library search path; on Windows this means `af.dll` and the selected backend DLLs. CPU compute in the Engine means ArrayFire CPU (`AF_BACKEND_CPU`); native C++ CPU fallback is a recorded compatibility/debug path, not the normal runtime. Check the resolved backend and physical device in **Preferences > Devices** and in runtime evidence; a GPU preference alone is not sufficient.

## Executables

A native build can produce:

- `cyxwiz-engine`: the desktop application;
- `cyxwiz-server-gui`: the Server Node graphical application;
- `cyxwiz-server-daemon`: the Server Node daemon;
- `cyxwiz-tests`: the Catch2 test executable (with `CYXWIZ_BUILD_TESTS=ON`).

Windows multi-configuration builds place them under `bin/Debug` or `bin/Release` inside the build tree (for example `build/bin/Debug` for direct presets or `build/windows-debug/bin/Debug` for the wrapper). Linux and macOS builds place them under the build tree's `bin/`.

## Tests

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

Use the matching preset on Linux and macOS (no `-C`). Tests prove only the paths and device placements they exercise.

## Troubleshooting

When configuration fails:

1. read the first CMake error rather than the final summary;
2. confirm the vcpkg toolchain path exists (run the setup script again if not);
3. disable unavailable optional integrations;
4. confirm the compiler matches the configured generator (Visual Studio 2026 for the Windows presets);
5. set `ArrayFire_DIR` only when ArrayFire is installed.

For reproducible problems, follow [SUPPORT.md](SUPPORT.md).
