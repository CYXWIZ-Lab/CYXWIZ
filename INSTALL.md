# Installing CyxWiz from source

CyxWiz is pre-release software. The supported installation path in this repository is a source build; published binary availability is not guaranteed.

## Requirements

- Git
- CMake 3.20 or newer
- a C++20 compiler
- platform OpenGL/GLFW development support
- enough disk space for the vcpkg dependency build
- ArrayFire for the desktop Engine runtime

The setup scripts clone and bootstrap vcpkg in the repository root. Python scripting is optional and the current CMake contract accepts Python 3.12 or 3.13.

ArrayFire is required when `CYXWIZ_BUILD_ENGINE=ON`. Install it separately and expose its CMake package to the build. A backend-only build with `CYXWIZ_BUILD_ENGINE=OFF` may compile without ArrayFire as a reduced native development/test configuration, but that is not the public Engine runtime path and does not provide ArrayFire CPU/CUDA/oneAPI/OpenCL placement.

ONNX Runtime, llama.cpp/GGUF, LibTorch, the assistant plugin, and MuJoCo are optional integrations. Disable integrations you do not have.

## Windows

Use the Visual Studio developer environment expected by `CMakePresets.json`, then:

```powershell
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
.\setup.bat
.\build.bat --debug
```

The Debug executables are produced under `build/windows-debug/bin/Debug` by the wrapper script. Direct preset builds use `build/bin/Debug`.

## Linux and macOS

Install a compiler and platform graphics headers first, then:

```bash
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
chmod +x setup.sh build.sh
./setup.sh
./build.sh --debug
```

Wrapper builds use `build/linux-debug` or `build/macos-debug`.

## Direct CMake build

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
ctest --test-dir build -C Debug --output-on-failure
```

Use the equivalent platform preset outside Windows.

For a minimal capability build:

```powershell
cmake --preset windows-debug -DCYXWIZ_ENABLE_ONNX=OFF -DCYXWIZ_ENABLE_GGUF=OFF -DCYXWIZ_ENABLE_PYTORCH=OFF -DCYXWIZ_BUILD_ASSISTANT_PLUGIN=OFF -DCYXWIZ_BUILD_MUJOCO_PLUGIN=OFF
cmake --build --preset windows-debug
```

For backend-only native development without the Engine GUI:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_SERVER_NODE=OFF -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
```

## ArrayFire discovery

If CMake cannot find an installed ArrayFire package, pass its package directory explicitly:

```powershell
cmake --preset windows-debug -DArrayFire_DIR="C:\path\to\ArrayFire\lib\cmake\ArrayFire"
```

At runtime, the ArrayFire libraries and selected backend libraries must be discoverable through the platform library search path. On Windows this includes `af.dll` and the selected backend runtime libraries. CPU compute in the Engine means ArrayFire CPU (`AF_BACKEND_CPU`); native C++ CPU fallback is a recorded compatibility/debug path, not the normal selected-device runtime. Verify the resolved backend and physical device in CyxWiz runtime evidence; a GPU preference alone is not sufficient.

## Executables

The native build can produce:

- `cyxwiz-engine` — desktop application;
- `cyxwiz-server-gui` — Server Node graphical application;
- `cyxwiz-server-daemon` — Server Node daemon.

Windows multi-configuration generators append `Debug` or `Release` below `build/bin`.

## Troubleshooting

When configuration fails:

1. read the first CMake error rather than the final summary;
2. confirm the vcpkg toolchain path exists;
3. disable unavailable optional integrations;
4. confirm the compiler matches the configured generator;
5. set `ArrayFire_DIR` only when ArrayFire is installed.

For reproducible problems, follow [SUPPORT.md](SUPPORT.md).
