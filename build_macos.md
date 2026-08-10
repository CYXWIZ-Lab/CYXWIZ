# Building CyxWiz on macOS

This guide documents the current, supported path for a focused CyxWiz Engine
build on macOS. It intentionally avoids historical workarounds, fixed build
times, and architecture-specific binary sizes.

## Scope

- The Engine is the desktop application built by the `cyxwiz-engine` target.
- This guide validates an Engine-only Release build. Build the Server Node only
  when it is needed; it has a broader dependency surface.
- Python scripting is optional. The Engine still runs when Python support is
  unavailable.

## Prerequisites

Install the Apple command-line developer tools, Git, CMake 3.20 or later, and
Ninja. Homebrew is one convenient way to install the command-line tools:

```bash
xcode-select --install
brew install cmake ninja autoconf autoconf-archive automake libtool
```

The Engine also requires an ArrayFire installation. Make its CMake package
discoverable before configuring. If CMake cannot find it, provide
`ArrayFire_DIR` with the directory containing `ArrayFireConfig.cmake`.

Download the macOS installer from [ArrayFire](https://arrayfire.com/download),
then run the downloaded package (or double-click it in Finder):

```bash
sudo installer -pkg ArrayFire_*_OSX.pkg -target /
```

The default installer location is `/opt/arrayfire`; its CMake package directory
is normally `/opt/arrayfire/share/ArrayFire/cmake`.

Optional Python scripting requires Python **3.12 or 3.13** with development
files, plus the `pybind11` package supplied through the project's vcpkg
manifest. Python 3.14 and newer are not supported for CyxWiz scripting in this
branch. They do not prevent a non-scripting Engine build, but they will not
enable the embedded Python bindings.

## Clone and prepare dependencies

Clone the repository and run the supplied setup script from its root:

```bash
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ
./setup.sh
```

`setup.sh` checks the toolchain, clones and bootstraps `vcpkg` when it is
missing, and leaves dependency installation to CMake's manifest mode. Do not
manually clone `imnodes` or check out `ImGuiColorTextEdit`: both are vendored
source directories in this revision, not Git submodules.

## Configure a focused Engine build

Run this command from the repository root. It keeps the build in
`build/macos-release`, builds the desktop Engine, and leaves optional heavy
integrations out of the common macOS path.

```bash
cmake -S . -B build/macos-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=OFF \
  -DCYXWIZ_BUILD_TESTS=OFF \
  -DCYXWIZ_ENABLE_CUDA=OFF \
  -DCYXWIZ_ENABLE_ONNX=OFF \
  -DCYXWIZ_ENABLE_GGUF=OFF \
  -DCYXWIZ_ENABLE_PYTORCH=OFF
```

The first configure may take a while because vcpkg restores manifest
dependencies into this build tree. Keep the build directory for incremental
builds.

### Enable Python scripting

First install a supported Python version. Then configure with its CMake root,
for example:

```bash
brew install python@3.12
cmake -S . -B build/macos-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DPython3_ROOT_DIR="$(brew --prefix python@3.12)" \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=OFF \
  -DCYXWIZ_BUILD_TESTS=OFF \
  -DCYXWIZ_ENABLE_CUDA=OFF \
  -DCYXWIZ_ENABLE_ONNX=OFF \
  -DCYXWIZ_ENABLE_GGUF=OFF \
  -DCYXWIZ_ENABLE_PYTORCH=OFF
```

At the end of configuration, confirm that CMake reports a Python 3.12 or 3.13
interpreter and that Python scripting is enabled. If the interpreter's
development files or `pybind11` are unavailable, CMake explicitly disables
scripting rather than failing the Engine build.

## Build and run

Build only the Engine target:

```bash
cmake --build build/macos-release --target cyxwiz-engine --parallel 4
```

Adjust `--parallel` to suit available memory and CPU. Launch the result:

```bash
open build/macos-release/bin/cyxwiz-engine
```

The Engine log is written beside the executable:

```bash
tail -n 100 build/macos-release/bin/engine_log.txt
```

Successful startup logs the selected start page or project window. A message
that Python scripting is disabled is informational when the build was made
without a compatible Python installation.

## Troubleshooting

### CMake cannot find ArrayFire

Install ArrayFire, then configure again with its package directory:

```bash
cmake -S . -B build/macos-release \
  -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake
```

Use the directory that contains `ArrayFireConfig.cmake`; do not point this
option at the ArrayFire executable or a Python package directory.

### Python scripting is disabled

Check the configure output and the cache:

```bash
cmake -LAH -N build/macos-release | grep -E 'Python3|CYXWIZ_HAS_PYTHON'
```

Use Python 3.12 or 3.13, set `Python3_ROOT_DIR` to that installation when
needed, and rerun the full configure command. Python 3.14 is intentionally
rejected by the current build policy.

### vcpkg is missing or not bootstrapped

Run `./setup.sh` again from the repository root. It is the supported way to
create the local `vcpkg` checkout expected by the CMake toolchain file.

### Reconfigure after changing dependencies or CMake options

Rerun the configure command. If the build cache itself is invalid, remove only
the selected build directory and configure again:

```bash
rm -rf build/macos-release
```

This removes generated build files and artifacts in that directory; it does
not remove source files or the vcpkg checkout.

## Server Node

The Server Node CMake target includes the current macOS-specific framework
links, including CFNetwork. Its full build is not part of this focused Engine
guide. Enable `CYXWIZ_BUILD_SERVER_NODE` only when the worker application is
required, then build its named targets after the Engine configuration succeeds.

## What to report with a build failure

Include the CMake configure error, the first compiler or linker error, the
macOS version and architecture, the CMake version, and whether Python
scripting was requested. Do not include tokens, credentials, or private
project files.
