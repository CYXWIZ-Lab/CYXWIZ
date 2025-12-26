# CyxWiz Installation Guide

## Binary Releases

Pre-built binaries are available for:
- **Windows x64** (Windows 10/11)
- **Linux x64** (Ubuntu 20.04+, Debian 11+, Fedora 36+)
- **macOS ARM64** (Apple Silicon M1/M2/M3)

Download the latest release from: https://github.com/CYXWIZ-Lab/CYXWIZ/releases

---

## Required Dependencies

### ArrayFire (Required - All Platforms)

ArrayFire is the GPU acceleration library used by CyxWiz.

#### Windows
1. Download ArrayFire 3.9.0 from https://arrayfire.com/download
2. Run the installer (default: `C:\Program Files\ArrayFire\v3`)
3. Add to PATH: `C:\Program Files\ArrayFire\v3\lib`

#### Linux
```bash
wget https://arrayfire.s3.amazonaws.com/3.9.0/ArrayFire-v3.9.0-Linux-x86_64.sh
chmod +x ArrayFire-v3.9.0-Linux-x86_64.sh
sudo ./ArrayFire-v3.9.0-Linux-x86_64.sh --prefix=/opt/arrayfire --skip-license

# Add to ~/.bashrc
export LD_LIBRARY_PATH=/opt/arrayfire/lib:$LD_LIBRARY_PATH
```

#### macOS
```bash
brew install arrayfire
```

---

## Optional Dependencies

### CUDA Toolkit (NVIDIA GPU Support)

For NVIDIA GPU acceleration, install CUDA Toolkit 12.x:
- Download: https://developer.nvidia.com/cuda-downloads
- Windows: Run installer, select "CUDA" components
- Linux: Follow distribution-specific instructions

### OpenCL (AMD/Intel GPU Support)

- **AMD**: Install AMD ROCm or AMDGPU-PRO drivers
- **Intel**: Install Intel oneAPI or OpenCL runtime

### Python (Scripting Support)

Python 3.8+ is required for the scripting engine:
- Windows: https://www.python.org/downloads/
- Linux: `sudo apt install python3-dev python3-pip`
- macOS: `brew install python@3.11`

---

## Running CyxWiz

### Windows

```powershell
# Extract the release
Expand-Archive cyxwiz-v0.4.0-windows-x64.zip -DestinationPath .

# Run the Engine (Desktop Client)
.\cyxwiz-v0.4.0-windows-x64\cyxwiz-engine.exe

# Run the Server Node (Compute Worker)
.\cyxwiz-v0.4.0-windows-x64\cyxwiz-server-node.exe
```

### Linux

```bash
# Extract the release
tar -xzf cyxwiz-v0.4.0-linux-x64.tar.gz
cd cyxwiz-v0.4.0-linux-x64

# Run the Engine
./run-engine.sh
# Or directly (ensure LD_LIBRARY_PATH is set):
./cyxwiz-engine

# Run the Server Node
./cyxwiz-server-node
```

### macOS

```bash
# Extract the release
unzip cyxwiz-v0.4.0-macos-arm64.zip
cd cyxwiz-v0.4.0-macos-arm64

# First run may be blocked by Gatekeeper
# Go to System Preferences > Security & Privacy > Allow

# Run the Engine
./cyxwiz-engine

# Run the Server Node
./cyxwiz-server-node
```

---

## Troubleshooting

### "DLL not found" / "Library not found"

Ensure ArrayFire is installed and its library path is in your system PATH:
- Windows: Add `C:\Program Files\ArrayFire\v3\lib` to PATH
- Linux: Export `LD_LIBRARY_PATH=/opt/arrayfire/lib:$LD_LIBRARY_PATH`
- macOS: ArrayFire from Homebrew should work automatically

### "CUDA driver not found"

- Install NVIDIA GPU drivers from https://www.nvidia.com/drivers
- CyxWiz will fall back to CPU if no GPU is available

### "OpenGL error"

Ensure your GPU drivers are up to date:
- Windows: Update via Windows Update or NVIDIA/AMD website
- Linux: `sudo apt install mesa-utils` and update drivers
- macOS: Update macOS to latest version

### macOS "App is damaged"

This happens when Gatekeeper blocks unsigned apps:
```bash
xattr -cr cyxwiz-engine
xattr -cr cyxwiz-server-node
```

---

## Building from Source

If you prefer to build from source, see [docs/mainbuild.md](docs/mainbuild.md).

Requirements:
- CMake 3.20+
- C++20 compiler (MSVC 2022, GCC 11+, Clang 14+)
- vcpkg (for dependencies)
- ArrayFire 3.9.0

```bash
# Clone with submodules
git clone --recursive https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CYXWIZ

# Install vcpkg dependencies
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Support

- Issues: https://github.com/CYXWIZ-Lab/CYXWIZ/issues
- Documentation: https://github.com/CYXWIZ-Lab/CYXWIZ/tree/master/docs
