# CyxWiz - Decentralized ML Compute Platform

CyxWiz is a revolutionary platform that combines the visual design capabilities of Unreal Engine with the computational power of MATLAB, creating a decentralized network for machine learning model training and deployment.

## 🎯 Overview

The platform consists of three main components:

1. **CyxWiz Engine** - Desktop client with rich GUI for building and designing ML models
2. **CyxWiz Server Node** - Also known as miners. Distributed compute nodes that train models
3. **[CyxWiz Central Server](cyxwiz-central-server/README.md)** - Orchestrator managing the decentralized network

## ✨ Features

- **Visual Node Editor**: Drag-and-drop interface for building ML models
- **Python Scripting**: Full Python support for advanced workflows
- **Decentralized Computing**: Leverage distributed GPU/CPU resources
- **Blockchain Integration**: Solana-based token economy (CYXWIZ coin)
- **Real-time Monitoring**: Track training progress with live visualizations
- **Cross-platform**: Windows, macOS, Linux support
- **Android Backend**: Run compute backend on Android devices

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌────────────────┐
│  CyxWiz Engine  │◄───────►│ CyxWiz Central   │◄───────►│ Server Node 1  │
│  (Desktop GUI)  │  gRPC   │    Server        │  gRPC   │  (Compute)     │
└─────────────────┘         │  (Orchestrator)  │         └────────────────┘
                            └──────────────────┘                  │
                                     │                            │
                                     │         ┌──────────────────┘
                                     ▼         ▼
                            ┌──────────────────────┐
                            │  Solana Blockchain   │
                            │  (CYXWIZ Token)      │
                            └──────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **C++ Compiler**: MSVC 2019+ (Windows), GCC 9+ (Linux), Clang 12+ (macOS)
- **CMake**: 3.20 or higher
- **vcpkg**: For dependency management
- **ArrayFire**: Download from https://arrayfire.com/download
- **Python**: 3.8+ (for scripting support)
- **Rust**: 1.70+ (for Central Server)

### Building on Windows

```bash
# 1. Clone the repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd cyxwiz

# 2. Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg
cd vcpkg
bootstrap-vcpkg.bat
cd ..

# 3. Configure CMake
cmake -B build/windows-release -S . -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCYXWIZ_BUILD_ENGINE=ON ^
  -DCYXWIZ_BUILD_SERVER_NODE=ON ^
  -DCYXWIZ_BUILD_TESTS=ON

# 4. Build (this will take 10-15 minutes on first run)
cmake --build build/windows-release --config Release

# 5. Run the applications
build\windows-release\bin\Release\cyxwiz-engine.exe
build\windows-release\bin\Release\cyxwiz-server-node.exe
```

**Note:** First build installs all dependencies via vcpkg (protobuf, gRPC, ImGui, etc.) and may take 15-20 minutes. Subsequent builds are much faster.

**Optional - ArrayFire GPU Acceleration:**
```bash
# Download ArrayFire from https://arrayfire.com/download
# Install and set ArrayFire_DIR environment variable
# Then rebuild with GPU support enabled
```

### Building on Linux/macOS

```bash
# 1. Install system dependencies
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install build-essential cmake git ninja-build

# macOS:
brew install cmake ninja

# 2. Clone the repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd cyxwiz

# 3. Install vcpkg
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
cd ..

# 4. Configure CMake
cmake -B build/linux-release -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DCYXWIZ_BUILD_ENGINE=ON \
  -DCYXWIZ_BUILD_SERVER_NODE=ON \
  -DCYXWIZ_BUILD_TESTS=ON

# 5. Build
cmake --build build/linux-release

# 6. Run
./build/linux-release/bin/cyxwiz-engine
./build/linux-release/bin/cyxwiz-server-node
```

### Troubleshooting

**"ArrayFire not found" warning:**
- This is expected. GPU acceleration is optional.
- The project builds successfully with CPU-only mode.
- To enable GPU: Install ArrayFire and rebuild.

**"vcpkg dependencies missing":**
```bash
cd vcpkg
./vcpkg install  # or .\vcpkg install on Windows
```

**"Ninja not found" (Windows):**
- Use Visual Studio generator instead (as shown in Windows instructions above)
- Or install Ninja: `choco install ninja`

## 📦 Project Structure

```
CyxWiz/
├── cyxwiz-backend/         # Shared compute library (ArrayFire-based)
├── cyxwiz-engine/          # Desktop client (ImGui + OpenGL)
├── cyxwiz-server-node/     # Compute worker node
├── cyxwiz-central-server/  # Orchestrator (Rust) - See README
├── cyxwiz-protocol/        # gRPC protocol definitions
├── scripts/                # Build and deployment scripts
├── docs/                   # Documentation
└── tests/                  # Unit and integration tests
```

## 🔧 Development

### Running Tests

```bash
cd build/windows-release
ctest --output-on-failure
```

### Building for Android (Backend Only)

```bash
cmake --preset android-release
cmake --build build/android-release
```

### Central Server (Rust)

See the [Central Server README](cyxwiz-central-server/README.md) for detailed documentation, TUI guide, and setup instructions.

```bash
cd cyxwiz-central-server
cargo build --release
cargo run
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Blockchain Integration](docs/blockchain.md)
- [Contributing Guide](docs/CONTRIBUTING.md)

## 🌐 Blockchain

CyxWiz uses Solana for its decentralized token economy:

- **Token**: CYXWIZ (SPL Token)
- **Use Cases**: Compute payments, node staking, governance
- **Smart Contracts**: Job escrow, payment streaming, rewards

See [Blockchain Architecture](docs/blockchain.md) for details.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- Website: https://cyxwiz.io
- Documentation: https://docs.cyxwiz.io
- Discord: https://discord.gg/cyxwiz
- Twitter: https://twitter.com/cyxwiz

## 🙏 Acknowledgments

- ArrayFire for GPU acceleration
- Dear ImGui for the GUI framework
- gRPC for network communication
- Solana for blockchain infrastructure
