# CyxWiz Project Structure

## Complete File Tree

```
CyxWiz_Claude/
├── CMakeLists.txt                      # Root build configuration
├── CMakePresets.json                   # Platform-specific build presets
├── vcpkg.json                          # C++ dependency manifest
├── .gitignore                          # Git ignore rules
├── README.md                           # Project overview
├── CLAUDE.md                           # ⭐ AI Assistant Guide (comprehensive)
├── PROJECT_STRUCTURE.md                # This file
│
├── cyxwiz-protocol/                    # gRPC Protocol Definitions
│   ├── CMakeLists.txt
│   ├── proto/
│   │   ├── common.proto                # Common types
│   │   ├── job.proto                   # Job management
│   │   ├── node.proto                  # Node communication
│   │   └── compute.proto               # Direct compute operations
│   └── common/
│       ├── version.h/cpp               # Version utilities
│       └── utils.h/cpp                 # Common utilities
│
├── cyxwiz-backend/                     # Shared Compute Library (DLL/SO)
│   ├── CMakeLists.txt
│   ├── include/cyxwiz/                 # Public API
│   │   ├── cyxwiz.h                    # Main header
│   │   ├── engine.h
│   │   ├── tensor.h                    # Tensor operations
│   │   ├── device.h                    # Device management
│   │   ├── optimizer.h                 # Optimizers (SGD, Adam, AdamW)
│   │   ├── loss.h                      # Loss functions
│   │   ├── activation.h                # Activation functions
│   │   ├── layer.h                     # Neural network layers
│   │   ├── model.h                     # Model interface
│   │   └── memory_manager.h            # Memory tracking
│   ├── src/
│   │   ├── core/                       # Core implementations
│   │   │   ├── engine.cpp
│   │   │   ├── tensor.cpp
│   │   │   ├── device.cpp
│   │   │   └── memory_manager.cpp
│   │   └── algorithms/                 # ML algorithms
│   │       ├── optimizer.cpp
│   │       ├── loss.cpp
│   │       ├── activation.cpp
│   │       ├── layer.cpp
│   │       └── model.cpp
│   └── python/                         # Python bindings
│       └── bindings.cpp                # pybind11 module
│
├── cyxwiz-engine/                      # Desktop Client (ImGui)
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.cpp                    # Entry point
│   │   ├── application.h/cpp           # Main application
│   │   ├── gui/                        # GUI components
│   │   │   ├── main_window.h/cpp       # Main dockable window
│   │   │   ├── node_editor.h/cpp       # Visual node editor
│   │   │   ├── console.h/cpp           # Command console
│   │   │   ├── viewport.h/cpp          # Training visualization
│   │   │   └── properties.h/cpp        # Property panel
│   │   ├── scripting/                  # Python scripting
│   │   │   ├── python_engine.h/cpp     # Embedded Python
│   │   │   └── script_manager.h/cpp    # Script management
│   │   └── network/                    # gRPC networking
│   │       ├── grpc_client.h/cpp       # gRPC client
│   │       └── job_manager.h/cpp       # Job management
│   └── resources/                      # Assets
│       ├── fonts/
│       └── shaders/
│
├── cyxwiz-server-node/                 # Compute Worker Node
│   ├── CMakeLists.txt
│   └── src/
│       ├── main.cpp                    # Entry point
│       ├── node_server.cpp             # gRPC server (TODO)
│       ├── job_executor.cpp            # Job execution (TODO)
│       └── metrics_collector.cpp       # Monitoring (TODO)
│
├── cyxwiz-central-server/              # Orchestrator (Rust)
│   ├── Cargo.toml                      # Rust dependencies
│   ├── build.rs                        # Build script
│   └── src/
│       └── main.rs                     # Entry point
│
├── tests/                              # Unit & Integration Tests
│   ├── CMakeLists.txt
│   └── unit/
│       ├── test_tensor.cpp
│       ├── test_device.cpp
│       └── test_optimizer.cpp
│
├── scripts/                            # Build Scripts
│   ├── build.bat                       # Windows build
│   └── build.sh                        # Linux/macOS build
│
└── docs/                               # Documentation (TODO)
    ├── architecture.md
    ├── blockchain.md
    └── CONTRIBUTING.md
```

## Component Summary

### ✅ Completed

1. **Project Infrastructure**
   - CMake build system with cross-platform presets
   - vcpkg dependency management
   - Build scripts for all platforms
   - Comprehensive documentation

2. **cyxwiz-protocol**
   - Complete gRPC protocol definitions
   - Common types, Job service, Node service, Compute service
   - Automatic code generation from .proto files

3. **cyxwiz-backend**
   - Core library structure
   - Tensor, Device, Optimizer APIs
   - Python bindings skeleton
   - ArrayFire integration framework
   - Memory tracking for debug builds

4. **cyxwiz-engine**
   - Complete GUI framework with ImGui
   - Docking, viewports, menu system
   - Node editor, Console, Viewport, Properties panels
   - Python scripting integration
   - gRPC client for server communication

5. **cyxwiz-server-node**
   - Basic structure
   - Main entry point
   - Placeholders for job execution

6. **cyxwiz-central-server**
   - Rust project setup
   - Cargo dependencies (Tokio, Tonic, Solana SDK)
   - Main entry point

7. **Tests**
   - Test framework setup (Catch2)
   - Sample unit tests

### 🚧 TODO (Marked in Code)

High-priority implementation tasks:

1. **Algorithm Implementations**
   - Complete optimizer implementations (SGD, Adam, AdamW)
   - Loss functions (MSE, Cross-Entropy, etc.)
   - Activation functions (ReLU, Sigmoid, Tanh, etc.)
   - Neural network layers (Dense, Conv2D, LSTM, etc.)

2. **GUI Enhancements**
   - Integrate ImNodes for visual node editing
   - Integrate ImPlot for real-time training plots
   - File dialogs (New, Open, Save)
   - Server connection dialog

3. **Server Node**
   - gRPC server implementation
   - Job executor using cyxwiz-backend
   - Docker containerization
   - btop TUI integration
   - Metrics collection

4. **Central Server**
   - gRPC service implementations
   - Node registry and discovery
   - Job scheduler
   - PostgreSQL/Redis integration
   - Solana payment processor

5. **Blockchain**
   - Smart contract development (JobEscrow, PaymentStreaming)
   - Token deployment
   - Payment flow integration

## Next Steps

### Immediate (Development Setup)

1. **Install Prerequisites**:
   ```bash
   # Install vcpkg
   git clone https://github.com/microsoft/vcpkg
   cd vcpkg && ./bootstrap-vcpkg.sh && cd ..

   # Install ArrayFire
   # Download from: https://arrayfire.com/download
   ```

2. **Build the Project**:
   ```bash
   # Windows
   scripts\build.bat

   # Linux/macOS
   chmod +x scripts/build.sh
   ./scripts/build.sh
   ```

3. **Run Tests**:
   ```bash
   cd build/<preset>
   ctest --output-on-failure
   ```

### Short-term (MVP Features)

1. Implement core ML algorithms in `cyxwiz-backend`
2. Complete gRPC client/server communication
3. Build basic job submission and execution flow
4. Add ImNodes for visual model building

### Medium-term (Network Features)

1. Implement Central Server orchestration
2. Add node discovery and registration
3. Implement job scheduling algorithm
4. Add metrics and monitoring

### Long-term (Blockchain & Marketplace)

1. Deploy Solana smart contracts
2. Integrate payment processor
3. Build model marketplace
4. Add governance and staking

## Key Commands Reference

```bash
# Build (Quick)
scripts/build.bat         # Windows
./scripts/build.sh        # Linux/macOS

# Build (Manual)
cmake --preset windows-release
cmake --build build/windows-release --config Release

# Run Components
./build/windows-release/bin/cyxwiz-engine        # Desktop client
./build/windows-release/bin/cyxwiz-server-node   # Compute node
cd cyxwiz-central-server && cargo run --release  # Orchestrator

# Test
cd build/windows-release && ctest

# Clean
rm -rf build
```

## Documentation

- **CLAUDE.md** - Comprehensive guide for AI assistants and developers
- **README.md** - Project overview and quick start
- **This file** - Complete project structure reference

## Questions?

Refer to CLAUDE.md for:
- Detailed architecture explanations
- Development workflows
- Adding new features
- Troubleshooting
- API references
