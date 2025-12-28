# Developer Guide

This guide provides comprehensive information for developers who want to build, modify, or contribute to the CyxWiz platform.

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Quick Start](quick-start.md) | Get up and running fast |
| [Building](building.md) | Complete build instructions |
| [Installation](installation.md) | Installing prerequisites |
| [Contributing](contributing.md) | Contribution guidelines |
| [Code Style](code-style.md) | Coding standards |
| [Testing](testing.md) | Testing strategies |

## Overview

CyxWiz consists of four main components:

| Component | Language | Build System |
|-----------|----------|--------------|
| cyxwiz-engine | C++20 | CMake |
| cyxwiz-server-node | C++20 | CMake |
| cyxwiz-central-server | Rust | Cargo |
| cyxwiz-backend | C++20 | CMake |

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| **CMake** | 3.20+ | C++ build system |
| **Visual Studio** | 2022+ | C++ compiler (Windows) |
| **GCC/Clang** | 10+/12+ | C++ compiler (Linux/macOS) |
| **Rust** | 1.70+ | Central Server |
| **Python** | 3.8+ | Scripting, bindings |
| **Git** | 2.0+ | Version control |

### Optional Tools

| Tool | Purpose |
|------|---------|
| **CUDA Toolkit** | NVIDIA GPU support |
| **ArrayFire** | GPU acceleration |
| **Docker** | Sandboxed execution |
| **Redis** | Central Server caching |
| **PostgreSQL** | Production database |

## Quick Start

### Clone and Build

```bash
# Clone repository
git clone https://github.com/CYXWIZ-Lab/CYXWIZ.git
cd CyxWiz

# Run setup (first time)
./setup.sh      # Linux/macOS
setup.bat       # Windows (from Developer Command Prompt)

# Build all components
./build.sh      # Linux/macOS
build.bat       # Windows
```

### Run Components

```bash
# Engine (Desktop GUI)
./build/linux-release/bin/cyxwiz-engine

# Server Node
./build/linux-release/bin/cyxwiz-server-node

# Central Server
cd cyxwiz-central-server && cargo run --release
```

## Project Structure

```
CyxWiz/
├── cyxwiz-backend/           # Shared compute library
│   ├── include/cyxwiz/       # Public headers
│   ├── src/                  # Implementation
│   └── python/               # Python bindings
├── cyxwiz-engine/            # Desktop client
│   ├── src/
│   │   ├── gui/              # ImGui panels
│   │   ├── core/             # Core functionality
│   │   └── scripting/        # Python integration
│   └── resources/            # Assets, fonts, icons
├── cyxwiz-server-node/       # Compute worker
│   ├── src/
│   │   ├── gui/              # GUI mode panels
│   │   ├── core/             # Core services
│   │   └── http/             # OpenAI API
│   └── tui/                  # Terminal UI
├── cyxwiz-central-server/    # Orchestrator (Rust)
│   └── src/
│       ├── api/              # gRPC/REST
│       ├── scheduler/        # Job scheduling
│       └── blockchain/       # Solana integration
├── cyxwiz-protocol/          # gRPC definitions
│   └── proto/                # .proto files
├── scripts/                  # Build scripts
├── docs/                     # Documentation
├── tests/                    # Test suites
└── vcpkg.json                # C++ dependencies
```

## Build Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CYXWIZ_BUILD_ENGINE` | ON | Build desktop client |
| `CYXWIZ_BUILD_SERVER_NODE` | ON | Build compute node |
| `CYXWIZ_BUILD_CENTRAL_SERVER` | ON | Build orchestrator |
| `CYXWIZ_BUILD_TESTS` | ON | Build test suites |
| `CYXWIZ_ENABLE_CUDA` | OFF | Enable CUDA backend |
| `CYXWIZ_ENABLE_OPENCL` | ON | Enable OpenCL backend |

### Build Configurations

| Preset | Flags | Use Case |
|--------|-------|----------|
| `windows-debug` | Debug symbols, logging | Development |
| `windows-release` | Optimized, no debug | Production |
| `linux-debug` | Debug symbols, logging | Development |
| `linux-release` | Optimized, no debug | Production |
| `android-release` | Cross-compile | Mobile backend |

## Development Workflow

### Feature Development

1. Create feature branch from `GUI` (development branch)
   ```bash
   git checkout GUI
   git pull origin GUI
   git checkout -b feature/my-feature
   ```

2. Make changes and commit
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

3. Run tests
   ```bash
   cd build/windows-release && ctest
   ```

4. Create pull request to `GUI` branch

### Code Review Checklist

- [ ] Code follows style guide
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No compiler warnings
- [ ] Cross-platform compatible
- [ ] No hardcoded paths

## Adding New Features

### Adding a New GUI Panel

1. **Create Header** (`panels/my_panel.h`)
   ```cpp
   #pragma once
   #include "../panel.h"

   namespace cyxwiz {

   class MyPanel : public Panel {
   public:
       MyPanel();
       ~MyPanel() override = default;
       void Render() override;

   private:
       // Panel state
   };

   } // namespace cyxwiz
   ```

2. **Create Implementation** (`panels/my_panel.cpp`)
   ```cpp
   #include "my_panel.h"
   #include <imgui.h>

   namespace cyxwiz {

   MyPanel::MyPanel() {
       name_ = "My Panel";
   }

   void MyPanel::Render() {
       if (!visible_) return;

       if (ImGui::Begin(name_.c_str(), &visible_)) {
           // Panel content
           ImGui::Text("Hello from MyPanel!");
       }
       ImGui::End();
   }

   } // namespace cyxwiz
   ```

3. **Add to CMakeLists.txt**
   ```cmake
   set(ENGINE_SOURCES
       ...
       src/gui/panels/my_panel.cpp
   )
   ```

4. **Integrate in MainWindow**
   ```cpp
   // In main_window.h
   std::unique_ptr<MyPanel> my_panel_;

   // In main_window.cpp constructor
   my_panel_ = std::make_unique<MyPanel>();

   // In Render()
   if (my_panel_) my_panel_->Render();
   ```

### Adding a New gRPC Service

1. **Define in .proto file**
   ```protobuf
   // cyxwiz-protocol/proto/myservice.proto
   syntax = "proto3";
   package cyxwiz.protocol;

   service MyService {
       rpc MyMethod(MyRequest) returns (MyResponse);
   }

   message MyRequest {
       string data = 1;
   }

   message MyResponse {
       string result = 1;
   }
   ```

2. **Rebuild** - CMake auto-generates code

3. **Implement in Rust** (Central Server)
   ```rust
   // src/api/grpc/my_service.rs
   use tonic::{Request, Response, Status};

   pub struct MyServiceImpl {
       // dependencies
   }

   #[tonic::async_trait]
   impl MyService for MyServiceImpl {
       async fn my_method(
           &self,
           request: Request<MyRequest>,
       ) -> Result<Response<MyResponse>, Status> {
           let req = request.into_inner();
           // Implementation
           Ok(Response::new(MyResponse {
               result: "processed".to_string(),
           }))
       }
   }
   ```

4. **Register in main.rs**
   ```rust
   let my_service = MyServiceImpl::new();
   let server = Server::builder()
       .add_service(MyServiceServer::new(my_service))
       .serve(addr);
   ```

### Adding a Backend Algorithm

1. **Define Header** (`include/cyxwiz/myalgo.h`)
   ```cpp
   #pragma once
   #include "api_export.h"
   #include "tensor.h"

   namespace cyxwiz {

   CYXWIZ_API Tensor MyAlgorithm(const Tensor& input, float param);

   } // namespace cyxwiz
   ```

2. **Implement** (`src/algorithms/myalgo.cpp`)
   ```cpp
   #include "cyxwiz/myalgo.h"

   namespace cyxwiz {

   Tensor MyAlgorithm(const Tensor& input, float param) {
       // Implementation using ArrayFire
       return result;
   }

   } // namespace cyxwiz
   ```

3. **Add Python Bindings** (`python/bindings.cpp`)
   ```cpp
   m.def("my_algorithm", &cyxwiz::MyAlgorithm,
         "My algorithm description",
         py::arg("input"), py::arg("param") = 1.0f);
   ```

## Debugging

### Debug Build

```bash
cmake --preset windows-debug
cmake --build build/windows-debug
```

### Debug Macros

```cpp
#ifdef CYXWIZ_DEBUG
    spdlog::debug("Debug message: {}", value);
#endif
```

### Visual Studio Debugging

1. Open `build/windows-debug/CyxWiz.sln`
2. Set startup project
3. F5 to debug

### GDB/LLDB

```bash
gdb ./build/linux-debug/bin/cyxwiz-engine
(gdb) break main
(gdb) run
```

## Testing

### C++ Tests (Catch2)

```bash
cd build/windows-release
ctest --output-on-failure

# Run specific test
./bin/cyxwiz-tests "[tensor]"
```

### Rust Tests

```bash
cd cyxwiz-central-server
cargo test
cargo test -- --nocapture  # Show output
```

### Writing Tests

```cpp
// tests/unit/test_tensor.cpp
#include <catch2/catch_test_macros.hpp>
#include <cyxwiz/tensor.h>

TEST_CASE("Tensor creation", "[tensor]") {
    cyxwiz::Tensor t({1.0f, 2.0f, 3.0f}, {3});

    REQUIRE(t.NumElements() == 3);
    REQUIRE(t.Shape() == std::vector<int>{3});
}
```

## Performance Profiling

### Built-in Profiler

Enable via `CYXWIZ_ENABLE_PROFILING`:

```cpp
CYXWIZ_PROFILE_FUNCTION();
// Code to profile
```

### External Tools

| Platform | Tool |
|----------|------|
| Windows | Visual Studio Profiler, Intel VTune |
| Linux | perf, Valgrind, gprof |
| macOS | Instruments |

## Cross-Platform Notes

### File Paths

```cpp
// Good - use std::filesystem
std::filesystem::path p = "data/models/model.h5";

// Bad - platform-specific
std::string p = "data\\models\\model.h5";
```

### Newlines

Use `\n` in code; Git handles conversion via `.gitattributes`.

### DLL Export/Import

```cpp
#ifdef _WIN32
    #ifdef CYXWIZ_BACKEND_EXPORTS
        #define CYXWIZ_API __declspec(dllexport)
    #else
        #define CYXWIZ_API __declspec(dllimport)
    #endif
#else
    #define CYXWIZ_API __attribute__((visibility("default")))
#endif
```

## Common Issues

### "ArrayFire not found"

- Install ArrayFire from https://arrayfire.com/download
- Set `ArrayFire_DIR` environment variable
- Or build with CPU-only: `-DCYXWIZ_ENABLE_CUDA=OFF`

### "vcpkg dependencies missing"

```bash
cd vcpkg
./vcpkg install
```

### "gRPC generation failed"

- Ensure protobuf is installed via vcpkg
- Check `.proto` syntax
- Rebuild from clean

### "Python not found"

- Install Python 3.8+
- Set `PYTHON_EXECUTABLE` in CMake

## Resources

- [Architecture](../overview/architecture.md)
- [Backend API](../backend/index.md)
- [Protocol Reference](../protocol/index.md)
- [GitHub Repository](https://github.com/CYXWIZ-Lab/CYXWIZ)

---

**Next**: [Quick Start](quick-start.md) | [Building](building.md)
