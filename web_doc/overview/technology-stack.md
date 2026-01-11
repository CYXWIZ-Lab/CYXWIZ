# Technology Stack

This document details all technologies, libraries, and frameworks used in the CyxWiz platform.

## Languages

### C++20

Primary language for Engine, Server Node, and Backend library.

**Standard Features Used:**
- `std::filesystem` - Cross-platform file operations
- `std::span` - Non-owning array views
- `std::format` - String formatting (where available)
- `constexpr` - Compile-time computation
- Concepts - Template constraints
- Ranges - Modern iteration patterns
- `std::jthread` - Automatic-joining threads
- Three-way comparison (`<=>`)

**Compiler Requirements:**
- MSVC 19.30+ (Visual Studio 2022+)
- GCC 10+
- Clang 12+

### Rust

Language for the Central Server (high-performance, memory-safe).

**Edition:** Rust 2021

**Key Features Used:**
- Async/await with Tokio runtime
- Trait-based polymorphism
- Error handling with `Result` and `?` operator
- Macro system for code generation
- Module system for organization

### Python

Embedded scripting language in the Engine.

**Version:** 3.8+

**Integration:** pybind11 for C++ bindings

## GUI Framework

### Dear ImGui

Immediate-mode GUI library for Engine and Server Node GUIs.

**Version:** 1.90+ (docking branch)

**Features:**
- Docking windows with split/merge
- Multiple viewports (pop-out windows)
- Customizable themes and styling
- Font system (TTF/OTF support)

**Configuration Flags:**
```cpp
ImGuiConfigFlags_DockingEnable      // Enable docking
ImGuiConfigFlags_ViewportsEnable    // Enable multi-viewports
```

### ImNodes

Node editor extension for Dear ImGui.

**Purpose:** Visual ML pipeline builder

**Features:**
- Draggable nodes
- Connection pins with types
- Link creation/deletion
- Pan and zoom
- Minimap

### ImPlot

Plotting extension for Dear ImGui.

**Purpose:** Real-time training visualization

**Plot Types:**
- Line plots (loss curves)
- Scatter plots (embeddings)
- Bar charts (metrics)
- Histograms (data distribution)
- Heatmaps (confusion matrices)

### GLFW

Cross-platform window and input library.

**Version:** 3.3+

**Features:**
- Window creation/management
- OpenGL context creation
- Keyboard/mouse input
- Multi-monitor support

### OpenGL

Graphics rendering backend.

**Version:** OpenGL 3.3+ Core Profile

**Used For:**
- ImGui rendering
- Future: GPU visualization

## GPU Computing

### ArrayFire

Unified interface for GPU/CPU computation.

**Version:** 3.8+

**Backends:**
| Backend | Hardware | API |
|---------|----------|-----|
| CUDA | NVIDIA GPUs | CUDA Toolkit |
| OpenCL | AMD/Intel/NVIDIA | OpenCL 1.2+ |
| CPU | Any CPU | Native threads |

**Key Features:**
- N-dimensional arrays
- BLAS/LAPACK operations
- Signal processing (FFT, convolution)
- Statistics and random numbers
- Automatic memory management

**Example Usage:**
```cpp
#include <arrayfire.h>

// Create tensors
af::array a = af::randu(1000, 1000);
af::array b = af::randu(1000, 1000);

// Matrix multiplication (auto GPU)
af::array c = af::matmul(a, b);

// Sync and get result
c.eval();
```

## Networking

### gRPC

High-performance RPC framework.

**Version:** 1.50+

**Features:**
- Protocol Buffers serialization
- HTTP/2 transport
- Bidirectional streaming
- Load balancing support
- TLS encryption

**Language Support:**
- C++ (Engine, Server Node, Backend)
- Rust via Tonic (Central Server)

### Protocol Buffers

Interface definition and serialization.

**Version:** 3.x (proto3 syntax)

**Files:**
| File | Purpose |
|------|---------|
| `common.proto` | Shared types |
| `job.proto` | Job management |
| `node.proto` | Node services |
| `compute.proto` | Compute operations |
| `wallet.proto` | Wallet operations |
| `deployment.proto` | Model deployment |

### Tonic (Rust)

gRPC framework for Rust.

**Features:**
- Async gRPC with Tokio
- Code generation from .proto
- Interceptors (middleware)
- TLS support

## Database

### PostgreSQL

Production database for Central Server.

**Version:** 13+

**Features Used:**
- JSONB for flexible schemas
- Full-text search
- Connection pooling via PgBouncer
- Index optimization

### SQLite

Development/testing database.

**Features:**
- Zero configuration
- Single file storage
- In-memory option
- WAL mode for concurrency

### SQLx (Rust)

Async database library.

**Features:**
- Compile-time query validation
- Connection pooling
- Migration support
- Multi-database support (Postgres, SQLite, MySQL)

### Redis

In-memory cache and message broker.

**Version:** 6+

**Use Cases:**
- Session caching
- Job queue metadata
- Real-time metrics buffering
- Rate limiting counters

## Blockchain

### Solana

High-performance blockchain for payments.

**Network:** Devnet (development), Mainnet-beta (production)

**SDK:**
- `solana-sdk` (Rust)
- `solana-client` (RPC communication)

**Features:**
- SPL Token program
- Transaction signing
- Account management

### SPL Token

Token standard for CYXWIZ coin.

**Operations:**
- Token transfers
- Account creation
- Balance queries

## Build System

### CMake

C++ build system orchestrator.

**Version:** 3.20+

**Features:**
- Presets for different configurations
- Cross-platform support
- vcpkg integration
- Protobuf code generation

**Key Presets:**
```json
{
  "windows-debug",
  "windows-release",
  "linux-debug",
  "linux-release",
  "macos-debug",
  "macos-release",
  "android-release"
}
```

### Cargo

Rust build system and package manager.

**Features:**
- Dependency resolution
- Build profiles
- Workspace support
- Testing framework

### vcpkg

C++ package manager.

**Mode:** Manifest mode (vcpkg.json)

**Managed Dependencies:**
```json
{
  "dependencies": [
    "imgui",
    "implot",
    "glfw3",
    "glad",
    "grpc",
    "protobuf",
    "spdlog",
    "nlohmann-json",
    "fmt",
    "sqlite3",
    "openssl",
    "pybind11",
    "catch2"
  ]
}
```

## Logging & Debugging

### spdlog

Fast C++ logging library.

**Features:**
- Multiple sinks (console, file, rotating)
- Log levels (trace, debug, info, warn, error)
- Format strings with fmt
- Thread-safe

**Example:**
```cpp
spdlog::info("Server started on port {}", port);
spdlog::error("Failed to connect: {}", error_msg);
```

### tracing (Rust)

Structured logging for Rust.

**Features:**
- Spans for context
- Async-aware
- Multiple subscribers
- Log levels

## Testing

### Catch2

C++ testing framework.

**Version:** 3.x

**Features:**
- BDD-style tests
- Section-based organization
- Matchers
- Benchmarking

**Example:**
```cpp
TEST_CASE("Tensor operations", "[tensor]") {
    SECTION("Addition") {
        Tensor a({1.0f, 2.0f, 3.0f});
        Tensor b({4.0f, 5.0f, 6.0f});
        Tensor c = a + b;
        REQUIRE(c[0] == 5.0f);
    }
}
```

### Rust Testing

Built-in test framework.

**Features:**
- Unit tests in modules
- Integration tests in `tests/`
- Doc tests
- Async test support

## Serialization

### nlohmann/json

Modern JSON library for C++.

**Features:**
- STL-like interface
- Automatic type deduction
- Pretty printing
- JSON Pointer support

### serde (Rust)

Serialization framework for Rust.

**Formats:**
- JSON (serde_json)
- TOML (toml)
- YAML (serde_yaml)
- MessagePack

## Python Integration

### pybind11

C++/Python binding library.

**Version:** 2.10+

**Features:**
- Automatic type conversion
- STL container support
- NumPy integration
- Exception translation

**Example:**
```cpp
PYBIND11_MODULE(pycyxwiz, m) {
    m.def("initialize", &cyxwiz::Initialize);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def("shape", &Tensor::Shape)
        .def("__add__", &Tensor::operator+);
}
```

## Security

### OpenSSL

Cryptography library.

**Features:**
- TLS/SSL
- Symmetric encryption (AES)
- Hashing (SHA-256, SHA-512)
- Random number generation

### JWT (JSON Web Tokens)

Authentication tokens.

**Libraries:**
- C++: jwt-cpp
- Rust: jsonwebtoken

## Dependencies Summary

### vcpkg Packages (C++)

| Package | Version | Purpose |
|---------|---------|---------|
| imgui | 1.90+ | GUI framework |
| implot | 0.16+ | Plotting |
| glfw3 | 3.3+ | Window/input |
| glad | 0.1+ | OpenGL loader |
| grpc | 1.50+ | RPC framework |
| protobuf | 3.21+ | Serialization |
| spdlog | 1.11+ | Logging |
| nlohmann-json | 3.11+ | JSON |
| fmt | 9.0+ | Formatting |
| sqlite3 | 3.40+ | Database |
| openssl | 3.0+ | Crypto |
| pybind11 | 2.10+ | Python bindings |
| catch2 | 3.0+ | Testing |

### Cargo Crates (Rust)

| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1.0 | Async runtime |
| tonic | 0.9 | gRPC framework |
| prost | 0.11 | Protobuf |
| sqlx | 0.7 | Database |
| redis | 0.23 | Cache |
| solana-sdk | 1.17 | Blockchain |
| jsonwebtoken | 9.0 | JWT |
| serde | 1.0 | Serialization |
| tracing | 0.1 | Logging |
| ratatui | 0.24 | TUI |

---

**Next**: [Data Flow](data-flow.md) | [Security Model](security.md)
