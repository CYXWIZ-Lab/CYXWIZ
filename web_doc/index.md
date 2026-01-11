# CyxWiz Documentation

Welcome to the comprehensive documentation for **CyxWiz** - a revolutionary decentralized ML compute platform that combines visual design capabilities with distributed GPU computing power.

## What is CyxWiz?

CyxWiz is an ecosystem of interconnected applications designed to democratize machine learning by providing:

- **Visual Model Building** - Drag-and-drop interface for creating ML architectures
- **Distributed Computing** - Leverage global GPU resources for training
- **Blockchain Economics** - Solana-based token system for compute payments
- **Cross-Platform Support** - Windows, macOS, Linux, and Android

## Platform Components

### [CyxWiz Engine](engine/index.md)
The desktop client with a rich graphical interface for designing, training, and deploying ML models. Features include:
- Visual node editor for building neural networks
- Python scripting integration
- Real-time training visualization
- 70+ analysis and data science tools

### [CyxWiz Server Node](node/index.md)
The distributed compute worker (also called "miners") that executes ML training jobs:
- GPU/CPU resource management
- OpenAI-compatible API server
- Secure job execution with Docker sandboxing
- Pool mining for collaborative training

### [CyxWiz Central Server](server/index.md)
The Rust-based network orchestrator managing the decentralized infrastructure:
- gRPC services for job and node management
- PostgreSQL/SQLite database integration
- Redis caching layer
- Solana blockchain integration for payments

## Quick Navigation

### Getting Started
- [Project Overview](overview/project-overview.md)
- [Architecture](overview/architecture.md)
- [Quick Start Guide](developer/quick-start.md)
- [Installation](developer/installation.md)

### For Users
- [Engine User Guide](engine/index.md)
- [Node Editor Tutorial](engine/node-editor/index.md)
- [Training Your First Model](engine/training.md)
- [Python Scripting](engine/scripting.md)

### Tool Categories
- [Data Science Tools](tools/data-science/index.md) - Profiling, correlation, outliers
- [Statistics Tools](tools/statistics/index.md) - Hypothesis testing, regression
- [Clustering Tools](tools/clustering/index.md) - K-Means, DBSCAN, GMM
- [Model Evaluation](tools/model-evaluation/index.md) - ROC, PR curves, confusion matrix
- [Data Transformations](tools/transformations/index.md) - Normalization, scaling
- [Linear Algebra](tools/linear-algebra/index.md) - SVD, eigendecomposition
- [Signal Processing](tools/signal-processing/index.md) - FFT, wavelets
- [Optimization](tools/optimization/index.md) - Gradient descent, LP/QP
- [Time Series](tools/time-series/index.md) - Forecasting, seasonality
- [Text Processing](tools/text-processing/index.md) - Tokenization, TF-IDF
- [Utilities](tools/utilities/index.md) - Calculator, converters

### For Developers
- [Build Instructions](developer/building.md)
- [Contributing Guide](developer/contributing.md)
- [Backend API Reference](backend/index.md)
- [gRPC Protocol Reference](protocol/index.md)
- [Code Style Guide](developer/code-style.md)

### Blockchain & Economics
- [Token Economics](blockchain/token-economics.md)
- [Payment Flow](blockchain/payment-flow.md)
- [Smart Contracts](blockchain/smart-contracts.md)
- [Solana Integration](blockchain/solana-integration.md)

## Technology Stack

| Component | Technologies |
|-----------|-------------|
| **Languages** | C++20, Rust, Python |
| **GUI Framework** | Dear ImGui, ImNodes, ImPlot |
| **GPU Computing** | ArrayFire (CUDA/OpenCL/CPU) |
| **Networking** | gRPC, Protocol Buffers |
| **Blockchain** | Solana, SPL Tokens |
| **Build System** | CMake 3.20+, Cargo, vcpkg |
| **Databases** | PostgreSQL, SQLite, Redis |

## Architecture Overview

```
+------------------+         +--------------------+         +------------------+
|  CyxWiz Engine   |<------->| CyxWiz Central     |<------->| Server Node 1    |
|  (Desktop GUI)   |  gRPC   |    Server          |  gRPC   |  (GPU Worker)    |
+------------------+         | (Rust Orchestrator)|         +------------------+
        |                    +--------------------+                  |
        |                            |                               |
        v                            v                               v
+------------------+         +--------------------+         +------------------+
| cyxwiz-backend   |         | PostgreSQL/SQLite  |         | Server Node 2    |
|    (DLL/SO)      |         | Redis Cache        |         |  (CPU Worker)    |
+------------------+         +--------------------+         +------------------+
                                     |
                                     v
                             +--------------------+
                             | Solana Blockchain  |
                             |  (CYXWIZ Token)    |
                             +--------------------+
```

## Version Information

- **Current Version**: 0.2.0
- **Engine**: v0.2.0 - Training infrastructure, data transforms, local training
- **Central Server**: v0.1.0 - Full gRPC services, TUI dashboard
- **Server Node**: v0.1.0 - GUI application, daemon mode, API server

## Documentation Conventions

Throughout this documentation:

- `Code blocks` indicate terminal commands or code snippets
- **Bold text** indicates UI elements or important terms
- *Italic text* indicates file paths or variable names
- > Blockquotes contain tips or important notes

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discord**: Real-time community support
- **Documentation**: You're here!

---

**Next**: [Project Overview](overview/project-overview.md) | [Quick Start](developer/quick-start.md)
