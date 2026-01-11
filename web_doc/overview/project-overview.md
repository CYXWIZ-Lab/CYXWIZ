# Project Overview

CyxWiz is a **decentralized machine learning compute platform** that revolutionizes how ML models are designed, trained, and deployed. By combining the visual design philosophy of game engines with the computational power of scientific computing platforms, CyxWiz creates a unique ecosystem for AI development.

## Vision

The CyxWiz platform aims to:

1. **Democratize ML Development** - Make advanced ML accessible through visual tools
2. **Distribute Computing Power** - Leverage idle GPUs worldwide for training
3. **Create Economic Incentives** - Reward compute providers through blockchain tokens
4. **Enable Collaboration** - Allow seamless sharing of models and resources

## Core Concepts

### Visual Model Building

Instead of writing complex neural network code, CyxWiz provides a visual node editor where you can:

- Drag and drop layer nodes (Dense, Conv2D, LSTM, Attention, etc.)
- Connect nodes to define data flow
- Configure parameters through property panels
- Generate production-ready code in PyTorch, TensorFlow, or Keras

### Decentralized Compute

Training ML models requires significant computational resources. CyxWiz creates a marketplace where:

- **Model Designers** submit training jobs
- **Compute Providers** (miners) execute jobs on their hardware
- **Smart Contracts** handle secure payment and verification
- **Central Server** orchestrates job distribution

### Blockchain Integration

The CYXWIZ token (on Solana) enables:

- Payment for compute services
- Staking for reputation and priority
- Governance participation
- Rewards distribution

## Platform Components

### 1. CyxWiz Engine

The flagship desktop application providing:

| Feature | Description |
|---------|-------------|
| **Node Editor** | Visual drag-and-drop model builder with 80+ node types |
| **Script Editor** | Python IDE with syntax highlighting and execution |
| **Training Dashboard** | Real-time loss/accuracy visualization |
| **Data Tools** | 70+ analysis panels (statistics, clustering, etc.) |
| **Asset Browser** | Project file management with filters |
| **Console** | Python REPL and log output |

**Technology**: C++20, Dear ImGui, ImNodes, ImPlot, OpenGL

### 2. CyxWiz Server Node

The compute worker application (also called "miner"):

| Feature | Description |
|---------|-------------|
| **Job Executor** | Runs training jobs with ArrayFire/GPU |
| **Hardware Monitor** | Tracks CPU, GPU, memory utilization |
| **OpenAI API** | Serves models via compatible REST API |
| **Pool Mining** | Collaborative training with other nodes |
| **Docker Sandbox** | Secure execution of untrusted code |

**Technology**: C++20, ArrayFire, gRPC, OpenAI API compatibility

### 3. CyxWiz Central Server

The network orchestrator managing the decentralized infrastructure:

| Feature | Description |
|---------|-------------|
| **Job Scheduler** | Matches jobs to available nodes |
| **Node Registry** | Tracks online nodes and capabilities |
| **Payment Processor** | Handles Solana token transfers |
| **TUI Dashboard** | Terminal interface for monitoring |
| **REST API** | Web dashboard integration |

**Technology**: Rust, Tokio, gRPC (Tonic), SQLx, Redis

### 4. cyxwiz-backend

The shared compute library used by Engine and Server Node:

| Feature | Description |
|---------|-------------|
| **Tensor Operations** | GPU-accelerated math via ArrayFire |
| **Neural Network Layers** | Dense, Conv, RNN, Attention implementations |
| **Optimizers** | SGD, Adam, AdamW, RMSprop |
| **Loss Functions** | MSE, CrossEntropy, BCE, Huber |
| **Python Bindings** | pycyxwiz module for scripting |

**Technology**: C++20, ArrayFire, pybind11

### 5. cyxwiz-protocol

The gRPC protocol definitions shared by all components:

| Protocol File | Purpose |
|--------------|---------|
| `common.proto` | Shared types (StatusCode, DeviceType) |
| `job.proto` | Job submission, status, results |
| `node.proto` | Node registration, heartbeat, metrics |
| `compute.proto` | Direct compute operations |
| `wallet.proto` | Wallet and payment operations |
| `deployment.proto` | Model deployment management |

## Use Cases

### 1. Individual Researcher

A data scientist can:
1. Design a neural network visually in the Engine
2. Load and explore data using built-in tools
3. Train locally or submit to the network
4. Export trained model in ONNX/PyTorch/TensorFlow format

### 2. Enterprise Team

A company can:
1. Set up private Server Nodes for internal training
2. Use the Central Server for job orchestration
3. Share models and datasets across the team
4. Track experiments with built-in logging

### 3. Compute Provider (Miner)

A GPU owner can:
1. Run the Server Node application
2. Register with the Central Server
3. Accept and execute training jobs
4. Earn CYXWIZ tokens for completed work

### 4. Model Consumer

A developer can:
1. Browse the model marketplace
2. Purchase trained models with tokens
3. Deploy models via OpenAI-compatible API
4. Fine-tune models for specific use cases

## Platform Benefits

### For Model Designers
- **Visual Design**: No coding required for basic models
- **Code Generation**: Export to production frameworks
- **Distributed Training**: Access to global GPU resources
- **Cost Efficiency**: Pay only for compute used

### For Compute Providers
- **Passive Income**: Monetize idle GPU resources
- **Easy Setup**: Single application to install
- **Transparent Payments**: Blockchain-based settlement
- **Reputation System**: Build trust over time

### For the Ecosystem
- **Decentralization**: No single point of failure
- **Transparency**: Open-source codebase
- **Scalability**: Add nodes to increase capacity
- **Interoperability**: Standard formats (ONNX, etc.)

## Technical Highlights

### Cross-Platform Support
- Windows (x64)
- macOS (x64, ARM64)
- Linux (x64)
- Android (backend library only)

### GPU Acceleration
- CUDA (NVIDIA GPUs)
- OpenCL (AMD, Intel GPUs)
- CPU fallback (for development)

### Modern C++
- C++20 standard
- RAII resource management
- Thread-safe designs
- Cross-platform abstractions

### Rust Backend
- Async runtime (Tokio)
- Memory safety guarantees
- High-performance networking
- Clean error handling

---

**Next**: [Architecture](architecture.md) | [Technology Stack](technology-stack.md)
