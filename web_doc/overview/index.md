# Overview

This section provides high-level documentation about the CyxWiz platform, its architecture, and core concepts.

## Contents

### [Project Overview](project-overview.md)
Introduction to CyxWiz, its vision, core concepts, and platform components. Start here if you're new to CyxWiz.

### [Architecture](architecture.md)
Comprehensive system architecture documentation including:
- Component relationships
- Data flows (job submission, node registration, training progress)
- Network topology
- Security architecture
- Database schema

### [Technology Stack](technology-stack.md)
Complete technology reference covering:
- Programming languages (C++20, Rust, Python)
- GUI frameworks (Dear ImGui, ImNodes, ImPlot)
- GPU computing (ArrayFire)
- Networking (gRPC, Protocol Buffers)
- Databases (PostgreSQL, SQLite, Redis)
- Blockchain (Solana)
- Build systems (CMake, Cargo, vcpkg)

### [Data Flow](data-flow.md)
Detailed data flow documentation:
- Job lifecycle
- Real-time updates
- File transfers
- Metrics collection

### [Security Model](security.md)
Security architecture and best practices:
- Authentication (JWT tokens)
- Authorization
- Sandboxing (Docker)
- Encryption (TLS)
- Blockchain verification

## Quick Links

| Topic | Description |
|-------|-------------|
| [Vision](project-overview.md#vision) | Platform goals and objectives |
| [Components](project-overview.md#platform-components) | Engine, Server Node, Central Server |
| [Use Cases](project-overview.md#use-cases) | Researcher, Enterprise, Miner scenarios |
| [Architecture Diagram](architecture.md#high-level-architecture) | Visual system overview |
| [Technology Table](technology-stack.md#dependencies-summary) | All dependencies |

## Key Concepts

### Decentralized Computing

CyxWiz distributes ML training across a network of compute providers (Server Nodes). Users submit jobs, and the Central Server orchestrates execution across available resources.

### Visual Model Building

The Engine provides a node-based interface for designing neural networks. Users connect layer nodes, configure parameters, and generate production code without writing boilerplate.

### Token Economics

The CYXWIZ token (on Solana) enables:
- Payment for compute services
- Node staking for reputation
- Governance participation
- Reward distribution

### Cross-Platform Design

All components are designed to work across:
- Windows (x64)
- macOS (x64, ARM64)
- Linux (x64)
- Android (backend library only)

---

**Next**: [Project Overview](project-overview.md)
