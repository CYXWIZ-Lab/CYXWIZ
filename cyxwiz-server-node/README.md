# CyxWiz Server Node

The Server Node is the worker-side CyxWiz application. This directory builds two executables:

- `cyxwiz-server-gui` for graphical configuration and monitoring;
- `cyxwiz-server-daemon` for daemon-style execution.

## Responsibilities

- report local hardware and runtime capability;
- host job, deployment, file, terminal, and node services;
- execute supported local worker jobs through shared backend contracts;
- expose metrics, status, and task lifecycle;
- connect to an external orchestrator through `cyxwiz-protocol`.

The external orchestration service is maintained in the separate [CyxCloud repository](https://github.com/CYXWIZ-Lab/cyxcloud). It is not built from this checkout. End-to-end distributed training remains pre-release and must not be represented as a verified production service.

## Configuration

Safe templates live in `resources/config/` and the root `config/` directory. Keep secrets and machine-local overrides outside Git. Placeholder JWT values are not suitable for deployed systems.

ArrayFire is optional for the Server Node build. When it is absent, GPU-related reporting and execution capability are reduced.

## Build

From the repository root:

```powershell
.\build.bat --server-node
```

Or build the targets directly:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_ENGINE=OFF -DCYXWIZ_BUILD_SERVER_NODE=ON
cmake --build --preset windows-debug --target cyxwiz-server-daemon cyxwiz-server-gui
```

See [the root README](../README.md), [installation guide](../INSTALL.md), and [security policy](../SECURITY.md).
