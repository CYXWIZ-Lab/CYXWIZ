# Project structure

This document defines the public repository boundary and intended dependency direction.

## Runtime components

| Component | Owns | Must not own |
| --- | --- | --- |
| `cyxwiz-backend` | tensors, computation, models, layers, losses, optimizers, data primitives, device abstraction | GUI state or network orchestration |
| `cyxwiz-engine` | desktop UI, graphs, Data Studio, local task orchestration, scripting integration | duplicate implementations of backend computation |
| `cyxwiz-protocol` | versioned protobuf/gRPC contracts | business logic or runtime state |
| `cyxwiz-server-node` | worker lifecycle, job execution services, hardware reporting | the external orchestration service |
| `plugins` | optional, isolated integrations | mandatory core behavior or silent fallback |

CyxCloud is a [separate repository](https://github.com/CYXWIZ-Lab/cyxcloud). The historical Central Server is not a component of this checkout.

## Dependency direction

```text
cyxwiz-engine ---------+
                       +--> cyxwiz-backend
cyxwiz-server-node ----+
          |
          +----------------> cyxwiz-protocol <---------------- CyxCloud

optional plugins --> explicit host contracts
```

The Engine may translate a graph into backend operations, but computation truth remains in the backend. Protocol types carry data between processes and must not become a second domain-model implementation.

## Supporting directories

- `tests/` contains public automated tests and focused benchmarks.
- `examples/` contains runnable or inspectable examples, not production state.
- `docs/` holds active, verified public documentation.
- `cmake/`, `vcpkg-ports/`, and `vcpkg-triplets/` hold build integration.
- `config/` contains safe templates only.
- `scripts/` contains maintained automation; generated dependency trees are excluded.

Build trees, package caches, logs, datasets, artifacts, checkpoints, credentials, internal tickets, business material, and licensed reference books are not public source and are excluded by `.gitignore`.

## Adding functionality

1. Identify the smallest owning component.
2. Reuse or extend an existing contract before creating another abstraction.
3. Keep optional capability discovery explicit.
4. Add tests at the owning layer and at any changed boundary.
5. Update documentation with verified behavior and known limitations.

New top-level directories require a clear owner, lifecycle, public purpose, and build or documentation entry point.
