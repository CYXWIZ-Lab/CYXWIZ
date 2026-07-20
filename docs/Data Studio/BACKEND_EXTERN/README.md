# BackendExtern Design Package

## Status

Planning only. No production code is authorized by this package.

## Decision in one sentence

Keep CyxWiz native graph execution on the current C++/CPU/ArrayFire path, and
add a versioned, out-of-process Python execution plane for curated external
models and user scripts.

`BackendExtern` is intentionally not a second implementation of the CyxWiz
graph executor. It is a small control plane and data-exchange contract that
lets a CyxWiz graph call a supported external model without making PyTorch,
JAX, or Flax core dependencies.

## Reading order

1. [overall.md](overall.md) - product decision, scope, and ownership.
2. [current_runtime.md](current_runtime.md) - current codebase facts that the
   design must preserve.
3. [architecture_design.md](architecture_design.md) - target architecture and
   ASCII component structure.
4. [runtime_contract.md](runtime_contract.md) - stable Engine-to-worker API.
5. [workflow.md](workflow.md) - lifecycle and user/runtime workflows.
6. [nodes.md](nodes.md) - proposed Data Studio and graph nodes.
7. [implementation.md](implementation.md) - staged implementation plan.
8. [ticket.md](ticket.md) - proposed work tickets and acceptance gates.
9. [lts_operations.md](lts_operations.md) - versioning, support, upgrades, and
   diagnostics.
10. [licensing.md](licensing.md) - commercial distribution and model policy.
11. [use_cases.md](use_cases.md) and [examples.md](examples.md) - concrete
   product scenarios.
12. [risk_register.md](risk_register.md) - risks that must stay visible before
   coding.

## Non-negotiable invariants

- Existing native graph compilation, materialization, training, and ArrayFire
  behavior remain valid when BackendExtern is absent or disabled.
- External framework code never links into the Engine process for supported
  product execution.
- The core owns a narrow, versioned protocol; each runtime owns its framework
  dependencies.
- A worker failure, timeout, or GPU problem does not crash the Engine.
- A successful model download is not a supported import or commercial-use
  approval.
- User scripts remain flexible, but only pinned managed runtimes receive a
  reproducibility and LTS promise.

