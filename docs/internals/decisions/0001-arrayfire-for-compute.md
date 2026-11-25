---
title: "Decision 0001: ArrayFire for compute"
description: Why CyxWiz executes tensors on ArrayFire with native providers, and why LibTorch is only the test oracle.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - README.md
  - ../architecture/compute-back-ends.md
  - ../parity-oracle.md
---

**Status:** accepted (2026-09-24).

## Context

The Engine and the Server Node must train on whatever GPU a user has:
NVIDIA, AMD, Intel or none. One code path has to run on CUDA, OpenCL,
oneAPI and the CPU, and the installer has to stay small enough to ship a
CPU base to everyone. Some layers (recurrent cells, Conv2D on ArrayFire's
CUDA back end) cannot run at useful sizes through ArrayFire's JIT.

## Decision

- Tensors execute on **ArrayFire** (CPU, CUDA, OpenCL and oneAPI back
  ends). Compute is written once on ArrayFire and stays on the device.
- Where ArrayFire falls short, **native neural providers** register behind
  one interface (`INeuralNetworkProvider`,
  `cyxwiz-backend/include/cyxwiz/neural_provider.h`) and are chosen per run
  by the target device: an NVIDIA provider (CyxWiz CUDA kernels with
  cuBLAS) and an OpenCL provider.
- The CPU fallback is ArrayFire's CPU back end, not hand-written CPU loops.
- **LibTorch is not a compute back end.** It is the numerical oracle of the
  parity tests and is never linked into `cyxwiz-backend`, `cyxwiz-engine`
  or `pycyxwiz`.

## Consequences

- The CPU base runs everywhere; each GPU runtime is an optional pack.
- No duplicate nodes per back end. Placement and any fallback are shown to
  the user, never silent.
- ArrayFire bugs on one back end are worked around once in the backend and
  listed in the [known issues](../../known-issues.md), not patched per layer.
- Reopening this needs a measured ArrayFire failure at a shape the product
  needs, a device platform neither ArrayFire nor the providers can serve, or
  a move to subgraph-level providers.
