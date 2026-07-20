# Implementation Strategy

## Phase 0 - Design closure

Deliver protocol schemas, node contracts, runtime ownership, supported platform
matrix, threat model, license policy, and a test plan. No framework install or
Hub download occurs in this phase.

Exit gate: architecture review accepts the core-service/provider split and the
LTS policy in this package.

## Phase 1 - Core control plane without frameworks

Implement small C++ types for runtime manifests, provider descriptors, run
requests/results/events, a runtime catalog, and a worker manager. Add a fake
test worker that validates lifecycle, cancellation, malformed responses, and
project reopen behavior.

Exit gate: all native graph tests remain unchanged; a fake worker crash cannot
crash the Engine; no Python framework is a build dependency.

## Phase 2 - Controlled worker and artifacts

Add a Python worker launcher, local IPC, approved-path artifact exchange,
hashing, timeout/termination, event streaming, and support-bundle records.
Start CPU-only on one platform. Reuse the Engine project/task lifecycle rather
than creating a second scheduler.

Exit gate: reproducible hello-model run and fault tests; installation and
uninstallation are recoverable; no token or secret appears in diagnostics.

## Phase 3 - One curated provider

Implement one small provider with a fixed input/output schema and a lockfile.
Choose a commercially approved model before productizing it. Add a single
Data Studio node and provenance panel. The model is inference-only.

Exit gate: deterministic reference fixture, schema failures, cache provenance,
and project reopen on exact runtime are covered.

## Phase 4 - GPU and second framework

Add GPU only after CPU is stable. Validate one GPU/platform/framework
combination at a time. Add the second framework only where it provides a real
model capability, not merely feature parity. Device reports and explicit
fallback reasons are mandatory.

## Phase 5 - Provider SDK decision

Only after two official providers share proven needs, publish a small provider
SDK. Do not design a broad third-party framework plugin system in advance.

## Explicitly deferred

- direct GPU zero-copy;
- arbitrary Python execution through the managed protocol;
- general training/fine-tuning;
- automatic model conversion;
- concurrent workers sharing a GPU by default;
- a multi-platform GPU matrix before Windows CPU/GPU support is proven.

