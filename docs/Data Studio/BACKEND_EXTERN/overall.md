# BackendExtern Overall Design and Product Decision

## Problem

CyxWiz has two valuable but different strengths:

- a native C++ graph runtime with CPU/ArrayFire execution; and
- project-scoped Python scripting and generated framework code.

Users also need access to ecosystems where useful models already exist,
including PyTorch and JAX/Flax. Treating those ecosystems as alternate native
backends for all CyxWiz nodes would multiply maintenance and weaken the native
runtime. Treating them as arbitrary console scripts would provide little
reproducibility, lifecycle control, or Data Studio UX.

## Product choice

Create `BackendExtern`: an Engine-owned, framework-neutral external-runtime
layer. It runs a managed Python worker process for a named runtime and exchanges
validated data, results, events, and provenance through a versioned protocol.

This gives three clearly labelled execution modes:

| Mode | Owner | LTS promise |
| --- | --- | --- |
| Native CyxWiz | C++ graph/runtime, CPU/ArrayFire | Full native support |
| Managed external model | BackendExtern plus a pinned runtime/model adapter | Full support for named versions and paths |
| User Python script | User project environment | Script transport works; package/model correctness is user-owned |

## What this is not

- Not a rewrite of ArrayFire or the native tensor system.
- Not a general Python remote-code execution interface.
- Not a promise to run arbitrary Hugging Face repositories.
- Not a promise that a PyTorch or JAX checkpoint can be converted to a native
  CyxWiz model.
- Not a general distributed training system in the first release.

## Value if done well

Data Studio gains a small catalog of high-value external models while keeping
their execution visible. Engineers get reproducible, inspectable runtime
records instead of undocumented environment assumptions. Researchers retain
the script editor as an escape hatch without putting its dependency churn into
the native engine.

## Ownership decision: core service plus optional providers

`BackendExtern` should ship as a small **core Engine service**, enabled only
when used. It needs core ownership because it coordinates project lifecycle,
task cancellation, graph compilation validation, diagnostics, provenance, and
secure worker startup. These are not safe to leave to an arbitrary plugin.

Model providers and their optional UI/node definitions can be plugins or
bundled provider packages. The first providers should be CyxWiz-maintained and
ship as optional installable components. Third-party providers are deferred
until the protocol and permission model have proven stable.

```text
Core Engine: protocol, worker manager, lifecycle, data/result types,
             diagnostics, provenance, graph validation
Official provider: model adapter, runtime lockfile, node metadata, tests
Third party: deferred; must use the same public protocol
```

## Success measure

The first useful release is not "JAX support". It is a user selecting one
supported external model in Data Studio, running it successfully or receiving a
precise failure, and saving a project that another supported installation can
reproduce from its runtime/model provenance.

