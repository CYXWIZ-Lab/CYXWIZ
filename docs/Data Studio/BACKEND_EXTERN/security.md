# BackendExtern Security Boundary

## Threat model

BackendExtern receives model metadata/artifacts from external sources, launches
an interpreter with heavyweight packages, and processes potentially sensitive
project data. The worker is a reliability boundary, not a complete security
sandbox. The product must be honest about that distinction.

## Trust zones

```text
Untrusted or conditional trust
  - model repository metadata and artifacts
  - user-provided table data and user scripts
  - external network responses

Trusted Engine control plane
  - validated runtime catalog
  - schema validator
  - approved path resolver
  - project permission decisions

Constrained worker
  - fixed runtime environment
  - named provider operation
  - approved input/model/output paths
  - private IPC endpoint
```

## Mandatory controls for v1

- Only Engine-created workers from an approved runtime executable may connect.
- IPC uses an unguessable per-run capability passed through a private launch
  channel; it is never written to project files or logs.
- Workers accept named provider operations, not arbitrary source code or shell
  commands.
- Paths are canonicalized and remain under permitted project artifact or model
  cache roots.
- The Engine enforces request/output size, event payload, timeout, and artifact
  count limits.
- The worker runs with the least practical OS permissions and no inherited
  credentials beyond approved model access.
- Model repositories never execute remote Python code. Unsafe serialization
  formats and custom code need a separate future review.
- Download tokens use the platform credential store and are redacted from logs,
  graph files, run reports, and support bundles.

## Important non-claims

- A same-user local worker is not a hostile-code sandbox.
- User-managed scripts are not sandboxed merely because the Engine starts
  Python for them.
- A model that is parseable is not necessarily safe, licensed, or approved for
  commercial use.

