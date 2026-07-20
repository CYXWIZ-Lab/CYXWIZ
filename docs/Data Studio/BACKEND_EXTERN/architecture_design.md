# BackendExtern Architecture Design

## Design principle

The Engine owns orchestration and contracts. A runtime worker owns framework
imports, framework device discovery, model loading, and model execution. They
communicate only through typed protocol messages and files under approved
project/cache paths.

## Component structure

```text
                         CyxWiz Engine process
 +------------------------------------------------------------------+
 | Data Studio / Node Editor                                        |
 |     |                                                            |
 |     v                                                            |
 | External Model Node + Properties                                 |
 |     |                                                            |
 |     v                                                            |
 | GraphCompiler / Materializer validation                          |
 |     |                                                            |
 |     v                                                            |
 | BackendExternService                                             |
 |   - RuntimeCatalog       - WorkerManager     - RunRegistry       |
 |   - Protocol validation  - Cancellation      - Provenance store  |
 |   - Diagnostics          - Permission gate   - Result importer   |
 +-------------------------------+----------------------------------+
                                 | local IPC, protocol v1
                                 | no C++ framework linkage
 +-------------------------------v----------------------------------+
 | Managed Python worker process                                     |
 |   runtime manifest + lockfile                                     |
 |   provider adapter                                                |
 |   PyTorch OR JAX/Flax OR another approved framework              |
 |   model cache access only                                         |
 +-------------------------------+----------------------------------+
                                 |
                                 v
                    approved model cache / project artifacts
```

## Placement rules

| Concern | Owner | Reason |
| --- | --- | --- |
| Native graph execution | existing Engine/backend | Preserve legacy behavior and performance |
| Worker process lifecycle | BackendExtern core service | Must integrate with project/task shutdown safely |
| Framework imports and CUDA/XLA state | worker | Isolate framework crashes and version conflicts |
| Model-specific parsing | provider adapter | Keep framework/model churn out of core |
| Node palette/panels | official provider plugin/package | Optional product capability |
| Runtime catalog schema | core | Stable LTS contract |
| Model license approval | product/legal catalog policy | Not inferable from technical format |

## Data exchange

The first protocol uses explicit files plus small JSON messages:

- tabular inputs: Arrow IPC/Parquet where the existing data path supports it;
- dense tensors: versioned `.npy` or a compact Engine-owned tensor artifact;
- control/result/event messages: JSON validated against a strict schema;
- model artifacts: read-only, provenance-checked cache paths.

Initial copies are intentional. GPU zero-copy and DLPack are deferred until
there is one proven same-device/same-driver use case with explicit ownership
rules. Premature zero-copy would couple ArrayFire, CUDA, PyTorch, and JAX
allocators and undermine worker isolation.

## Provider topology

```text
BackendExternService
  -> RuntimeCatalog entry: cyxwiz-tabular-2026.1
      -> provider: cyxwiz.tabfm
      -> framework choices: pytorch | jax_flax
      -> exact Python and package lock
      -> allowed model revisions
      -> declared input/output schema
```

The provider never receives an arbitrary shell command or raw Python string
from the graph. It receives a validated `RunRequest` for a known operation.

## Failure containment

```text
worker returns structured error / exits / exceeds timeout
  -> WorkerManager marks run failed or cancelled
  -> no result is materialized
  -> Engine stays alive
  -> RunRegistry records reason, runtime identity, and safe diagnostics
  -> node displays an actionable state
```

