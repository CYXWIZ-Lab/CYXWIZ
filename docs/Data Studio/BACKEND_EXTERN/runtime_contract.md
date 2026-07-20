# BackendExtern Runtime Contract v1

## Contract goals

- stable across framework upgrades;
- small enough to test end-to-end;
- no framework object, Python object, GPU pointer, or Engine internal pointer
  crosses the boundary;
- explicit about input/output schemas, cancellation, and provenance.

## Runtime manifest

Each managed runtime is declared by an Engine-owned manifest, conceptually:

```json
{
  "id": "cyxwiz-tabular-2026.1",
  "protocol_version": 1,
  "python": "3.12.x",
  "provider_ids": ["cyxwiz.tabfm"],
  "frameworks": {
    "pytorch": "pinned version",
    "jax": "pinned version",
    "flax": "pinned version"
  },
  "platforms": ["windows-x64"],
  "devices": ["cpu", "cuda-if-validated"],
  "lockfile_sha256": "...",
  "support_until": "YYYY-MM-DD"
}
```

The manifest is metadata, not an instruction to execute arbitrary installers.
Installation is performed by a controlled runtime installer with verified
package sources and recorded hashes.

## Run request

```json
{
  "protocol_version": 1,
  "run_id": "uuid",
  "runtime_id": "cyxwiz-tabular-2026.1",
  "provider_id": "cyxwiz.tabfm",
  "operation": "predict",
  "device_policy": "auto",
  "input_artifacts": [{"kind": "arrow_table", "path": "approved/path", "sha256": "..."}],
  "model": {"repository": "owner/model", "revision": "immutable-id", "artifact_manifest": "approved/path"},
  "parameters": {"task": "classification"},
  "limits": {"timeout_seconds": 300, "max_output_bytes": 0}
}
```

Core validates every enum, path, size, hash, provider/operation pair, and
parameter schema before starting a worker. The worker repeats validation; it
does not trust the caller merely because it is local IPC.

## Events and result

Workers emit bounded events: `started`, `progress`, `log`, `warning`,
`completed`, `failed`, and `cancelled`. Each has a run ID, monotonic sequence
number, timestamp, and bounded payload.

`RunResult` returns only declared result artifacts plus metadata:

```text
status, result artifact paths/hashes, schema, metric summary,
runtime/provider/framework versions, device used, model provenance,
duration, warnings, safe error code/message
```

The Engine imports results only after schema and hash verification. A worker
cannot create arbitrary graph nodes or mutate project state.

## Cancellation and timeout

The Engine sends cooperative cancellation first, waits a short bounded grace
period, then terminates the worker if necessary. A terminated worker result is
always `cancelled` or `failed`; partial output is never treated as successful.

## Versioning

Protocol major version changes are incompatible. Runtime updates do not modify
an existing runtime ID: a new runtime ID is created, tested, and offered as an
opt-in project migration. Old projects remain pinned until their runtime LTS
window ends.

