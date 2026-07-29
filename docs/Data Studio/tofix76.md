# To Fix 76 - Persisted Training Run History and Reproducible Run Manifests

## Status

Open - Track70 follow-up. Depends on the stable run/checkpoint identity contract
introduced by To Fix 75.

## Decision statement

Make Training Dashboard history project-scoped and restart-safe. A run shown
after restart must come from a validated persisted run manifest, not from a
reconstructed or fabricated in-memory record.

## Current truth

Run Comparison accurately shows completed training runs from the current engine
session. Checkpoint loading for testing is shown as active model state and is
not inserted as a fake training run. This is honest but incomplete: closing the
engine loses comparison history and the relationship between a run, its graph,
data partitions, metrics, checkpoints, and test results.

## Storage contract

Use one bounded project-owned directory:

```text
<project>/runs/<run_id>/
  run.json
  graph.cyxgraph
  metrics.json
  events.jsonl       optional bounded event stream
```

Large model and data payloads remain in checkpoints, artifacts, or caches and
are referenced by stable identity and hash. The run directory must not duplicate
them.

`run.json` records:

- run ID, display name, status, timestamps, and parent/resume relationship;
- graph and compiled-plan fingerprints;
- engine/backend/build identity;
- objective family, model, loss, optimizer, scheduler, and precision settings;
- dataset role identities and exact partition manifest;
- preprocessing-state artifact identities;
- checkpoint references, including best and last;
- metric schema, summary, test-result references, warnings, and failure reason;
- reproducibility classification and any user-approved overrides.

## Lifecycle

Run state transitions are explicit:

```text
preparing -> running -> completed
                     -> early_stopped
                     -> cancelled
                     -> failed
                     -> interrupted
```

The manifest is written atomically at meaningful transitions. On startup, a
stale `running` record is classified as `interrupted`; it is never silently
reported as completed.

## GUI behavior

- Run Comparison loads validated project history lazily.
- Users can filter, compare, rename, inspect, and remove a selected run record.
- Missing referenced checkpoints/artifacts are shown as unavailable, without
  deleting historical metrics.
- `Test` results attach to the run/checkpoint actually evaluated.
- `Resume` is enabled only when To Fix 75 proves exact-resume compatibility.
- `Clear All` clearly distinguishes clearing the view from deleting persisted
  run records and requires confirmation for deletion.

## Implementation phases

1. Define a versioned `RunManifest` and atomic `RunStore`.
2. Persist lifecycle transitions and bounded metrics from TrainingManager.
3. Load project history without blocking the GUI.
4. Connect checkpoints, tests, artifacts, and resumed-run ancestry.
5. Replace session-only Run Comparison storage with the validated store while
   retaining an in-memory cache for rendering.

## Acceptance criteria

- Completed, failed, cancelled, early-stopped, and interrupted runs survive an
  engine restart with truthful status.
- Run Comparison after restart reports the same metric summaries and partition
  fingerprints as the original session.
- A malformed manifest is isolated and reported; it cannot crash startup or
  hide other valid runs.
- Concurrent metric updates cannot expose partial JSON.
- Clearing a UI filter does not delete disk history.
- Deleting one selected run never deletes shared checkpoints or artifacts
  unless an explicit ownership/reference check permits it.

## Non-goals

- a server database or cloud experiment tracker;
- duplicating full datasets, models, or checkpoint tensors;
- treating imported checkpoints without run metadata as completed runs;
- unbounded per-batch telemetry retention.
