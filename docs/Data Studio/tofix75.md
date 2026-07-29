# To Fix 75 - Checkpoint Format v2 and Exact Training Resume

## Status

In progress - first Track70 follow-up. The format-capability guard, v2 manifest
contract, backend Adam state contract, and verified model/Adam payload archives
are implemented. Complete checkpoint assembly, runtime state, and the Resume
Training GUI are not implemented yet.

## Implemented foundation - 2026-07-29

The first two foundation slices are complete:

- `CheckpointManager::InspectCheckpoint` now reports separate load-for-test,
  warm-start, and exact-resume capabilities.
- Format v1 is explicitly test-load/warm-start only and explains the missing
  optimizer, scheduler, RNG/sampler, and cursor state.
- Unknown or future metadata versions fail before model parameters are read or
  mutated; a future `2.0` marker cannot be interpreted accidentally as v1.
- The backend now owns a typed `OptimizerState` envelope and transactional
  export/import interface instead of requiring engine code to access private
  optimizer fields.
- Adam exports and validates its learning rate, step count, hyperparameters,
  and paired first/second moment tensors. A failed import leaves the optimizer
  unchanged.
- AdamW and other optimizers report state serialization as unsupported until
  their full private state contract is implemented; no optimizer silently
  claims exact-resume support.

Release verification passed:

- the Adam state round-trip reproduces the exact next parameter update and
  rejects an incomplete moment pair transactionally;
- the checkpoint/debug suite preserves all v1 round-trips and rejects an
  unsupported version without model mutation;
- the complete backend unit executable passed 2,366 assertions in 272 cases.

The v2 manifest and atomic inventory slice is also implemented:

- `CheckpointManifestV2` records run/checkpoint identity, graph/dataset/
  partition fingerprints, training cursor, algorithm identities, runtime-state
  declarations, and typed payload descriptors;
- validation separates a structurally valid manifest from one that declares
  all exact-resume state, so an incomplete inventory cannot enable Resume;
- payload paths must be safe and relative, payload sizes must be non-zero, and
  SHA-256 inventory values must have the expected form;
- model, optimizer, runtime, graph, and dataset payloads are required for an
  exact-resume declaration, with conditional scheduler and early-stopping
  state plus mandatory RNG and sampler state;
- `manifest.json` is written through a unique temporary file and atomic rename,
  and an already published manifest is immutable;
- focused coverage proves complete/incomplete classification, path traversal
  rejection, atomic publication, round-trip loading, and fail-closed future
  schema handling.

The first v2 payload slice is also implemented:

- model parameters and backend-owned optimizer state share one typed named-
  tensor archive instead of duplicating tensor formats;
- each tensor records its name, full shape, data type, and exact byte size, so
  payload I/O no longer assumes every tensor is `Float32`;
- payloads are written to a unique temporary path, hashed with streaming
  SHA-256, measured, and atomically renamed into an immutable final path;
- reads verify the descriptor path, file type, exact size, and SHA-256 before
  parsing or mutating a model/optimizer;
- model parameter count, names, shapes, and data types are checked before
  installation, while Adam delegates final transactional validation to its
  backend state contract;
- Adam state at step zero is valid even though it has no moment tensors;
- focused Release coverage proves model round-trip, Adam exact-next-step
  equivalence, step-zero round-trip, hexadecimal hash-case compatibility, and
  corrupted-payload rejection without model mutation.

The full Release engine builds successfully with this payload layer. The next
slice is runtime payload state: training cursor, RNG, sampler/shuffler,
scheduler, precision/gradient scaler, and early stopping, followed by complete
manifest assembly and transactional checkpoint publication. The engine must
not expose `Resume Training` until end-to-end interrupted-versus-uninterrupted
equivalence also passes.

## Decision statement

Introduce a versioned checkpoint v2 contract that can resume training exactly,
while preserving checkpoint v1 as a load-for-testing format.

`Resume Training` must mean continuation of the same run state. Loading model
weights into a newly initialized optimizer is warm-starting and must not be
presented as exact resume.

## Current truth

Checkpoint v1 provides a useful tested workflow:

- model parameters and training metrics are saved;
- parameter name, count, shape, and type are checked before installation;
- a compatible checkpoint can be loaded transactionally for `Tools > Test`;
- the active graph fingerprint prevents testing a changed graph accidentally;
- optimizer learning rate may be restored.

It does not persist the full state required to reproduce the next training
step:

- optimizer moment/momentum tensors and optimizer step counters;
- scheduler state;
- gradient-scaler state when mixed precision is used;
- engine, backend, sampler, shuffler, and augmentation RNG state;
- epoch, batch, accumulation, and sampler cursor;
- resolved dataset identities and partition manifest;
- immutable graph/configuration snapshot;
- early-stopping state and best-checkpoint relationship.

## Smallest production contract

Each v2 checkpoint directory owns one atomic manifest plus typed payloads:

```text
checkpoint/
  manifest.json
  model/
  optimizer/
  scheduler/        optional
  runtime_state/
```

The manifest records:

- format and schema version;
- run ID, checkpoint ID, parent checkpoint ID, and creation time;
- engine/backend version and compute/device portability metadata;
- graph fingerprint and a canonical graph/configuration snapshot reference;
- dataset and partition fingerprints;
- model, optimizer, scheduler, loss, and precision identities;
- completed epoch, next batch, optimizer step, and accumulation position;
- RNG/sampler state inventory and payload hashes;
- early-stopping/best-model state;
- payload sizes, hashes, and required/optional status.

Payload publication is transactional: write to a temporary checkpoint, verify
it, then rename it into place. Cancellation or failure must not expose a
partially valid checkpoint.

## Compatibility rules

- v1 remains loadable for testing and explicit warm start.
- v1 cannot be selected for exact resume.
- unknown future versions fail closed with a clear compatibility message.
- exact resume rejects graph, model, optimizer, scheduler, dataset, partition,
  or state mismatches before mutating the active run.
- device relocation is permitted only for state types proven portable.
- changing allowed runtime controls such as a larger final epoch count must be
  explicit and recorded as a resume override.

## Implementation phases

1. Define `CheckpointManifestV2`, payload inventory, canonical serialization,
   and compatibility diagnostics.
2. Add backend state export/import contracts for each supported optimizer;
   unsupported optimizers report `exact resume unavailable`.
3. Persist scheduler, precision, early-stopping, RNG, sampler, and cursor state.
4. Add transactional save/load and a read-only checkpoint inspector.
5. Add `Resume Training...` as a background task distinct from
   `Load Checkpoint for Testing...` and `Warm Start...`.
6. Connect resumed runs to persisted run history from To Fix 76.

## Acceptance criteria

- An uninterrupted deterministic training run and a run interrupted at a
  checkpoint then resumed produce the same next batches, optimizer steps,
  parameters, and metrics within the declared numerical tolerance.
- The test covers a stateful optimizer such as Adam, shuffled batching, and
  early stopping.
- A corrupted or incomplete payload is rejected without changing the active
  model or run.
- Cancelling save or load publishes no checkpoint and installs no partial
  state.
- v1 load-for-testing regressions continue to pass.
- The GUI and logs distinguish exact resume, warm start, and load for testing.

## Non-goals

- distributed multi-worker resume before a distributed state contract exists;
- arbitrary cross-version migration of unknown backend state;
- embedding complete datasets inside checkpoints;
- silently approximating resume when a required state cannot be restored.

## Dependencies

This is the first follow-up. To Fix 76 may share its run/checkpoint identity
types, but checkpoint serialization must remain a focused module rather than a
general project database.
