# To Fix 77 - Typed Preprocessing-State Ports and Fit/Transform Graph Semantics

## Status

Open - Track70 follow-up. The existing file-based fitted-state workflow remains
the supported production slice until this ticket is implemented.

## Decision statement

Represent fitted preprocessing state as a typed immutable graph artifact, not
as an arbitrary file-path convention.

Fit-capable preprocessing nodes must make data flow and state flow explicit:

```text
Train Dataset -> Fit + Transform -> Train Dataset'
                         |
                         +-> PreprocessingState

Test Dataset + PreprocessingState -> Transform -> Test Dataset'
```

Dataset and state pins are different types and cannot be connected
interchangeably.

## Current truth

`FillMissing` and `StandardScaler` already support:

- Fit + Transform and Transform Only modes;
- persisted state paths and overwrite policy;
- feature/label selection;
- training-derived state reused on external Test data;
- operator, configuration, schema, and numeric-width validation;
- rejection of fitting behind explicit evaluation roles.

This correctly prevents leakage but requires users to coordinate paths and run
ordering manually. State identity is not part of a first-class graph edge or
artifact registry.

## Core types

Introduce a narrow common artifact descriptor:

```text
PreprocessingStateRef
  artifact_id
  operator_type
  schema_version
  fitted_schema_fingerprint
  configuration_fingerprint
  training_dataset_fingerprint
  payload_hash
  created_by_node
```

Operator payloads remain private to their implementation. The core understands
identity, compatibility, provenance, storage, and typed routing; it must not
accumulate every scaler/imputer field.

## Graph behavior

- A fit-capable node has an explicit state output.
- Transform mode has an explicit state input and cannot fit.
- A combined Fit + Transform node may output both transformed data and state.
- Evaluation-role paths cannot reach a fit operation.
- One fitted state may fan out to Dev, Test, and Inference transforms.
- State artifacts are immutable. Refitting creates a new artifact ID.
- Saved graphs store references and provenance, not machine-specific absolute
  paths where a project-relative artifact is available.
- The compiler validates state type, operator identity, schema compatibility,
  configuration fingerprint, and train/evaluation direction before execution.

## Migration

Existing `state_path` graphs remain readable through a legacy file adapter.
The UI should offer an explicit import into the project artifact store. New
graphs use typed state pins; they must not silently emit hidden side files.

## Implementation phases

1. Add the state artifact type, registry, serialization envelope, and pin type.
2. Convert `FillMissing` and `StandardScaler` as the reference operators.
3. Add compiler provenance/leakage checks and cache identity integration.
4. Add graph UI affordances for fit, transform, state inspection, and
   compatibility errors.
5. Migrate additional fit-capable preprocessing operators individually.

## Acceptance criteria

- One graph can fit on Train and transform Dev/Test from the exact same state.
- Test values use Train statistics, proven by deterministic fixtures.
- Connecting Dataset to a state pin, using scaler state in an imputer, or
  feeding Test into Fit fails at compile time with corrective diagnostics.
- Changing fit configuration, schema, or training data produces a different
  artifact identity and invalidates stale materialization.
- State save is atomic and cancellation cannot publish a partial artifact.
- Legacy path-based graphs continue to execute with a visible compatibility
  notice.

## Non-goals

- one universal JSON payload containing every operator implementation;
- implicit fitting selected by whichever input executes first;
- mutable state shared across concurrent runs;
- automatic refitting on Test or Inference after a compatibility failure.
