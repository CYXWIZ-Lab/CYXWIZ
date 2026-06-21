# To Fix 29 - Experiment Management And Hyperparameter Search

## Purpose

Create a first-class experiment-management layer above the lightweight
run-comparison ledger added during `done12.md`.

## Why This Exists

`done12.md` added practical run comparison for completed training runs. That is
useful for immediate workflow visibility, but it is not a persistent experiment
database, benchmark manager, architecture scorer, or hyperparameter-search
system.

This file owns those larger capabilities so they are not mixed into checkpoint
fine tuning, DataLoader runtime semantics, or pretrained-model import.

## Scope

- Persistent experiment/run database.
- Benchmark-run management across datasets, model families, and domains.
- Automatic hyperparameter search.
- Hyperparameter-search UI.
- Architecture preset scoring.
- Run grouping, filtering, comparison, export, and import.
- Budget-aware search execution with explicit user confirmation.

## Out Of Scope

- Pretrained transformer import and fine tuning. That belongs with the
  `done19.md` follow-up scope.
- DataLoader runtime ownership for `validation_freq`, `grad_accum_steps`,
  `seed`, `pin_memory`, and `log_interval`. That belongs in `tofix16.md`.
- Replacing the lightweight `TrainingRunComparisonRecord`.
- Running automatic search before manual queued experiments work.

## Design Guardrails

- Keep `TrainingRunComparisonRecord` as a lightweight completed-run event or UI
  row. Add persistence as a separate layer.
- Use a typed, versioned schema for persistent experiment records.
- Keep search orchestration separate from `TrainingExecutor`; the executor
  should run one configured training job, not own global search policy.
- Store only truthful runtime facts. GPU, CPU, memory, checkpoint, split, and
  metric fields must come from runtime state, not inference.
- Require explicit user confirmation and budget limits before automatic search
  launches more than one training run.
- Prefer a manual candidate queue before adding optimization algorithms.

## Proposed Record Shape

At minimum, persistent experiment records should preserve:

- experiment id and run id,
- graph/model identity and version,
- dataset identity, split policy, and effective seed if owned,
- training policy and optimizer settings,
- checkpoint used at start,
- checkpoint produced at end,
- validation and test metric availability flags,
- final metrics and best metrics,
- run status,
- backend/device summary,
- timestamps and duration,
- error code/log reference if the run failed.

## Implementation Order

1. Audit existing run-comparison records, debugger run records, crash/debug log
   records, and hyperparameter-search UI/code sketches.
2. Define a versioned persistent experiment schema.
3. Add a local storage adapter.
4. Import completed `TrainingRunComparisonRecord` rows into the persistent
   ledger.
5. Add a browser UI for stored experiments and runs.
6. Add run grouping and benchmark-set creation.
7. Add manual candidate queue execution.
8. Add export/import for experiment ledgers.
9. Add automatic search only after manual orchestration and persistence are
   stable.
10. Add architecture preset scoring after benchmark grouping is real.

## Acceptance Criteria

- Completed runs persist across Studio sessions.
- Runs can be grouped under an experiment.
- Users can compare runs across model families without losing split,
  checkpoint, validation, or test metadata.
- Users can export and import an experiment ledger.
- Failed runs preserve status, error code, and log reference.
- Automatic search cannot start without explicit user confirmation and a
  visible budget.
- Tests cover schema versioning, persistence round trip, sorting/filtering,
  import from completed run records, and export/import.
