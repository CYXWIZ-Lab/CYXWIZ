## 24) Phase 5 design register (observability, ops, and lifecycle quality)

### 24.1 Goal
Make runtime behavior observable and reproducible without adding heavyweight new infrastructure.

### 24.2 Execution trace and request identity
Minimum traceability for each run:
- `run_id` (unique per launch),
- `graph_id` (stable graph/project identifier),
- `compile_id` (cache key from compile input),
- `executor_mode` (legacy/arrow/parquet/external/sequence),
- `started_at` / `ended_at`.

Usage:
- every UI callback and major phase should include run identifiers,
- runtime failures should reference `compile_id` and `executor_mode`.

### 24.3 Callback contract
Callbacks currently carry progress and status; for Phase 5 they should be explicit:

```text
Event stream contract
  -> TRAINING_STARTED
  -> EPOCH_STARTED(epoch_idx)
  -> BATCH_PROCESSED(step_idx, loss, metrics...)
  -> VALIDATION_RESULT(epoch_idx, metric_map)
  -> CHECKPOINT_SAVED(path, metric, mode)
  -> TRAINING_PAUSED/RESUMED
  -> TRAINING_STOPPED(reason)
  -> TRAINING_ERROR(error_code, message)
  -> TRAINING_COMPLETED(best_metric, epochs_done)
```

Event contracts should include:
- run_id,
- severity,
- source component (`compiler`, `preflight`, `materializer`, `executor`, `batcher`),
- optional stack/diagnostic context.

### 24.4 Logging policy
- Use phase-separated logs:
  - `compile` logs: graph extraction, connector decisions, warnings.
  - `materializer` logs: source detection, operator mapping, blocked nodes.
  - `training` logs: epoch/batch progression, validation cadence, stop reasons.
- Do not log full payload tensors in default mode.
- Keep logs redact-safe for paths/connection secrets and credentials.

### 24.5 Reproducibility contract
At minimum, store:
- compiler version/hash,
- dataset split seed,
- random seeds for:
  - dataset order,
  - model initialization,
  - augmentation operations,
- backend placement selection.

Recommendations:
- if seeds are omitted, mark run as `non-deterministic` explicitly.
- checkpoint restore should restore with matching seed context and split policy.

### 24.6 Checkpoint and best-model policy
Current behavior includes checkpoint creation and best-model restore logic.
Phase-5 hardening:
- define:
  - checkpoint cadence source (`epoch interval` / `batch interval`),
  - naming template and retention strategy,
  - restore rule:
    - `restore_last`,
    - `restore_best_metric`,
    - `no_restore`.
- record in run metadata:
  - path,
  - metric key used,
  - value used for restore decision.

### 24.7 Failure code taxonomy
Introduce an explicit failure code space grouped by layer:
- `C_*` compile codes
- `P_*` preflight codes
- `M_*` materialization codes
- `E_*` executor codes
- `R_*` resource/runtime codes

Suggested minimal set:
- `C_MISSING_SOURCE`, `C_MISSING_LOSS`, `C_MISSING_OPTIMIZER`
- `C_CYCLE_DETECTED`, `C_PIN_MISMATCH`, `C_SHAPE_CONFLICT`
- `P_DATA_READY_FAIL`, `P_LABEL_MISSING`
- `M_UNSUPPORTED_NODE`, `M_UNSUPPORTED_SOURCE`, `M_OP_MAPPING_FAIL`
- `E_BATCHER_FAIL`, `E_BACKEND_FAIL`, `E_EPOCH_FAIL`, `E_DIVERGED`
- `R_CUDA_UNAVAILABLE`, `R_OOM_RISK`, `R_IO_FAILURE`

Policy:
- every code maps to one concise user message + one recovery hint.

### 24.7.1 Failure recovery playbook

```text
Code                   Failure class   Recovery
---------------------  --------------  --------------------------------------------
C_MISSING_SOURCE       compile         Add a valid dataset/source node and re-run compile.
C_MISSING_LOSS         compile         Connect model output to a supported loss node.
C_MISSING_OPTIMIZER    compile         Attach optimizer and wire loss->optimizer.
C_CYCLE_DETECTED       compile         Remove cycle and compile as DAG.
C_PIN_MISMATCH         compile         Fix source/target pin directions.
P_LABEL_MISSING        preflight       Re-map labels source and re-run preflight.
M_UNSUPPORTED_NODE     materializer    Remove unsupported preprocessing node from active path.
M_OP_MAPPING_FAIL      materializer    Replace node family with supported operation or plugin extension.
E_BATCHER_FAIL         executor        Retry with supported batcher source mode.
E_BACKEND_FAIL         executor        Check backend diagnostics and retry on CPU path.
E_EPOCH_FAIL           runtime         Reduce learning-rate/gradient scale; resume if safe.
R_OOM_RISK            resource        Reduce batch/hidden size and enable checkpointed runs.
R_IO_FAILURE           runtime         Validate filesystem paths and dataset availability.
```

### 24.8 Operational health rules
1. Every blocked launch writes a deterministic reason.
2. Every recoverable runtime error should emit both technical message and user action hint.
3. Pause/stop transitions are explicit state events.
4. Every successful run writes:
   - train/val final metric summary,
   - best checkpoint identity,
   - mode and backend placement outcome.

### 24.9 Ops readiness and hardening plan
- add standardized metric schema (`loss`, `accuracy`, `val_loss`, `val_accuracy` plus optional custom metrics),
- add lightweight health counters:
  - `pause_count`,
  - `stop_request_count`,
  - `failed_batches`,
  - `checkpoint_count`,
- expose counters in UI summary panel.

### 24.10 Phase 5 completion criteria
- run_id and phase-scoped event metadata is in place for compile/preflight/materializer/executor events,
- failure codes are referenced in compile popup, preflight UI, and runtime UI message rendering,
- checkpoint restore mode and seed metadata are persisted in run summary,
- logging no longer depends on implicit state and includes component + severity.
