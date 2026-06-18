# To Fix 12 - Fine Tuning and Sentiment Model Improvement

This note defines how we should think about fine tuning in CyxWiz and
what we can improve next in the engine without confusing retraining,
checkpointing, and pretrained-model fine tuning.

Current relationship to `tofix19.md`: this file tracks current
sentiment/text-classifier improvement strategy. Pretrained transformer
import, decoder/generative training, and broad PyTorch-parity model
families are tracked in `tofix19.md`.

Short version:
- In the current engine, fine tuning means training the existing graph
  better.
- We can already tune architecture, optimizer settings, checkpoints,
  validation behavior, and test evaluation.
- We do not yet have full pretrained transformer fine tuning inside the
  engine.
- The next work should improve comparability, control, and evaluation
  rather than hardcoding more training behavior into the backend.

## What Fine Tuning Means Here

There are two different meanings that should stay separate.

### 1. Engine-level fine tuning

This is what the current CyxWiz engine already does:
- train the current graph from scratch
- resume from saved checkpoints
- change node parameters and training configuration
- compare different runs on the same dataset split
- export the best checkpoint from training

This is the practical meaning of fine tuning for the current sentiment
project.

### 2. Pretrained-model fine tuning

This is the classic ML meaning:
- load a pretrained model such as BERT, RoBERTa, DistilBERT, or
  BERTweet
- attach a classification head
- freeze some layers or train all layers
- adapt the pretrained weights to the target sentiment dataset

The current backend does not yet provide that as a first-class path.

## Current Engine State

Already implemented or available:
- train/validation/test split for the text dataset
- DataLoader UI controls for training policy
- save-best-checkpoint support
- early stopping support
- pause, continue, and cooperative early stop in the training dashboard
- test execution path after training
- GRU and LSTM recurrent layers
- bidirectional recurrent support in the graph model
- Studio debugger run records for training progress and backend failures
- model export from the current trained session

That means the engine can already support real training experiments,
but it still needs better control and comparison tooling.

## What We Can Test Now

We can already run controlled comparisons on the same split:

1. GRU vs LSTM
2. bidirectional on vs off
3. one layer vs multiple layers
4. different hidden sizes
5. different dropout values
6. different batch sizes
7. different learning rates
8. different `num_workers`
9. early stopping patience values
10. best validation checkpoint vs final epoch checkpoint

For sentiment specifically, the deciding metrics should be:
- validation accuracy
- validation loss
- held-out test accuracy
- training time per epoch
- total training stability

Training accuracy alone is not enough.

## What We Should Improve In The Engine

### 1. Make training control fully user-owned

Training policy should stay in the UI and graph settings, not hidden as
hardcoded backend values.

The backend should accept:
- batch size
- epochs
- learning rate
- dropout
- hidden size
- number of layers
- bidirectional flag
- num workers
- save-best-checkpoint flag
- early stopping patience
- checkpoint directory

### 2. Make evaluation explicit

After training, the user should be able to run:
- validation evaluation
- test-set evaluation
- sample-level inference

The engine should show:
- the checkpoint used
- the exact split used
- the resulting accuracy and loss

### 3. Keep best-checkpoint behavior visible

When training improves and then degrades, the engine should make it
obvious which checkpoint is best.

The deployable model should come from:
- best validation epoch, or
- a user-selected checkpoint

not just the last batch of the last epoch.

### 4. Add comparison tooling

The Studio debugger and training dashboard should let us compare runs:
- GRU vs LSTM
- different bidirectional settings
- different training lengths
- different validation curves
- different test outcomes

### 5. Separate speed problems from model quality

If a model is slow, the debugger should tell us whether the bottleneck
is:
- GPU path
- CPU fallback
- preprocessing
- recurrent layer shape/layout
- UI polling

That matters because a slow model is not always a bad model.

## What We Still Do Not Have

Not yet implemented as a first-class engine feature:
- pretrained transformer import and fine tuning
- layer freezing controls
- per-layer learning rate schedules
- automatic hyperparameter search
- benchmark-run management across multiple model families
- architecture-specific preset scoring for transformer models

## Suggested Next Order

1. Keep the current train/val/test split and checkpoint flow stable.
2. Compare GRU and LSTM on the same dataset and same training budget.
3. Keep best-validation checkpointing as the default deployment path.
4. Use the held-out test set only after training is complete.
5. Expose more training controls in the UI if any setting still feels
   hardcoded.
6. Decide later whether the next model family should be transformer-
   based fine tuning rather than more recurrent tuning.

## How To Use The Implemented Workflow

1. Build a graph with a dataset source, preprocessing nodes if needed, model
   layers, loss, optimizer, and DataLoader.
2. In the DataLoader dialog, set the runtime-owned training policy:
   `epochs`, `batch_size`, `num_workers`, `prefetch_factor`,
   `save_best_checkpoint`, `early_stopping_patience`, and `checkpoint_dir`.
3. Keep train/validation/test split settings fixed when comparing model
   changes so each run uses the same evaluation basis.
4. Start training from Studio and monitor the Training Dashboard.
5. Use the dashboard curves to inspect train/validation behavior during the
   run, including best validation points and generalization gap.
6. After training completes, open the Run Comparison table in the Training
   Dashboard.
7. Compare runs by dataset, domain, sequence mode, split ratios, actual split
   sample counts, architecture, training policy, checkpoint policy, best
   validation metrics, held-out test metrics, checkpoint used, elapsed time,
   and run status.
8. Use Export Run CSV from the Run Comparison table to persist the in-session
   comparison ledger.
9. Treat `final epoch model state` as meaning no best checkpoint was restored;
   explicit checkpoint paths mean the executor restored that checkpoint for
   final evaluation.
10. Do not treat partial/future DataLoader fields such as `validation_freq`,
    `grad_accum_steps`, `seed`, `pin_memory`, and `log_interval` as runtime
    behavior until they are explicitly implemented and tested.

## What This Note Is Not

This is not the Studio debugger architecture note in `tofix9.md`.
It is also not the memory cleanup note in `tofix10.md`, or the text
vocabulary workflow note in `tofix11.md`.

This file exists to keep model-improvement strategy explicit so we can
compare runs and extend the engine without mixing the training policy
into unrelated backend changes.

## 2026-06-18 Engine Truth Audit

Status: active target, but the implementation target is narrower than the
original wording suggests.

What the engine already has:
- `TrainingConfiguration` already carries train/validation/test split ratios,
  DataLoader loop settings, optimizer settings, checkpoint policy, early
  stopping, and final held-out test metrics.
- `GraphCompiler` already reads DataSplit and DataLoader parameters into the
  compiled training configuration, including `epochs`, `batch_size`,
  `num_workers`, `prefetch_factor`, `save_best_checkpoint`,
  `early_stopping_patience`, and `checkpoint_dir`.
- `TrainingExecutor` already creates train/validation/test batchers for Arrow
  and Parquet, runs validation during training, saves the best validation
  checkpoint when enabled, restores the best checkpoint before final test
  evaluation, and writes final test loss/accuracy into `TrainingMetrics`.
- The training plot panel already shows train/validation curves, best
  validation summary, custom metrics, and CSV export.
- The test results panel already displays test accuracy/loss, confusion
  matrix, per-class metrics, predictions, and export actions.

Resolved by the 2026-06-18 batches:
- `DataLoaderDialog` now exposes the runtime-supported training-loop fields
  targeted here: `epochs`, `save_best_checkpoint`,
  `early_stopping_patience`, and `checkpoint_dir`.
- Runtime-owned DataLoader fields are covered by focused compiler/launcher
  contract tests so training dispatch preserves the intended policy.
- Partial/future DataLoader fields are labeled truthfully instead of being
  presented as fully owned runtime behavior.
- A lightweight first-class run-comparison ledger now exists for completed
  runs, with CSV export and an in-session Training Dashboard table.
- The ledger is general-purpose rather than GRU/LSTM-only, and includes
  dataset/domain identity, split policy and sample counts, architecture,
  training policy, checkpoint policy, checkpoint used, validation/test metric
  availability, best validation epochs, final metrics, elapsed time, and
  deterministic sorting.

Still intentionally out of scope for this note:
- Pretrained transformer fine tuning, freeze/unfreeze controls, per-layer
  learning-rate schedules, and automatic hyperparameter search.
- A persistent experiment database or heavyweight benchmark manager.
- Making partial/future DataLoader fields such as `validation_freq`,
  `grad_accum_steps`, `seed`, `pin_memory`, and `log_interval` runtime-owned
  before their executor semantics are designed and tested.

Lean implementation target:
1. Done: do not rebuild the training loop.
2. Done: do not add automatic hyperparameter search here.
3. Done: do not add pretrained transformer fine tuning here; that belongs to
   `tofix19.md`.
4. Done: align the DataLoader dialog with the already-supported runtime
   contract targeted by this note.
5. Done: add validation/tests that prove compiler/launcher policy preservation.
6. Done: add the smallest run-comparison artifact needed to compare completed
   runs by config, split, best validation metrics, final test metrics,
   checkpoint used, and elapsed time.

Target batches:
1. Done: DataLoader dialog parity: expose `epochs`, `save_best_checkpoint`,
   `early_stopping_patience`, and `checkpoint_dir` in the dialog because those
   are already compiler/runtime-supported.
2. Done: runtime truth labeling: clearly mark UI-only/future fields such as
   `grad_accum_steps`, `validation_freq`, `pin_memory`, and `log_interval` if
   they are not yet consumed by `TrainingExecutor`.
3. Done: contract tests: cover DataLoader parameter propagation from node
   params to `TrainingConfiguration` and from launch result to dispatch.
4. Done: run comparison ledger: add a lightweight local record/table/export for
   completed training runs before building any heavier experiment manager.
   This ledger must be general enough for all model families. GRU/LSTM
   recurrent fields are optional details, not the core schema.

## 2026-06-18 Progress

- Implemented DataLoader dialog parity for runtime-supported fields:
  `epochs`, `save_best_checkpoint`, `early_stopping_patience`, and
  `checkpoint_dir`.
- Marked partial/future DataLoader property fields truthfully in the inline
  properties editor: `grad_accum_steps`, DataLoader `seed`, `pin_memory`,
  `log_interval`, and `validation_freq`.
- Extended the GUI training launch contract test so checkpoint policy fields
  are preserved into training dispatch along with epochs and batch size.
- Added the first lightweight run-comparison contract:
  `TrainingRunComparisonRecord`, stable CSV header/row helpers, best
  validation metric extraction, metric-presence flags, checkpoint-used field,
  run status, generic architecture summary, primary layer type, model layer
  count, and optional recurrent-family fields for GRU/LSTM comparisons.
- Added a minimal CSV write helper for the run-comparison ledger so completed
  runs can be persisted without introducing a larger experiment-manager
  subsystem.
- Added deterministic comparison sorting that prefers held-out test accuracy
  when present, then validation availability/accuracy/loss, then elapsed time.
  This keeps run ranking explicit and small.
- Added a minimal Training Dashboard UX for run comparison: completed runs are
  recorded in-session, ranked, shown in a table, and exportable as CSV. This
  keeps comparison visible without introducing a separate experiment-manager
  subsystem.
- Polished the run-comparison table so the visible UX separates model family
  from architecture summary and shows elapsed training time.
- Made checkpoint visibility explicit in run-comparison records: explicit
  checkpoint paths are shown when known, explicit checkpoint roots are shown
  when configured, and otherwise the default `.cyxwiz/checkpoints` run folder
  is labeled instead of appearing empty.
- Wired the actual restored best-checkpoint path into `TrainingMetrics` so
  run-comparison records can show the checkpoint used for final evaluation
  when `TrainingExecutor` restores one.
- Added train/validation/test split ratios to the run-comparison record, CSV
  export, focused contract test, and Training Dashboard table so completed
  runs show the exact split policy used for the comparison.
- Exposed checkpoint policy in the Training Dashboard comparison table by
  showing whether best-checkpoint saving was enabled and the configured early
  stopping patience beside the resolved checkpoint path.
- Extended the compiler contract test for selected DataLoader nodes so runtime
  training-loop fields are proven to flow into `TrainingConfiguration`:
  `epochs`, `batch_size`, `num_workers`, `prefetch_factor`,
  `save_best_checkpoint`, `early_stopping_patience`, and `checkpoint_dir`.
- Made metric availability explicit by adding validation/test presence flags
  to `TrainingMetrics` and using those flags in run comparison. This avoids
  treating zero-valued loss or accuracy as "metrics missing."
- Added actual train/validation/test sample counts to `TrainingMetrics`, the
  run-comparison record, CSV export, focused comparison test, and Training
  Dashboard table so split reporting is not limited to configured ratios.
- Corrected checkpoint display truthfulness: when no checkpoint was restored
  and best-checkpoint saving is disabled, run comparison now reports the final
  epoch model state instead of implying a default checkpoint path was used.
- Added best-validation epoch reporting to the run-comparison record, CSV
  export, focused comparison test, and Training Dashboard table so users can
  see which epoch produced the best validation loss and accuracy.
- Added preprocessing-domain visibility to run-comparison records, CSV export,
  focused comparison test, and the Training Dashboard table so comparisons
  distinguish tabular, text, image, audio, time-series, and general runs.
- Added sequence-batch mode visibility to run-comparison records, CSV export,
  focused comparison test, and the Training Dashboard table so sequence tasks
  are not conflated with plain sample-level classification runs.
- Made run-comparison sorting deterministic for identical metrics by using
  `run_id` as the final tie-breaker after test metrics, validation metrics,
  and elapsed time.

Generalized run-comparison rule:
- The comparison record is for every training run, not just sentiment GRU/LSTM
  experiments.
- Core fields must describe dataset, architecture summary, training settings,
  checkpoint policy, checkpoint used, validation/test metric availability,
  final metrics, run status, and elapsed time.
- Architecture-specific fields such as recurrent `hidden_size`, `num_layers`,
  and `bidirectional` are optional detail fields. They must not define the
  whole schema.
