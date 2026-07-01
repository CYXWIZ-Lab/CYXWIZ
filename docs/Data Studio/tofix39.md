# tofix39 - Engine Computation Truth, Numerical Parity, and Training Lifecycle

## Status

Open. Current sentiment-stack computation evidence is complete; broad whole-engine computation-truth coverage remains active.

## Why this exists

The sentiment TF-IDF graph exposed a broader engine requirement: the system must prove that its computation is numerically correct, stable, and lifecycle-honest before we trust training accuracy.

The narrow bug was TF-IDF materialization allocating a full `rows x full_vocab` matrix before `max_features`. That path is now bounded, but the observed training result still raises broader questions:

- TF-IDF + Dense + CrossEntropy trains, but accuracy stayed near random for 7 classes.
- Current run ended with `status=completed` at epoch 8 even though the graph configured `epochs=20`.
- There was no explicit failure reason, cancellation marker, or user stop in the trace.
- `current_run.json` recorded `epoch=8`, `epochs=20`, `status=completed`.

This ticket is the broad "truth computation" effort for the engine.

## Current conclusion for sentiment graph

For the current TF-IDF sentiment graph, we now have enough narrow evidence to proceed with training/model-quality investigation:

- TF-IDF bounded materialization is shape-correct.
- TF-IDF values match deterministic reference math on a tiny corpus.
- Label encoding is deterministic.
- CrossEntropy forward loss matches deterministic reference math.
- Linear forward and backward math match deterministic reference math.
- AdamW first-step update matches deterministic reference math.
- Early-stop reporting now distinguishes `early_stopped` from `completed`.
- TF-IDF feature selection now keeps high corpus/document-frequency terms before rare-IDF tie-breaking, instead of selecting the rarest terms first.

This does not prove the whole engine yet, but it reduces suspicion that the current sentiment graph is failing because of basic TF-IDF, Linear, CrossEntropy, or AdamW arithmetic.

If sentiment accuracy remains poor, next likely areas are:

- dataset label distribution and class imbalance
- class balancing / weighted sampler behavior
- feature selection quality from TF-IDF `max_features`
- train/validation/test split behavior
- metric aggregation over full epochs
- model capacity and hyperparameters

## Sentiment graph update for next accuracy run

Updated on 2026-06-30:

- Fixed `TFIDFVectorizerOperator` feature selection:
  - before: selected highest-IDF rare terms first, which can feed mostly one-off noise words to the classifier.
  - now: selects by corpus frequency, then document frequency, then IDF/name tie-breakers.
- Updated `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph`:
  - TF-IDF max features: `2000 -> 5000`
  - DataLoader batch size: `64 -> 128`
  - epochs: `20 -> 30`
  - early stopping patience: `5 -> 8`
  - validation frequency remains `1`
  - best checkpoint remains enabled
- Rebuilt and passed:
  - `test_computation_truth_tfidf_loss`
  - Release `cyxwiz-engine`

Expected effect:

- The next training run should receive more predictive TF-IDF features.
- If accuracy remains near random, the next investigation should move to label distribution, sampler/class balancing, split behavior, or metric aggregation rather than basic vectorizer/loss/optimizer arithmetic.

## Goal

Build a deterministic computation-truth test suite that compares CyxWiz engine math and training behavior against stable references.

Primary references:

- PyTorch for tensor math, Dense/Linear, activations, CrossEntropy, BCE, optimizers, and gradient steps.
- scikit-learn for TF-IDF vectorization when available.
- Small in-repo reference implementations when external packages are unavailable.

## Test home

Use:

`cyxwiz-engine/tests/computation_truth/`

This is engine-level numerical correctness, so it belongs under engine tests, not examples or GUI docs.

## Required coverage

This section is the broad whole-engine target and remains pending beyond the current sentiment-stack evidence.

### 1. TF-IDF materialization truth

Status: partially done for a deterministic tiny corpus.

Validate on tiny fixed corpora:

- Done: vocabulary selection is deterministic for the covered case.
- Done: `max_features` bounds output width before dense allocation.
- Done: output shape is exactly `[rows, max_features]` plus optional `y`.
- Pending: l1/l2 normalization reference coverage.
- Done: `use_idf=true` and `smooth_idf=true` covered.
- Pending: `use_idf=false` and `smooth_idf=false` variants.
- Done: label encoding is deterministic and stable for the covered case.
- Done by implementation: no full `rows x full_vocab` dense allocation happens internally.

### 2. Dense and loss parity

Status: partially done for Linear and CrossEntropy.

For fixed inputs, labels, weights, and biases:

- Done: Linear forward output matches deterministic reference.
- Done: Linear backward gradients match deterministic reference.
- Pending: DenseLayer-specific parity if we decide to test both DenseLayer and LinearLayer.
- Pending: ReLU/Dropout eval-mode behavior.
- Done: CrossEntropy forward loss matches deterministic reference equivalent to PyTorch semantics.
- Done: Softmax/logit interpretation covered through CrossEntropy reference.
- Pending: Accuracy calculation over batches/epochs.

### 3. Backward and optimizer parity

Status: partially done for Linear backward and AdamW first step.

For one small batch:

- Done: Linear weight gradients match deterministic reference.
- Done: Linear bias gradients match deterministic reference.
- Pending: DenseLayer-specific gradients if required.
- Done: AdamW one-step update matches documented reference.
- Pending: Adam one-step update.
- Pending: multi-step AdamW state progression.
- Pending: gradient accumulation behavior for configured `grad_accum_steps`.

### 4. Training lifecycle truth

Status: implementation fix landed; automated lifecycle test pending.

The engine must prove:

- Pending test: configured `epochs=N` executes exactly N epochs unless stopped by an explicit reason.
- Done implementation: early stopping records a clear reason.
- Pending test: early stopping records the stopping epoch.
- Done implementation: user cancellation records a clear reason.
- Done implementation: plugin early stop records a clear reason.
- Pending: best-checkpoint restore must not make a 20-epoch run look like it naturally completed at an earlier epoch.
- Done implementation: final `current_run.json` distinguishes:
  - completed all epochs
  - early stopped
  - cancelled
  - failed
  - restored best checkpoint

### 5. GPU/CPU parity

Status: barely started.

For small deterministic operations:

- Partially observed: Linear parity test ran while ArrayFire GPU was active.
- Pending: explicit CPU vs GPU forward value comparison.
- Pending: CPU vs GPU loss value comparison.
- Pending: CPU vs GPU one-step optimizer comparison where supported.
- Pending: unsupported GPU fallback is explicit in the trace.

## Immediate investigation from current run

Observed on 2026-06-30:

- Graph: `examples/cyxgraph/Sentiment analysis/sentiment_analysis_tfidf_mlp_classifier.cyxgraph`
- Configured graph epochs: `20`
- Debug run: `build/bin/Release/.cyxwiz/debug_runs/current_run.json`
- Final recorded state: `epoch=8`, `epochs=20`, `status=completed`
- Trace tail: epoch 8, batch `341/341`, `EpochComplete`, status `ok`
- Final training accuracy around `17.27%`
- 7-class random baseline is about `14.29%`
- Loss around `1.88`; random-softmax baseline is `ln(7) ~= 1.946`

This must be explained before using the result as model-quality evidence.

## Stop-feedback fix landed

Implemented on 2026-06-30:

- Early stopping is no longer recorded as generic `completed`.
- Training metrics now carry:
  - `terminal_status`
  - `terminal_reason`
- Crash/debug run JSON now carries:
  - `terminal_reason`
  - `failure_reason`
- Recorder status now distinguishes:
  - `completed`
  - `early_stopped`
  - `cancelled`
  - `failed`
- Validation plateau early stop records:
  - `status=early_stopped`
  - `terminal_reason=validation_loss_plateau_patience_N`
- Plugin early stop records:
  - `status=early_stopped`
  - `terminal_reason=plugin_requested_early_stop`
- Normal completion records:
  - `status=completed`
  - `terminal_reason=completed_all_epochs`
- User cancellation records:
  - `status=cancelled`
  - `terminal_reason=user_cancelled`

Files changed:

- `cyxwiz-engine/src/core/crash_run_recorder.h`
- `cyxwiz-engine/src/core/crash_run_recorder.cpp`
- `cyxwiz-engine/src/core/training_executor.h`
- `cyxwiz-engine/src/core/training_executor.cpp`

Release engine rebuild succeeded after the fix.

Remaining lifecycle follow-up:

- Add a deterministic test that proves a configured `epochs=20`, `early_stopping_patience=5` run records `early_stopped`, not `completed`.
- Add UI copy that displays `terminal_reason` clearly in the training dashboard/debugger.

## Computation-truth harness started

Implemented on 2026-06-30:

- Created `cyxwiz-engine/tests/computation_truth/`.
- Added `test_computation_truth_tfidf_loss`.
- Added CMake target `test_computation_truth_tfidf_loss`.
- Test validates:
  - TF-IDF bounded materialization shape.
  - TF-IDF does not emit beyond `max_features`.
  - TF-IDF deterministic values for a tiny corpus.
  - deterministic string label encoding.
  - CrossEntropy mean loss against hand reference math equivalent to PyTorch semantics.
  - Linear forward output against hand reference `input @ weight^T + bias`.
  - Linear backward gradients against hand reference:
    - grad input
    - mean-reduced grad weight
    - mean-reduced grad bias
  - AdamW first-step parameter update against hand reference:
    - decoupled weight decay
    - first moment
    - second moment
    - bias correction

Validated:

```powershell
cmake --build build --config Release --target test_computation_truth_tfidf_loss -- /m:4 /v:minimal
build\bin\Release\test_computation_truth_tfidf_loss.exe
```

Result:

```text
Computation truth TF-IDF + CrossEntropy + Linear + AdamW checks passed
```

Latest observed run also exercised `LinearLayer` with ArrayFire GPU active.

## Current done list

- Created computation-truth folder.
- Added `test_computation_truth_tfidf_loss`.
- Added CMake target.
- Built and ran the target successfully.
- Implemented bounded TF-IDF materialization before this ticket.
- Proved current TF-IDF tiny-corpus values.
- Proved current CrossEntropy forward loss.
- Proved current Linear forward/backward math.
- Proved current AdamW first-step update.
- Fixed early-stop terminal status/reason reporting.
- Rebuilt Release engine after lifecycle fix.

## Current pending list

- Add lifecycle automated test for early stop terminal status and reason.
- Add full epoch accuracy aggregation truth test.
- Add train/validation/test split truth test.
- Add class balancing / weighted sampler truth test.
- Add l1/l2 TF-IDF normalization variants.
- Add `use_idf=false` and `smooth_idf=false` TF-IDF variants.
- Add ReLU/Dropout eval-mode parity.
- Add Adam and multi-step AdamW optimizer parity.
- Add gradient accumulation parity.
- Add CPU/GPU explicit parity matrix.
- Add optional Python-side PyTorch/sklearn parity script.
- Surface `terminal_reason` in training dashboard/debugger UI.
- Extend this whole-engine truth matrix to all computational nodes over time.

Build note:

- Current CMake configuration reports `PyTorch/LibTorch: OFF`.
- Until LibTorch is available, C++ parity tests should use exact hand references.
- Python-side PyTorch/sklearn comparison can still be added as an optional external parity script.

## Acceptance criteria

- A focused computation-truth test binary exists and is easy to run.
- Tests can run without the GUI.
- Tests print expected vs actual values on failure.
- TF-IDF parity covers materialized feature values, not only shape.
- Dense/CrossEntropy parity proves values against PyTorch or a deterministic reference.
- Training lifecycle test fails if a run configured for 20 epochs reports completed at 8 without an explicit stop reason.
- Training lifecycle UI/debugger displays `terminal_reason` in plain user-facing language.
- Debug output records the terminal reason unambiguously.

## Notes

This ticket is intentionally broader than the sentiment graph. The sentiment model is only the first visible symptom. The engine needs a general truth harness so future GPU, JIT, optimizer, and preprocessing work can be trusted.
