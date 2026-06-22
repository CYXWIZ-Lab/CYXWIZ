# To Fix 16 - Deferred Model Performance And Loader Throughput

**Created:** 2026-06-04
**Source:** Follow-up work split out from `done1.md` Priority 5 after
GRU, TransformerEncoder, LSTM baseline, and explicit `num_workers`
contract work were completed.

## Boundary

This file tracks larger performance/features that were intentionally not
mixed into the Priority 5 completion slice. These are not current
correctness blockers for the verified GRU, TransformerEncoder
text-classification, or existing synchronous batcher paths.

Do not mark any item complete based on wiring alone. Each item needs a
focused smoke/benchmark, a clear before/after result, and UI/runtime
wording that matches the actual behavior.

## Priority 1 - Async Data Prefetch

Current batchers can split work across synchronous per-batch workers, but
they join before returning the batch. `prefetch_factor` is serialized for
future compatibility but is currently ignored.

2026-06-11 first slice:

- Arrow training and validation now log aggregate batch-fetch timing
  (`avg_fetch` and `max_fetch`) once per epoch/evaluation pass. This is
  baseline instrumentation only; it does not implement prefetching or
  change batcher behavior.
- Use these timings to decide whether async prefetch is likely to help a
  specific graph. If fetch time is small relative to forward/backward time,
  prefetch should not be prioritized for that path.

2026-06-12 async prefetch slice:

- `prefetch_factor` is now compiled into `TrainingConfiguration`.
- Arrow and Parquet training setup can wrap train/validation/test
  `IBatcher` instances with a bounded async `PrefetchBatcher` queue.
- `prefetch_factor=0` keeps the old synchronous behavior. A positive value
  is the queue depth in completed batches.
- This overlaps batch construction with model compute, but does not change
  `num_workers`: worker threads inside Arrow/Parquet batchers still do
  synchronous per-batch conversion before each batch enters the queue.
- This does not implement pinned host memory.

2026-06-12 UI wording sync:

- DataLoader property help and compile summaries now describe `num_workers`
  and `prefetch_factor` separately: workers are synchronous per-batch
  conversion, while prefetch is the bounded async queue for supported
  Arrow/Parquet batchers.

2026-06-12 rich dialog parity:

- The DataLoader rich configuration dialog now reads, edits, resets, and
  saves `prefetch_factor` with the same semantics as the properties panel.
- Values below 0 are clamped to 0; dialog input is capped at 64 to avoid
  accidental oversized queues.

2026-06-12 Parquet prefetch coverage:

- Focused batcher setup tests now exercise Parquet prefetch wrapper
  ownership, sample-count preservation, queued batch consumption,
  end-of-epoch behavior, and reset.

**Goal:** add a real async prefetch queue only where it improves training
latency without making shutdown, cancellation, or dataset lifetime unsafe.

**Candidate paths:**
- ArrowDatasetBatcher
- ParquetArrowBatcher
- ImageDatasetBatcher
- AudioDatasetBatcher
- TextDatasetBatcher delegated Arrow path

**Completion criteria:**
- bounded queue with deterministic shutdown,
- clear cancellation behavior when training stops,
- no UI freeze or leaked worker threads,
- measured improvement on at least one realistic dataset,
- `prefetch_factor` UI/logs updated only after behavior is real.

## Priority 1.5 - Runtime Ownership For Deferred DataLoader Fields

`done12.md` intentionally left several DataLoader UI fields as partial/future
because the current runtime does not yet own their behavior end to end.

**Goal:** make these fields active only after `TrainingExecutor`, compiler
metadata, launcher summaries, dashboard reporting, and tests all agree on the
same semantics.

**Fields to design before activation:**
- `validation_freq`: define whether validation cadence is epoch-based,
  batch-based, or both, and how it affects best-checkpoint selection.
- `grad_accum_steps`: define loss scaling, optimizer-step timing, progress
  reporting, pause/stop behavior, and checkpoint boundaries.
- `seed`: define deterministic split, shuffle, batch-order, and backend RNG
  ownership across CPU and ArrayFire paths.
- `pin_memory`: keep tied to Priority 2; do not expose as active unless there
  is a real backend transfer behavior.
- `log_interval`: wire batch/epoch logging cadence without hiding true
  training progress or flooding the UI.

**Completion criteria:**
- compiler contract tests prove node values reach the runtime config,
- executor tests prove each field changes behavior when enabled,
- dashboard/log summaries show the effective values truthfully,
- unsupported backend behavior is explicit instead of silent,
- UI labels stop saying partial/future only after runtime tests pass.

2026-06-22 runtime policy slice:

- `validation_freq` is now runtime-owned as an epoch-based validation cadence.
  Values below 1 compile to 1, and the final epoch always validates so final
  metrics and best-checkpoint selection are not stale.
- Checkpoint saving and early stopping evaluate only on epochs where validation
  actually ran.
- `log_interval` is now runtime-owned for legacy, sequence, Arrow, and Parquet
  training epoch loops. Batch 1 is always logged; `log_interval=0` disables
  periodic follow-up batch logs.
- DataLoader properties, the rich DataLoader dialog, and compile summaries now
  describe `validation_freq` and `log_interval` as active runtime fields.
- Focused compiler and TrainingExecutor tests cover DataLoader propagation and
  scheduled validation history behavior.

2026-06-22 deterministic DataLoader seed slice:

- DataLoader `seed` is now compiled into `TrainingConfiguration` as
  `dataloader_seed`.
- Legacy, Arrow, Parquet, image, audio, and text DataLoader batchers now receive
  the compiled seed for deterministic shuffle/split order.
- Text DataLoader uses the same seed for tokenized train/validation/test
  partition assignment and delegated Arrow training-batch shuffling.
- Compile summaries and DataLoader property help now describe `seed` as an
  active DataLoader runtime field.

2026-06-22 gradient accumulation slice:

- `grad_accum_steps` is now compiled into `TrainingConfiguration`.
- `TrainingExecutor` accumulates model gradients across backward passes, averages
  them at the configured accumulation boundary, and applies one optimizer step.
- Final partial accumulation windows are stepped at epoch end so no completed
  batch is silently dropped.
- Runtime metrics now report `optimizer_step_count`, and focused executor
  coverage proves `grad_accum_steps=2` changes optimizer-step cadence.
- DataLoader properties, the rich DataLoader dialog, runtime logs, and compile
  summaries now describe `grad_accum_steps` as active.

2026-06-22 pin-memory decision slice:

- `pin_memory` remains serialized for graph compatibility, but is not presented
  as an active runtime feature because there is no current pinned host-memory
  allocator or owned async CPU-to-device transfer boundary in the training
  batchers.
- Inline DataLoader properties and the rich DataLoader dialog now show
  `pin_memory` as unsupported/read-only. Loaded graphs with
  `pin_memory=true` keep the saved value but receive an explicit unsupported
  warning instead of a normal toggle.
- The compiler warning now says `pin_memory=true` is unsupported by current
  batchers and will be ignored.

Still deferred:

- Real pinned host-memory transfers remain a separate Priority 2 backend
  project; implementing them requires an owned pinned allocator, explicit
  CPU-to-device transfer points, and before/after profiling.

## Priority 2 - Pinned Host Memory

`pin_memory` is serialized for graph compatibility but current training
batchers ignore it and the UI marks it unsupported.

**Goal:** decide whether pinned host-memory transfers are useful in the
current ArrayFire/Tensor flow and implement only if the backend can make
the promise real.

**Completion criteria:**
- backend capability detection,
- no-op or warning on unsupported backends,
- measured GPU transfer benefit on CUDA/OpenCL path,
- UI tooltip changed from reserved to active only after verification.

## Priority 3 - LSTM ArrayFire Performance

LSTM correctness is verified and the focused Debug backend baseline is
about 3.2 seconds for `cyxwiz-tests.exe "[lstm]"` on the current
workstation. There is no dedicated `[lstm][arrayfire]` tagged test yet.

**Goal:** profile real recurrent workloads before touching LSTM internals.

**Completion criteria:**
- add an AF-specific recurrent benchmark or tagged smoke test,
- identify whether the hotspot is layout conversion, gate math, cache
  construction, backward pass, or host/device sync,
- optimize one measured hotspot at a time,
- preserve CPU/AF numerical behavior and existing LSTM/GRU smoke tests.

2026-06-22 recurrent AF profiling slice:

- Added `test_recurrent_af_profile_smoke` as an explicit recurrent
  forward/backward profiling smoke target for the small LSTM shape that the
  shared placement estimator keeps ArrayFire-CUDA eligible.
- The smoke target reports active ArrayFire backend, placement reason,
  estimated CUDA formal-parameter pressure, forward timing, backward timing,
  and the current hotspot candidates:
  input projection, per-step gate math, AF-to-Tensor cache materialization,
  and CPU backward.
- First local CUDA smoke result for `batch=8, seq_len=8, input_size=16,
  hidden_size=8`: forward about 128 ms, backward about 510 ms. The next
  measured optimization should start with the backward/cache boundary rather
  than changing gate math blindly.
- No LSTM math was changed in this slice. Optimization remains gated on
  comparing repeated timings from this smoke target against real recurrent
  workloads.

## Priority 4 - TransformerDecoder And Generation

TransformerEncoder text classification is verified. Decoder/generation is
not.

**Goal:** implement TransformerDecoder only when there is a graph-level
training or inference contract for sequence generation.

**Completion criteria:**
- decoder module/API contract,
- graph node shape contract,
- causal mask behavior,
- smoke graph for at least one tiny sequence task,
- explicit documentation that this is generation/seq2seq, not the already
  completed encoder classifier path.

## Priority 5 - Pretrained Transformer Import / Fine-Tuning

No pretrained transformer import, checkpoint conversion, tokenizer
compatibility layer, or LLM fine-tuning contract exists yet.

**Goal:** design this as a separate model-import/fine-tuning project, not
as a small extension of the current TransformerEncoder classifier.

**Completion criteria:**
- supported checkpoint format chosen,
- tokenizer/vocabulary compatibility defined,
- parameter mapping and shape validation,
- memory/runtime constraints documented,
- one minimal import smoke test before any UI claims.
