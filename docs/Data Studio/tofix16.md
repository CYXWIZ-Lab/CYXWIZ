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

## Priority 2 - Pinned Host Memory

`pin_memory` is serialized for future compatibility but current training
batchers ignore it.

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
