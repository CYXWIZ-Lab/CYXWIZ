# Data Studio follow-ups after tofix16

Status: planning notes for follow-up projects after the `tofix16.md`
runtime/truthfulness pass.

These are not loose ends from `tofix16.md`. They are larger projects that
need separate design, implementation, tests, and profiling before the UI or
compiler should claim support.

## 1. Real pinned host-memory backend

`pin_memory` is currently serialized for graph compatibility only. It is not a
runtime feature yet.

Pinned host memory means CPU memory pages are locked so the OS cannot move or
page them out. GPU drivers can transfer from pinned CPU buffers to device
memory more efficiently, and in some APIs can overlap host-to-device transfer
with compute.

What it is not:

- It is not just a boolean on the DataLoader.
- It is not normal `std::vector` or `new` allocation.
- It is not useful unless there is a real CPU-to-device transfer boundary.
- It does not make CPU preprocessing faster by itself.

Why `pin_memory` is currently not implemented:

- Current batchers build `Tensor` batches and hand them to the training path,
  but they do not own a pinned allocator.
- The runtime does not have an explicit host-pinned staging buffer abstraction.
- The training path does not expose a clear async copy/compute overlap point.
- Without those pieces, enabling `pin_memory=true` would be a UI lie and could
  add complexity without improving performance.

What a real implementation needs:

- A small pinned host-buffer abstraction owned by the backend/runtime.
- Platform/backend-specific allocation and free paths, for example CUDA pinned
  host allocation where CUDA is available.
- A fallback path to regular host memory when pinned allocation is unavailable.
- A clear ownership contract: which batcher owns the pinned buffer, when it is
  reused, and when the tensor view becomes invalid.
- Explicit CPU-to-device copy points so profiling can prove whether pinned
  memory helps.
- A benchmark comparing regular host memory versus pinned host memory on at
  least one realistic Arrow/Parquet or image/audio training workload.

Acceptance criteria:

- `pin_memory=true` changes an actual transfer path, not just logs or config.
- Unsupported backends degrade safely to regular host memory with a warning.
- Tests cover allocation fallback and batch lifecycle ownership.
- Profiling shows whether transfer latency or overlap improves on a real
  workload.
- Only after that should the DataLoader UI expose `pin_memory` as active.

## 2. LSTM ArrayFire performance phase 2

`tofix16.md` added `test_recurrent_af_profile_smoke` and one measured CPU BPTT
scratch-buffer optimization.

Current local result for the small CUDA smoke:

- Before scratch reuse: backward about 510 ms.
- After scratch reuse: backward about 74 ms.

Next work should stay measurement-gated.

Potential hotspots:

- AF-to-Tensor cache materialization after AF forward.
- Forward-side `eval()` / host-device synchronization overhead.
- CPU backward loops that still do unnecessary scalar work.
- Repeated tensor allocation in forward/backward caches.

Acceptance criteria:

- Add repeated-run timing or a small benchmark mode before deeper changes.
- Optimize one measured hotspot at a time.
- Preserve CPU/AF numerical behavior.
- Keep existing recurrent placement tests and LSTM/GRU smoke tests passing.

## 3. TransformerDecoder generation and seq2seq contract

Current support is decoder-only causal self-attention through
`TransformerDecoderModule::Forward(input)`.

The compiler now rejects selected training paths where `TransformerDecoder`
has connected `Memory`, because that would imply seq2seq/cross-attention
support that the graph/runtime do not yet own.

Required future contract:

- Encoder-memory graph edge semantics.
- Target input versus target label wiring.
- Shifted-token target construction.
- Causal-mask ownership and shape validation.
- Token-level cross-entropy and ignore-index behavior.
- Greedy or sampling generation inference path.
- Tiny smoke graph for a sequence-generation task.

Acceptance criteria:

- Decoder-only and seq2seq modes are explicit and separate.
- Connected `Memory` has tested runtime behavior.
- Generation is documented as generation/seq2seq, not encoder
  classification.

## 4. Pretrained transformer import and fine-tuning pipeline

Current import UI scope is model inspection plus `.cyxmodel` graph extraction.
It does not load checkpoint weights into a trainable Studio model.

Required future contract:

- Supported checkpoint format decision, for example native `.cyxmodel`,
  safetensors, ONNX weights, or another scoped format.
- Parameter-name mapping from checkpoint tensors to Studio layers.
- Shape validation and useful mismatch diagnostics.
- Tokenizer/vocabulary compatibility and packaging.
- Freeze/unfreeze ownership in optimizer parameter selection.
- Resume-training behavior for optimizer state, if supported.
- Minimal import smoke test before UI claims fine-tuning support.

Acceptance criteria:

- Importing weights changes a real trainable model.
- Unsupported checkpoint formats fail clearly.
- Freeze/unfreeze changes optimizer-visible parameters.
- Tokenizer/preprocessor package is loaded with the model, not guessed.
- UI transfer-learning controls are reintroduced only after the above is real.

## 5. Import dialog follow-up

The Import Model dialog currently avoids overclaiming. It inspects model
metadata and extracts graph JSON when available.

Future UI work should remain tied to runtime capability:

- Keep inspection mode as the default safe path.
- Add a separate "Import for training" path only when a trainable model import
  contract exists.
- Show format-specific blockers before import starts.
- Keep graph extraction separate from weight loading.
- Avoid exposing freeze/fine-tune controls until they affect real optimizer
  behavior.

Acceptance criteria:

- UI language distinguishes inspection, graph extraction, inference import,
  and training import.
- A successful training import cannot be produced by metadata probe alone.
- Logs and callbacks use the same truthful terminology as the UI.
