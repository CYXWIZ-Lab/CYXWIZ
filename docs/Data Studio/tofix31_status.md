# tofix31 status - lean decomposition complete

## Current status

`tofix31.md` was too broad to implement as one ticket. It covered multiple
future-facing capabilities with different risk profiles. The ticket has now
been decomposed into focused, validated slices.

## Completed slices

### `tofix31a` - LSTM ArrayFire profiling matrix

- Added repeated-run profiling controls for recurrent ArrayFire smoke testing.
- Added warmup/run-count/shape environment overrides.
- Added matrix-mode profiling across representative LSTM shapes.
- Added profiling-only backward eval-barrier interval override.
- Validated the focused recurrent profiling target and matrix runs.
- Kept production behavior unchanged because interval tuning was not
  consistently better across shapes.

### `tofix31b` - TransformerDecoder runtime contract

- Made Studio TransformerDecoder docs match the compiler/runtime boundary.
- Current support is decoder-only causal self-attention for tested LM-style
  stacks.
- Encoder-decoder Memory/cross-attention and autoregressive generation loops
  remain future contract work.
- Validated the focused graph compiler deferred-node test.

### `tofix31c` - Pretrained/import truthfulness boundary

- Tightened wording around imported/pretrained model support.
- Import dialog remains metadata/graph inspection, not trainable checkpoint
  import.
- AdamW docs no longer imply imported Transformer fine-tuning is first-class.
- Compiler already rejects unsupported imported/pretrained fine-tuning sketches.

### `tofix31d` - Pinned-memory boundary

- Kept `pin_memory` serialized for compatibility.
- Added a compiler warning issue when selected `DataLoader.pin_memory=true` is
  ignored.
- Added focused compiler-test assertion for the warning.
- Validated the focused graph compiler deferred-node test.

### `tofix31e` - Import dialog inspection-only controls

- Renamed import options to inspection options.
- Replaced active optimizer/training-history loading controls with
  inspection-only status text.
- Disabled strict matching and shape-mismatch controls until real
  import-to-training exists.
- Validated the full `cyxwiz-engine` GUI target build.

## What the engine has now

- Truthful TransformerDecoder boundary in compiler tests and Studio UI.
- Truthful pretrained/import boundary in compiler checks and import dialog UI.
- Truthful pinned-memory boundary through UI text, compiler warning issues, and
  tests.
- LSTM ArrayFire profiling harness for future performance decisions.
- No fake implementation of broad future features.

## What remains future work

These are intentionally not implemented by `tofix31`:

- Real pinned host-memory allocation and H2D transfer path.
- First-class seq2seq/cross-attention graph contract.
- Autoregressive generation runtime.
- Pretrained transformer checkpoint import and fine-tuning.
- Freeze/unfreeze optimizer ownership for imported weights.
- Arbitrary tokenizer/preprocessor packaging for imported pretrained models.

## Closure recommendation

Close `tofix31` as a planning/truthfulness/performance-baseline ticket.

Do not claim the broad future capabilities are implemented. The correct claim is
that the engine now exposes truthful boundaries and has focused follow-up
documents for each remaining implementation area.
