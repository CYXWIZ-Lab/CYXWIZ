# Track 52 - Transformer Backward Parity And Training Correctness

## Status

In progress.

## Ticket

`tofix52.md`

## Completed Slices

### Single-Block Backward Parity

- Fixed post-LN residual/LayerNorm gradient routing in
  `TransformerEncoderLayer::Backward`.
- Fixed post-LN residual/LayerNorm gradient routing in single-input
  `TransformerDecoderLayer::Backward`.
- Kept the implementation inside the existing CPU-backed transformer layer
  path; no GPU kernels or new runtime dependencies were added.
- Added deterministic computation-truth checks for encoder and causal decoder
  input gradients plus representative attention, layer norm, and feed-forward
  parameter gradients.

### Two-Block Encoder Stack Backward Parity

- Added a two-block `SequentialModel` encoder stack truth test.
- Set both transformer blocks through the normal `layerN.*` parameter naming
  contract.
- Verified stack input gradient and representative layer-indexed gradients from
  both blocks against PyTorch-derived constants.

### Two-Block Decoder Stack Backward Parity

- Added a two-block `SequentialModel` causal decoder stack truth test.
- Set both transformer decoder blocks through the normal `layerN.*` parameter
  naming contract.
- Verified stack input gradient and representative layer-indexed gradients from
  both blocks against PyTorch-derived constants.

### Masked Attention Backward Parity

- Extended the existing `MultiHeadAttentionLayer` mask fixture from forward-only
  checks into backward parity.
- Verified masked attention input gradient plus all projection and bias
  gradients against PyTorch-derived constants.

### Tiny Transformer Training-Step Sanity

- Added a deterministic `SequentialModel` fixture with a
  `TransformerEncoderModule` followed by `TimeDistributedDenseModule`.
- Verified token-shaped `CrossEntropyLoss` with `Reduction::Mean`,
  `ignore_index=-100`, and `label_smoothing=0.1` against a PyTorch-derived
  scalar loss constant.
- Verified loss backward produces encoder attention gradients and token-head
  gradients through the normal layer-indexed gradient contract.
- Verified one SGD step keeps the updated loss finite and lowers the smoothed
  token cross-entropy on the tiny fixture.

## Validation

Passed:

```powershell
cmake --build D:\Dev\CyxWiz_Claude\build --config Debug --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal
D:\Dev\CyxWiz_Claude\build\bin\Debug\test_computation_truth_transformer_primitives.exe
```

Observed test output:

```text
Computation truth transformer primitive checks passed
```

Related regression set already passed after the transformer backward fix:

```powershell
D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_transformer_encoder_roundtrip.exe
D:\Dev\CyxWiz_Claude\build\bin\Debug\test_cyxmodel_transformer_decoder_roundtrip.exe
D:\Dev\CyxWiz_Claude\build\bin\Debug\test_debug_executor.exe
```

## Remaining Scope

The ticket is not complete yet. Remaining work:

- Cross-attention decoder backward parity, if claimed by graph/runtime paths.
- Documentation of unsupported backward paths as fail-closed or CPU-only.
- Optional broader loss-training coverage beyond this ticket's tiny transformer
  sanity fixture, only if the ticket owner wants reduction/ignore-index/
  label-smoothing matrix coverage outside transformer training.

## Next Pickup

Recommended next slice:

1. Decide whether cross-attention decoder backward is a supported runtime path.
2. If supported, add a focused PyTorch-derived decoder cross-attention backward
   parity fixture.
3. If unsupported, document the unsupported path clearly and make sure it fails
   closed instead of silently training with wrong gradients.