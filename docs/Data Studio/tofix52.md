# To Fix 52 - Transformer Backward Parity And Training Correctness

## Purpose

Extend transformer correctness from forward/runtime contracts into full
training correctness.

Current coverage is strong for primitive forward behavior and selected export
paths, but full transformer training claims require backward parity and optimizer
behavior across stacked layers.

## Scope

- LayerNorm backward parity.
- MultiHeadAttention backward parity.
- TransformerEncoder stack backward parity.
- TransformerDecoder stack backward parity.
- Feed-forward block gradients.
- Residual connection gradients.
- Masked attention gradients.
- Cross-entropy training step sanity.

## Reference

Use PyTorch as the numerical reference for shape, loss, and gradients.

The comparison should be narrow and deterministic:

- Small tensors.
- Fixed seed.
- CPU reference first.
- Explicit tolerances.
- No broad randomized property tests until the fixed cases are stable.

## Engine requirements

- Gradients must preserve tensor shapes.
- Parameter names must map clearly between source and imported models.
- Loss reduction behavior must match documented settings.
- Ignore index and label smoothing must be tested where applicable.

## Tests

Add computation-truth tests for:

- Single encoder block backward.
- Single decoder block backward.
- Two-block stack backward.
- Causal mask backward.
- Attention mask backward.
- Tiny training step loss decreases or matches reference update.

## Non-goals

- Do not add GPU kernels in this ticket.
- Do not test large models.
- Do not claim pretrained transformer compatibility.

## Completion criteria

- CPU transformer training has defensible backward parity for the supported
  graph contracts.
- Any unsupported backward path fails closed or is documented as CPU-only and
  forward-only.
