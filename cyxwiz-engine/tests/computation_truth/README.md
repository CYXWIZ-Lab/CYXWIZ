# CyxWiz Engine Computation Truth Tests

This folder is for deterministic numerical parity tests.

Purpose:

- Compare CyxWiz engine computations against stable references.
- Catch silent math drift in preprocessing, model forward passes, losses, gradients, optimizers, and training lifecycle.
- Keep correctness tests independent of the GUI.

Reference stack:

- PyTorch for tensor/model/loss/optimizer parity.
- scikit-learn for TF-IDF parity when available.
- Small in-repo reference implementations when external Python packages are unavailable.

Initial target cases:

- TF-IDF bounded materialization values and shape.
- Dense forward parity.
- CrossEntropy loss parity.
- One-step Adam/AdamW parity.
- Training lifecycle: configured epochs vs completed/stopped/cancelled reason.

The broad tracking ticket is:

`docs/Data Studio/tofix39.md`

## Current tests

`test_computation_truth_tfidf_loss`

- Builds an Arrow text table in memory.
- Runs `TFIDFVectorizerOperator`.
- Verifies bounded output width: `max_features` columns plus optional `y`.
- Verifies deterministic TF-IDF values against a hand reference.
- Verifies deterministic label encoding.
- Verifies `CrossEntropyLoss` mean against hand reference math equivalent to PyTorch `cross_entropy`.
- Verifies `LinearLayer` forward output against hand reference `input @ weight^T + bias`.
- Verifies `LinearLayer` backward values:
  - gradient with respect to input
  - mean-reduced weight gradients
  - mean-reduced bias gradients
- Verifies `AdamWOptimizer` first-step parameter update with decoupled weight decay against a hand reference equivalent to PyTorch AdamW first-step semantics.

Run:

```powershell
cmake --build build --config Release --target test_computation_truth_tfidf_loss -- /m:4 /v:minimal
build\bin\Release\test_computation_truth_tfidf_loss.exe
```

Note: some local builds report `PyTorch/LibTorch: OFF`; those builds should
use hand references or optional Python-side PyTorch checks. When LibTorch is
enabled, keep it in computation-truth tests and outside the engine runtime
boundary.

Latest observed run also exercised `LinearLayer` while ArrayFire GPU was active.

`test_computation_truth_transformer_primitives`

- Verifies embedding, positional encoding, attention, layer norm, encoder, and
  decoder primitive parity.
- Verifies tiny causal-LM vocabulary logits and loss against PyTorch linear and
  cross-entropy semantics.
- Verifies BERT-style CLS extraction, sequence-classifier logits, and
  token-classifier logits against PyTorch indexing and linear-head semantics.
- Verifies GPT-style generation candidate probabilities for temperature,
  top-k, top-p, and greedy selection against PyTorch softmax/top-k reference
  behavior, with hard-coded PyTorch-derived constants when LibTorch is not
  enabled.
- Verifies deterministic multinomial replay over the PyTorch-verified
  candidate distribution. CyxWiz does not require its C++ RNG stream to match
  `torch.multinomial` exactly.

Run:

```powershell
cmake --build build --config Debug --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal
build\bin\Debug\test_computation_truth_transformer_primitives.exe

cmake --build build --config Release --target test_computation_truth_transformer_primitives -- /m:4 /v:minimal
$env:PATH = "<local-libtorch>\lib;$env:PATH"
build\bin\Release\test_computation_truth_transformer_primitives.exe
```
