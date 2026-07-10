# To Fix 49 - Full LM-Stack Inference Contract

## Purpose

Define and implement the full language-model inference contract after the
`tofix28` foundation.

`tofix28` proved generation controls, `.cyxmodel` generation metadata,
tokenizer packaging, compiler sequence shape handling, and
`TransformerDecoder -> TimeDistributedDense` import/export logits shaped
`[batch, seq, vocab]`.

This ticket covers the next layer: a complete trained LM stack that can be
loaded, validated, prompted, and executed through the normal inference path.

## Scope

- Load a trained causal-LM `.cyxmodel`.
- Load packaged tokenizer assets from the same package.
- Validate model-family metadata.
- Validate runtime output contract: `Float32 [1, seq, vocab]`.
- Encode a text prompt with the packaged tokenizer.
- Run generation with the imported model.
- Decode generated token IDs back to text.
- Surface contract errors in Studio and logs.

## Engine contract

The runtime must know:

- Model family: `causal_lm`.
- Tokenizer assets are present and usable.
- Prompt IDs are rank-2 compatible: `[1, seq]`.
- Model output is rank-3 logits: `[1, seq, vocab]`.
- `vocab` matches or is compatible with tokenizer vocabulary size.
- EOS handling is deterministic and visible.

## Studio contract

The Language Model Generation panel should show:

- Active package path.
- Model family.
- Generation support flag.
- Output contract.
- Tokenizer vocabulary size.
- Max sequence length.
- EOS token ID.
- Compatibility check result.
- Clear error if the package is a classifier, encoder, missing tokenizer, or
  has wrong output shape.

## Tests

Add focused tests for:

- Package load with model + tokenizer.
- Text prompt encode.
- Generation runtime on imported `.cyxmodel`.
- Decode generated IDs.
- Failure for missing tokenizer assets.
- Failure for non-generation package.
- Failure for `[batch, classes]` classifier output.

## Non-goals

- Do not implement a full GPT model claim here.
- Do not add cross-attention or encoder-decoder support here.
- Do not optimize GPU kernels here.

## Completion criteria

- A trained or fixture causal-LM package can be loaded and used for generation
  through the same contract Studio exposes.
- Contract failures are clear before generation starts.
- Tests prove success and failure paths.
