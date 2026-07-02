# To Fix 54 - CBOW First-Class Small Model Family

## Purpose

Add CBOW as a first-class small NLP model family.

CBOW is useful because it is much smaller than transformer models and gives the
engine a practical word-embedding training path for CPU-first correctness.

## Scope

- CBOW graph pattern.
- CBOW dataset/materializer contract.
- Context window extraction.
- Embedding layer.
- Projection/head layer.
- Negative sampling or full softmax decision.
- Training smoke test.
- Export/import.
- Inference/query utility for learned embeddings.

## Engine contract

The engine should define:

- Input: context token IDs.
- Target: center token ID.
- Vocabulary asset.
- Context window size.
- Embedding dimension.
- Loss behavior.
- Output: token logits or embedding lookup.

## Tests

Add tests for:

- Context window materialization.
- Graph compile.
- One training step.
- Export/import parameter roundtrip.
- Inference embedding lookup.
- Failure for missing vocabulary.

## Non-goals

- Do not build Word2Vec skip-gram unless needed.
- Do not add approximate nearest-neighbor search in this ticket.
- Do not add large corpus training benchmarks.

## Completion criteria

- CBOW appears as a real model-family pattern, not a Dense placeholder.
- Training, export/import, and basic inference behavior are tested.
- Documentation explains when CBOW is preferable to transformer models.
