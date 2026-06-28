# To Fix 28 - LLM/Text Generation Follow-Up Work

This document tracks follow-up work split out from `done8.md` after the focused
`tofix8` implementation completed.

`done8.md` established a minimal, native, reference-tested LLM/text-generation
path in CyxWiz. This file is not a restart of that roadmap. It is for production
hardening and broader capability work that should stay separate from the closed
minimal path.

---

## Goal

Harden and extend the tested text-generation path without bloating the engine or
claiming production-scale LLM support before the required pieces are real.

The essential rule remains:

- keep native CyxWiz components small and auditable
- use reference-tested fixtures before broad enablement
- keep ArrayFire/GPU as first-class compute where it is proven stable
- preserve CPU fallback and capability truth where GPU/runtime support is not
  proven

---

## Follow-Up Areas

### 1. Production dataset ingestion

Scope:

- real HuggingFace or HF-style dataset loading beyond deterministic row mapping
- streaming or chunked ingestion for large instruction datasets
- split handling for train/validation/test
- robust schema detection for prompt/query/answer/response variants

Non-goal:

- do not add a large dataset framework inside the core engine unless the boundary
  is narrow and testable.

### 2. Generation controls

Scope:

- temperature sampling
- top-k sampling
- top-p / nucleus sampling
- EOS stopping behavior
- max-new-token validation
- deterministic seeded generation tests

Non-goal:

- do not expose UI claims for sampling modes until server/runtime tests prove
  the behavior.

### 3. Standalone attention graph contract

Scope:

- define whether `MultiHeadAttention` is a real standalone graph node or only an
  internal layer used by encoder/decoder modules
- align compiler, model builder, runtime, metadata, and UI behavior
- add tests before enabling standalone graph usage

Non-goal:

- do not silently unblock standalone attention just because decoder-internal
  attention is now tested.

### 4. Larger model and runtime stability

Scope:

- stress tests for longer sequence lengths and larger vocab/logit shapes
- memory behavior under ArrayFire CUDA
- clear GPU-to-CPU fallback diagnostics
- avoid blanket `eval()` calls unless a specific kernel/materialization issue is
  proven

Non-goal:

- do not hide CUDA kernel failures behind vague success states.

### 5. Performance and backend placement

Scope:

- profile minimal causal-LM forward/generation paths
- identify hotspots in attention, token loss, embedding, and time-distributed
  dense operations
- preserve backend placement truth for unsupported or CPU-routed paths

Non-goal:

- do not optimize by adding broad abstractions or duplicating algorithm paths.

### 6. Artifact and serving hardening

Scope:

- richer `.cyxmodel` text metadata validation
- backward-compatible tokenizer/vocab manifest versioning
- clearer `/v1/generate` error codes and response contracts
- model capability reporting that separates classification from generation

Non-goal:

- do not merge classification and generation response shapes into one ambiguous
  contract.

---

## Relationship To Other Docs

- `done8.md`: closed minimal LLM/text-generation implementation roadmap.
- `tracker8_done.md`: implementation history and validation record for `done8.md`.
- `done14.md`: structural/modularity debt such as splitting oversized layer
  implementation files into cleaner translation units.
- `tofix27.md`: Data Studio runtime/materializer parity follow-up from `done7.md`,
  not an LLM follow-up.

---

## Done Criteria

This follow-up is done only when each enabled production feature has:

- a narrow design boundary
- reference-driven tests or deterministic fixtures
- CPU correctness coverage where applicable
- ArrayFire/GPU coverage where support is claimed
- truthful compiler/runtime/UI capability metadata
- clear unsupported states for anything not implemented
