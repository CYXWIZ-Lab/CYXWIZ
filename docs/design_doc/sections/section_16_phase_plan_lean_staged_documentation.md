## 16) Phase plan (lean, staged documentation)

### Phase 0 - Current baseline (this doc)
- architecture map
- boundaries
- compile -> preflight -> materialize -> execute

### Phase 1 - Node catalog and contract matrix
- enumerate every `NodeType` by stable status:
  - `Supported in compiler`
  - `Supported in materializer`
  - `Supported in training executor`
  - `Legacy / unsupported`

### Phase 2 - Training contract
- fully document:
  - batch semantics,
  - sequence/temporal input contract,
  - shape validation matrix per loss type.

### Phase 3 - Runtime capability matrix
- per backend (CPU/GPU/CUDA/ArrayFire), per node family.
- explicit unsupported behavior with reason codes.

### Phase 4 - End-to-end example walks
- vision example
- text example
- time-series example
- external dataset example

### Phase 5 - Operational quality
- trace IDs for callback events,
- reproducibility toggles and checkpoint semantics,
- failure-code enum unification.

---
