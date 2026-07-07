## 13) Node-contract matrix (high-level)

Current system has a large node surface; important observation:
- Not every cataloged node has a stable training/runtime mapping.
- Compiler and materializer define the actual runtime boundary.

```text
editor visibility
     |
     +-- NodeType enum (broad)
     |
     +-- pin schema + params
     |
     +-- compile-time capability check
            |
            +-- Graph path support
            |
            +-- Pipeline operator support
            |
            +-- fail-closed unsupported marker
            |
            +-- runtime node executor/provider
```

Important practical rule:
- If a node is present in editor, check its exact category in:
  - `pipeline_runtime_capabilities*` (operator/runtime allow list),
  - compiler sequential blockers (`GraphCompiler`),
  - training path extraction logic.

---
