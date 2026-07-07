## 15) Lean guardrail assessment (applied to this design)

### 15.1 complexity risks
- Very large NodeType surface in editor relative to fully supported runtime set.
- Multiple compatibility layers and aliases can be a tax if not governed by explicit deprecation plans.
- Deep domain coverage increases cognitive load for the compiler.

### 15.2 simplification opportunities
- Keep core surface to:
  - supported model layers,
  - deterministic preprocessors,
  - stable training nodes.
- Move rarely used legacy nodes to explicit compatibility packages or plugin adapters.
- Make unsupported-node behavior deterministic and user-visible at compile time.

### 15.3 strong boundary checks (must stay narrow)
- `GraphCompiler` and `PipelineRuntimeCapabilities` are the guard boundary.
- `TrainingExecutor` should only assume validated plans, not interpret arbitrary graph semantics.
- `INodeProvider` boundaries should never bypass compile-time capability checks.

---
