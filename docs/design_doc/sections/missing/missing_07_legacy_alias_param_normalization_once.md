# M-07) Legacy alias and node-parameter normalization idempotence

## M-07.1 Question
Are legacy alias parameters normalized exactly once, without hidden second-pass mutation?

## M-07.2 Current alias surface

```text
Node catalog -> node alias metadata
  -> compiler normalization pass
  -> runtime strategy selection:
       - canonicalize operator
       - compatibility handling
       - legacy fallback/alias translation
  -> materialization/runtime capability checks
```

## M-07.3 Boundary risk
- If alias normalization and canonical mapping run multiple times across separate passes, parameters can drift.
- If parameter normalization and pin correction are performed in both graph-compilation and executor construction, behavior can become order-dependent.

## M-07.4 Design guidance (lean-safe)
- Single ownership of normalization:
  - canonicalization in exactly one preprocessing pass,
  - runtime to treat normalized fields as immutable inputs.
- Add assertions that key fields (normalized op id, pinned numeric coercions, compatibility flags) are unchanged across passes.
- Add duplicate-test case for one alias collision pair where alias name and canonical name both exist.

## M-07.5 Evidence anchors
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:32` (alias_type_name / canonical_type_name fields)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:34` (canonical node type mapping)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.h:276` (`ResolvePipelineLegacyAliasDecision` API)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1048` (legacy alias decision catalog entries)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1050` (default alias decision fallback)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1054` (legacy alias handling behavior)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1223` (legacy node classification lookup)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:1230` (`ResolvePipelineLegacyAliasDecision` implementation)
- `cyxwiz-engine/src/core/graph_compiler.cpp:1375` (single Build() pass over graph)
- `cyxwiz-engine/src/core/graph_compiler.h:305` (compiled parameter contract source of truth)
- `cyxwiz-engine/src/core/graph_compiler.cpp:1375` (single Build() pass over nodes and passes parameters through one contract object)
- `cyxwiz-engine/src/core/model_builder.h:42` (`BuildSequentialFromConfig` / config consumption entrypoint)
- `cyxwiz-engine/src/core/model_builder.cpp:1308` (BuildSequentialFromConfig path)
