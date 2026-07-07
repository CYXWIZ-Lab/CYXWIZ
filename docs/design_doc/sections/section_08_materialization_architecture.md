## 8) Materialization architecture

Materialization is explicit and opt-in per source/runtime path.

ASCII:

```
Compile training config + operator chain
              |
              v
   PipelineMaterializer::Apply()
              |
      +-------+-------+
      | source detect  |
      |  Arrow/Parquet |
      +-------+-------+
              |
   +----------+----------+
   | linear path + DAG   |
   | node compatibility  |
   +----------+----------+
              |
      +-------+-------+
      | construct     |
      | pipeline ops  |
      +-------+-------+
              |
           success -> register materialized source
           fail    -> error + block
```

Key behavior:
- If source is Arrow:
  - walks data input path,
  - checks runtime support via pipeline capabilities,
  - applies operators in order,
  - materializes as `<source>__materialized` marker only when operators applied.
- If non-Arrow and unsupported by current materializer path:
  - returns blocked/failure result with actionable reason.
- Text node combinations are folded into tokenizer/padding/materialization context.

Failure policy:
- Prefer fail-closed for unsupported mappings to avoid silent behavioral divergence.

---
