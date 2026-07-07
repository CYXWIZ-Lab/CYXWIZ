# M-06) `materializer_source_kind == Unknown` edge contract

## M-06.1 Question
When does `PipelineMaterializerSourceKind::Unknown` appear, and is it always intentional?

## M-06.2 Present logic

```text
ResolvePipelineMaterializerSourceKind(dataset_name, data_loader_registry):
  if dataset_name matches registered loader name -> known source kind
  else if name indicates sequence/audio/image/text alias -> explicit mapped kind
  else -> Unknown

Materialize(config):
  if source kind is unsupported:
    mark skipped_unsupported_source = true
    fill unsupported_source_reason
    return success path without hard failure (unless other path errors)
```

## M-06.3 Contract interpretation
- `Unknown` is an explicit “compatibility/legacy fallback marker”, not a normal validated data source.
- It is tied to warning/skip behavior and avoids crashing on unexpected source labels.
- It should be restricted to intended compatibility branches to keep runtime deterministic.

## M-06.4 Required governance follow-up
- Add a startup/integration assertion log:
  - enumerate all `Unknown` usages,
  - require explicit allowlist for legacy branches,
  - convert unknown-only flows to hard failure if policy changes demand stricter guarantees.

## M-06.5 Evidence anchors
- `cyxwiz-engine/src/core/pipeline_materializer.h:11` (Materializer source-kind enum/contract)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:53` (ResolvePipelineMaterializerSourceKind entrypoint)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:65` (explicit registry-based source matching)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:71` (Unknown fallback)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:90` (Materialize() stamps source_kind)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:98` (skipped_unsupported_source + reason)
- `cyxwiz-engine/src/core/pipeline_materializer.cpp:104` (warn-path for unsupported source)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:879` (AudioDataset capability notes for materialization backend)
- `cyxwiz-engine/src/core/pipeline_runtime_capabilities.cpp:988` (storage backend mapping includes Unknown)
- `cyxwiz-engine/src/core/graph_compiler.h:307` (`TrainingConfiguration` includes materializer_source_kind field)
- `cyxwiz-engine/src/core/graph_compiler.cpp:501` (audio/domain inference helper used by materialization)
- `cyxwiz-engine/src/core/graph_compiler.cpp:2880` (dataset selection and registry coupling)
