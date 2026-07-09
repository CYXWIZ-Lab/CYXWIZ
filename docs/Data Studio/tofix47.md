# To Fix 47 - Persistent Materialization Cache And Prepared Dataset Reuse

## Why this matters

Training currently runs materialization before the model starts. For small graphs this is acceptable, but for large preprocessing graphs it can be expensive enough to make the Studio appear frozen or waste time every run.

Example from the sentiment TF-IDF path:

- Read CSV text.
- Tokenize text.
- Count document and term frequencies.
- Select vocabulary.
- Build a TF-IDF matrix.
- Create an Arrow table/dataset.
- Register the prepared dataset for training.

If the user trains today, closes CyxWiz, and trains the same graph tomorrow, the engine should not repeat this work when the source data and preprocessing graph are unchanged.

The required capability is simple: save the prepared materialized dataset and reuse it safely.

## Current engine truth

Based on the current code:

- `PipelineMaterializer` materializes into memory.
- The output dataset name is currently derived as `<source_dataset_name>__materialized`.
- `PipelineMaterializer::Materialize` registers the output table through `DataRegistry::RegisterArrowTable`.
- Unsupported source kinds are skipped explicitly; the current materializer path is Arrow-table focused.
- `MaterializeResult` reports effective dataset name, operators applied, source kind, skip status, success, and errors.
- There is no persistent cache key.
- There is no disk artifact for a prepared dataset.
- There is no materialization manifest.
- There is no cache hit/miss/stale status in the training result.
- There is no Studio control for reusing, rebuilding, inspecting, or deleting a prepared dataset.
- `ArrowDataset` already supports persistent import/export paths including Parquet and Feather.

This means the engine already has the storage primitives needed for a lean implementation. We should not add a new database or a broad storage subsystem.

## Target behavior

When training starts:

1. The engine computes a materialization cache key from the source dataset and preprocessing graph.
2. If a valid prepared dataset exists, the engine loads it and registers it as the effective training dataset.
3. If no valid prepared dataset exists, the engine runs the current materialization path.
4. After successful materialization, the engine saves the prepared dataset and a manifest.
5. The training dashboard and debugger report whether the prepared dataset was reused, rebuilt, skipped, or failed.

The user should see messages such as:

- `Preparing materialized dataset...`
- `Using cached prepared dataset.`
- `Materialization cache miss; rebuilding prepared dataset.`
- `Materialization cache stale because source file changed.`
- `Materialization completed and saved.`

## Cache key

The cache key must prove that a prepared dataset still matches the requested training graph.

Minimum inputs:

- Source dataset name.
- Source dataset identity.
- Source file path when available.
- Source file size and modification time when available.
- Source schema fingerprint.
- Preprocessing/materializer node ids, types, names, and parameters.
- Preprocessing links that affect materialization order.
- Materializer operator version.
- Engine materialization cache schema version.

The first implementation can use a single global materializer cache schema version. Per-operator versions can be added later if needed.

## Manifest

Each cached materialization should write a small JSON manifest beside the dataset artifact.

Recommended fields:

- `cache_key`
- `source_dataset_name`
- `effective_dataset_name`
- `artifact_path`
- `artifact_format`
- `row_count`
- `column_count`
- `schema_fingerprint`
- `operators_applied`
- `engine_version`
- `materializer_cache_schema_version`
- `created_at`
- `last_used_at`
- `cache_status`
- `stale_reason`

The manifest is for truth and diagnostics. It should be visible from the Studio debugger and usable in logs.

## Storage layout

Use a project or workspace cache directory, not the source repository.

Suggested layout:

```text
.cyxwiz/
  cache/
    materialized/
      <cache_key>/
        manifest.json
        data.parquet
```

Default artifact format:

- Prefer Parquet for large/wide materialized tables.
- Keep Feather/Arrow IPC as an optional faster local format if existing support is strong.

Do not invent a custom binary format for this ticket.

## Engine changes

Add a focused cache module instead of growing `pipeline_materializer.cpp`.

Suggested files:

- `cyxwiz-engine/src/core/materialization_cache.h`
- `cyxwiz-engine/src/core/materialization_cache.cpp`

Small integration edits:

- `cyxwiz-engine/src/core/pipeline_materializer.h`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp`
- Training launch path that currently invokes materialization.
- Training progress/debug event path.
- Studio training dashboard and debugger panels.

## Proposed data types

Add a cache policy/config object:

```cpp
enum class MaterializationCacheMode {
    Disabled,
    Auto,
    Rebuild,
    RequireHit
};
```

Add cache status:

```cpp
enum class MaterializationCacheStatus {
    Disabled,
    Miss,
    Hit,
    Stale,
    Saved,
    SaveFailed,
    Corrupt,
    Unsupported
};
```

Extend `MaterializeResult` with:

- `MaterializationCacheStatus cache_status`
- `std::string cache_key`
- `std::string cache_artifact_path`
- `std::string cache_message`
- `bool loaded_from_cache`
- `bool saved_to_cache`

Keep these fields narrow. Do not expose cache internals through the whole engine.

## Materialization flow

Current flow:

```text
resolve source dataset
load source table
apply materializer operators
register <source>__materialized
return effective dataset name
```

Target flow:

```text
resolve source dataset
compute cache key
if cache enabled:
  read manifest
  validate source, schema, graph, and engine cache version
  if valid:
    load cached Arrow dataset
    register <source>__materialized
    return cache hit

load source table
apply materializer operators
register <source>__materialized
if cache enabled:
  export materialized table
  write manifest atomically
return cache miss/saved
```

Atomic write rule:

- Write to a temporary directory/file first.
- Replace the final manifest last.
- Never leave a half-written artifact marked as valid.

## Studio behavior

Training dashboard should show:

- Cache status.
- Cache key short hash.
- Artifact path.
- Row count.
- Column count.
- Operators applied.
- Build or load duration.
- Whether source/graph changes invalidated cache.

Useful actions:

- `Use cached prepared dataset`
- `Rebuild prepared dataset`
- `Inspect prepared dataset`
- `Delete prepared dataset`
- `Open cache location`

Debugger should record:

- Materialization started.
- Cache lookup started.
- Cache hit/miss/stale/corrupt.
- Cache validation reason.
- Materialization stages.
- Cache artifact saved.
- Training started using effective dataset name.

## Failure policy

The cache must never silently return stale data.

Rules:

- If the source changed, rebuild.
- If preprocessing node parameters changed, rebuild.
- If graph links changed in a way that affects materialization, rebuild.
- If schema changed, rebuild.
- If manifest is corrupt, rebuild and report.
- If artifact is missing, rebuild and report.
- If saving the cache fails, continue training with the in-memory materialized table and report `SaveFailed`.
- If the user selects `RequireHit` and no valid cache exists, fail before training with a clear error.

## Tests

Required tests:

- Cache miss materializes, registers, saves artifact, and writes manifest.
- Cache hit loads artifact and does not apply operators again.
- Changing a node parameter invalidates cache.
- Changing source file size or modification time invalidates cache.
- Changing schema invalidates cache.
- Corrupt manifest rebuilds safely.
- Missing artifact rebuilds safely.
- Save failure does not block training in `Auto` mode.
- `RequireHit` fails clearly when no cache exists.
- Unsupported source kind remains an explicit pass-through skip.

## Lean guardrails

Keep this ticket focused.

Do:

- Reuse existing ArrowDataset Parquet/Feather persistence.
- Add one focused cache module.
- Keep cache reporting explicit and inspectable.
- Make invalidation conservative.
- Keep UI controls minimal but truthful.

Do not:

- Add a database.
- Add a new custom storage format.
- Cache every internal intermediate table in the first version.
- Hide stale-cache decisions.
- Make the materializer responsible for unrelated dataset conversion features.
- Solve distributed cache or cloud cache in this ticket.

## Implementation order

1. Add materialization cache manifest/key structs and hashing utility.
2. Add cache validation and artifact load/save helpers.
3. Integrate cache lookup/save around `PipelineMaterializer::Materialize`.
4. Extend `MaterializeResult` with cache status fields.
5. Emit training progress/debug events for cache status.
6. Add Studio dashboard/debugger display for materialization cache truth.
7. Add usage documentation for prepared dataset reuse.
8. Add tests for hit, miss, stale, corrupt, save failure, and explicit rebuild.

## Resume 2026-07-09 - Current Pushed State And Next Pickup

This ticket is mostly complete in code on branch `Nodes_Implementation`.
The relevant pushed commits are:

- `4a849e3d` - Add materialization cache and async launcher fixes.
- `f942a82a` - Surface materialization cache details in training panel.

What is already in place:

- New focused cache module at `cyxwiz-engine/src/core/materialization_cache.*`.
- `PipelineMaterializer` cache lookup/save flow with manifest validation.
- Training launch integration with cache-aware progress reporting.
- Training panel Materialization summary now shows cache key and cache artifact
  path, with a copy button for the artifact path.
- Regression coverage for cache behavior and launcher async behavior.
- Main engine build and targeted regression tests pass.

Latest verified commands:

- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target test_text_gui_training_launch test_graph_training_sequence_preflight`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_text_gui_training_launch.exe`
- `D:\Dev\CyxWiz_Claude\build\bin\Release\test_graph_training_sequence_preflight.exe`
- `cmake --build D:\Dev\CyxWiz_Claude\build --config Release --target cyxwiz-engine`
- `git diff --check`

What is left, if we continue this ticket:

- `Open cache location` is now implemented for cache inspection. Rebuild/delete
  controls remain product decisions and should require explicit confirmation if
  they are added later.
- Add more debugger/inspection detail only if product wants cache manifest fields
  beyond the training panel summary and open-location action.
- Tighten diagnostics only if a later bug report exposes a stale, corrupt, or
  save-failure gap.
- Keep unrelated dirty worktree noise in `docs/Data Studio/` out of tofix47
  commits unless the user explicitly asks to handle those files.

How to pick this up next time:

1. Start from `git status --short` and confirm only intended files are touched.
2. Read `cyxwiz-engine/src/core/materialization_cache.*`,
   `cyxwiz-engine/src/core/pipeline_materializer.*`,
   `cyxwiz-engine/src/gui/graph_training_launcher.cpp`, and
   `cyxwiz-engine/src/gui/panels/training_plot_panel.*`.
3. Re-run `test_materialization_cache`, `test_pipeline_materializer_cache`,
   `test_graph_training_sequence_preflight`, and `test_text_gui_training_launch`
   before any further cache behavior changes.
4. If adding UI actions, keep them narrow and non-surprising: inspect/open are
   lower risk than delete/rebuild, and destructive actions should require clear
   confirmation.

## Acceptance criteria

- A repeated training run with unchanged source data and preprocessing graph reuses the saved prepared dataset.
- A changed source file or preprocessing parameter invalidates the cache.
- The dashboard explains whether materialization was built, loaded, skipped, stale, or failed.
- The debugger records enough detail for an engineer to understand why the cache was or was not used.
- Training still works if cache saving fails in automatic mode.
- No new broad storage subsystem is introduced.

## Resume 2026-07-09 - Cache Inspection Metadata Handoff

Continuation after `track47.md` updates.

What is now in place:

- `MaterializeResult` reports `cache_manifest_path`, `cache_row_count`, and
  `cache_column_count` in addition to cache key, artifact path, status, and
  message.
- `PipelineMaterializer` populates manifest path and prepared dataset dimensions
  on cache save and cache hit from the manifest.
- Pass-through materialization with zero applied operators clears predicted
  artifact/manifest paths because no cache artifact is written.
- `graph_training_launcher.cpp` forwards explicit cache manifest metadata to
  `TrainingPlotPanel`.
- The training panel materialization summary and stage history show cache key,
  prepared dataset artifact path, manifest path, and row/column counts.
- `test_pipeline_materializer_cache` now asserts manifest path and prepared
  dataset dimensions on cache save and cache hit.

Latest verified commands:

```powershell
cmake --build build --config Release --target test_pipeline_materializer_cache test_text_gui_training_launch cyxwiz-engine
build\bin\Release\test_materialization_cache.exe
build\bin\Release\test_pipeline_materializer_cache.exe
build\bin\Release\test_text_gui_training_launch.exe
build\bin\Release\test_graph_training_sequence_preflight.exe
git diff --check
```

Next pickup guidance:

1. Cache inspection now exposes copy actions plus `Open cache location` from the
   materialization summary.
2. Keep future cache UI actions non-destructive unless product explicitly asks
   for rebuild/delete controls; destructive actions need clear confirmation.
3. Re-run `test_materialization_cache`, `test_pipeline_materializer_cache`,
   `test_text_gui_training_launch`, and `test_graph_training_sequence_preflight`
   after touching materialization cache reporting.
4. Keep unrelated dirty `docs/Data Studio` churn out of the tofix47 commit.
