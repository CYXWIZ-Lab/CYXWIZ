# To Fix 18 - Deferred Data Studio Pipeline Canvas Cleanup

**Created:** 2026-06-05
**Source:** Follow-up split out while closing `done1.md` Priority 8.

## Boundary

This file tracks Data Studio `PipelineCanvas` cleanup that is not part of
the current main graph/training path.

Do not mix this work back into `done1.md`. `done1.md` is closed for the
main engine graph, backend, and frontend contract work. Only promote an
item from this file when the Data Studio pipeline canvas is actively used
to create or execute training graphs again.

## Priority 1 - Time-Series Placeholder Param Names

`cyxwiz-engine/src/gui/data_studio/pipeline_canvas.cpp` still has older
placeholder parameter names for internal `TSWindow` and `TSFeatures`
nodes:

- `TSWindow`: `window_size`, `stride`, `target_column`
- `TSFeatures`: `columns`, `rolling_window`, `lag_features`,
  `rolling_features`

The main Node Editor graph path is already fixed:

- `TimeSeriesWindow`: `value_col`, `feature_cols`, `time_col`,
  `input_width`, `label_width`, `shift`
- `TimeSeriesFeatures`: `value_col`, `lag_values`, `rolling_windows`,
  `rolling_aggregations`
- old main-graph saved params migrate on load

### Why Deferred

The Data Studio `PipelineCanvas` surface is separate from the main
Node Editor training graph path. Renaming its placeholders without a
clear execution bridge would add churn without proving a runtime benefit.

### Next Step

Before editing code, decide whether `PipelineCanvas` nodes are:

1. UI-only placeholders,
2. an independent Data Studio pipeline format, or
3. a source for real Node Editor training graphs.

Only after that decision should the param names be migrated or bridged.

### Completion Criteria

- The chosen `PipelineCanvas` role is documented.
- Any migrated param names map cleanly to the canonical Cat-1 operator
  contract.
- Old serialized Data Studio pipeline JSON either migrates on load or is
  explicitly marked incompatible.
- A focused save/load smoke test covers one `TSWindow` and one
  `TSFeatures` node if this surface remains active.

## Status 2026-06-22

Decision: `PipelineCanvas` is an active independent Data Studio pipeline surface, not a UI-only placeholder and not the main Node Editor training graph. It saves/loads Data Studio pipeline JSON, executes through `PipelineExecutor`, and can expose deployment readiness for handoff to the Node Editor.

Cleanup completed:
- the quick-add palette already creates canonical `TimeSeriesWindow` and `TimeSeriesFeatures` node type strings,
- direct legacy `TSWindow` and `TSFeatures` canvas creation now uses canonical Cat-1 parameter names,
- old saved Data Studio canvas JSON loaded through `PipelineCanvas::LoadPipeline` migrates legacy time-series parameter names into canonical names,
- `PipelineExecutor` keeps legacy `TSWindow`/`TSFeatures` aliases executable for compatibility and now reads canonical params first with legacy-key fallback.

Compatibility policy:
- keep legacy type aliases so old pipeline JSON can still route,
- do not promote legacy aliases in the quick-add palette,
- prefer canonical params on all newly created or loaded canvas nodes,
- do not remove legacy executor fallback until saved-pipeline migration has a versioned incompatibility policy.

This closes the `tofix18` parameter-name cleanup without touching the main Node Editor graph path.
