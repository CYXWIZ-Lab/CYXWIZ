# Track 47 - Persistent Materialization Cache Continuation

## 2026-07-09 Continuation

Continued from `tofix47.md` after the core materialization cache work was
already in place.

### Change

- Kept the cache UX non-destructive.
- Extended the training materialization summary to show the prepared dataset
  manifest path derived from the cached artifact path.
- Added copy actions for both the prepared dataset artifact path and manifest
  path.
- Added the manifest path to per-stage materialization events so the debugger
  timeline exposes the same cache truth as the summary.

### Files touched

- `cyxwiz-engine/src/gui/panels/training_plot_panel.h`
- `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp`
- `docs/Data Studio/track47.md`

### Verification

Passed:

```powershell
cmake --build build --config Release --target cyxwiz-engine
cmake --build build --config Release --target test_materialization_cache test_pipeline_materializer_cache test_graph_training_sequence_preflight test_text_gui_training_launch
build\bin\Release\test_materialization_cache.exe
build\bin\Release\test_pipeline_materializer_cache.exe
build\bin\Release\test_graph_training_sequence_preflight.exe
build\bin\Release\test_text_gui_training_launch.exe
git diff --check
```

### Notes

- No rebuild/delete cache actions were added. This keeps the first Studio
  surface inspectable and low-risk.
- Existing unrelated dirty worktree noise in `docs/Data Studio/` remains
  untouched.

## 2026-07-09 Continuation 2

Tightened the cache inspection plumbing so the UI no longer infers manifest
details from the artifact path.

### Change

- Extended `MaterializeResult` with:
  - `cache_manifest_path`
  - `cache_row_count`
  - `cache_column_count`
- Populated those fields on cache save and cache hit from the materializer
  manifest.
- Cleared artifact and manifest paths when no materializer operators apply,
  because no cache artifact is written in that pass-through case.
- Passed explicit manifest metadata through `graph_training_launcher.cpp` into
  `TrainingPlotPanel`.
- Displayed prepared dataset row/column counts in the materialization summary
  and stage history.
- Added regression assertions for manifest path and prepared dataset dimensions.

### Additional files touched

- `cyxwiz-engine/src/core/pipeline_materializer.h`
- `cyxwiz-engine/src/core/pipeline_materializer.cpp`
- `cyxwiz-engine/src/gui/graph_training_launcher.cpp`
- `cyxwiz-engine/tests/test_pipeline_materializer_cache.cpp`

### Additional verification

Passed:

```powershell
cmake --build build --config Release --target test_pipeline_materializer_cache test_text_gui_training_launch cyxwiz-engine
build\bin\Release\test_materialization_cache.exe
build\bin\Release\test_pipeline_materializer_cache.exe
build\bin\Release\test_text_gui_training_launch.exe
build\bin\Release\test_graph_training_sequence_preflight.exe
```

## 2026-07-10 Continuation 3

Closed the remaining non-destructive cache inspection action.

### Change

- Added `Open cache location` to the Training Dashboard materialization summary
  when a prepared dataset artifact or manifest path is available.
- Opens the cache artifact directory through the platform file browser and
  records success/failure as panel events.
- Kept rebuild/delete cache actions out of this track.

### Additional files touched

- `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp`

### Additional verification

Passed:

```powershell
cmake --build build --config Release --target cyxwiz-engine
cmake --build build --config Release --target test_materialization_cache test_pipeline_materializer_cache test_graph_training_sequence_preflight test_text_gui_training_launch
build\bin\Release\test_materialization_cache.exe
build\bin\Release\test_pipeline_materializer_cache.exe
build\bin\Release\test_graph_training_sequence_preflight.exe
build\bin\Release\test_text_gui_training_launch.exe
```

Note: the first `test_graph_training_sequence_preflight.exe` run hit the 120s
command timeout without output; rerunning with a longer timeout passed.
