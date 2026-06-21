# Tracker 13 - Mechanical Modularity Plan

## Purpose

Track the active implementation plan for `tofix13.md`.

`tofix13.md` is the goal document: make the engine and Studio code easier to
read, navigate, test, and maintain by splitting oversized files into coherent
translation units, headers, and folders.

This tracker is the working plan. It must stay focused on organization only.

## Non-Negotiable Rule

This is not a redesign task.

Allowed:

- move existing code into clearer files,
- split headers and source files by concept,
- update includes and build files,
- preserve umbrella headers for compatibility,
- add focused tests only to prove behavior did not change,
- document ownership and file boundaries.

Not allowed:

- algorithm changes,
- runtime behavior changes,
- UI behavior changes,
- public API changes unless required to preserve compatibility after a move,
- new abstraction layers that only rename complexity,
- duplicate wrappers around existing logic,
- opportunistic bug fixes mixed into a mechanical split.

## Working Method

Each batch should follow this order:

1. Inventory the target file or module family.
2. Identify natural ownership boundaries already present in the code.
3. Move one coherent group at a time.
4. Keep public compatibility through umbrella headers where practical.
5. Update build wiring.
6. Run the smallest relevant build/test proof.
7. Record what moved and what remained.

## Progress Board

| Area | Status | Notes |
| --- | --- | --- |
| Codebase inventory | Complete | Initial oversized-file inventory captured below. |
| Backend layers | Complete | `layer.cpp` implementation has been split into focused layer-family translation units; public `layer.h` remains the compatibility umbrella. |
| Backend losses | Complete | `loss.cpp` implementation has been split into regression, probability, classification, and metric-learning translation units; public `loss.h` remains the compatibility umbrella. |
| Backend optimizers | Complete | `optimizer.cpp` implementation has been split into optimizer-family translation units; public `optimizer.h` remains the compatibility umbrella. |
| Backend metrics/evaluation | Complete | Existing `model_evaluation.cpp` implementations were split into cross-validation, classification metrics, and ROC/PR curve translation units. |
| Backend data/batch pipeline | Complete | Batches 5A-5I split algorithm data loaders plus core DuckDB data-loading paths into focused translation units. |
| Backend serialization/model artifacts | Complete | Batches 6A-6C moved `.cyxmodel` I/O, distributed checkpoint persistence, and tokenizer vocabulary persistence into focused translation units. |
| Studio panels/dialogs | In progress | Batches 7A-7J moved TableViewer clipboard, dialogs, editing/save, visualization handoff, stats/sidebar helpers, file loading, tab/session management, display transforms, quick plotting, and context menus into focused translation units. |
| Studio graph/runtime adapters | Pending | `pipeline_executor.cpp` is highest line-count risk but should not be first. |
| Build/header compatibility | Pending | Keep old include paths working where possible. |
| Regression proof | Pending | Record build/test command for each mechanical split batch. |

## Batch 0 - Inventory And Ranking

Goal: identify the largest and highest-risk files before moving code.

Tasks:

- list oversized backend files by line count,
- list oversized Studio/frontend files by line count,
- tag each file by responsibility groups already visible in the code,
- identify files that are too risky to split first,
- choose the first low-risk mechanical split.

Acceptance:

- tracker has a ranked target list,
- no source code moved in this batch unless the inventory reveals a trivial
  header-only cleanup,
- first implementation batch is small enough to audit.

### Batch 0 Findings - 2026-06-19

Largest backend code files:

| Lines | File | Initial classification |
| --- | --- | --- |
| 6010 | `cyxwiz-backend/src/algorithms/layer.cpp` | Best first target; already contains clear layer-family bounda

### Batch 7C - TableViewer Cell Editing And Save Split

Status: complete.

Moved:

- `TableViewerPanel::BeginCellEdit`
- `TableViewerPanel::EndCellEdit`
- `TableViewerPanel::SaveTable`
- `TableViewerPanel::HasUnsavedChanges`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_editing.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private method declarations remain in `table_viewer.h`.
- Cell edit lifecycle remains unchanged.
- Save dispatch by file extension remains unchanged.
- CMake now builds the editing translation unit.

Validation:

- Engine build passed: `cmake --build build --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7J - TableViewer Context Menu Split

Status: complete.

Moved:

- `TableViewerPanel::RenderColumnContextMenu`
- `TableViewerPanel::RenderCellContextMenu`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_context_menus.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private context-menu declarations remain in `table_viewer.h`.
- Existing sort, filter, color-map, formatting, quick-plot, and clipboard menu behavior remain unchanged.
- CMake now builds the context-menu translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Correction validation: initial Batch 7J attempt left the context-menu translation unit empty; follow-up moved the methods into `table_viewer_context_menus.cpp`, rebuilt, and reran the same passing test suite.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7I - TableViewer Quick Plot Split

Status: complete.

Moved:

- `TableViewerPanel::ShowQuickPlotPopup`
- `TableViewerPanel::RenderQuickPlot`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_plot.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private quick-plot declarations remain in `table_viewer.h`.
- Existing quick-plot rendering, visualizer handoff, and Python-script clipboard behavior remain unchanged.
- CMake now builds the quick-plot translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7H - TableViewer Display Transform Split

Status: complete.

Moved:

- `TableViewerPanel::ApplyColorMap`
- `TableViewerPanel::ClearColorMap`
- `TableViewerPanel::GetColorMapColor`
- `TableViewerPanel::ApplyFilter`
- `TableViewerPanel::ClearFilter`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_transforms.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private transform helper declarations remain in `table_viewer.h`.
- Existing column color-map and row-filter behavior remain unchanged.
- CMake now builds the transform translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7G - TableViewer Tab Session Split

Status: complete.

Moved:

- `TableViewerPanel::SetTable`
- `TableViewerPanel::SetTableByName`
- `TableViewerPanel::CloseCurrentTab`
- `TableViewerPanel::CloseTab`
- `TableViewerPanel::CloseAllTabs`
- `TableViewerPanel::IsFileOpen`
- `TableViewerPanel::FocusTab`
- `TableViewerPanel::FindTabByPath`
- `TableViewerPanel::GetActiveTab`
- `TableViewerPanel::Clear`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_tabs.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private tab helper declarations remain in `table_viewer.h`.
- Existing tab focus, close, lookup, and in-memory table behavior remain unchanged.
- CMake now builds the tab/session translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7F - TableViewer File Loading Split

Status: complete.

Moved:

- `TableViewerPanel::LoadCSV`
- `TableViewerPanel::LoadTXT`
- `TableViewerPanel::LoadHDF5`
- `TableViewerPanel::LoadExcel`
- `TableViewerPanel::LoadFileAsync`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_loading.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private async loading declaration remains in `table_viewer.h`.
- Existing lazy-loading and standard in-memory loading behavior remain unchanged.
- CMake now builds the loading translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7E - TableViewer Stats And Sidebar Split

Status: complete.

Moved:

- `TableViewerPanel::ComputeColumnStats`
- `TableViewerPanel::SortByColumn`
- `TableViewerPanel::RenderStatsSidebar`
- `TableViewerPanel::RenderMiniHistogram`
- `TableViewerPanel::GetColumnAsDoubles`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_stats.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private method declarations remain in `table_viewer.h`.
- Column statistics, sorting, sidebar display, and mini histogram behavior remain unchanged.
- CMake now builds the stats translation unit.

Validation:

- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.

### Batch 7D - TableViewer Visualization Handoff Split

Status: complete.

Moved:

- `TableViewerPanel::SendToVisualizer`

From:

- `cyxwiz-engine/src/gui/panels/table_viewer.cpp`

To:

- `cyxwiz-engine/src/gui/panels/table_viewer_visualization.cpp`

Compatibility:

- Public panel API remains unchanged.
- Private method declaration remains in `table_viewer.h`.
- Visualizer handoff behavior remains unchanged.
- CMake now builds the visualization translation unit.

Validation:

- Recovery note: A timed-out write truncated `table_viewer.cpp`; recovered it from `HEAD` and re-applied Batches 7A-7D mechanically.
- Recovery note: `cyxwiz-engine/CMakeLists.txt` was also truncated; recovered it from `HEAD` and re-applied the TableViewer source-list additions.
- Engine build passed: `cmake --build build\cyxwiz-engine --config Debug --target cyxwiz-engine`.
- Tests passed: `build\bin\Debug\cyxwiz-tests.exe`.
- Result: 1623 assertions in 224 test cases.
- Existing build warnings remain unchanged: LibTorch not found; NCCL not found.
- Existing runtime recurrent CPU-routing warnings remain expected and unchanged.
