# To Fix 3 - CyxWiz Engine Frontend Review for Engineering Pickup

This document captures a focused review of the current `cyxwiz-engine`
frontend and editor UX.

It is based on the current implementation, not a blank-sheet redesign.
The goal is to give engineers a practical backlog covering:

- frontend architecture
- UX and workflow gaps
- misleading or placeholder behavior
- maintainability concerns
- concrete high-value features to build next

This should be read alongside:

- `docs/Data Studio/done1.md`
- `docs/Data Studio/done2.md`

Those two cover backend/runtime issues. This document covers the engine
frontend and how it exposes backend behavior to users.

---

## Executive Summary

The engine frontend already has substantial surface area:

- start page
- node editor
- data input/config dialogs
- training dashboards and plots
- properties panel
- many standalone analysis/tool panels
- script editor and debugger
- plugin management

The main frontend problem is not lack of features. It is product
coherence.

Right now the UI behaves like a large collection of landed panels and
tools, with uneven workflow quality between them. The biggest issues are:

- graph/debug feedback is still fragmented
- some panels show placeholder or synthetic state that can mislead users
- core workflows like project start, data loading, and training status
  are more complex than they should be
- too many tool panels exist outside the graph workflow
- several core frontend files are very large and overloaded

The highest-value frontend work is:

1. make graph execution and debug results visible and trustworthy
2. tighten the Data Input flow
3. unify training/status dashboards
4. reduce tool/panel sprawl by pushing more workflows into node-driven UX
5. clean up large frontend files and inconsistent entry points

---

## Priority 0: Product Truthfulness

### 1. Training dashboard seeds fake sample metrics on construction

**Severity:** High
**Status:** Fixed on 2026-06-05

`cyxwiz-engine/src/gui/panels/training_dashboard.cpp`

Audit result: this issue was still present. The
`TrainingDashboardPanel` constructor populated loss, accuracy, and
throughput history with synthetic sample data immediately after plot
initialization.

That means the dashboard can look active before any training run has
happened.

This is a product-trust issue, not just a cosmetic one.

**Impact:**

- users can mistake placeholder charts for real training state
- debugging training telemetry becomes harder
- screenshots and demos can be misleading

**Recommendation:**

- removed synthetic data from production panel initialization
- added an explicit empty state until real metrics arrive
- cleared underlying realtime plot datasets on metric reset so stale
  chart data cannot leak into the next run
- if sample/demo mode is needed later, gate it behind a debug flag or dev-only
  build option

---

### 2. There are two training panels with overlapping responsibility

**Severity:** High
**Status:** Fixed on 2026-06-05

Relevant files:

- `cyxwiz-engine/src/gui/panels/training_dashboard.cpp`
- `cyxwiz-engine/src/gui/panels/training_plot_panel.cpp`

Observations:

- `TrainingDashboardPanel` renders charts, metrics, controls, and RL tabs
- `TrainingPlotPanel` also renders loss/accuracy/custom metrics and
  training status
- `TrainingPlotPanel` uses the window title `"Training Dashboard"`

This creates product ambiguity and implementation overlap.

**Impact:**

- two panels compete to represent training state
- duplicated chart logic
- duplicated controls and state concepts
- risk of inconsistent metrics or empty/filled state behavior

Audit result: this issue was still present. `TrainingPlotPanel` was the
graph/Python-connected supervised training surface, while
`TrainingDashboardPanel` was created from the node editor for RL training
but still used the same `"Training Dashboard"` title and carried supervised
loss/accuracy/throughput chart code.

Fix applied:

- kept `TrainingPlotPanel` as the canonical supervised training dashboard
- renamed the node-editor-created panel to `"RL Training Dashboard"`
- removed supervised chart/state methods from `TrainingDashboardPanel`
- limited the RL dashboard UI to episode and policy diagnostic metrics
- removed local start/pause controls that only toggled panel-local state
- fixed RL launch ordering so reset happens before marking RL training active

**Follow-up:**

- keep supervised metric ingestion and lifecycle centralized in
  `TrainingPlotPanel`
- keep RL-specific metric streaming in `TrainingDashboardPanel` unless it is
  later promoted into the canonical training dashboard as an owned child view

---

### 3. Some UI entry points claim missing features that already exist

**Severity:** Medium
**Status:** Fixed on 2026-06-05

Relevant files:

- `cyxwiz-engine/src/gui/panels/toolbar.cpp`
- `cyxwiz-engine/src/gui/panels/python_settings_panel.cpp`

Audit result: this issue was still present. The toolbar logged:

- `Python settings panel not yet implemented in preferences`

But `python_settings_panel.cpp` already exists and has a real UI.

This is an integration gap.

**Impact:**

- users are told a feature is missing when it is not
- preferences surface feels incomplete
- internal feature discoverability is poor

**Recommendation:**

- wired the existing Python settings panel into Preferences -> Python
- removed the stale "not yet implemented" button/log path
- audit other menu items for similar stale placeholder messaging
- treat unhooked-but-implemented panels as integration bugs

---

## Priority 1: Graph and Debug UX

### 4. Debug infrastructure exists, but the graph-facing UI is still too thin

**Severity:** High
**Status:** Fixed on 2026-06-05

Relevant files:

- `cyxwiz-engine/src/core/debug_executor.h`
- `cyxwiz-engine/src/core/debug_executor.cpp`
- `cyxwiz-engine/src/core/graph_compiler.cpp`

The backend/frontend boundary already exposes useful debug structures:

- `DebugResult`
- `LayerTrace`
- `GradNormEntry`
- `DebugStage`
- compile and validation issues

This is strong raw capability. The missing piece is a first-class UI for
it.

Audit result: this issue has been implemented as the `Studio Debugger`
panel rather than a separate `Debug Results` panel.

Implemented behavior:

- `Local Debug` is available from the graph toolbar and `F6`
- debug execution is gated by the same compile pass used before training
- `DebugExecutor` results are pushed into `StudioDebuggerPanel`
- the panel renders stage/failure summaries, layer traces, shape comparison,
  NaN/Inf state, timing, gradient lenses, issue lists, recommendations, and
  run history
- trace rows and selected trace details can focus the corresponding graph node
- the panel is registered in the sidebar as `Studio Debugger`
- successful debug runs cache the current graph hash for training staleness
  checks

**Follow-up:**

- reduce dependence on the legacy compile-result popup for Local Debug; the
  popup can remain as a short status handoff, but the durable result surface is
  `Studio Debugger`

---

### 5. Graph validation is still structurally correct but too shallow for ML workflow guidance

**Severity:** High
**Status:** Partially fixed on 2026-06-05

Relevant file:

- `cyxwiz-engine/src/gui/node_editor_validation.cpp`

Audit result: this item was partly stale. `node_editor_validation.cpp` is still
mostly structural, but the main compile gate now carries deeper semantic
validation in `GraphCompiler`.

Current graph/editor validation covers:

- empty graph
- input/output presence
- cycle detection
- reachability
- required pins
- a specific 4D -> Dense without Flatten warning
- data-loaded state via the loader registry
- required input and output pin connectivity
- label stream reaches loss targets
- prediction stream reaches loss predictions
- optimizer loss input reaches a real loss node
- unsupported/template nodes on the selected training path
- DataSplit ratio and batch-size sanity checks
- image resize/memory-risk checks
- shape-op parameter checks

Additional fix applied:

- added loss/output compatibility checks in `GraphCompiler`
- `BCELoss` / `BCEWithLogits` now block model paths with more than one
  prediction output
- `CrossEntropyLoss` now blocks single-logit outputs and mismatches between
  Output node class count and model output size
- covered the new checks in `test_graph_compiler_deferred_nodes`
- added selected-path preprocessing/domain validation in `GraphCompiler`
- domain-specific preprocessing nodes now block compile when they do not match
  the selected DataInput category, e.g. image `Resize` on a tabular data path
- covered the domain mismatch check in `test_graph_compiler_deferred_nodes`

That is useful, but still not enough for a visual ML tool.

**Missing validation depth:**

- dataset/task-type mismatches
- label tensor wiring problems
- train/val/test split semantic checks
- loss/input shape compatibility
- dtype warnings
- suspicious data path warnings beyond the selected training path
- invalid training graphs that are graph-valid but compile-invalid

**Recommendation:**

- move validation toward semantic pipeline validation, not just graph
  topology validation
- surface blocking vs warning vs info levels clearly
- add auto-fix actions where possible
- render issues on-canvas, not only through generic error messages

---

### 6. Node editor and graph tooling appear split across many files without a clear UX contract

**Severity:** Medium
**Status:** Fixed on 2026-06-05

Relevant files include:

- `node_editor.cpp`
- `node_editor_nodes.cpp`
- `node_editor_io.cpp`
- `node_editor_connection.cpp`
- `node_editor_context_menu.cpp`
- `node_editor_codegen.cpp`
- `node_editor_validation.cpp`
- `node_editor_shape_inference.cpp`
- `node_config_dialog.cpp`
- `node_documentation.cpp`

This is not automatically a problem, but the editor feature set is
spread across many implementation units with separate behaviors.

The risk is not file count by itself. The risk is that user interactions
become inconsistent:

- add/search/configure
- validate
- debug
- generate code
- inspect docs

Audit result: this issue was still present. Existing docs covered Data Studio
handoff and debugger architecture, but there was no concise source-of-truth
contract for the node editor lifecycle itself.

Fix applied:

- added `docs/Data Studio/node_editor_workflow_contract.md`
- defined the canonical add/configure/connect/validate/compile/debug/train
  lifecycle
- documented source-of-truth ownership for graph, node params, data registry,
  compile state, debug state, and training state
- documented command, shortcut, and feedback rules to keep future UI additions
  aligned

**Follow-up:**

- keep new node-editor entry points aligned with the documented lifecycle
- update the contract when a stage changes ownership or feedback surface
- use the contract during future toolbar/menu/command-palette audits

---

## Priority 2: Data Input and Dataset Workflow

### 7. `DataInputDialog` is powerful but overgrown

**Severity:** High
**Status:** Fixed on 2026-06-06

Relevant file:

- `cyxwiz-engine/src/gui/data_input_dialog.cpp`

This dialog already covers a lot:

- file / ML dataset / database / cloud sources
- tabular / image / audio / video / text / time series categories
- previews
- memory estimates
- async apply/load behavior
- registry restore logic
- text/audio/image layout variants

That capability is good. The implementation and UX are overloaded.

**Observed signs:**

- very large file
- many persistence/restore edge-case fixes embedded inline
- multiple source types and category-specific branches
- async state coordination mixed with render/config logic

**Product risk:**

- harder to keep behavior consistent across modalities
- easier to regress persisted state
- users face a dialog that does too much in one place

Audit result: this issue was still real after earlier Data Input fixes. The
dialog had pure metadata, capability, detection, preview parsing, and label
distribution logic embedded beside rendering and async apply code.

Fix applied:

- added `gui/data_input_capabilities.{h,cpp}`
- moved source-type enum ownership, apply support checks, preview support
  checks, unsupported-status messages, preview-unavailable messages, and byte
  formatting into the helper
- moved file/source parameter mapping, file type/category detection, source
  labels, apply-path summaries, backend labels, and dataset-name generation
  into the same helper
- added `gui/data_input_preview.{h,cpp}` for preview table parsing and label
  distribution calculation
- kept `DataInputDialog` methods as thin wrappers around the extracted helpers
  so existing render/apply code paths and user-visible behavior stay unchanged

Follow-up boundary left intentionally narrow: the dialog still owns ImGui
rendering and async apply orchestration, but the non-UI metadata/model logic is
now outside the monolithic file. Future modality-specific UI components can be
added without redoing this core split.

**Recommendation:**

- split by concern:
  - state restore / persistence
  - loader dispatch / apply
  - preview loading
  - category-specific UI sections
  - status/memory reporting
- consider modality-specific child views/components for:
  - tabular
  - image
  - audio
  - text
  - time series

---

### 8. Data Input still contains unsupported or partial experiences in the main workflow

**Severity:** High
**Status:** Fixed on 2026-06-05

Relevant file:

- `cyxwiz-engine/src/gui/data_input_dialog.cpp`

Observed gaps:

- video path explicitly reports unsupported
- audio preview is not yet implemented
- image preview says coming soon when no preview textures exist
- database connection test is TODO
- cloud download is TODO
- preview loading TODO notes remain for several source types

These are acceptable during development, but they should not all sit in
the primary configuration surface without clearer capability boundaries.

Audit result: this issue was still present. The dialog exposed planned
database, cloud, ML dataset, and video flows as primary actions, and some
buttons still reported simulated success.

Fix applied:

- labelled planned source/category modes directly in the selector
- kept planned modes selectable so existing saved configurations remain
  inspectable
- blocked Apply for ML dataset, database, cloud, and video paths with explicit
  "not wired yet" status instead of silent no-op/fake success
- disabled database connection/query and cloud listing action buttons
- removed simulated database connection and ML dataset download success states
- disabled preview loading where no preview implementation exists, while
  preserving supported tabular/text/time-series preview behavior

**Follow-up:**

- explicitly mark experimental or unavailable source modes in the UI
- disable unsupported modes before selection where possible
- separate "supported today" vs "planned" capabilities
- avoid letting unsupported flows feel one click away from working

---

### 9. Data preview and memory reporting are useful, but should become more operational

**Severity:** Medium
**Status:** Fixed on 2026-06-06

Relevant files:

- `cyxwiz-engine/src/gui/data_input_dialog.cpp`
- `cyxwiz-engine/src/gui/properties.cpp`

The frontend already exposes:

- previews
- estimated RAM
- loaded memory bytes
- disk-backed cache behavior
- pinned-memory settings

That is a good foundation.

What is still missing is higher-confidence operational feedback:

- actual load time
- actual parsed row count vs estimated row count
- column type inference summary
- class distribution summary
- split summary before training
- "what will happen when I click Apply" summary

Audit result: this issue was still present. The dialog had preview rows,
memory-tab actuals, loader status messages, and audit details, but the user
had to move between surfaces to understand the current source, preview state,
Apply path, loaded backend, loaded size, and load duration.

Fix applied:

- added a compact `DATASET SUMMARY` section to `DataInputDialog`
- before Apply, the summary shows selected source, category, preview row/column
  count, disk-size estimate, and the exact Apply path
- while Apply is running, the summary shows loading state, dataset name,
  source, and Apply path
- after Apply, the summary shows authoritative loaded dataset name,
  rows/samples, columns, backend, actual or estimated footprint, audit counts,
  and UI-measured Apply duration
- distinguished actual RAM, disk-backed storage, and lazy-loader fully-cached
  estimates in one place

**Follow-up:**

- add split summaries once train/validation/test split state has a single
  authoritative owner
- add richer class/column-type distribution summaries from loaded registry
  metadata rather than re-parsing in the dialog

---

## Priority 3: Start Page and Project Entry

### 10. Start page exists, but onboarding is still shallow

**Severity:** Medium
**Status:** Partially fixed on 2026-06-06

Relevant file:

- `cyxwiz-engine/src/gui/dialogs/start_page.cpp`

Current start page supports:

- recent projects grouped by time
- search
- create new project
- open project
- open folder
- clone repository
- continue without project

Previous audit found several actions that were exposed before they were fully
implemented:

- open folder
- clone repository
- create project dialog

Also, the page is still project-management oriented rather than workflow
oriented.

Audit result: the start page still had clickable TODO actions for create
project, open folder, and clone repository.

Fix applied:

- rendered the existing start-page create-project modal instead of setting an
  unused flag
- wired create-project to `ProjectManager::CreateProject`
- added location browsing through `FileDialogs::SelectFolder`
- changed "Open a folder" to "Open a project folder" and wired it to open a
  `.cyxwiz` project file from the selected folder
- marked repository cloning as planned/unavailable instead of exposing it as a
  clickable no-op
- aligned `ProjectSelectionDialog` with the same project-entry contract:
  - create-project location browsing now uses `FileDialogs::SelectFolder`
  - open-project now uses the shared `FileDialogs::OpenProject`
  - newly created projects select `ProjectManager::GetProjectFilePath`
  - project selection accepts either a `.cyxwiz` file or a project folder and
    resolves to the authoritative `.cyxwiz` file
  - removed the ineffective virtual-environment checkbox from that dialog
- centralized `.cyxwiz` project-file resolution in
  `ProjectManager::ResolveProjectFilePath`, then reused it from command-line
  startup, the start page, and `ProjectSelectionDialog`

Additional fix applied on 2026-06-06:

- added a `Starter graphs` section to the start page using real
  `examples/cyxgraph/*.cyxgraph` files
- listed bounded first-run workflows for vision, NLP, and audio examples
- added a distinct `ExampleGraphSelected` start-page result so `.cyxgraph`
  templates are not treated as `.cyxwiz` project files
- routed start-page graph opening through the existing `NodeEditor::LoadGraph`
  implementation via `MainWindow::OpenGraphInNodeEditor`
- reused the same graph-open wrapper from the asset browser to keep graph
  loading behavior centralized

**What is missing:**

- "new tabular project / new vision project / new NLP project"
- recent examples or guided demos
- "resume last graph" / "continue last training session"
- repository clone workflow

**Recommendation:**

- add a template gallery as a first-class entry path
- make project creation domain-aware
- keep recent projects, but do not let them be the whole first-run UX

---

### 11. Start-page implementation still carries thread-unsafe time conversion helpers

**Severity:** Low
**Status:** Fixed on 2026-06-05

Relevant file:

- `cyxwiz-engine/src/gui/dialogs/start_page.cpp`

Audit result: this issue was still present. The code used `std::localtime`,
including a month-grouping path that called it twice and compared pointers to
the same static buffer.

Fix applied:

- added a small platform-safe local-time helper using `localtime_s` on Windows
  and `localtime_r` elsewhere
- copied `std::tm` values before comparison in `IsThisMonth`
- reused one timestamp formatter for all recent-project groups

This is not the biggest issue in this file, but it is a recurring
quality smell in UI code that may eventually run across multiple
background-driven surfaces.

**Follow-up:**

- keep UI formatting helpers centralized

---

## Priority 4: Properties Panel and Configuration Surfaces

### 12. `properties.cpp` is carrying too much responsibility

**Severity:** High
**Status:** Partially fixed on 2026-06-06

Relevant file:

- `cyxwiz-engine/src/gui/properties.cpp`

The properties panel appears to handle:

- node parameter editing
- shape/memory information
- training-related options
- pinned memory
- previews
- presets
- dataset-related messaging

This file is very large and likely acting as a generic sink for
configuration logic.

Audit result: this issue is still real. `properties.cpp` had pure generic
parameter policy mixed into the panel implementation: hidden-parameter rules,
strict numeric parsing, numeric range parsing, and metadata validation.

Fix applied:

- added `gui/properties_parameter_rules.{h,cpp}`
- moved generic parameter hiding policy, strict int/float parsing, validation
  range parsing, and metadata parameter validation into that helper
- removed the `Properties::ValidateParameter` member so the panel keeps the
  ImGui rendering while the reusable parameter rules live outside the panel
- added `gui/properties_shape_info.{h,cpp}`
- moved shape formatting, batch-size lookup, dataset-shape resolution,
  output-shape inference, layer-parameter counting, and graph shape traversal
  into the shape helper
- kept the existing `Properties` shape methods as thin wrappers so rendering
  code and existing call sites stay stable
- added `gui/properties_presets.{h,cpp}`
- moved built-in preset discovery and placeholder save/load policy out of
  `Properties`, leaving the panel responsible only for rendering preset
  controls and invalidating shape state after a load
- added `gui/properties_executor.{h,cpp}`
- moved executor availability checks, executor config/result rendering, and
  placeholder executor input setup out of `Properties`, removing the executor
  factory dependency from the panel header
- added `gui/properties_advanced.{h,cpp}`
- moved initial-position, graph connection counts, and raw debug parameter
  rendering out of `Properties` while keeping the section open-state owned by
  the panel
- added `gui/properties_node_editors.{h,cpp}`
- moved the long node-type-specific parameter editor switch into a helper with
  an explicit render context for shape invalidation, dynamic pins, and signal
  scope demo state
- added `gui/properties_metadata_editor.{h,cpp}`
- moved metadata-backed parameter grouping, advanced-parameter grouping, typed
  ImGui controls, default reset handling, and inline validation rendering out
  of `Properties`
- removed the Windows file-dialog dependency from `properties.cpp`; the helper
  now owns file-parameter browsing alongside the rest of metadata parameter UI
- moved tensor-shape and learnable-parameter rendering into
  `gui/properties_shape_info.{h,cpp}` next to the shape computation and
  formatting helpers
- removed private shape formatting/inference wrappers from `Properties`, so
  the panel no longer mirrors helper APIs it does not own

Remaining responsibility still in `properties.cpp`:

- section composition and high-level panel orchestration

**Impact:**

- higher regression risk
- difficult testing
- difficult to reason about ownership between node config dialogs and
  generic properties editing

**Recommendation:**

- split the panel by domain:
  - generic property editors
  - model/layer properties
  - dataset node properties
  - execution/training properties
  - preset management
- define what belongs in node-specific config dialogs vs the global
  properties panel

---

### 13. Parameter editing still looks more implementation-driven than user-driven

**Severity:** Medium
**Status:** Partially fixed on 2026-06-06

Relevant files:

- `cyxwiz-engine/src/gui/properties.cpp`
- `cyxwiz-engine/src/gui/node_config_dialog.cpp`

The engine already has many parameters and node types. The UX risk is
that parameter editing becomes a direct reflection of internal fields
instead of a task-oriented control surface.

Audit result: the existing metadata-driven renderer already had a usable
schema foundation through `ParameterDefinition` fields for type, default,
description, enum values, and validation. The right fix was to improve that
path rather than create a second renderer.

Fix applied:

- extended `ParameterDefinition` with optional UI label, group, required, and
  advanced fields while preserving existing aggregate initializers
- parsed the new optional fields from JSON node templates
- improved the generic properties renderer to show human-readable labels,
  required markers, grouped parameters, and an advanced-parameters subgroup
- added reset-to-default controls for metadata-backed parameters
- made bounded integer parameters render as sliders
- changed validation feedback from a bare marker to the actual inline error
  message
- hid underscore-prefixed/internal metadata keys from generic editing

**What to improve:**

- stronger typed controls
- grouped settings with better defaults
- hidden advanced/internal settings
- validation beside controls
- reset-to-default support
- required vs optional distinction

**Recommendation:**

- formalize a parameter schema for UI rendering where possible
- let nodes describe control type, grouping, help text, and validation
  metadata

---

## Priority 5: Tool and Panel Sprawl

### 14. The engine has many standalone analysis panels, which weakens the graph-first product story

**Severity:** High
**Status:** Partially fixed on 2026-06-06

Observed under:

- `cyxwiz-engine/src/gui/panels/*`

There are many standalone panels for:

- clustering
- decomposition
- normalization
- TF-IDF
- tokenization
- forecasting
- statistics
- signal and matrix tools
- many other one-off analysis surfaces

This is useful capability, but it creates two product modes:

1. graph-based ML workflow builder
2. toolbox of disconnected panels

Those two modes can coexist, but right now they appear insufficiently
unified.

Audit result: this issue is still real, but many of the apparent standalone
tool areas already have graph-facing `NodeType` coverage and several have
registered pipeline operators. The main gap is a current migration contract,
not another competing implementation surface.

Fix applied:

- added `docs/Data Studio/tool_panel_graph_migration_plan.md`
- classified which panels should remain standalone because they are inspectors,
  utilities, monitoring surfaces, or project/system management surfaces
- classified graph-backed tool families that should stop gaining duplicated
  panel-local configuration
- documented migration rules for future menu, command-palette, metadata, and
  pipeline-operator work

**Impact:**

- harder onboarding
- duplicated configuration concepts
- harder reproducibility
- harder project persistence across workflows

**Recommendation:**

- decide which tools should become node-backed workflows
- prefer graph-backed configuration and output persistence where
  possible
- keep truly standalone utilities only when they are clearly separate
  from the ML pipeline story

---

### 15. Panel discoverability is improving, but the command surface needs rationalization

**Severity:** Medium
**Status:** Partially fixed on 2026-06-06

Relevant files:

- `cyxwiz-engine/src/gui/panels/toolbar.cpp`
- `toolbar_*.cpp`

The toolbar and command palette are substantial. That is good.

Audit result: one concrete shortcut conflict was still present. `F6` is wired
globally to Local Debug, and the graph toolbar tooltip also presents F6 as
Local Debug, but the Train menu labelled `Pause` as `F6`.

Fix applied:

- added `Train -> Local Debug` with the `F6` shortcut
- wired the Train-menu Local Debug action to `MainWindow::LocalDebugGraphAndReport`
- removed the false `F6` shortcut from `Pause Training`
- marked toolbar TODO commands as planned and unavailable instead of exposing
  them as clickable no-ops:
  - `Simulation -> Run/Pause/Stop/Step/Load MJCF`
  - `Script -> Run Script`
  - `Deploy -> Quantize`
  - `Deploy -> Publish to Marketplace`
  - `Help -> Documentation/Keyboard Shortcuts/API Reference/Check for Updates`
- marked remaining exposed command TODOs as planned/unavailable:
  - `View -> Fullscreen`
  - `Find -> Find Previous`
  - `Replace in Files -> Replace All`
- preserved wired actions in the same menus, including MuJoCo plugin panel
  toggles, script creation/opening, model export, server connect/deploy, issue
  reporting, tutorials, and About

The remaining problem is command coherence:

- some menu actions are fully wired
- some are placeholders
- some launch panels
- some imply future execution paths

Additional fix applied on 2026-06-06:

- extended command-palette `ToolEntry` metadata with a typed command surface
  and availability state
- annotated graph-backed standalone panel families in the existing
  `all_tools_` list instead of adding a second command registry
- rendered command-surface badges in the command palette so users can
  distinguish commands, panels, graph-backed panels, and utilities
- blocked future planned/unavailable command entries from executing through the
  palette if they are added as visible search results
- removed the duplicate `Theme Editor` command-palette entry and kept the View
  command as the canonical entry

**Recommendation:**

- audit every command-palette and toolbar action
- classify as:
  - working
  - experimental
  - placeholder
  - hidden until ready
- do not expose commands that cannot complete their core user promise

---

## Priority 6: Implementation Hygiene and Maintainability

### 16. Several frontend files are large enough to slow safe iteration

**Severity:** High
**Status:** Partially fixed on 2026-06-06

Notable files:

- `cyxwiz-engine/src/gui/data_input_dialog.cpp`
- `cyxwiz-engine/src/gui/properties.cpp`
- `cyxwiz-engine/src/gui/theme.cpp`
- `cyxwiz-engine/src/gui/panels/toolbar.cpp`
- `cyxwiz-engine/src/gui/panels/script_editor.cpp`

Large files are not automatically wrong, but these files appear to mix:

- rendering
- state transitions
- persistence
- integration logic
- business rules
- async coordination

That makes changes slower and more error-prone.

Audit result: this is still real. `properties.cpp` and `DataInputDialog` have
already had several concern-specific helpers extracted, and `toolbar.cpp`
already had menu-specific split files. The next low-risk boundary was command
palette search/rendering, which was still embedded in the main toolbar file.

Fix applied:

- added `gui/panels/toolbar_command_palette.cpp`
- moved command-palette open/shortcut handling, fuzzy search, result filtering,
  and palette rendering out of `toolbar.cpp`
- kept `InitializeToolEntries()` in `toolbar.cpp` for now because it owns the
  callback wiring and command list declaration
- reduced `toolbar.cpp` from roughly 2446 lines to roughly 2229 lines without
  changing command behavior
- added `gui/panels/script_editor_edit.cpp`
- moved Script Editor edit/navigation/line/text transformation operations out
  of `script_editor.cpp`, including undo/redo, cut/copy/paste/delete,
  go-to-line, duplicate/move line, indent/outdent, case transforms, sort, and
  join lines
- reduced `script_editor.cpp` from roughly 4642 lines to roughly 4077 lines
  without changing the public Script Editor command API
- added `gui/data_io_dialogs.cpp`
- moved `DataOutputDialog`, `DataLoaderDialog`, and `DataSplitDialog`
  implementations out of `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` from roughly 3651 lines to roughly 3331
  lines while keeping the existing node configuration dialog API
- added `gui/data_input_dialog_preview.cpp`
- moved Data Input preview rendering and text label distribution UI out of
  `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 3059 lines while keeping
  preview loading and parsing on the existing helper paths
- added `gui/data_input_dialog_profile.cpp`
- moved Data Input profiling tab rendering and Arrow column-statistics
  computation out of `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 2816 lines while keeping
  loaded-dataset state cleanup in the main dialog implementation
- added `gui/data_input_dialog_helpers.cpp`
- moved Data Input file detection, preview loading, browse dialogs, source
  labeling, capability messages, and dataset-name helpers out of
  `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 2585 lines while keeping
  Apply, async result polling, and source/options rendering in the main file
- added `gui/data_input_dialog_filter_tabs.cpp`
- moved Data Input transformation, row-limit, and encoding tab rendering out
  of `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 2496 lines while leaving
  memory, audit, and dataset-source rendering in the main file for now
- added `gui/data_input_dialog_status_tabs.cpp`
- moved Data Input memory-management and dataset-audit tab rendering out of
  `data_input_dialog.cpp`, including audit issue parsing helpers
- reduced `data_input_dialog.cpp` further to roughly 2201 lines while leaving
  Apply, async polling, and dataset-source rendering in the main file
- added `gui/data_input_dialog_ml_dataset.cpp`
- moved Data Input ML dataset source/options rendering out of
  `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 1989 lines while leaving
  Apply, async polling, database/cloud rendering, and summary rendering in the
  main file
- added `gui/data_input_dialog_external_sources.cpp`
- moved Data Input database and cloud source rendering out of
  `data_input_dialog.cpp`
- added `gui/data_input_dialog_summary.cpp`
- moved Data Input dataset summary rendering out of `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 1926 lines while leaving
  Apply, async polling, and file/source option rendering in the main file
- added `gui/data_input_dialog_source_options.cpp`
- moved Data Input source selector, file source picker, and tabular/image/
  audio/video/text option rendering out of `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 1111 lines while leaving
  Apply, async polling, content composition, reset, and unload ownership in
  the main file
- added `gui/data_input_dialog_apply.cpp`
- moved Data Input Apply dispatch and async load result polling out of
  `data_input_dialog.cpp`
- reduced `data_input_dialog.cpp` further to roughly 428 lines while leaving
  constructor restore, content composition, reset, and unload ownership in
  the main file
- added `gui/theme_presets.cpp`
- moved Theme preset color definitions out of `theme.cpp`
- reduced `theme.cpp` from roughly 3010 lines to roughly 1004 lines while
  leaving preset selection, style config, ImNodes styling, and dock styling in
  the main file
- added `gui/panels/toolbar_search.cpp` and
  `gui/panels/toolbar_file_dialogs.cpp`
- moved Toolbar find-in-files helpers and file dialog wrappers out of
  `toolbar.cpp`
- reduced `toolbar.cpp` further to roughly 2455 lines while leaving the main
  render body and command initialization in place

**Recommendation:**

- break large files by workflow or concern
- keep render functions smaller and domain-local
- move non-UI logic into reusable service/helper classes

---

### 17. There are stale backups and notes inside the source tree

**Severity:** Medium
**Status:** Fixed on 2026-06-05

Observed files:

- `toolbar.cpp.backup`
- `toolbar.h.backup`
- `node_editor_enhancements.txt`
- `PLOT_MENU_REDESIGN.md`

Audit result: this issue was still present. All four files were tracked.

Fix applied:

- removed obsolete tracked backup copies:
  - `cyxwiz-engine/src/gui/panels/toolbar.cpp.backup`
  - `cyxwiz-engine/src/gui/panels/toolbar.h.backup`
- moved design notes out of source-adjacent directories:
  - `docs/Data Studio/legacy_notes/PLOT_MENU_REDESIGN.md`
  - `docs/Data Studio/legacy_notes/node_editor_enhancements.txt`

Some of these notes are still useful historical context, but they no longer
live next to active source files.

**Impact:**

- unclear source of truth
- easier accidental drift
- higher maintenance noise

**Follow-up:**

- keep source-adjacent directories focused on active implementation
- add new design notes under docs instead of beside `.cpp` / `.h` files

---

### 18. Frontend state ownership needs stronger boundaries

**Severity:** Medium
**Status:** Fixed on 2026-06-05

Current behavior suggests state is spread across:

- node parameters
- dialog local state
- data registry
- engine config
- project state
- panel-local caches

Some of that is inevitable. The problem is lack of a clearly documented
state model for critical workflows.

The `DataInputDialog` code in particular contains multiple restore and
re-sync paths because persisted params, registry state, and async apply
results can diverge.

Audit result: this issue was still present. The new node-editor workflow
contract defined some ownership boundaries, but the broader frontend state
model was not documented as an explicit source-of-truth matrix.

Fix applied:

- added `docs/Data Studio/frontend_state_ownership.md`
- documented authoritative owners for project/session, graph structure, node
  configuration, loaded datasets, async load state, compile results, debug
  results, training runs, layout state, and generated output
- documented Data Input restore/apply rules, graph compile invalidation,
  debugger snapshot rules, and training run ownership
- added a review checklist for future panels, dialogs, commands, and async
  tasks

**Follow-up:**

- use the ownership matrix during future Data Input and Properties refactors
- update the matrix when a workflow gains a new durable run/session store
- keep UI caches explicit and invalidated when their owner changes

---

## Suggested Feature Roadmap

### Phase 1: Trust and Feedback

1. Remove fake training-dashboard sample data
2. Consolidate training dashboard and training plot panel
3. Build a proper Debug Results panel using `DebugResult`
4. Add on-canvas compile/debug issue rendering
5. Wire Python settings into preferences

### Phase 2: Core Workflow UX

1. Refactor `DataInputDialog` into smaller modality-aware components
2. Add dataset summary and stronger pre-Apply feedback
3. Add starter templates to the start page
4. Tighten properties panel with typed/grouped controls

### Phase 3: Product Coherence

1. Decide which standalone tools become graph-backed nodes
2. Reduce placeholder commands from the main toolbar/command palette
3. Split oversized frontend files by responsibility
4. Clean stale source-adjacent backups/notes

---

## Best First Engineering Tasks

These are good pickup items that are valuable and bounded.

### Task 1: Fix training dashboard truthfulness

Scope:

- `gui/panels/training_dashboard.cpp`
- possibly `gui/panels/training_plot_panel.*`

Deliverables:

- no synthetic metrics in normal startup
- clean empty state
- clear distinction between real and absent training data

### Task 2: Wire Python settings into preferences

Scope:

- `gui/panels/python_settings_panel.*`
- `gui/panels/toolbar.cpp`
- preferences integration path

Deliverables:

- preferences entry opens the real panel
- stale "not yet implemented" log removed

### Task 3: Add Debug Results panel

Scope:

- `core/debug_executor.*`
- panel creation and toolbar/graph integration

Deliverables:

- latest debug run visible in a dedicated panel
- per-layer trace table
- grad norm table
- clickable node navigation

### Task 4: Split `DataInputDialog`

Scope:

- `gui/data_input_dialog.cpp`

Deliverables:

- category-specific UI blocks extracted
- preview/loading/apply/state logic separated
- no behavior change during first refactor pass

### Task 5: Start-page workflow upgrade

Scope:

- `gui/dialogs/start_page.*`

Deliverables:

- template entry points
- create-project flow actually implemented
- unsupported actions either implemented or hidden

---

## Closing Assessment

The engine frontend already has a lot of real product work in it. The
problem is not that it is empty. The problem is that too many features
landed as separate surfaces without enough consolidation around the core
CyxWiz workflow:

- create/open project
- load data
- build graph
- validate graph
- debug graph
- train model
- inspect results

That workflow should be the product spine.

The next frontend cycle should optimize for:

- trust
- clarity
- fewer overlapping surfaces
- stronger workflow continuity
- smaller, more maintainable frontend modules
