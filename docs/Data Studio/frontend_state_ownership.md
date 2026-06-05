# Frontend State Ownership

This document defines which object is authoritative for critical CyxWiz Studio
frontend state. It complements `node_editor_workflow_contract.md`.

The rule is simple: UI panels may cache state for rendering, but workflow truth
must have one owner.

## Ownership Matrix

| Workflow state | Authoritative owner | UI mirrors/caches | Persistence hint | Conflict resolution |
| --- | --- | --- | --- | --- |
| Project/session identity | Project manager and current project path | Start page recent list, window title, asset browser | Recent-project metadata and project file | If the project cannot be opened, UI recents are stale and must not create a session |
| Graph structure | `NodeEditor` nodes, links, and graph snapshot | Canvas layout, selection, mini-map, node browser filters | Saved `.cyxgraph` graph data | Saved graph loads replace editor graph; panel-local selections are discarded if invalid |
| Node configuration | Node `parameters` plus node-specific config dialog schema | Properties panel controls, node description, config dialogs | Node parameters in saved graph | Dialog edits commit to node parameters; cancelled dialogs do not mutate workflow truth |
| Loaded dataset | `DataRegistry` entry keyed by dataset name | Data Input status, Properties summary, Data Studio dataset dropdowns | Node `dataset_name` and `data_loaded` hints | Registry wins; if params say loaded but registry has no dataset, the UI must show not loaded |
| Data load in progress | `AsyncTaskManager` task plus loader `AsyncLoadState` | Data Input progress/status text | None until completion | In-flight state is provisional; Train must see `data_loaded=false` until completion |
| Compile result | `GraphCompiler` result for the current graph snapshot | Compile popup, issue list, node issue decoration | None by default | Graph edits invalidate old compile results |
| Debug result | Studio Debugger session and immutable graph snapshot | Debugger panel lenses, run history, node trace highlighting | Debug run store where available | Debug output only applies to the graph snapshot it was run against |
| Training run | Training executor/manager and compiled training configuration | Training dashboard, task/progress panels, plot panels | Run metadata/checkpoints where available | Graph/data changes after launch do not mutate the active run configuration |
| Panel visibility/layout | Docking/layout manager | Individual panel `visible` flags | UI layout settings | Layout state never changes graph, data, debug, or training truth |
| Generated output | The producing workflow or run record | Output panels, script console, plot viewer | Output artifacts/checkpoints where available | Empty or placeholder output must be labelled as absent, not shown as real result data |

## Data Input Rules

- `DataRegistry` is the source of truth for whether a dataset is loaded.
- `DataInputDialog` local state is only draft UI state until Apply starts.
- During async Apply, node parameters should mark `data_loaded=false`.
- After async Apply succeeds, the registry entry and node parameters are
  synchronized on the UI thread.
- Reopening the dialog must restore from node parameters, then verify loaded
  state against the registry.
- Unsupported or planned source modes must not set `data_loaded=true`.

## Graph And Compile Rules

- Graph edits invalidate compile/debug readiness for the edited graph.
- Compiler output is tied to a graph snapshot, not to the mutable canvas.
- Editor-local validation can warn early, but compile-time semantic validation
  owns the final blocking decision before Debug or Train.
- Canvas issue markers should reflect the latest validation/compile result only.

## Debug Rules

- Studio Debugger owns durable debug sessions and trace state.
- Local Debug may be launched from toolbar, menu, or F6, but all entry points
  must use the same callback and produce the same session shape.
- A successful debug run marks only that graph snapshot as debug-clean.
- A failed or stale debug result must not authorize training.

## Training Rules

- Training launch consumes a compiled configuration and a registry-backed
  dataset reference.
- Training dashboards render run state supplied by the training path. They must
  not invent metrics or reuse stale metric buffers after reset.
- User edits to the graph while training is active create a future graph state;
  they do not alter the active run.
- Result panels should identify which run or graph snapshot they display when
  that context is available.

## Review Checklist

Use this checklist when adding a panel, menu item, dialog, or async task:

- What state does it read?
- What state does it write?
- Which object is authoritative after the write?
- Is any UI cache invalidated when the owner changes?
- Can the user see when a cached result is stale?
- Does Apply/Run/Train fail closed when authoritative state is missing?
- Does the feature avoid simulated success, fake metrics, and placeholder
  output as real state?
