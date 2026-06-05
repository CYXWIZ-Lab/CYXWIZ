# Node Editor Workflow Contract

This document defines the canonical CyxWiz Studio node-editor workflow. It is
not a redesign proposal; it is the contract the current implementation should
keep converging toward.

## Product Spine

The node editor is the primary workflow surface for model work:

1. Add nodes.
2. Configure nodes.
3. Connect the graph.
4. Validate the graph.
5. Compile the graph.
6. Debug the graph.
7. Train the model.
8. Inspect results and iterate.

Standalone panels can support this workflow, but they should not create a
second, conflicting model-building path.

## Stage Contract

| Stage | Primary user action | Source of truth | Feedback surface | Blocking rule |
| --- | --- | --- | --- | --- |
| Add node | Node browser, context menu, pattern insert, template insert | `NodeEditor` node list | Canvas node appears with expected pins | Template/unsupported nodes must be labelled or disabled |
| Configure node | Properties panel or node config dialog | Node parameters | Inline controls, validation hints, node description | Required parameters must be visible before Compile |
| Connect graph | Canvas links | Node/link graph | Pin color, connection validation, issue list | Invalid pin type/direction links must not persist |
| Validate graph | Automatic checks plus Validate/Compile entry points | `node_editor_validation` and compiler preflight | On-canvas issues plus compile/debug issue list | Structural errors block Compile, Debug, and Train |
| Compile graph | Compile button or compile gate before Debug/Train | `GraphCompiler` result | Compile popup and Studio Debugger handoff | Compile errors block Local Debug and Train |
| Debug graph | Local Debug button, Train menu Local Debug, or F6 | `DebugExecutor` plus immutable graph snapshot | Studio Debugger, trace rows, node focus | Failed debug marks the graph not ready for Train |
| Train model | Start Training button, Train menu, or F5 | Compiled training configuration and loaded dataset registry | Training dashboard, task/progress panels, graph staleness gate | Train requires clean compile and current successful Local Debug |
| Inspect results | Training plots, Studio Debugger, output panels | Training run state and persisted debug/training traces | Canonical supervised/RL dashboards plus debugger lenses | Result panels must not synthesize metrics as real data |

## Ownership Rules

- `NodeEditor` owns graph structure, selection, canvas state, and graph-local
  visual feedback.
- Node config dialogs own modal edits for a single node, but commit through
  node parameters.
- The Properties panel owns lightweight selected-node edits and summaries. It
  should not become a second data registry or training state owner.
- Data loading state is owned by the loader registry and reflected into node
  parameters only as a project/persistence hint.
- Compile state is owned by the compiler result for the current graph snapshot.
- Debug state is owned by Studio Debugger sessions and should reference the
  immutable graph snapshot used for the run.
- Training state is owned by the training execution path and its dashboards.

## Entry Point Rules

- Every visible command should map to one of the product-spine stages above.
- If a command is planned but not executable, label it as planned and disable
  execution controls.
- If a command opens a panel, the panel should clearly advance or inspect the
  graph workflow.
- Keyboard shortcuts, toolbar buttons, and menu items for the same action must
  call the same callback.
- Popups may provide immediate status, but durable debug and training results
  belong in the relevant panel.

## Feedback Rules

- Blocking errors should be visible before the user reaches Train.
- Compile, Local Debug, and Train should reuse the same severity language:
  error blocks progress, warning needs review, info explains context.
- Canvas feedback should point to the node or link responsible for an issue
  whenever possible.
- The Studio Debugger is the durable surface for Local Debug and graph trace
  details.
- Empty states must be explicit. No panel should show sample metrics, simulated
  success, or placeholder output as real workflow state.

## Implementation Boundaries

The current file split can stay as long as responsibilities remain clear:

- `node_editor.cpp`: canvas orchestration and top-level toolbar interactions
- `node_editor_nodes.cpp`: node catalog and node creation
- `node_editor_connection.cpp`: link creation and connection queries
- `node_editor_context_menu.cpp`: context creation/search entry points
- `node_editor_validation.cpp`: editor-local structural validation
- `graph_compiler.cpp`: compile-time semantic validation and training config
- `node_config_dialog.cpp`: modal node configuration
- `node_documentation.cpp`: node help/documentation
- `studio_debugger_panel.cpp`: durable debug/run inspection

When a new feature crosses these boundaries, prefer a narrow callback or typed
result object over adding another broad shared state path.
