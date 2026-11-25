---
title: Add a built-in node
description: Add a node to the Engine's catalogue, give it pins, settings and a runtime, and prove it with the contract tests.
type: how-to
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - architecture/overview.md
  - testing.md
  - parity-oracle.md
  - coding-standards.md
  - ../concepts/how-graphs-run.md
---

Use this when: you add a node type that ships inside the Engine. A node
that ships in a plugin uses the SDK instead; see
[Plugin developer guide](../extending/plugin-developer-guide.md).

A built-in node has one source of truth: its entry in the node metadata
registry. The Nodes panel, the search palette (Ctrl+Space), the canvas
right-click menu, the Properties panel and the node factory all read it.
You do not add the node to any of those surfaces by hand.

## Before you start

- A source build that compiles; see [Build from source](build-from-source.md).
- Know which band the node belongs to (data preparation, partitioning,
  phase-aware iteration, or model); see
  [The data pipeline](../concepts/data-pipeline.md). Nodes are real
  operations: data flows through them. Do not add a node that only carries
  settings for the compiler to scan.
- For a numerical operation, a PyTorch or scikit-learn reference to compare
  against; see [The parity oracle](parity-oracle.md).

## Steps

1. **Give the node a type.** In `cyxwiz-engine/src/core/graph_model.h`, add
   a value to `enum class NodeType` at the end of the list. The numeric
   value is written into every `.cyxgraph` file, so the enum is append
   only: never insert, reorder or reuse a value.
2. **Register its metadata.** In
   `cyxwiz-engine/src/core/node_metadata_registry.cpp`, add a
   `RegisterNode({...})` call to the `Initialize*Nodes()` function of its
   family (for example `InitializeActivationNodes()`). The `NodeMetadata`
   fields (`core/node_metadata.h`) are, in order:
   - type, category, display name, icon (a FontAwesome macro);
   - search keywords;
   - usage count and favourite flag (`0, false`);
   - a one-line description, a longer help text and an example;
   - input and output ports (`PortDefinition`: name, `PinType`, required,
     description, variadic, minimum and maximum connections);
   - parameters (`ParameterDefinition`: name, type, default, description,
     enum values, validation range such as `"0-100"` or `">0"`, display
     name, group, required, advanced, consumption);
   - status (`NodeImplementationStatus::Implemented`), badge, support axes
     and the Properties editor (`NodePropertiesEditor::Automatic` unless
     the node has a dialog).

   Parameter types are `string`, `multiline`, `password`, `int`, `float`,
   `enum`, `bool`, `file`, `directory`, `channels`, `fixed` and `column`.
   Mark a parameter `ParameterConsumption::UiOnly` only when no runtime
   code reads it.
3. **Check the pins and defaults the canvas gets.** `CreateGraphNode` in
   `cyxwiz-engine/src/core/graph_node_factory.cpp` builds every registered
   node's pins and default parameters from the metadata (its `default:`
   case). Add a `case` only for behaviour the metadata cannot express, and
   still call `PopulateStaticNodeContractFromMetadata` in it. The factory
   is shared by the editor and headless hosts, so a saved graph loads into
   the same pins everywhere.
4. **Give it an import name.** Patterns and other files that name node
   types by string resolve them through one table,
   `kNodeTypeImportNames` in
   `cyxwiz-engine/src/gui/node_type_import_registry.cpp` (used by both
   `node_editor_io.cpp` and `patterns/pattern_library.cpp`). Add
   `{"YourNode", NodeType::YourNode}`. Then raise the expected count in
   `CheckGraphImportNameContract` in
   `cyxwiz-engine/tests/test_node_contract_gates.cpp`; the test pins the
   count so a lost entry is caught.
5. **Give it a runtime.** Pick the path that matches the band:

   | Node kind | Where it runs | What to add |
   | --- | --- | --- |
   | Data step (Band 1 or 2) | `PipelineExecutor` | A class implementing `IPipelineOperator` (`core/node_executors/pipeline_operator.h`: `GetName`, `GetBand`, `Configure`, `Apply` on an Arrow table; optional streaming and sparse output) in `core/node_executors/`, registered with `RegisterCreator` in `PipelineOperatorFactory::RegisterDefaults()` |
   | Model layer or activation (Band 4) | Training | Shape inference and parameter resolution in `core/graph_compiler.cpp`, the backend module in `core/model_builder.cpp`, the layer's name and placement in `core/backend_placement_capabilities.h`, its training role in `core/pipeline_runtime_capabilities.cpp` |
   | Not runnable yet | Nowhere | A fail-closed entry in `core/pipeline_runtime_capabilities.cpp` with a reason longer than a few words; the registry then shows the node as blocked with that reason |

   The compute itself belongs in `cyxwiz-backend` (written once on
   ArrayFire, device resident), not in the Engine. See
   [Compute back ends inside the backend](architecture/compute-back-ends.md).
6. **Optional: code export.** If the node can be written as PyTorch, add it
   to `core/code_export.cpp`; otherwise the export refuses the graph and
   names the node.
7. **Choose how its settings are edited.** With
   `NodePropertiesEditor::Automatic`, the Properties panel draws one typed
   field per parameter from the metadata. A node that needs a preview,
   a file picker or validation gets a dialog: derive from
   `NodeConfigDialog`, register it in the `NodeConfigDialogFactory`
   constructor (`gui/node_config_dialog.cpp`, `RegisterDialog`), and set
   `NodePropertiesEditor::Dialog`. The **Open Dialog...** button appears
   because a dialog is registered; there is no separate list to update.
8. **Write its help.** Add an entry in
   `NodeDocumentationManager::InitializeDocumentation()`
   (`gui/node_documentation.cpp`): title, description, usage, a line per
   parameter, tips and category. The Node Info panel shows it.
9. **Test it.**
   - Build and run `test_node_contract_gates` directly; it is not in ctest.
     It checks every registered node: the parameter and port schema, that
     the factory creates exactly the metadata's pins and parameters, the
     Properties path, that an implemented node declares an implementation
     owner and a support state, and the import names. Pass
     `--inventory-json <file>` to write the full node inventory as JSON.
   - For a numerical node, add a parity test against the reference; see
     [The parity oracle](parity-oracle.md).
   - Add a graph that uses the node to a test or an `examples/` project
     when the node changes what a graph can do.
10. **Document it in the same change.** The node's reference page is
    generated: rebuild `tool_node_reference_dump` and run
    `python scripts/docs/generate_node_reference.py` (ctest
    `node_reference_docs` fails until you do). Hand-written notes for the
    node or its family go in `docs/reference/nodes/_notes/`; how-to text goes
    in the area's user guide. Add a release-note fragment in `docs/release-notes/next/` from
    `docs/internals/docs/templates/release-note-fragment.md`. Run
    `python scripts/docs/check_docs.py`.

## If something goes wrong

**"No node metadata registered for type N"** in the log: the type has no
`RegisterNode` entry, so the factory cannot build its pins. Add step 2.

**"Unknown node type 'X'"** when a pattern loads: the string is missing
from `kNodeTypeImportNames` (step 4).

**"accepted graph-import name count drifted"** from
`test_node_contract_gates`: you added or removed an import name without
updating the expected count.

**"implemented node has no declared implementation owner"**: the node is
marked implemented but nothing claims to run it. Add its runtime (step 5)
or mark it fail-closed.

**"Serialized node 'X' has unsupported node type id N"** when a graph opens:
the file was saved by a build with a newer enum value. This is why the enum
is append only.

## Related

- [Architecture overview](architecture/overview.md)
- [How graphs run](../concepts/how-graphs-run.md)
- [Tests](testing.md)
- [Coding standards](coding-standards.md)
- [Decision: the four pipeline bands](decisions/0006-four-pipeline-bands.md)
