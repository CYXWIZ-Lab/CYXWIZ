# To Fix 48 - Graph Node Property Truth For All Editable Nodes

## Purpose

Make the Properties panel truthful for every editable graph node.

The current problem is broader than transformer or generation work. Many graph
nodes expose incomplete, stale, placeholder, or misleading properties. Users
should be able to click any editable canvas node and see the real contract that
the engine compiler, runtime, export/import path, debugger, and documentation
actually support.

## Core rule

Node properties must be generated from, or validated against, the same contract
used by compiler, runtime, export/import, debugger, and documentation.

No property should appear as editable unless the engine can actually consume it.
No implemented parameter should be hidden from the user when safe editing is
expected.

## Applies to all editable node families

- Data input and dataset nodes.
- Data split and dataloader nodes.
- Preprocessing and materializer nodes.
- Classical ML nodes.
- Model layer nodes.
- Recurrent and sequence nodes.
- Transformer and NLP nodes.
- Loss nodes.
- Optimizer nodes.
- Metric nodes.
- Debugger/probe nodes.
- Export/import/inference-related nodes when exposed on the graph.
- Plugin/custom nodes.

This is an engine-wide UI contract, not a ticket for one model family.

## Current issues to audit

- Properties panel fields may not match `GraphCompiler` parameters.
- Some implemented backend/runtime parameters are not exposed.
- Some UI fields appear editable but are ignored downstream.
- Some node metadata exists but does not drive the properties UI.
- Some advanced parameters need a dialog instead of crowding the side panel.
- Some node properties need status truth, not only editable values.
- Some graph nodes need explicit training/export/import/debugger support truth.

## Target behavior

When a user selects a node:

1. The Properties panel shows the current node type and current parameter values.
2. Editable fields are only shown when backed by real compiler/runtime support.
3. Read-only truth fields explain implementation status.
4. Unsupported or planned properties are hidden, disabled, or clearly labelled.
5. Large property groups open a focused dialog instead of overflowing the panel.
6. Changes update the graph node parameters using the same names consumed by the
   compiler and runtime.
7. The panel surfaces relevant warnings before training, export, or inference.

## Required truth fields

Each editable node should expose, where applicable:

- Node type and display name.
- Compiler support status.
- Runtime support status.
- Training support status.
- Export/import support status.
- Inference support status.
- Debugger/probe support status.
- Backend placement truth: CPU, GPU, mixed, unknown, unsupported.
- Expected input shape.
- Expected output shape.
- Required inputs.
- Optional inputs.
- Required parameters.
- Optional parameters.
- Default values.
- Valid ranges or allowed enum values.
- Ignored/deprecated parameters.
- Error/warning reason codes.

## Implementation plan

### 1. Contract inventory

Audit every editable node and record:

- Node metadata entry.
- Properties-panel implementation.
- Compiler extraction path.
- Runtime/module construction path.
- Export/import behavior.
- Debugger/backend placement behavior.
- Tests that prove the contract.

### 2. Central schema

Prefer a central property schema per node type instead of scattered hardcoded UI.

The schema should define:

- Parameter name.
- Display label.
- Type.
- Default value.
- Range or enum.
- Whether editable.
- Whether required.
- Whether advanced.
- Whether runtime-supported.
- Whether export/import-supported.
- Help text.

### 3. Properties panel rendering

Update the Properties panel so it renders from the schema where possible.

Do not keep separate UI labels that drift from compiler parameter names unless
there is an explicit mapping table.

### 4. Advanced dialogs

Use focused dialogs for nodes with large contracts, such as:

- DataInput.
- DataLoader.
- TF-IDF/vectorizers.
- Transformer blocks.
- Export/inference/generation controls.
- Debugger/probe nodes.
- Plugin nodes with many parameters.

### 5. Validation and warnings

The panel should surface contract warnings early:

- Parameter is serialized but ignored.
- Parameter is planned but not implemented.
- GPU requested but node is CPU-backed.
- Export/import does not preserve this setting.
- Runtime shape depends on another node.
- Training path differs from inference path.

## Initial priority order

1. DataInput and DataLoader.
2. Preprocessing/materializer nodes.
3. Loss, optimizer, and metric nodes.
4. Core model layers.
5. Recurrent and sequence nodes.
6. Transformer and NLP nodes.
7. Debugger/probe/inference/export nodes.
8. Plugin/custom nodes.

## Acceptance criteria

- Every editable graph node has a known property schema or an explicit
  unsupported/planned status.
- Properties panel does not show fake editable fields.
- Implemented compiler/runtime parameters are visible and editable where safe.
- Parameter names round-trip through graph save/load.
- Compiler consumes the same parameter names shown by the UI.
- Tests cover at least one representative node per family.
- Documentation explains the property truth rule.
- Any remaining unsupported nodes are listed with reason and follow-up path.

## Non-goals

- Do not redesign the full node editor visual style in this ticket.
- Do not implement missing algorithms just to make a property real.
- Do not add broad reflection frameworks unless a lean schema table is
  insufficient.
- Do not expose unsafe low-level parameters without validation.

## Notes

This ticket is connected to the broader debugger and engine-truth direction:
engineers need the graph to describe what will actually happen. The Properties
panel is the first place users should see that truth.
