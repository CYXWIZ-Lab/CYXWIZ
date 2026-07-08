# tofix38 - Properties Panel Truth and Node Configuration UX

## Purpose

Make the Properties panel a truthful, concise, editable view of the selected
node's effective configuration.

The current panel can show stale, generic, incomplete, or misleading values.
That creates real workflow errors: users believe a node has one setting while
the compiler, loader, materializer, or runtime reads a different key.

This ticket defines the Properties panel as a source-of-truth surface:

- show what the node will actually use,
- show enough context to understand where the value comes from,
- allow safe quick edits for small properties,
- route large or complex configuration into dedicated dialogs.

## Problem Examples

Recent sentiment/TF-IDF graph work exposed the issue:

- DataInput had `label_column=status`, but the text loader needed
  `text_label_column=status`.
- The graph appeared labeled to the user, but the text dataset registered as
  `0 classes`.
- Output/CrossEntropy mismatch happened because the visible model output was
  `Dense 7`, while the Output node still contributed `num_classes=10`.
- Recurrent node names such as `GRU 32` became false when `hidden_size` was
  changed to `8`.

These are not algorithm problems. They are truth-surface problems.

## Design Principle

Properties must show **effective truth**, not just raw stored strings.

For every selected node, the panel should distinguish:

- raw saved parameter,
- default value,
- effective value after aliases/defaults,
- validation status,
- compiler/runtime meaning,
- whether the value is editable inline or needs a dialog.

## Scope

### 1. Property source-of-truth mapping

Create a canonical property schema per node type.

Each property entry should define:

- canonical key,
- accepted aliases,
- display label,
- type: string, int, float, bool, enum, path, column, expression, JSON, list,
- default value,
- required/optional,
- validation rule,
- owning subsystem: UI, compiler, loader, materializer, runtime, exporter,
- whether it is quick-editable in the Properties panel,
- whether it requires a dedicated dialog.

Example:

```text
DataInput/Text
canonical: text_label_column
aliases: label_column, label_col
display: Label column
owner: text loader + dataset audit
effective value: status
quick editable: yes
dialog: Data Input dialog
```

### 2. Effective-value display

Properties panel should show the value the engine will actually use.

Examples:

```text
Label column: status
Source key: text_label_column
Aliases present: label_column=status
Status: OK
```

```text
Output classes: 10
Model output width: 7
Status: Error - CrossEntropy expects class count to match model output width
```

```text
LSTM hidden size: 64
Placement: CPU
Reason: cuda_jit_param_overflow_risk
```

### 3. Concise by default

The default panel should not dump every raw parameter.

Show a compact "truth summary" first:

- node name,
- node type,
- role in graph,
- required properties,
- effective values,
- current validation state,
- compile/runtime warnings that directly affect this node.

Raw parameters should be available in an advanced foldout, not mixed into the
main editing surface.

### 4. Inline edits only for small/simple properties

Keep the panel fast for common edits:

- names,
- numeric hyperparameters,
- booleans,
- enums,
- small text fields,
- selected column names,
- dropout rate,
- hidden size,
- batch size,
- learning rate,
- class count.

Edits must update the canonical key and any required compatibility aliases.

Example:

Changing DataInput text label column should write:

```text
text_label_column=status
label_column=status
```

when both are still required by current code paths.

### 5. Use dialogs for large or complex properties

If a property is large, structured, or needs preview/inspection, the Properties
panel should show a concise summary and an "Open..." action.

Use dedicated dialogs for:

- DataInput source/file/schema/preview configuration,
- tokenizer vocabulary building and inspection,
- TF-IDF vocabulary/feature preview,
- large JSON blobs,
- model artifact paths,
- optimizer advanced options,
- scheduler configuration,
- augmentation chains,
- column-selection lists with many columns,
- expressions/formulas,
- debugger/profiler traces.

The Properties panel should not become a giant form.

### 6. Validation and truth badges

Add small status indicators per property:

- OK,
- Missing,
- Defaulted,
- Alias used,
- Stale,
- Conflicting,
- Runtime-only,
- Compiler-only,
- Unsupported,
- Requires dialog.

Examples:

```text
Label column: status [OK]
label_column alias also present [Alias used]
```

```text
pin_memory: true [Unsupported]
No pinned host-memory transfer backend exists yet.
```

```text
GRU backend: CPU [Runtime truth]
Reason: gru_arrayfire_cuda_probe_required
```

### 7. Graph-aware property truth

Some node properties are only true when compared with graph context.

The panel should surface graph-aware truth for:

- CrossEntropy class count vs final Dense output width,
- Output node `num_classes` vs dataset class count,
- DataLoader class balancing vs train-only sampler behavior,
- TextTokenizer max_length vs compiled input shape,
- TFIDFVectorizer max_features vs compiled input shape,
- recurrent hidden size/sequence length vs backend placement,
- label column presence vs dataset audit result.

### 8. Raw parameter inspector

Add an advanced raw inspector for debugging.

It should show:

- raw key/value map,
- canonical mapping result,
- aliases,
- unknown keys,
- stale keys,
- values that will not be read by any subsystem.

Unknown/stale keys should be visible but not scary. The user should be able to
clean them when safe.

### 9. Save/load and template compatibility

The truth layer must work for:

- live GUI nodes,
- saved graph JSON,
- pattern template JSON,
- imported graph formats,
- nodes created from the node browser,
- nodes created from tool-to-node workflows.

When loading old graphs, aliases should be normalized lazily and safely. Do not
break older saved projects just because they use old parameter names.

## Implementation Notes

Likely areas:

- `node_metadata_registry` for canonical property schemas,
- `properties_node_editors` and Properties panel rendering,
- node config dialogs for complex editors,
- graph compiler parameter extractors,
- pipeline materializer parameter extraction,
- DataInput dialog apply/load alias synchronization,
- node save/load normalization.

The schema should be shared. Avoid adding another disconnected list of property
names that can drift from compiler/runtime usage.

## Acceptance Criteria

- Selecting a node shows a concise effective configuration summary.
- DataInput text label truth shows `text_label_column` and does not hide behind
  generic `label_column`.
- Output/CrossEntropy mismatch is visible from the relevant node properties
  before the user has to read raw compile logs.
- TFIDFVectorizer shows `max_features` as the effective model input width.
- LSTM/GRU properties show backend placement truth when a compile report exists.
- Large/complex configs show a summary plus an "Open dialog" action.
- Raw parameter inspector shows aliases, unknown keys, and stale keys.
- Editing a property updates the canonical key and required compatibility
  aliases.
- Existing saved graphs continue to load.
