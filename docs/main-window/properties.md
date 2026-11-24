---
title: Properties panel
description: The panel that shows the selected node's settings, where each value comes from, what the compiler built and what the file stores.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - ../user-guide/graph-editor/node-properties.md
  - node-dialogs.md
  - editing-values.md
redirect_from:
  - ../usage/properties.md
---

The **Properties** panel (right column in the default layout; sidebar
**Properties**, or **View > Panels > Workspace > Properties**) shows the
node selected on the CyxWiz Studio canvas: what it is, its settings and
where each value comes from, what the compiler built for it, and what the
graph file stores. For the steps of changing settings, see
[Set node properties](../user-guide/graph-editor/node-properties.md).

With nothing selected it shows **Select a node to edit its properties** and
the graph in one line: name, node and link counts, and the compile state
with the first error's node. It also says "Double-click a node to open its
dialog.", which applies only to the nodes that have a dialog (below).

## The header

| Part | Meaning |
| --- | --- |
| Icon and name | Click the name to rename the node; Enter or clicking away saves it. |
| type · category · #id | The node type, its group and its id in the graph. |
| Status chip | Quiet when *Implemented*; *Planned* (cannot run yet), *Deprecated* and *External* are coloured. |
| Badge | The type's badge when it has one, for example *Coming Soon*. |
| Primary button | **Open Dialog...** on nodes set up in a window of their own (Data Input, Data Output, Data Convert, Data Loader, Data Split, Embedding, Tokenizer, Row Filter, SQL Query, Quality Analyzer, Word Embeddings, and the older Vocabulary and Padding nodes), **Open Plot window** (Plot) or **Open Dashboard** (Dashboard). |
| **Actions** | The menu below. |

### Actions

| Item | Does |
| --- | --- |
| **Apply preset** | The type's built-in presets (Dense: Small (64), Medium (256), Large (1024); Conv2D: VGG-style, ResNet-style, MobileNet-style; Adam: Default, Fast learning, Fine-tuning) and the ones you saved. Hover a preset for its values. |
| **Save current as preset...** | Saves the node's settings under a name, for this node type, in `%APPDATA%\CyxWiz\node_presets.json` on Windows and `~/.cyxwiz/node_presets.json` on Linux and macOS. A saved preset with a built-in's name replaces it. |
| **Delete saved preset** | Removes one of your presets. |
| **Reset all settings to defaults** | Every declared setting back to its default. |
| **Copy settings as JSON** | Name, type and parameters, to the clipboard. |
| **Remove unused keys** | Removes stored keys that no setting uses any more. |
| **Show node in canvas** | Moves the view to the node. |

## Settings

One row per setting: the label, the editor and, at the right, a status mark
when the Engine's rules know the setting.

| Mark | Meaning |
| --- | --- |
| ✓ | The value is used as shown. |
| *dialog* | The value is set in the node's dialog. |
| *default* | The node has no value of its own; the default applies. |
| *alias* | The value came from an older key the rules still read. |
| *missing*, *conflict*, *stale*, *unsupported* | Need attention. |
| *runtime*, *compiler* | The value is decided when the graph compiles or runs. |

Hover a mark for the source key, the owner (UI, Compiler, Loader,
Materializer, Runtime or Exporter) and the rule's message. **▸ Details**
under the rows shows them for every row.

- Settings are grouped as the node type declares (Training, Generation
  previews, ...); rarely used ones are under a folded **Advanced
  settings**. A group's words are dropped from its labels ("Max new tokens"
  under Generation previews).
- Hover a label for the full name, description, default and **Required** /
  **Optional**.
- A reset arrow follows the label of a row whose value differs from its
  default (beside the value when the label fills its column). Hover it for
  the default; click it to go back.
- A value is written to the node only when you change it; looking at a node
  changes nothing.
- Numbers are typed fields with **-** / **+** steps. A setting with a range
  keeps its value in the range; its step is 0.01 for a range up to 1, 0.1
  up to 10, and 1 beyond. See [Editing values](editing-values.md).
- A problem with a value shows as a note under its row: "Required",
  "Invalid integer", "Invalid number", "Value must be >= N" / "Value must
  be <= N", "Invalid option", "Unsupported value; expected X".

Settings the rules know but no editor shows (for example a Data Input's
label column, owned by the loader) are listed read-only below the rows.

A **Data Input** also gets a **Data** row: the file, whether it is loaded
("Not loaded" with a clock until you **Apply** in its dialog), and the rows
× columns (or images and classes) once it is. Its **Dataset classes** row
counts the classes of the label column: text labels count their distinct
values, true / false labels are 2, and whole numbers from 0 are class
numbers, so the count is the largest one plus 1 (labels 0 to 3: 4
classes). Fractional or negative labels are a regression target: the row
says *unknown* and why. An Output node's class count is checked against it.

A node with no settings says **No editable settings for this node type**.
Plot and Dashboard nodes show the settings saved by their window, in words;
change them in the window.

## As compiled

The card shows what the compiler built for this node from the latest
background compile of the canvas:

| Status | Meaning |
| --- | --- |
| *Compiled* | The node compiled. |
| *Compiled, data not loaded* | It compiled, but the data is not loaded; shapes come from the data's saved description. Compile (F7) reports the unloaded data as an error. |
| *Compile failed* | The node has a compile error; the card lists it. |
| *Compiling...* | A compile is running; the card shows the previous one until it ends. |
| *Not compiled yet* | The graph compiles in the background after an edit. |
| *Not compiled for training* | The canvas is in Data Pipeline mode; the training compile does not apply. |
| *No model to compile* | The graph has no model layers, loss or optimizer (a data-only graph), so there is nothing to train. |

For a model layer the card also gives the input and output shapes per
sample and per batch, the output memory per batch, and the learnable
parameters and their memory. **▸ Details** adds the layer's parameters and
the node type's support (local training, remote training, export). The
shapes come from the compiler.

A Data Split on an image or audio graph shows its parts instead: a row per
part and a column per class, the images left out, and **Show the split in
Data Studio** (see [Image split](../usage/image_split.md)).

A Concatenate fed by an Embedding also gets the *Word + POS fusion* card
with the compiler's verdict and the fix when it will not compile.

A composite layer node shows **Inside this layer**: a table of its steps,
outputs and weights.

## Advanced

Position and connections, then **Raw parameters**: every key the graph
file stores for this node, its value, which setting it maps to, the
cleanup advice, and **Remove** where the rules allow it.

## Limits

- Changing a setting is not an undo step of its own; **Ctrl+Z** on the
  canvas also takes back settings changed since the last canvas change.
- The panel does not show which settings change results and which only
  change how an editor shows the node; the owner in a mark's tooltip is
  the nearest hint.

## Changes

- Changed in 1.0.8: **Reset all settings to defaults** in **Actions**, and
  the image and audio split view in the compiled card.

## Related

- [Set node properties](../user-guide/graph-editor/node-properties.md)
- [Node dialogs](node-dialogs.md)
- [Editing values](editing-values.md)
