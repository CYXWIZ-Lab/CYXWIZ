---
title: Set node properties
description: Change a node's settings in the Properties panel or its dialog, return to defaults, and use presets.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - ../../main-window/properties.md
  - ../../main-window/node-dialogs.md
  - ../../main-window/editing-values.md
  - compile-and-run.md
---

Use this to change what a node does: a Dense layer's units, a Dropout rate,
the file a Data Input reads. Every node keeps its settings in the graph, and
they are saved with the `.cyxgraph` file.

## Properties panel or dialog

There are two places to set a node, and the node decides which:

| Where | Used for | How to open |
| --- | --- | --- |
| **Properties** panel | Most nodes: layers, activations, losses, optimizers, most data steps. One row per setting. | Click the node on the canvas. If the panel is hidden: **View > Panels > Workspace > Properties**. |
| Node dialog | Nodes that read or preview data: **Data Input**, **Data Output**, **Data Convert**, **Data Loader**, **Data Split**, **Tokenizer**, **Embedding**, **Row Filter**, **SQL Query**, **Quality Analyzer**, **Word Embeddings**, and the older Vocabulary and Padding nodes. | Double-click the node, right-click it and choose **Configure...**, or click **Open Dialog...** at the top of Properties. |

A node with a dialog still shows in Properties; when everything is set in
the dialog the panel says **This node is set up in its dialog.**

Double-click on a node without a dialog only selects it. The empty
Properties panel says "Double-click a node to open its dialog.", but that
works only for the nodes listed above; for the rest, click the node and use
the Properties panel.

Some dialog titles differ from the node's name: **Tokenizer** opens **Text
Tokenizer Configuration**, **Embedding** opens **Embedding Configuration**
and **SQL Query** opens **SQL Workbench**.

## Change a setting in Properties

1. On the canvas, click the node. The **Properties** panel shows its name,
   type and settings, grouped as the node declares them. Rarely used ones
   are folded under **Advanced settings**.
2. Hover a setting's label for its full name, what it does, its default,
   and whether it is **Required** or **Optional**.
3. Change the value:
   - a number: type it, or use the **-** and **+** steps;
   - a checkbox: click it;
   - a choice or a column: pick from the list;
   - a file or folder: type the path, or click the folder button to browse.

   How each kind of field behaves is in
   [Editing values](../../main-window/editing-values.md).
4. Read the note under the row if one appears, such as "Value must be <=
   1" or "Required". A number with a range is kept inside its range as you
   type.

The value is written to the node only when you change it; looking at a node
changes nothing. **Compile** checks the new value; see
[Compile and run](compile-and-run.md).

## Change settings in a dialog

1. Double-click the node (or click **Open Dialog...** in Properties).
2. Set the fields on each tab.
3. Click **Apply** to save and keep the dialog open, or **OK** to save and
   close. **Cancel** puts back what the node had when the dialog opened.

For a Data Input, **Apply** also loads the data. Until it does, Properties
shows the node as **Not loaded**. Each dialog is described in
[Node dialogs](../../main-window/node-dialogs.md).

## Go back to the defaults

- A row whose value differs from its default shows a reset arrow. Hover it
  for the default; click it to go back.
- **Actions > Reset all settings to defaults** at the top of Properties
  resets every setting of the node.
- A new node starts with each setting's default. A setting you never
  changed is not stored; the default applies.

## Use a preset

1. Click **Actions** at the top of Properties.
2. Choose **Apply preset** and a preset, for example Dense **Small (64)**,
   **Medium (256)** or **Large (1024)**, or Adam **Default**, **Fast
   learning** or **Fine-tuning**. Hover a preset to see its values.

To keep your own, choose **Actions > Save current as preset...** and give
it a name. It is offered for every node of that type, in every project.

## Settings that are used, and settings only an editor reads

Almost every setting is read when the graph runs: by the compiler, the data
loader, the data steps, training or export. A few settings only change how
an editor or view shows the node and do not change results. Properties does
not mark which is which today. The status mark at the right of a row is the
nearest hint: hover it for **Owner**, which names the part that reads the
setting (for example Compiler, Loader or Runtime, or UI for a setting
only an editor reads).

## Nodes that cannot run yet

A node marked **Planned** in the Properties header (or **Coming soon** in the
Nodes panel) is in the list of nodes the Engine cannot run in this version.
A saved graph with such a node still opens and says why the node cannot
run. Replace it with a node that runs. See
[Find and add nodes](find-and-add-nodes.md#nodes-you-cannot-add).

## Undo and settings

Changing a setting is not an undo step of its own. **Ctrl+Z** on the canvas
goes back to the graph before the last canvas change (adding, moving,
linking or deleting), and that also takes back every setting you changed
since. After an undo, check the settings you edited.

## Related

- [The Properties panel](../../main-window/properties.md): every part of
  the panel, the status marks and the compiled card.
- [Node dialogs](../../main-window/node-dialogs.md)
- [Editing values](../../main-window/editing-values.md)
