---
title: Find and add nodes
description: Every way to find a node and put it on the canvas, and what to do when a node cannot be added.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - ../../main-window/nodes-panel.md
  - canvas.md
  - connect-nodes.md
  - node-properties.md
---

Use this when you know what a step of your model or data preparation
should do and want the node for it on the canvas.

## The ways to add a node

| Way | How | Where the node lands |
| --- | --- | --- |
| **Nodes** panel | Double-click a card or row, or right-click it and choose **Add to Canvas**. | A free spot near the middle of the view; the view moves to it. |
| **Nodes** panel, drag | Drag a card or row onto the canvas. | Where you drop it. |
| **Favourites** and **Recent** chips | Click a chip in the Nodes panel (one click), or drag it. | A free spot near the middle of the view, or the drop point. |
| Canvas **Search nodes...** box | Click the box at the top right of the canvas, type, then press Enter or click a result. | A free spot near the middle of the view. |
| Right-click menu | Right-click empty canvas, then **Add Node** and a group, or type in its **Search nodes...** field. | Where you right-clicked. |
| **Nodes > Add Layer** | In the menu bar: **Dense / Linear**, **Convolutional**, **Pooling**, **Dropout**, **Batch Normalization** or **Attention**. | A free spot near the middle of the view. |
| **Pattern Browser** | **View > Panels > Workspace > Patterns** (**Ctrl+Shift+P**), pick a pattern, then **Insert full graph** or **Insert block**, then **Replace the canvas** or **Add beside it**. | Beside the graph, or in place of it. |
| Example graph | On an empty canvas, **Open an example graph**. | Opens a whole graph. |
| From your data | **Asset Browser**: right-click a data file and choose **Create Data Input**. | A new Data Input set to that file. |
| From a dialog or the compile report | Buttons such as **Add a Resize** or **Add a Text Tokenizer**. | After the node that needs it, already linked. |

Plugin nodes and your own Python nodes appear in the same places: the
**Plugins** group of the Nodes panel, the canvas search and the right-click
menu.

**Nodes > Add Layer** adds a preset named after its value, for example
"Dense (128)" or "Dropout (0.5)". The name keeps that number after you
change the value in Properties; rename the node if that confuses you.

**Create Data Input** in the Asset Browser only sets the file. It does not
load the data: open the node's dialog and click **Apply**. See
[Node dialogs](../../main-window/node-dialogs.md).

## Find a node by name

There are three search boxes, and each matches a little differently:

- **Nodes** panel, **Search nodes (/)**: names that contain your words
  first, then a **Related** list of nodes that only mention them in a
  keyword or description. Press **/** while the panel has focus to jump to
  the box, **Esc** to clear it.
- Canvas box, **Search nodes...**: a loose match on the name, then the
  group and keywords; the best 15 results. Use the Up and Down arrows and
  Enter. Matches that cannot run are counted at the end ("2 more can't run
  yet") with **Show**.
- Right-click menu, **Search nodes...**: plain text match on the name,
  group and keywords.

No keyboard shortcut opens the canvas search box; click it. The box
updates its list a moment after you stop typing. If you type quickly and
press Enter at once, it can add the top match of what you typed before:
wait for the list to show your word, then press Enter.

## Learn what a node does before adding it

- Hover a card in the Nodes panel for its description, its inputs and
  outputs, and where it runs ("Runs in training", "Runs as a data step",
  "Can't run yet: ...").
- Click a card once to show every detail in the **Info** panel (**View >
  Panels > Workspace > Node Info**). Click does not add the node; double-click
  does.
- Hover a node already on the canvas for its description and settings.

## Nodes you cannot add

Some nodes are listed but cannot be added:

- **Can't run yet**: nodes the Engine cannot run in this version, marked
  **Coming soon** or **Blocked**. In the Nodes panel and the right-click menu
  they sit in a folded **Can't run yet** group; in the canvas search they
  show **[Coming Soon]** or **[Blocked]** with the reason. Hover one for
  why. A graph saved with such a node still opens and says why it cannot
  run.
- **Subgraph** is made from a selection (right-click canvas, **Actions >
  Create Subgraph**, **Ctrl+Shift+U**), not added from a list.
- Deprecated nodes are hidden.

The same state has several names: **Planned** in the Properties header,
**Coming soon** on cards, **[Coming Soon]** in search and **Can't run yet**
as the group. They all mean the node cannot run yet.

If you drop a blocked node onto the canvas, nothing is added and the only
message is a line in **Console > Logs**.

## Advanced operations

**Advanced operations** holds the table steps a Preparation Recipe is made
of: Rows, Columns, Text, Combine, and SQL and checks. Data Studio's
**Prepare** adds them for you; add one here to build a data step by hand.
It is a folded group at the end of the Nodes panel and a submenu of the
right-click menu; the canvas search shows at most five of them, marked
"Advanced operation · Rows" and so on.

## Related

- [The Nodes panel](../../main-window/nodes-panel.md)
- [Connect nodes](connect-nodes.md)
- [Set node properties](node-properties.md)
