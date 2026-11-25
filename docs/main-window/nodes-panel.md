---
title: Nodes panel
description: The panel that lists every node by group, with search, filters, favourites, recent nodes and Studio Tools.
type: reference
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - ../user-guide/graph-editor/find-and-add-nodes.md
  - properties.md
  - ../user-guide/graph-editor/groups-frames-and-names.md
redirect_from:
  - nodes-panel.md
---

The **Nodes** panel lists every node you can put on the CyxWiz Studio
canvas, by group, with search and filters. It is a tab beside the Asset
Browser in the default layout. Open it from the sidebar, **View > Panels >
Workspace > Nodes**, or **Open the Nodes panel** on an empty canvas.

For every way to add a node, including the canvas search and the
right-click menu, see [Find and add nodes](../user-guide/graph-editor/find-and-add-nodes.md).

## Layout, from the top

| Part | What it does |
| --- | --- |
| **Search nodes (/)** | Lists nodes whose name holds your words first (the whole name, then the start of a name, then the rest), then under **Related** the nodes that only mention them in a keyword or description. **/** jumps here while the panel has focus; **Esc** or the **x** clears it. (**Ctrl+F** is the canvas's Find.) |
| Filter | **All** (every node that runs, grouped), **Trains a model** (layers, losses, optimizers), **Data steps** (loading, preparing, exporting), **Views and panels** (dashboards, plots, notes), **Can't run yet** (previews and blocked, with why), **Favourites**. Beside it the count: "250 nodes" (folds at the end not counted) or "19 found" while searching. With a filter other than **All**, every group that has a node is open. |
| Grid or list | Cards with icons (as many columns as the width holds, names on up to two lines), or one row per node. **File > Preferences > Nodes panel > View** sets how a session starts. |
| **Studio Tools** | **Note** (double-click, or drag onto the canvas), **Group** (double-click: groups the selected nodes, as **Ctrl+G**; off until nodes are selected) and **Frame** (drag onto the canvas). |
| **Favourites**, **Recent** | Chips; see below. |
| Groups | **I/O**, **Analytics**, **ML Layers** and the rest, each with its count. Click a group's name to open or fold it; the panel remembers which are open (I/O and Training on a first run). A group with nothing that runs is not listed. |
| Folded at the end | **Plugins**, **Advanced operations**, **Can't run yet**, **Examples** (below). |

## Clicks on a card or row

| Action | Result |
| --- | --- |
| Click | Shows the node in the **Info** panel. Does not add it. |
| Double-click | Adds the node at a free spot near the middle of the view; the view moves to show it. |
| Drag onto the canvas | Adds the node where you drop it. |
| Right-click | **Add to Canvas**, **Add to Favourites** / **Remove from Favourites**, **Show details**, **Copy Node Name**. |
| Hover | What the node does, its inputs and outputs (with what each carries) and where it runs: "Runs in training", "Runs as a data step", "Runs in training and as a data step", "Shows what reaches it", "Runs in part: ..." or "Can't run yet: ...". |
| Star | Adds the node to Favourites, or takes it off. |

A small dot on a card marks a node that runs only in part (amber), is
blocked (red) or whose support is not settled yet (grey). A **Coming soon**
strip marks a node that cannot run yet.

## Favourites and Recent

- **Favourites**: the nodes you starred, as chips, in the order you starred
  them. Right-click a chip for **Add to Canvas**, **Remove from
  Favourites** and **Show details**.
- **Recent**: the last nodes you added to the canvas from any place, newest
  first.

**One click on a chip adds the node** (unlike a card, where one click only
shows it in Info). You can also drag a chip onto the canvas.

Both are saved with your Engine settings, so they are the same in every
project. Plugin nodes are not kept in them. **File > Preferences > Nodes
panel** has **Show Recent**, **Show the last** N **nodes** (8 by default,
1 to 20) and **Clear Recent**.

## The folded groups at the end

- **Plugins**: nodes from loaded plugins and Python node bundles. Choosing
  a plugin under **Apps** opens this group narrowed to it; **Show all
  plugins** removes the narrowing, and **Open Plugin Manager** opens the
  Plugin Manager.
- **Advanced operations**: the table steps a Preparation Recipe is made of,
  by kind: *Rows* (Row Filter, Sorter, Duplicate Remover, Row Sampler,
  Missing Value, Table Cropper, Row to Column Names), *Columns* (Column
  Filter, Rename Columns, Column Appender, Cell Extractor, Cell Updater,
  String Manipulation, Math Formula, Rule Engine, Unpivot), *Text* (Text
  Clean, JSONPath, Regex Tester), *Combine* (GroupBy, Joiner, Row Appender)
  and *SQL and checks* (SQL Query, Check, Data Validator). Data Studio's
  Prepare adds these for you.
- **Can't run yet**: nodes the Engine cannot run in this version, marked
  **Coming soon** or **Blocked**. Hover one for the reason. A node leaves
  this list when it can run. A saved graph that has one still opens and
  says why it cannot run.
- **Examples**: small graphs to start from. Double-click one to open it in
  the Pattern Browser, where you set it up and insert it.

## The Info panel

The **Info** panel (menu name **Node Info**: **View > Panels > Workspace >
Node Info**) shows a node's description, support, ports with their types,
parameters and an example. Hover a card to preview a node there ("Preview -
click the node to keep it here"); click a card, or a node on the canvas, to
keep it. A canvas node that is removed leaves the Info panel.

The Info panel lists parameters by their stored key (for example `units`),
while Properties shows the readable name ("Units").

## Limits

- The Nodes panel search, the canvas **Search nodes...** box and the
  right-click menu's search match differently, so the same word can give
  different lists.
- No key opens the canvas search box; **/** works only in this panel.

## Changes

- Changed in 1.0.8: name-first search with a **Related** list, and the
  filter menu.

## Related

- [Find and add nodes](../user-guide/graph-editor/find-and-add-nodes.md)
- [Properties panel](properties.md)
- [Organise a graph with groups, frames, notes and names](../user-guide/graph-editor/groups-frames-and-names.md)
