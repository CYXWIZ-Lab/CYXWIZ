---
title: Move around the canvas
description: Pan, zoom, select, delete and undo on the CyxWiz Studio canvas, and start a graph from an empty one.
type: how-to
area: graph-editor
audience: user
status: stable
since: 1.0.7
owner: graph-editor
last_reviewed: 2026-11-24
related:
  - find-and-add-nodes.md
  - connect-nodes.md
  - groups-frames-and-names.md
  - ../../reference/shortcuts.md
---

The canvas is the **CyxWiz Studio** window: the grid where you place nodes
and link them into a graph. Its tab names the open graph ("Untitled" when
there is none) and shows a dot while it has unsaved changes.

Most canvas keys work only while the canvas has focus. Click an empty part
of the canvas first if a key seems to do nothing.

## Start from an empty canvas

An empty canvas shows **Start a graph** with three buttons:

- **Search nodes**: puts the cursor in the **Search nodes...** box at the
  top right of the canvas. Type a name and press Enter.
- **Open the Nodes panel**: shows the **Nodes** panel. Drag a node from it
  onto the canvas.
- **Open an example graph**: lists five starter graphs: **Binary image
  classification**, **Multiclass image classification**, **Text
  classification**, **Audio classification** and **Time-series
  forecasting**. Each trains at once on a small sample in
  `examples/datasets`, offline. The graph opens untitled, so **Save** asks
  for a name.

The example button appears only when the Engine finds the `examples/cyxgraph`
folder, which release packages and source checkouts both include. If you
started the Engine from somewhere else, the hint still mentions example
graphs but the button is missing; open an example with **File** or the
start page instead.

See [Find and add nodes](find-and-add-nodes.md) for every other way to add
a node.

## Pan and zoom

| To | Do this |
| --- | --- |
| Pan | Left-drag on an empty part of the canvas. |
| Zoom toward the pointer | Turn the mouse wheel. |
| Zoom around the centre | Press **+** or **-** (the keypad keys work too). |
| Zoom to a set level | Studio bar **View** menu: the **-** and **+** buttons beside **Zoom**, or **Zoom to 100%**. |
| See every node | Press **F** with nothing selected, or **View > Fit all nodes** on the Studio bar. |
| See the selected nodes | Press **F** with nodes selected. |
| Overview | Press **M** for the minimap, or **View > Minimaps > Studio Minimap**. Click the minimap to move the view; right-click it to move it to a corner or hide it. |

Zoom runs from 25% to 200%. The Studio bar's **View** button shows the
current level.

Left-drag on empty canvas always pans. There is no box (rubber-band)
selection: use Ctrl+click or **Ctrl+A** instead.

## Select nodes

- Click a node to select it. The **Properties** panel shows its settings.
- **Ctrl+click** a node to add it to the selection or take it out.
- **Ctrl+A** selects every node.
- **Esc** closes the find bar or an open menu first; pressed again, it
  clears the selection.

Hover a node for its description, its settings and, during or after a run,
its run state (Executing..., Completed, Error or Pending...).

## Find a node

1. Press **Ctrl+F**, or choose **View > Find in the graph** on the Studio
   bar. A bar opens in the canvas's top-left corner.
2. Type part of a node's name, its type, or a setting's name or value
   (`dense`, `relu`, `0.001`). Every match gets a ring; the view centres on
   the current one.
3. Press **Enter** or **F3** for the next match, **Shift+Enter** or
   **Shift+F3** for the previous one, **Esc** to close the bar.

## Move, delete and duplicate

- Drag a node by its title to move it.
- Press **Delete** to remove the selected nodes and their links, or
  right-click a node and choose **Delete**. **Delete** does not remove a
  link; see [Connect nodes](connect-nodes.md#remove-a-link).
- **Ctrl+D** duplicates the selection; **Ctrl+C**, **Ctrl+X** and
  **Ctrl+V** copy, cut and paste it. The copies land at a fixed spot near
  the top-left of the graph area, not beside the originals; press **F** if
  you cannot see them.
- Right-click empty canvas and choose **Actions > Align / Distribute** to
  line up two or more selected nodes, or **Auto Layout (Grid)** to tidy
  them.
- To empty the canvas, click the graph name on the Studio bar and choose
  **Clear the canvas...**. The question names how many nodes and links go;
  **Ctrl+Z** brings them back.

## Undo and redo

- **Ctrl+Z** undoes, **Ctrl+Y** or **Ctrl+Shift+Z** redoes. The canvas keeps
  the last 50 steps.
- Undo covers adding, moving and deleting nodes, links, subgraphs and notes.
- Changing a setting in **Properties** or a node dialog is not an undo step
  of its own. **Ctrl+Z** goes back to the graph as it was before the last
  canvas change, so it also takes back every setting you changed since
  then. Check your settings after an undo.
- Groups and frames are not part of undo; see
  [Organise a graph](groups-frames-and-names.md).

## Save the graph

With the canvas focused, **Ctrl+S** saves the graph as a `.cyxgraph` file
and **Ctrl+Shift+S** saves it under a new name. On an empty canvas the
Engine says **Nothing to save yet**. **File > Save** in the menu bar saves
the graph together with your scripts, layout and project settings.

## Organise a large graph

Groups, frames, notes and node names keep a big graph readable. See
[Organise a graph with groups, frames, notes and names](groups-frames-and-names.md).

## Changes

- Changed in 1.0.8: the empty canvas shows **Start a graph** with its three
  buttons.

## Related

- [Find and add nodes](find-and-add-nodes.md)
- [Connect nodes](connect-nodes.md)
- [Keyboard shortcuts](../../reference/shortcuts.md)
